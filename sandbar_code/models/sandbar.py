"""SandBAR network.

This file contains the architecture used by the paper:

* SandBAR's encoder-decoder transformer backbone.
* The band-asymmetric sandwich block.
* SPARK, implemented as ``FreqEnhFFN`` in the post-attention FFN slot.
* Low-frequency guidance through ``LowFreqExtractor`` + ``SCM`` + ``FAM``.
* Three output heads used by the multi-scale spatial-frequency objective.

In the CLI, ``post_ffn=freqenh`` selects SPARK and ``post_ffn=gated``
selects the deployed scaffold-free graph left by PaST strip-and-fold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from contextlib import contextmanager, nullcontext
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .sfnet_components import dynamic_filter, SFconv


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Restormer-style channel LayerNorm: normalises across the channel
    dimension at each (h, w) position."""

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Low-Frequency Extractor — depthwise Gaussian blur, outputs 3ch low-freq RGB
class LowFreqExtractor(nn.Module):
    def __init__(self, kernel_size=31, init_sigma=10.0):
        super(LowFreqExtractor, self).__init__()
        self.blur = nn.Conv2d(3, 3, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2,
                              groups=3, bias=False)
        with torch.no_grad():
            k = kernel_size
            ax = torch.arange(k, dtype=torch.float32) - k // 2
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * init_sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            kernel = gaussian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            self.blur.weight.copy_(kernel)

    def forward(self, x):
        return self.blur(x)


##########################################################################
## SCM — Shallow CNN Module (encodes 3ch low-freq into feature-dim representation)
## Adapted from SFNet
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, out_plane // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_plane // 2, out_plane, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        return self.main(x)


##########################################################################
## FAM — Feature Attention Module (fuses two feature streams)
## Adapted from SFNet
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU()
        )

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


##########################################################################
## Feed-Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Frequency-Decomposed FFN variants
class SqueezeExcite(nn.Module):
    """Lightweight channel attention used inside FreqEnhFFN's low-band branch."""

    def __init__(self, dim, reduction=8, bias=False):
        super(SqueezeExcite, self).__init__()
        hidden = max(dim // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class FreqEnhFFN(nn.Module):
    """Frequency-enhanced FFN: decompose -> enhance both bands -> SFconv fuse -> GDFN.

    Both bands pass learnable enhancement, are recombined by SFconv
    (channel-attention weighted sum + 1x1 projection), and the fused feature
    feeds a single unmodified GDFN.
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FreqEnhFFN, self).__init__()
        self.freq_split = dynamic_filter(inchannels=dim)
        self.low_enh = nn.Sequential(
            SqueezeExcite(dim, reduction=8, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        self.high_enh = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        self.fuse = SFconv(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        low, high = self.freq_split.decompose(x)
        low = self.low_enh(low)
        high = self.high_enh(high)
        x_fused = self.fuse(low, high) + x
        return self.ffn(x_fused)


class FreqEnhGaussFFN(nn.Module):
    """SPARK with the content-adaptive splitter replaced by a FIXED Gaussian
    low-pass -- the splitter-provenance control.

    Ablation (b) of Table 4. Everything downstream of the split --
    asymmetric branches, SFconv fusion, residual, GDFN -- is identical to
    FreqEnhFFN, so the comparison isolates the content-adaptive splitter. The
    fixed split removes the dynamic filter's kernel-predictor (conv+bn, ~2.4K
    params per module); everything else matches.
    """

    def __init__(self, dim, ffn_expansion_factor, bias, kernel_size=3, sigma=0.8):
        super(FreqEnhGaussFFN, self).__init__()
        k = kernel_size
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).expand(dim, 1, k, k).clone()
        self.register_buffer('gauss_kernel', g)
        self.pad = nn.ReflectionPad2d(k // 2)
        self.low_enh = nn.Sequential(
            SqueezeExcite(dim, reduction=8, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        self.high_enh = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )
        self.fuse = SFconv(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        low = F.conv2d(self.pad(x), self.gauss_kernel, groups=x.shape[1])
        high = x - low
        low = self.low_enh(low)
        high = self.high_enh(high)
        x_fused = self.fuse(low, high) + x
        return self.ffn(x_fused)


class FreqEnhSymFFN(nn.Module):
    """SPARK with SYMMETRIC per-band branches -- the asymmetry control.

    Both bands receive the low-band treatment (squeeze-excite + depthwise 5x5
    + pointwise). This control has MORE parameters than FreqEnhFFN (the high
    branch gains an SE block and a larger kernel), so if it underperforms,
    the asymmetric design is validated against a strictly stronger-capacity
    symmetric alternative -- the clean direction for the claim.
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FreqEnhSymFFN, self).__init__()
        self.freq_split = dynamic_filter(inchannels=dim)

        def branch():
            return nn.Sequential(
                SqueezeExcite(dim, reduction=8, bias=bias),
                nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias),
                nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            )
        self.low_enh = branch()
        self.high_enh = branch()
        self.fuse = SFconv(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        low, high = self.freq_split.decompose(x)
        low = self.low_enh(low)
        high = self.high_enh(high)
        x_fused = self.fuse(low, high) + x
        return self.ffn(x_fused)


class DFFN(nn.Module):
    """Discriminative frequency-domain FFN from FFTformer (Kong et al., CVPR
    2023), ported verbatim from the official repository
    (basicsr/models/archs/fftformer_arch.py): per-patch rFFT with a learnable
    per-channel spectral filter inside a gated FFN. Prior-art control for the
    post-attention slot. UIE patch: reflection-pad H/W to multiples of
    patch_size before the per-patch FFT and crop back afterwards so arbitrary
    evaluation resolutions work; at the 256-crop training size it is a no-op.
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size,
                                            self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                      patch1=self.patch_size, patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x


def build_ffn(dim, ffn_expansion_factor, bias, ffn_type):
    """Build the module that occupies a feed-forward slot.

    ``gated``         GDFN -- the permanent, scaffold-free slot content.
    ``freqenh``       SPARK -- the removable band-asymmetric scaffold.
    ``freqenh_gauss`` SPARK with a fixed Gaussian split (ablation (b)).
    ``freqenh_sym``   SPARK with symmetric band operators (ablation (c)).
    ``dffn``          DFFN, the second module in the placement study.
    """
    ffn_type = ffn_type.lower()
    if ffn_type == 'gated':
        return FeedForward(dim, ffn_expansion_factor, bias)
    if ffn_type == 'freqenh':
        return FreqEnhFFN(dim, ffn_expansion_factor, bias)
    if ffn_type == 'freqenh_gauss':
        return FreqEnhGaussFFN(dim, ffn_expansion_factor, bias)
    if ffn_type == 'freqenh_sym':
        return FreqEnhSymFFN(dim, ffn_expansion_factor, bias)
    if ffn_type == 'dffn':
        return DFFN(dim, ffn_expansion_factor, bias)
    raise ValueError(
        f"Unknown ffn_type={ffn_type!r}; expected gated, freqenh, "
        "freqenh_gauss, freqenh_sym or dffn"
    )


@contextmanager
def preserve_bn_running_stats(module):
    """Keep BatchNorm running stats single-updated under activation checkpointing.

    PyTorch checkpoint recomputes forward during backward. If the checkpointed
    module contains BatchNorm, that recompute would otherwise update running
    statistics twice. We preserve stats only during recompute while keeping
    train-mode batch statistics for identical forward values.
    """
    states = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.track_running_stats:
            states.append((
                child,
                child.running_mean.detach().clone(),
                child.running_var.detach().clone(),
                child.num_batches_tracked.detach().clone(),
            ))
    try:
        yield
    finally:
        for child, running_mean, running_var, num_batches_tracked in states:
            child.running_mean.copy_(running_mean)
            child.running_var.copy_(running_var)
            child.num_batches_tracked.copy_(num_batches_tracked)


def run_with_optional_checkpoint(module, x, use_checkpoint, *extra):
    if not use_checkpoint or not x.requires_grad:
        return module(x, *extra)

    def forward_fn(inp, *e):
        return module(inp, *e)

    try:
        return checkpoint(
            forward_fn,
            x,
            *extra,
            use_reentrant=False,
            context_fn=lambda: (nullcontext(), preserve_bn_running_stats(module)),
        )
    except TypeError:
        return checkpoint(forward_fn, x, *extra)


##########################################################################
## Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
                                     groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.upsample(y)
        out = y * out
        out = self.project_out(out)
        return out


##########################################################################
## Sandwich Block: FFN -> Attention -> FFN
class SandwichBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias,
                 pre_ffn_type='gated', post_ffn_type='gated', use_checkpoint=False):
        super(SandwichBlock, self).__init__()
        self.norm1_1 = LayerNorm(dim)
        self.ffn1 = build_ffn(dim, ffn_expansion_factor, bias, pre_ffn_type)
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = build_ffn(dim, ffn_expansion_factor, bias, post_ffn_type)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        checkpoint_enabled = self.use_checkpoint and self.training
        x = x + run_with_optional_checkpoint(self.ffn1, self.norm1_1(x), checkpoint_enabled)
        attn_update = run_with_optional_checkpoint(self.attn, self.norm1(x), checkpoint_enabled)
        x = x + attn_update
        x = x + run_with_optional_checkpoint(self.ffn, self.norm2(x), checkpoint_enabled)
        return x


##########################################################################
## Gated Embedding Layer
class GatedEmb(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(GatedEmb, self).__init__()
        self.gproj1 = nn.Conv2d(in_c, embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.gproj1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return x


##########################################################################
## Downsample — Conv + PixelUnshuffle
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## Upsample — Conv + PixelShuffle
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## SandBAR: encoder-decoder backbone with low-frequency guidance
class SandBAR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 lf_kernel_size=31,
                 lf_init_sigma=10.0,
                 pre_ffn_type='gated',
                 post_ffn_type='gated',
                 use_checkpoint=False,
                 use_lf_guidance=True,
                 ):
        super(SandBAR, self).__init__()

        # Low-frequency guidance path. use_lf_guidance=False ablates the whole
        # path (extractor + SCMs + FAMs) for ablation (d) of Table 4.
        self.use_lf_guidance = use_lf_guidance
        if use_lf_guidance:
            self.lf_extractor = LowFreqExtractor(kernel_size=lf_kernel_size, init_sigma=lf_init_sigma)
            self.scm2 = SCM(int(dim * 2))    # outputs dim*2 channels at 1/2x
            self.scm1 = SCM(int(dim * 4))    # outputs dim*4 channels at 1/4x
            self.fam2 = FAM(int(dim * 2))    # fuses at 1/2x scale
            self.fam1 = FAM(int(dim * 4))    # fuses at 1/4x scale
        else:
            self.lf_extractor = self.scm2 = self.scm1 = self.fam2 = self.fam1 = None

        # Patch embedding (RGB only, 3 channels)
        self.patch_embed = GatedEmb(inp_channels, dim)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 4))
        self.latent = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[3])])

        # Decoder
        self.up4_3 = Upsample(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            SandwichBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                          bias=bias, pre_ffn_type=pre_ffn_type,
                          post_ffn_type=post_ffn_type, use_checkpoint=use_checkpoint)
            for _ in range(num_blocks[0])])

        # Multi-scale output heads
        self.output_level3 = nn.Conv2d(int(dim * 4), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output_level2 = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Multi-scale input for residual connections
        inp_2x = F.interpolate(inp_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        inp_4x = F.interpolate(inp_img, scale_factor=0.25, mode='bilinear', align_corners=False)

        # Extract low-frequency RGB and encode at each scale (ablatable)
        if self.use_lf_guidance:
            lf = self.lf_extractor(inp_img)              # [B, 3, H, W]
            lf_2x = F.interpolate(lf, scale_factor=0.5, mode='bilinear', align_corners=False)
            lf_4x = F.interpolate(lf, scale_factor=0.25, mode='bilinear', align_corners=False)
            z2 = self.scm2(lf_2x)                        # [B, dim*2, H/2, W/2]
            z4 = self.scm1(lf_4x)                        # [B, dim*4, H/4, W/4]

        # Encoder Level 1 (full resolution)
        inp_enc_level1 = self.patch_embed(inp_img)    # [B, dim, H, W]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # Encoder Level 2 (1/2x) — downsample then fuse with SCM features
        inp_enc_level2 = self.down1_2(out_enc_level1) # [B, dim*2, H/2, W/2]
        if self.use_lf_guidance:
            inp_enc_level2 = self.fam2(inp_enc_level2, z2) # fuse low-freq features
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # Encoder Level 3 (1/4x) — downsample then fuse with SCM features
        inp_enc_level3 = self.down2_3(out_enc_level2) # [B, dim*4, H/4, W/4]
        if self.use_lf_guidance:
            inp_enc_level3 = self.fam1(inp_enc_level3, z4) # fuse low-freq features
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Bottleneck (1/8x)
        inp_enc_level4 = self.down3_4(out_enc_level3) # [B, dim*8, H/8, W/8]
        latent = self.latent(inp_enc_level4)

        # Decoder with multi-scale outputs
        outputs = []

        # Decode Level 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        outputs.append(self.output_level3(out_dec_level3) + inp_4x)

        # Decode Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        outputs.append(self.output_level2(out_dec_level2) + inp_2x)

        # Decode Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        outputs.append(self.output(out_dec_level1) + inp_img)

        return outputs


def build_net(dim=32, num_blocks=None, pre_ffn_type='gated', post_ffn_type='gated',
              ffn_expansion_factor=2.66, use_checkpoint=False, use_lf_guidance=True):
    """Build the SandBAR network with the paper configuration:
    dim=32, blocks (2, 4, 4, 6), heads [1, 2, 4, 8], FFN expansion 2.66.
    """
    if num_blocks is None:
        num_blocks = [2, 4, 4, 6]
    return SandBAR(dim=dim, num_blocks=num_blocks, pre_ffn_type=pre_ffn_type,
                   post_ffn_type=post_ffn_type, ffn_expansion_factor=ffn_expansion_factor,
                   use_checkpoint=use_checkpoint, use_lf_guidance=use_lf_guidance)

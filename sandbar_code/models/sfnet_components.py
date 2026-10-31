"""SFNet components verbatim adapted from:
  Cui, Y., Tao, Y., Bing, Z., Ren, W., Gao, X., Cao, X., Huang, K., Knoll, A.
  "Selective Frequency Network for Image Restoration", ICLR 2023.
  Source: SFNet/Motion_deblurring/models/layers.py (https://github.com/c-yn/SFNet)

Two classes are reproduced here unchanged:
  * SFconv          - channel-attention recombination of low/high feature streams
  * dynamic_filter  - content-aware low-pass filter; decomposes feature into
                      (low_part, out_high) where out_high = x - low_part

The only addition is `dynamic_filter.decompose(x)`, which returns the
intermediate (low_part, out_high) without applying SFconv recombination, so
SPARK can route each band through its own operator. The original `forward()`
is preserved unchanged from SFNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFconv(nn.Module):
    """SFconv - Selective Frequency channel-attention recombination of two
    feature streams (low/high frequency). Verbatim from SFNet layers.py:194-238.

    The original SFconv selects between AdaptiveAvgPool2d and a fixed-size
    AvgPool2d. SandBAR evaluates at native resolution, so only the adaptive
    branch is implemented here.
    """

    def __init__(self, features, M=2, r=2, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.features = features
        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for _ in range(M):
            self.fcs.append(nn.Conv2d(d, features, 1, 1, 0))
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)
        fea_z = self.fc(emerge)
        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)
        fea_high = high * high_att
        fea_low = low * low_att
        out = self.out(fea_high + fea_low)
        return out


class dynamic_filter(nn.Module):
    """Dynamic content-aware low-pass filter. Verbatim from SFNet
    layers.py:152-190 (Cui et al., ICLR 2023).

    Forward: applies a learnable softmax-normalized K x K kernel (computed
    per-image from GAP(x)) to extract a low-pass response, then SFconv-
    recombines low + high streams.

    Added: `decompose(x)` method exposes the intermediate (low_part, out_high)
    used inside forward(), so SPARK can route each band separately.
    """

    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels)

    def decompose(self, x):
        """Return the (low_part, out_high) decomposition produced inside
        forward() BEFORE SFconv recombination.

        Both tensors share the same shape as the input `x`. By construction,
        `low_part + out_high == x` (out_high = x - low_part).
        """
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            n, self.group, c // self.group, self.kernel_size ** 2, h * w
        )

        n_, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n_, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x_unfold * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high

    def forward(self, x):
        # Verbatim SFNet behavior: decompose then SFconv-recombine.
        low_part, out_high = self.decompose(x)
        out = self.modulate(low_part, out_high)
        return out

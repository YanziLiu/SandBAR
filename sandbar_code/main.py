"""SandBAR training and evaluation entry point.

The default architecture is the scaffold-free permanent graph. Use
``--post_ffn freqenh`` during training to enable SPARK in the post-attention
slot. Use ``tools/export_deployed.py`` to strip SPARK and fold its residual
bias for PaST deployment.

Test mode uses the repository's evaluate.py and the public Dataset layout.
"""

import os
import random
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.backends import cudnn
from models.sandbar import build_net
from train import _train
from eval import _eval

def parse_blocks(blocks):
    values = [int(x.strip()) for x in blocks.split(',') if x.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError('--model_blocks must contain 4 comma-separated integers')
    return values


def main(args):
    if args.mode == 'test':
        return _eval(args)

    # Reproducibility: fix all RNG seeds. cudnn.benchmark stays True (SFNet style),
    # so this gives "same seed, similar results" rather than strict bit-equivalence.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    blocks = parse_blocks(args.model_blocks)
    model = build_net(
        dim=args.model_dim,
        num_blocks=blocks,
        pre_ffn_type=args.pre_ffn,
        post_ffn_type=args.post_ffn,
        ffn_expansion_factor=args.ffn_expansion_factor,
        use_checkpoint=args.use_checkpoint,
        use_lf_guidance=not args.no_lf_guidance,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print('Model config: dim=%d blocks=%s pre_ffn=%s post_ffn=%s ffn_expansion=%.4f checkpoint=%s' % (
        args.model_dim, blocks, args.pre_ffn, args.post_ffn, args.ffn_expansion_factor, args.use_checkpoint))
    print('Parameter count: %.3f M' % (param_count / 1e6))

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='SandBAR', type=str)
    parser.add_argument('--data_dir', type=str, default='dataset/Dataset',
                        help='Dataset root containing train/ and testset(ref)/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Model / ablation controls
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--model_blocks', type=str, default='2,4,4,6')
    # gated = GDFN (permanent graph); freqenh = SPARK (training scaffold);
    # freqenh_gauss / freqenh_sym = Table 5 ablations (b) / (c);
    # dffn = the second module in the Table 4 placement study.
    ffn_choices = ['gated', 'freqenh', 'freqenh_gauss', 'freqenh_sym', 'dffn']
    parser.add_argument('--pre_ffn', default='gated', choices=ffn_choices, type=str)
    parser.add_argument('--post_ffn', default='gated', choices=ffn_choices, type=str)
    parser.add_argument('--ffn_expansion_factor', type=float, default=2.66)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.add_argument('--no_lf_guidance', action='store_true',
                        help='ablate the low-frequency guidance path (extractor + SCM + FAM)')

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    # Test
    parser.add_argument('--test_model', type=str,
                        default=str(Path(__file__).resolve().parents[1] / 'weights/sandbar_checkpoint.pkl'))
    parser.add_argument('--save_image', action='store_true',
                        help='write restored PNGs to --output_root/results/<model_name>/<domain>')
    parser.add_argument('--output_root', type=str, default='')

    args = parser.parse_args()
    if args.output_root:
        args.output_root = os.path.abspath(args.output_root)
        args.model_save_dir = os.path.join(args.output_root, 'results', args.model_name, 'Training-Results')
        args.result_dir = os.path.join(args.output_root, 'results', args.model_name)
    else:
        args.model_save_dir = os.path.join('results', args.model_name, 'Training-Results')
        args.result_dir = os.path.join('results', args.model_name)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    print(args)
    main(args)

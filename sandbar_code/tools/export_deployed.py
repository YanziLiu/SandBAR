"""Export the PaST deployed graph.

SandBAR is trained with SPARK in the post-attention FFN slot
(`--post_ffn freqenh`). For deployment, PaST removes the SPARK apparatus and
folds the constant residual bias from `fuse.out.bias` into the preceding
post-slot LayerNorm bias. The exported checkpoint loads into the permanent
scaffold-free graph (`--post_ffn gated`) with the same backbone settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sandbar import build_net  # noqa: E402


APPARATUS_SEGS = (
    ".freq_split.",
    ".low_enh.",
    ".high_enh.",
    ".branch_a.",
    ".branch_b.",
    ".fuse.",
    ".gauss_kernel",
)


def parse_blocks(blocks: str) -> list[int]:
    values = [int(x.strip()) for x in blocks.split(",") if x.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--model_blocks must contain 4 integers")
    return values


def load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return state


def strip_and_fold(src: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    deployed: dict[str, torch.Tensor] = {}
    for key, val in src.items():
        if ".ffn." in key and any(seg in key for seg in APPARATUS_SEGS):
            continue
        deployed[key.replace(".ffn.ffn.", ".ffn.")] = val.detach().clone()

    folded = 0
    for key, val in src.items():
        if key.endswith(".ffn.fuse.out.bias"):
            norm_key = f"{key[:-len('.ffn.fuse.out.bias')]}.norm2.body.bias"
            deployed[norm_key] = deployed[norm_key] + val.detach()
            folded += 1
    return deployed, folded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Training-time SandBAR checkpoint with post_ffn=freqenh")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output checkpoint for the deployed post_ffn=gated graph")
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--model_blocks", default="2,4,4,6")
    parser.add_argument("--ffn_expansion_factor", type=float, default=2.66)
    args = parser.parse_args()

    source = load_state(args.input)
    deployed, folded = strip_and_fold(source)

    model = build_net(
        dim=args.model_dim,
        num_blocks=parse_blocks(args.model_blocks),
        pre_ffn_type="gated",
        post_ffn_type="gated",
        ffn_expansion_factor=args.ffn_expansion_factor,
        use_checkpoint=False,
    )
    model.load_state_dict(deployed, strict=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": deployed, "past_export": "strip_and_fold"}, args.output)
    print(f"folded {folded} SPARK fuse biases")
    print(f"wrote {args.output}")
    print(f"deployed parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()


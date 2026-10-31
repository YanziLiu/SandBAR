"""Route test mode through the released-checkpoint evaluator."""

from pathlib import Path
import subprocess
import sys

import torch


def _eval(args):
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "evaluate.py"),
        "--data-root", args.data_dir,
        "--checkpoint", args.test_model,
        "--output", args.result_dir,
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    if args.save_image:
        command.append("--save-images")
    return subprocess.run(command, check=True)

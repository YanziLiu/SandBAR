"""Restore a single RGB image or a directory with the deployed SandBAR model."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from sandbar_code.models.sandbar import build_net

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHT = ROOT / "weights/sandbar_checkpoint.pkl"


def load_model(checkpoint, device):
    model = build_net()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def read_image(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


@torch.inference_mode()
def restore(model, image, device):
    height, width = image.shape[-2:]
    image = F.pad(image.to(device), (0, (-width) % 16, 0, (-height) % 16), "reflect")
    return model(image)[2][..., :height, :width].clamp(0, 1).cpu()


def save_image(prediction, path):
    rgb = prediction[0].permute(1, 2, 0).numpy()
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.device)
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        prediction = restore(model, read_image(path), args.device)
        destination = args.output / (path.stem + ".png")
        save_image(prediction, destination)
        print(destination, flush=True)


if __name__ == "__main__":
    main()

"""Evaluate the released checkpoint on the 490 paired benchmark images."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import features
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from inference import ROOT, DEFAULT_WEIGHT, load_model, read_image, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="The downloaded Dataset directory containing testset(ref)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if features.version_codec("jpg") != "9.0":
        raise RuntimeError("Paper evaluation requires IJG libjpeg 9. Use environment.yml; pip Pillow wheels use a different JPEG decoder.")
    torch.set_num_threads(8)
    model = load_model(args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    started = time.monotonic()
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "filename", "psnr", "ssim"])
        writer.writeheader()
        for domain in ["EUVP", "LSUI", "UIEB"]:
            names = (ROOT / "data/splits" / f"{domain}.txt").read_text().splitlines()
            base = args.data_root / "testset(ref)" / f"test-{domain}"
            target_dir = base / "target"
            scores = []
            for index, name in enumerate(names, 1):
                image = read_image(base / "input" / name)
                target = read_image(target_dir / name)[0].numpy()
                prediction = restore(model, image, args.device)[0].numpy()
                psnr = float(peak_signal_noise_ratio(target, prediction, data_range=1))
                ssim = float(structural_similarity(target, prediction, data_range=1, channel_axis=0))
                scores.append((psnr, ssim))
                writer.writerow(dict(domain=domain, filename=name, psnr=psnr, ssim=ssim))
                if index % 50 == 0 or index == len(names):
                    handle.flush()
                    print(f"{domain} {index}/{len(names)}", flush=True)
            mean = np.mean(scores, axis=0)
            summary[domain] = dict(images=len(names), psnr=float(mean[0]), ssim=float(mean[1]))
            print(f"{domain}: PSNR {mean[0]:.8f}, SSIM {mean[1]:.8f}", flush=True)
    summary["equal_domain_mean"] = {
        metric: float(np.mean([summary[domain][metric] for domain in ["EUVP", "LSUI", "UIEB"]]))
        for metric in ["psnr", "ssim"]}
    summary["parameters"] = sum(p.numel() for p in model.parameters())
    summary["seconds"] = time.monotonic() - started
    summary["torch"] = torch.__version__
    summary["jpeg_decoder"] = features.version_codec("jpg")
    summary["device"] = torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

# SandBAR: A Band-Asymmetric Architecture for Underwater Image Restoration with Removable Frequency Scaffolding (WACV 2027 Accepted)

Yanzi Liu, Haoran Hou<sup>†</sup>, Jiachi Tong<sup>†</sup>, Shen Chen<sup>†</sup>, and Shuang Chen<sup>*</sup>

<sup>†</sup> Haoran Hou, Jiachi Tong, and Shen Chen contributed equally. <sup>*</sup> Corresponding author.

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](papers/2026411507.pdf)
[![Supplementary](https://img.shields.io/badge/Supplementary-Material-red)](papers/01450_supp.pdf)
[![Models](https://img.shields.io/badge/Pretrained-Model-green)](#pretrained-model)

#### News

- SandBAR has been accepted to **WACV 2027**! 🎉

<hr />

> **Abstract:** *Reliable underwater imagery supports marine monitoring, ecological assessment and autonomous exploration, yet light scattering, absorption and turbidity jointly degrade image clarity, distort colour and suppress fine structure. Although these effects are physically interwoven, they commonly appear as complementary frequency tendencies: large-scale, channel-dependent appearance distortions and local high-frequency detail loss. We refer to this tendency as band asymmetry; it motivates different forms of restoration for global colour and local structure. Our primary contribution is SandBAR, a novel unified band-asymmetric architecture for underwater image restoration. Rather than treating frequency processing as an isolated module, SandBAR organises the restoration pipeline around band asymmetry at the block, encoder and objective levels. Within this architecture, we introduce a novel band-asymmetric sandwich block with a reserved post-attention repair slot and the Split-band Parallel Asymmetric Repair Kernel (SPARK), a novel module that applies channel-aware large-support correction to the low band, which predominantly carries large-scale appearance distortions, and lightweight local refinement to the detail-bearing high band, thereby restoring global colour while preserving fine structure. To deliver the global cast to the depth at which the network represents global structure, a learnable low-frequency guidance path injects a low-pass view of the input into the deep encoder, and a three-scale spatial-frequency objective supervises the same axis throughout the decoder. Building on the reserved post-attention slot, we further introduce Post-attention Scaffold Training (PaST), a novel removable nonlinear scaffolding scheme that trains the permanent SandBAR graph under band-specific capacity and runs it at inference without that capacity. SandBAR reaches 29.89 / 28.92 / 22.77 dB on EUVP / LSUI / UIEB and establishes a new three-domain mean state of the art on the benchmark family, exceeding the strongest published result by approximately 1.0 dB. PaST adds no parameters or FLOPs at inference and preserves matched latency.*

<hr />

## Network Architecture

![SandBAR architecture](assets/architecture.png)

The architecture combines the band-asymmetric sandwich block, SPARK, low-frequency guidance, and a three-scale spatial-frequency objective. PaST trains the model with the SPARK scaffold and removes its auxiliary branches for inference.

## Installation

See [INSTALL.md](INSTALL.md) for the evaluation environment and installation instructions.

## Demo

### Pretrained model

Download `SandBAR_code_and_checkpoint.zip` from this repository's **Releases** section. The archive contains the code and the pretrained deployment checkpoint at `SandBAR/weights/sandbar_checkpoint.pkl`.

If you cloned the repository or downloaded only the source code, copy the archive's `weights/` folder into the repository root. This checkpoint has already undergone PaST strip-and-fold and can be loaded directly for inference.

Restore a directory of images:

```bash
python inference.py --input /path/to/images --output outputs
```

Restore a single image:

```bash
python inference.py --input /path/to/image.jpg --output outputs
```

The restored images are saved as PNG files. Add `--device cpu` for CPU inference.

## Training and Evaluation

| Task | Training | Evaluation | Data |
|---|---|---|---|
| Underwater image restoration | [Instructions](#training) | [Instructions](#evaluation) | [Dataset and splits](#dataset) |

### Dataset

Download [UnderwaterBenchmarkDataset](https://www.kaggle.com/datasets/xuhangc/underwaterbenchmarkdataset), linked by the [GuidedHybSensUIR benchmark repository](https://github.com/CXH-Research/GuidedHybSensUIR). Keep its original directory structure:

```text
Dataset/
  train/input/
  train/target/
  testset(ref)/test-EUVP/input/
  testset(ref)/test-EUVP/target/
  testset(ref)/test-LSUI/input/
  testset(ref)/test-LSUI/target/
  testset(ref)/test-UIEB/input/
  testset(ref)/test-UIEB/target/
```

`data/splits/` contains the exact 4,800 training filenames and the 200 EUVP, 200 LSUI and 90 UIEB test filenames. The downloadable archive contains 5,600 training pairs; the supplied training loader uses the 4,800-name list, excluding the 800 duplicate UIEB copies present in the archive. Keep filenames and extension capitalization unchanged.


### Training

The original training implementation and its configuration are included:

```bash
DATA_DIR=/path/to/Dataset OUT_ROOT=outputs bash scripts/train_sandbar.sh
```

This enables SPARK with dimension 32, blocks [2,4,4,6], expansion 2.66, batch size 4, Adam at 1e-4, 3 warm-up epochs, cosine decay to 1e-6, and 650 training epochs. It uses 256×256 random crops, horizontal flips, three-scale L1 + 0.1 FFT supervision, and gradient clipping at 0.01. The release verification below evaluates the supplied pretrained checkpoint; it does not represent a new training run.

Export a training checkpoint to the deployed graph:

```bash
python sandbar_code/tools/export_deployed.py \
  --input /path/to/training_checkpoint.pkl \
  --output weights/my_deployed_checkpoint.pkl
```


### Evaluation

```bash
python evaluate.py --data-root /path/to/Dataset
```

The command writes per-image PSNR/SSIM to `results/evaluation/per_image.csv` and domain means to `summary.json`. It uses native-resolution RGB float32 inputs, right/bottom reflection padding to multiples of 16, the full-resolution output head, cropping to the original size, and clipping to [0,1]. PSNR and SSIM are calculated directly from floating-point predictions; saved PNG images do not enter this calculation. SSIM uses the scikit-image 7×7 uniform window with `data_range=1`. The overall PSNR is the equal-weight mean of the three domain means.


## Results

The released deployment checkpoint reproduces the paper's main PSNR results on EUVP, LSUI, and UIEB. The three-domain mean gives equal weight to each domain.

| Dataset | Test images | PSNR (dB) |
|---|---:|---:|
| EUVP | 200 | **29.89** |
| LSUI | 200 | **28.92** |
| UIEB | 90 | **22.77** |
| Three-domain mean | 490 in total | **27.19** |

The deployed model has **24,088,479 parameters**. PaST adds no parameters or FLOPs at inference.

<details>
<summary><strong>Detailed evaluation results</strong> (click to expand)</summary>

| Dataset | PSNR (dB) | SSIM |
|---|---:|---:|
| EUVP | 29.88560774 | 0.90964295 |
| LSUI | 28.92391658 | 0.92160711 |
| UIEB | 22.76943044 | 0.91354280 |
| Three-domain mean | 27.19298492 | 0.91493095 |

Per-image measurements and runtime details are included in [results/reproduced/](results/reproduced/).

</details>

<details>
<summary><strong>Visual comparison</strong> (click to expand)</summary>

![Visual comparison from the paper](assets/qualitative.png)

</details>

## Citation

If you use SandBAR in your research, please cite:

```bibtex
@inproceedings{liu2027sandbar,
  title={SandBAR: A Band-Asymmetric Architecture for Underwater Image Restoration with Removable Frequency Scaffolding},
  author={Liu, Yanzi and Hou, Haoran and Tong, Jiachi and Chen, Shen and Chen, Shuang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2027}
}
```


## Contact

For questions, please contact [Yanzi Liu](mailto:yanzi.liu@kcl.ac.uk) or [Shuang Chen](mailto:shuang.chen@durham.ac.uk).

**Acknowledgment:** The implementation includes components adapted from HINT and SFNet. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

The code is released under the [MIT License](LICENSE). Please obtain the benchmark from its original source and follow its data terms.

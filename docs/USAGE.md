# Usage

Run the commands below from the repository root. See [INSTALL.md](../INSTALL.md) for environment setup.

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

## Dataset

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

## Training

The original training implementation and its configuration are included:

```bash
DATA_DIR=/path/to/Dataset OUT_ROOT=outputs bash scripts/train_sandbar.sh
```

This enables SPARK with dimension 32, blocks [2,4,4,6], expansion 2.66, batch size 4, Adam at 1e-4, 3 warm-up epochs, cosine decay to 1e-6, and 650 training epochs. It uses 256×256 random crops, horizontal flips, three-scale L1 + 0.1 FFT supervision, and gradient clipping at 0.01. The release verification in the [README](../README.md#results) evaluates the supplied pretrained checkpoint; it does not represent a new training run.

Export a training checkpoint to the deployed graph:

```bash
python sandbar_code/tools/export_deployed.py \
  --input /path/to/training_checkpoint.pkl \
  --output weights/my_deployed_checkpoint.pkl
```

## Evaluation

```bash
python evaluate.py --data-root /path/to/Dataset
```

The command writes per-image PSNR/SSIM to `results/evaluation/per_image.csv` and domain means to `summary.json`. It uses native-resolution RGB float32 inputs, right/bottom reflection padding to multiples of 16, the full-resolution output head, cropping to the original size, and clipping to [0,1]. PSNR and SSIM are calculated directly from floating-point predictions; saved PNG images do not enter this calculation. SSIM uses the scikit-image 7×7 uniform window with `data_range=1`. The overall PSNR is the equal-weight mean of the three domain means.

Return to the [repository homepage](../README.md).

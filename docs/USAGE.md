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
  testset(ref)/test-EUVP/input/
  testset(ref)/test-EUVP/target/
  testset(ref)/test-LSUI/input/
  testset(ref)/test-LSUI/target/
  testset(ref)/test-UIEB/input/
  testset(ref)/test-UIEB/target/
```

`data/splits/` contains the exact 200 EUVP, 200 LSUI and 90 UIEB test filenames used for checkpoint evaluation. Keep filenames and extension capitalization unchanged.

## Evaluation

```bash
python evaluate.py --data-root /path/to/Dataset
```

The command writes per-image PSNR/SSIM to `results/evaluation/per_image.csv` and domain means to `summary.json`. It uses native-resolution RGB float32 inputs, right/bottom reflection padding to multiples of 16, the full-resolution output head, cropping to the original size, and clipping to [0,1]. PSNR and SSIM are calculated directly from floating-point predictions; saved PNG images do not enter this calculation. SSIM uses the scikit-image 7×7 uniform window with `data_range=1`. The overall PSNR is the equal-weight mean of the three domain means.

Return to the [repository homepage](../README.md).

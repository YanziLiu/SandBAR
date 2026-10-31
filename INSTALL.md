# Installation

The released checkpoint evaluation environment is Linux, Python 3.13, PyTorch 2.11 with CUDA 12.8, and an NVIDIA RTX 5090.

```bash
conda env create -f environment.yml
conda activate sandbar
```

Evaluation uses Pillow 12.1.0 linked against IJG libjpeg 9 (`jpeg=9f` from `defaults`), as pinned in `environment.yml`. Keep this conda Pillow installation to match the reported checkpoint results. The evaluation command checks this dependency before running.


Return to the [README](README.md).

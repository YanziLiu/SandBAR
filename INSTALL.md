# Installation

The paper evaluation environment is Linux, Python 3.13, PyTorch 2.11 with CUDA 12.8, and an NVIDIA RTX 5090.

```bash
conda env create -f environment.yml
conda activate sandbar
```

**JPEG decoding matters for this benchmark.** The paper uses Pillow linked against IJG libjpeg 9. The environment specifies `pillow=12.1.0` and `jpeg=9f` from `defaults`. A pip Pillow wheel commonly uses libjpeg-turbo and decodes some input and reference pixels differently. Keep the conda Pillow installation for paper evaluation. The evaluation command checks this dependency before running.


Return to the [README](README.md).

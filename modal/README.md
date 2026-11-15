# Modal test harness

This directory is **not production code**. `src/sightline` never imports it. It exists so heavy CPU or GPU checks run on Modal instead of the local machine.

## WP6 K2.6 parity

`wp6_parity.py` compares the vendored official K2.6 MoonViT tower against the community standalone packaging `immanuelpeter/MoonViT-K2.6` on 20 images at 448, 672 and 896 pixels, using eager attention.

```bash
modal run modal/wp6_parity.py -- --resolutions 448 672 896 --count 20
```

Sizing: **T4 GPU, 4 CPU, 8 GB RAM**. Bump to 16 GB or A10G only if the run OOMs or hits a compatibility issue. The function returns elapsed time, peak RSS, CPU seconds, peak GPU memory and per-image max-abs / max-rel / cosine / allclose metrics.

The image installs the cu130 torch build, mounts `src/` and the extracted K2.6 tower, and downloads the community reference from Hugging Face at run time. The Modal token stays in the CLI profile; it is not stored here.
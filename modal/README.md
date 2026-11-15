# Modal test harness

This directory is **not production code**. `src/sightline` never imports it. It exists so heavy CPU or GPU checks run on Modal instead of the local machine.

## WP6 K2.6 parity

`wp6_parity.py` compares the vendored official K2.6 MoonViT tower against the community standalone packaging `immanuelpeter/MoonViT-K2.6` on 20 images at 448, 672 and 896 pixels, using eager attention.

```bash
modal run modal/wp6_parity.py -- --resolutions 448 672 896 --count 20
```

Sizing: **T4 GPU, 4 CPU, 8 GB RAM**. Bump to 16 GB or A10G only if the run OOMs or hits a compatibility issue. The function returns elapsed time, peak RSS, CPU seconds, peak GPU memory and per-image max-abs / max-rel / cosine / allclose metrics.

The image installs the cu130 torch build, mounts `src/` and the extracted K2.6 tower, and downloads the community reference from Hugging Face at run time. ## WP10 overfit

`wp10_overfit.py` trains the WP7 projector on the 32 WP9 scenes against the frozen Qwen3-1.7B reader and the frozen K3 MoonViT-V2 tower.

```bash
python3 modal/wp10_overfit.py --seed 0
```

It runs on **A10G (24 GB)**: the T4 OOMs at micro-batch 8 (13.75 GiB) and micro-batch 4 (13.24 GiB). The A10G fits micro-batch 8. The function returns the exact match, blank/shuffled/swapped controls, answer loss gap, per-step loss, elapsed time and memory.

## Conformance

`conformance.py` runs the heavy K3/K2.6 MoonViT tower parity tests on a T4.

```bash
python3 modal/conformance.py
```

## M1 training

`m1_train.py` trains the projector on the M1 synthetic data (Baseten recipe, 2048 steps) on an **A10G**. The T4 OOMs at micro-batch 8 and 4.

```bash
python3 modal/m1_train.py --seed 0 --calibrate   # 5-step calibration
python3 modal/m1_train.py --seed 0               # full 2048 steps
```

The function returns elapsed time, peak RSS, CPU seconds, peak GPU memory and per-100-step losses. Checkpoints go to the `sightline-m1` Modal Volume.

The Modal token stays in the CLI profile; it is not stored here.
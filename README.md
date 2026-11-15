# Sightline

A research and learning project on post-hoc visual retrofitting: attach a frozen vision encoder (Kimi MoonViT) to a frozen, text-only LLM through a small trainable projector, and measure honestly what vision capability results and where it stops. The long-term target is DeepSeek-V4-Flash-0731; the first reader is Qwen3-1.7B.

**Status:** M0 complete — G0 signed off 2026-11-15 (`DECISIONS/0027`). M1 in progress: M1a data done; M1b training.

## Start here

1. `CHARTER.md`: the question, non-goals, milestones, honesty rules.
1. `docs/sightline-project-document.md`: the full plan, evidence and gates.
1. `docs/m0-task-breakdown.md`: the M0 work packages (all done).
1. `reports/g0.md`: the signed-off G0 report.
1. `EXPERIMENTS.md`: the experiment log (EXP-0001).
1. `AGENTS.md`: rules for anyone (human or agent) writing code here.
1. `DECISIONS/`: dated decision records; they override the documents.

## Repository layout

```
CHARTER.md  README.md  EXPERIMENTS.md  AGENTS.md  CLAUDE.md
docs/          project documents and small manifests
DECISIONS/     dated decision records (see DECISIONS/README.md)
tools/         standalone scripts (K3/K2.6 vision extraction, palette, bf16 calibration)
modal/         test-only Modal execution harness (not imported by src)
src/sightline/ the package: config, data, vision, projector, backbone, objective, train, evaluate, record
tests/unit/        unit tests
tests/conformance/ tests every reader adapter must pass
experiments/   one folder per experiment: config and run records (committed)
reports/       generated reports (small, committed): g0.md
artifacts/     local only, not in git: weights, extracted towers, checkpoints
data/          local only, not in git: datasets and caches
```

## M0 result

- Frozen Qwen3-1.7B reader + frozen K3 MoonViT-V2 tower + projector (fp32, 4096→4096→2048, 25.2 M).
- Golden splice, generation parity, chunked prefill, parameter test, resume: pass.
- 32-example overfit: 3/3 seeds; 32/32 by step 125/125/150; blank 0%, shuffled 6.25%, swapped 6.25% (chance 12.5%); vision-minus-blind 100%.
- K3/K2.6 tower parity: pass (CPU and Modal T4).
- Report: `reports/g0.md`; decisions: `DECISIONS/0025`–`0028`; run records: `experiments/`.

## Setup

Requires Python 3.12+ (developed on 3.14.8) and `uv`. The pinned environment is in `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The target GPU is an RTX PRO 6000 (sm_120). Before a local GPU run, confirm `torch.cuda.get_arch_list()` includes `sm_120` and `torch.cuda.is_available()` is true. The current CPU-only shell exposes no NVIDIA device: WP0–WP4 and WP7–WP9 run on CPU; the K3/K2.6 tower conformance and the WP10 overfit run on Modal (T4/A10G).

## Commands

```bash
pytest                                          # default suite (slow tests deselected)
pytest -m slow                                  # conformance/tower/slow tests
ruff check src tests modal                      # lint
ruff format --check src tests modal             # format check
python -m sightline.record --dry-run --out experiments/_dryrun
python3 tools/calibrate_bf16_tolerance.py       # bf16 precision calibration
python3 tools/extract_k3_vision.py --repo moonshotai/Kimi-K2.6 --revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb --prefixes vision_tower.,mm_projector. --expect-tower 329 --expect-projector 6 --out artifacts/k26_vision_7eb5002f.safetensors --manifest docs/manifests/k26_vision_7eb5002f.manifest.json
python3 modal/wp10_overfit.py --seed 0          # WP10 overfit on Modal A10G
python3 modal/conformance.py                    # K3/K2.6 tower conformance on Modal T4
```

Without `uv sync`, prefix Python commands with `PYTHONPATH=src`.

## Reproducibility

Every run writes a run record (commit and dirty flag, resolved config, reader/encoder/tokenizer revisions, chat-template hash, seeds, library versions, attention backend, dtypes, metrics as JSONL, GPU-hours). Seeds are 0, 1 and 2. The reader is `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; the K3 vision tower is extracted at `f831ab66814297da540d832a5235f8e904f29d06` and published at `keypa/MoonViT-V2-K3-f831ab66`; the K2.6 tower is extracted at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`.

## Licenses

The license for this repository's own code is decided before any release (DECISIONS/). The Kimi K3 weights and code are under the Kimi K3 License, whose notice must accompany any copy. The K2.6 MoonViT code is under the Modified MIT License. Qwen3 is Apache-2.0. Each data source's license is recorded before it is downloaded.
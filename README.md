# Sightline

A research and learning project on post-hoc visual retrofitting: attach a frozen vision encoder (Kimi MoonViT) to a frozen, text-only LLM through a small trainable projector, and measure honestly what vision capability results and where it stops. The long-term target is DeepSeek-V4-Flash-0731; the first reader is Qwen3-1.7B.

**Status:** M0 (plumbing) in progress: WP0, WP1, WP2, WP3, WP5 and WP6 done; WP4 next. See CHARTER.md for the milestones.

## Start here

1. `CHARTER.md`: the question, non-goals, milestones, honesty rules.
2. `docs/sightline-project-document.md`: the full plan, evidence and gates.
3. `docs/m0-task-breakdown.md`: the work packages for the current stage.
4. `AGENTS.md`: rules for anyone (human or agent) writing code here.
5. `DECISIONS/`: dated decision records; they override the documents.

## Repository layout

```
CHARTER.md  README.md  EXPERIMENTS.md  AGENTS.md  CLAUDE.md
docs/          project documents and small manifests
DECISIONS/     dated decision records (see DECISIONS/README.md)
tools/         standalone scripts (K3 vision extraction, palette generator)
modal/         test-only Modal execution harness (not imported by src)
src/sightline/ the package: config, data, vision, projector, backbone, objective, train, evaluate, record
tests/unit/        unit tests
tests/conformance/ tests every reader adapter must pass
experiments/   one folder per experiment: config and run records (committed)
reports/       generated reports (small, committed)
artifacts/     local only, not in git: weights, extracted towers, checkpoints
data/          local only, not in git: datasets and caches
```

## Setup

Requires Python 3.12+ (developed on 3.14.8) and `uv`. The pinned environment is in `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

The target GPU is an RTX PRO 6000 (sm_120). Before any GPU run, confirm `torch.cuda.get_arch_list()` includes `sm_120` and `torch.cuda.is_available()` is true. The current CPU-only shell exposes no NVIDIA device, so WP0 and WP1 run on CPU.

## Commands

```bash
pytest                                          # run the test suite
ruff check src tests                            # lint
ruff format --check src tests                   # format check
python -m sightline.record --dry-run --out experiments/_dryrun
```

Without `uv sync`, prefix Python commands with `PYTHONPATH=src`.

## Reproducibility

Every run writes a run record (commit, resolved config, model and tokenizer revisions, seeds, library versions, attention backend, dtypes, metrics, GPU-hours). Seeds are 0, 1 and 2. Model and tokenizer revisions are pinned; the K3 vision tower is extracted at a pinned revision with `tools/extract_k3_vision.py`.

## Licenses

The license for this repository's own code is decided before any release (DECISIONS/). The Kimi K3 weights and code are under the Kimi K3 License, whose notice must accompany any copy. Each data source's license is recorded before it is downloaded.

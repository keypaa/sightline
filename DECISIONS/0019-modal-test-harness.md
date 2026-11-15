# DECISION-0019: Modal is a test harness, not production

Date: 2026-11-15
Status: accepted
Work package: WP6

## Context
Heavy CPU/GPU checks should run on Modal to keep the local machine free. Modal must not become a production dependency.

## Decision
- `modal/` is a test/execution harness. `src/sightline` never imports it.
- `modal/wp6_parity.py` defines a T4 function (4 CPU, 16 GB RAM after the 8 GB smoke test peaked at 8.18 GB) that mounts `src/` and the extracted K2.6 tower, installs torch 2.13.0+cu130, torchvision 0.28.0 and transformers 5.12.1, downloads the community reference, and returns per-image metrics plus time, RAM, CPU and GPU measurements.
- The Modal token stays in the CLI profile; it is not stored in the repo.

## Evidence
- 448 smoke test: Tesla T4, 8.18 GB peak, max_abs 0.0. [V]
- Full 60-image run: 180.8 s, peak RSS 8.16 GB, CPU 177.3 s, max GPU 5.68 GB, max_abs 0.0. [V]
- Modal client 1.5.4, authenticated as `keypaa`. [V]

## Consequences
Future heavy tests use `modal/`. Production code remains local and pure Python. The 16 GB bump is recorded; A10G is only for compatibility or OOM.

## Open items
none
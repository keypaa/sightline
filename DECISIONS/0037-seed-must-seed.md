# DECISION-0037: `--seed` must actually seed the run

Date: 2026-11-19
Status: accepted
Work package: M1b

## Context

Seeds 0 and 1 were run at LR 1e-3 with identical settings and produced very
different held-out results: colour 0.855 vs 0.965, shape 0.260 vs 0.955, count
0.318 vs 0.258. That was read as seed variance.

Reading `tools/run_m1.py` showed why that reading is wrong. `--seed` was used
only for output paths and record labels. The dataset was always
`generate_m1_dataset(seed=0)` and the projector was always initialised under
`torch.manual_seed(123)`. Seeds 0, 1 and 2 therefore had identical data and
identical initialisation, and differed only by GPU nondeterminism.

That makes the three-seed protocol invalid as implemented: "all 3 seeds
positive" would have been a statement about run-to-run nondeterminism, not about
independent initialisations.

## Decision

- `set_seeds(seed)` seeds `random`, `numpy`, `torch` and the CUDA generators, and
  is called per trainer construction from `args.seed`.
- The base batch order is shuffled with `random.Random(args.seed)` after the
  batches are built, so the order of presentation varies per seed.
- The **dataset stays at seed 0**. The M1a manifest is pinned at
  `2fe44a9d…`; the benchmark split is not a random effect. What varies per seed
  is the initialisation and the presentation order.
- Seeds 0, 1 and 2 must be re-run. The existing runs are not three
  independent seeds and cannot support a G1 claim.

## Evidence

- `--seed` appeared only in paths and record labels; `generate_m1_dataset(seed=0)`
  and `torch.manual_seed(123)` were unconditional. [V]
- Seeds 0 and 1 differed markedly with identical configuration. [V]
- `set_seeds` reproducibility and distinctness are unit tested. [V]

## Consequences

The three runs at LR 1e-3 must be repeated from scratch. That is about 1.2 h per
seed on the RTX PRO 6000. The existing per-seed numbers are kept as evidence of
nondeterminism, not as seeds.

## Open items

- [U] How much of the seed-0 vs seed-1 spread was nondeterminism rather than
  seed. Closing step: the re-run, where seeds differ by construction.
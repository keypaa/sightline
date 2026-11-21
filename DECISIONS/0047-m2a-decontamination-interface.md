# DECISIONS 0047: M2a decontamination interface

Date: 2026-11-21
Status: accepted
Work package: M2a

## Context

`DECISIONS/0046` measurement #1 requires SSCD decontamination against MMMU-Pro and BLINK,
because FineVision's 66-benchmark dedup includes neither [V — lmms-eval-embeddings file
listing]. Task 10 implemented it. Two interface choices were not decided anywhere and are
recorded here, per AGENTS.md rule 4.

## Decision

1. **`write_report(path, rows, n_records, *, checked_benchmarks=None)`.** When
   `checked_benchmarks` is given, the report carries a `per_benchmark` entry for every
   benchmark examined, including zeros. Without it, behaviour is unchanged (`{}` when no
   rows were found), so the Task 10 brief's original assertion still holds. The CLI always
   passes the checked names, so a real run proves both MMMU-Pro and BLINK were examined
   rather than one being silently skipped.
2. **`--sscd-threshold` defaults to 0.95** and is recorded in the run report. The value is
   frozen by FineVision's methodology; the flag exists so a re-run can record a deliberate
   change rather than hiding it.
3. **`benchmark_images` (sha256 → Path) is the injection point** for the MMMU-Pro and BLINK
   image sets. The tool refuses to run a similarity comparison without it, and refuses
   `--similarity` without an SSCD callable. It never silently degrades to exact-hash-only.
4. **`dhash`/`distance` are not used** by the decontamination path, despite the Task 10
   brief listing them as consumed. SSCD is the method the 0.95 threshold is defined
   against, and `dhash` is colour-blind (`src/sightline/m2a/images.py`), so it is a weaker
   signal that would need its own tolerance.

## Consequences

- A clean decontamination report names both benchmarks with zero counts, which is auditable
  evidence rather than an absence.
- The real run needs the pinned MMMU-Pro and BLINK image sets and an importable SSCD
  callable. Until then the tool runs exact-hash-only, which catches byte-identical reuse but
  not resized or recompressed copies.

## Open items

- [U] Whether the MMMU-Pro and BLINK image sets are obtainable at a pinned revision.
  Closing step: locate the test splits, record their hashes in the dedup run record.
  Owner: Keylhan Paumard--André.
# DECISIONS 0051: M2a annotation concurrency

Date: 2026-11-22
Status: accepted
Work package: M2a

## Context

The 20k annotation run needs a `--workers` value. Published benchmarks for this model
on this card [S] suggested a large gain up to c=32, but the project's rule 9 requires
measuring before committing. A 200-conversation probe was run at c=8, 16 and 32;
results are in `reports/m2a-annotator-concurrency-probe.md`.

## Decision

**Use `--workers 16` for the 20k annotation run.**

Measured [V]:

| workers | conversations/hour |
| --- | --- |
| 8 | 8,298 |
| **16** | **21,900** |
| 32 | 23,736 |

Rationale: the recipe's frozen rule is to pick the lowest concurrency within ~10% of
the best. c=16 delivers 92.3% of c=32's throughput. Going from 16 to 32 buys 8.4% more
throughput — about 15 minutes over a 3.3-hour run — while doubling client-side load
and the chance of a transient failure. The knee is at 16.

This supersedes the placeholder `--workers 32` in the earlier recipe draft, which was
taken from the published table rather than measured.

## Consequences

- 19,962 conversations project to ~3.3 hours at the chosen setting [I].
- The value is for this annotator, this server configuration and this image size. A
  change to any of those needs a new probe, not a reused number.
- The probe itself is evidence: 600/600 conversations parsed, `unparsed=0` at every
  concurrency, so the setting is not trading quality for speed.

## Open items

None.

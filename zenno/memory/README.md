# Project memory

Short, opinionated notes so the same dead end is not walked twice. This is the
repo's institutional memory; `DECISIONS/` records choices, `docs/` holds
documents, `reports/` holds generated artifacts.

## What belongs here

- Something that **worked**, with the measurement that showed it.
- Something that **did not work**, and the reason — not just "it failed".
- A trap that cost time, so the next person recognises it early.
- A question still open, with an owner and a closing step.

## What does not belong here

- Anything already written down properly elsewhere. Link to it instead.
- Numbers without a source. Tag them.

## Conventions

Every non-obvious claim carries an evidence tag, same as the rest of the repo:

- **[V]** verified: read the primary source or ran the check.
- **[S]** secondary: a summary, port or third-party report.
- **[I]** inference.
- **[U]** unknown.

Each entry says **why**, because a result without a mechanism is a coincidence
waiting to be rediscovered.

## Index

| file | contents |
| --- | --- |
| [findings.md](findings.md) | running log: what works, what does not, what is open |
| [`reports/m1b-feature-probe.md`](../../reports/m1b-feature-probe.md) | the decisive probe: answers are in the frozen features |

Detailed write-ups live in `docs/measurement-pitfalls.md`,
`reports/m1b-seed0-postmortem.md` and `DECISIONS/`; this folder points at them
rather than copying them.
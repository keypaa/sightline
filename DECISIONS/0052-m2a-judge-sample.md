# DECISIONS 0052: M2a judge sample size

Date: 2026-11-23
Status: accepted
Work package: M2a
Owner instruction: probe the verifier first, then choose the judging sample from
the measured call cost.

## Context

`DECISIONS/0049` froze the gate thresholds but not the number of claims to
judge. The recipe's `judge` command defaults to `--sample 0` (all claims). The
regenerated corpus has 45,699 real and 38,888 planted claims, so an unsampled
run is 84,587 separate `opencode run` calls — infeasible on a metered verifier.

A 20+20 probe was run with `--workers 1` against `opencode/hy4-flash-free`:

- 40 calls took **228 s** (~5.7 s/call); two re-asked calls after a transient
  HTTP 429 took 16 s. **[V]** `experiments/m2a_grounding/judge_probe.log`,
  `judge_probe_retry.log`.
- Real: 17 true / 3 false (15% contradicted at n=20 — too small to conclude;
  the Wilson upper bound is far above the 0.15 ceiling). **[V]**
- Planted: 18 rejected / 20, 1 accepted, 1 `can't-tell` (90% rejection, exactly
  the floor at n=20). **[V]**
- The first probe hit "Free usage limit reached" on calls 39-40; the quota was
  temporary and a retry succeeded. **[V]**

## Decision

1. Judge **500 real and 500 planted claims**, `--sample-seed 0`, `--workers 1`,
   verifier `opencode/hy4-flash-free`. Expected cost ~1,000 calls, ~100 min.
2. Write to `experiments/m2a_grounding/verdicts.jsonl`. Resume by re-running the
   same command; `judge` drops `error` rows and re-asks them.
3. The gates stay as frozen in `DECISIONS/0049`: contradicted upper 95 ≤ 0.15,
   planted rejection ≥ 0.90, human agreement ≥ 0.85.
4. The 20+20 probe rows are **not** folded into the gate sample: they use a
   different `sample_claims` draw, so mixing them would make the denominator
   hard to describe. They remain as measurement evidence only.

## Deviation

`DECISIONS/0049` describes a pilot of 500 images / ~1,500 claims. The measured
call cost makes 1,500 per layer ~3,000 calls (~5 h) with a real risk of a second
quota trip. 500 per layer is a quota-driven reduction, recorded here rather
than applied silently.

## Consequences

- A 15% point estimate on 20 real claims cannot establish the 0.15 ceiling
  either way. If it holds at n=500, G2 fails and the corpus needs filtering or
  an instrument change — report it, do not tune the threshold.
- Human review samples 50 items from the verdicts and reports Wilson intervals;
  agreement remains calibration evidence, not a precision estimate.

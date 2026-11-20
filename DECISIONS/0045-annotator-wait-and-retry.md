# DECISIONS 0045: M2a annotator wait-and-retry on quota

Date: 2026-11-20
Status: accepted
Work package: M2a (20k iteration corpus)
Owner instruction: "Keep Kilo, add wait-and-retry".

## Evidence

- Kilo Gateway documents **200 free-model requests per hour per IP** [V —
  https://kilo.ai/docs/gateway/usage-and-billing, read 2026-11-20]. The 429 body
  does not carry a retry time or reset timestamp.
- Observed: two separate runs hit 429 after roughly 145 successful calls [V —
  `reports/m2a-annotator-quota-probe.md`]. The gap between the documented 200 and
  the observed ~145 is consistent with other traffic from the same IP [I].
- Remaining after the bounded 500-row probe: 19,706 annotations. At 200 per hour
  that is about 99 quota windows, or 4.1 days of continuous waiting [I].

## Decision

1. **Transport stays Kilo CLI** with `--archived`, per `DECISIONS/0040` and
   `DECISIONS/0044`. No provider change.
2. **On a confirmed rate limit, wait 3,600 s (one documented window) and resume.**
   The wait is fixed, not exponential: the documented limit is a fixed hourly
   quota, and no retry-after value is exposed by the CLI.
3. **Safety cap: 120 waits (5 days).** If the quota has not recovered by then the
   stage stops with a non-zero exit and keeps its partial results. This prevents
   an unbounded process if the account policy changes.
4. **Only rate limits wait.** Any other `TransportError` stops the stage at once.
   A run of tokenless responses (the secondary guard in `DECISIONS/0044`) also
   stops without waiting, because it is not a confirmed quota error.
5. **Resume is by id.** Only rows written in the current window count as done;
   the failed record is retried in the next window. Partial rows are always
   kept. Each wait is reported with its window number and rows completed so far.
6. The values in items 2 and 3 are **frozen before the first run**. Changing them
   needs a new DECISIONS entry and a recorded reason.

## Consequence

This makes the 20k run unattended but slow: roughly four days at the documented
quota. It does not change the 20k target or the corpus plan. The transport-switch
item in `DECISIONS/0044` stays open; this decision buys a run without revisiting
provider provenance.

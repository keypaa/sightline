# DECISION-0040: M2a annotator, measured cost against the section 16 cap

Date: 2026-11-21
Status: accepted
Work package: M2a entry criteria

## Context

Section 16 item 8 makes the annotator an M2 entry criterion: choose one, check
its terms, and estimate tokens against a cap of about 200M tokens a day with no
cost in the hundreds of dollars. The project document assumed 1.6k tokens per
conversation.

Three pilot runs on 30 M1 images measured the real cost (`reports/m2a-annotator-pilot.md`).

## Measurement

- **3,179 input tokens per conversation**, deterministic on repeat, at prompt
  hash `b6d8d625b4ffc424`.
- 30/30 parsed as JSON; 27/30 grounded (0.900) against a pre-declared bar of 0.90.
- 40k conversations: **127M tokens** — inside the 200M/day cap.
- 131k conversations: **~417M tokens** — over the cap, needs two days or a
  higher tier.
- The same images cost 16,126 tokens per conversation in the first run and 3,179
  in the third. Provider-side and variable; [U].

## Decision

- The annotator for M2a is `kilo/nvidia/nemotron-3.5-super-vl:free`, accepted at
  the measured cost and quality.
- The **iterative** M2a set is 20k–40k conversations, which fits the daily cap.
- The **131k G2 reproduction** does not fit the free tier's implied cap at the
  measured rate and needs either a paid tier or a two-day run. That is a
  scheduling decision for M2a's end, not a blocker for starting.
- The prompt requires the target attribute to be stated explicitly. This is
  recorded as the annotator contract.

## Deviation from the project document

The document's 1.6k tokens per conversation is **wrong by about 2×** at the
measured 3,179, and by up to 10× if the provider's high-token mode returns. The
budget in section 16 and the M2 row of the data table must be updated to 3,179
with the variance noted. This entry records the correction; the document is
re-exported when M1 completes.

## Consequences

M2a can start on the free tier for the iterative set. The reproduction run needs a
cost decision before it is scheduled.

## Open items

- [U] Provider pricing. Closing step: read the pricing page, multiply by 127M.
- [U] The 5× token variance across runs. Closing step: two pilots in one session.
- [U] Count elicitation is 8/10. Closing step: another prompt iteration if count
  data quality matters.
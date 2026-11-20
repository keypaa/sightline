# DECISIONS 0044: M2a annotation rate-limit guard and archived sessions

Date: 2026-11-20
Status: accepted
Work package: M2a (20k iteration corpus)
Owner instruction: "let's wait a bit, still add --archived" and "leave them alone"
(existing sessions).

## Problem

The 20k annotation run was stopped after ~500 calls. The fetch stage was good:
`in=20000 kept=19962 dropped=38 {'duplicates': 32}`. The annotation stage was
not: `[annotate] 500/19962 unparsed=386`.

Root cause, [V] from the killed run and a live reproduction:

- `opencode run` reports a rate limit as a structured error event with exit code
  1, e.g. `{"type":"error","error":{"type":"provider.quota","status":429,
  "message":"Rate limit exceeded", ... "Free model usage limit reached"}}`.
- The old `_call` in `tools/build_m2a.py` collected only `text` and `step_finish`
  events. It discarded the return code, stderr and `error` events. A 429
  therefore became empty text and no token usage, which `parse_messages` marked
  `parsed=False`.
- 433 of the 434 unparsed rows in `experiments/m2a_iter20k/conversations.jsonl`
  have `tokens_in=None` and `tokens_out=None`; every parsed row has real counts.
  That is the signature of a failed call, not a bad answer.
- Every `opencode run` creates a session and the call did not use `--archived`,
  so the scripted calls appeared in the session list.

## Decision

1. **Transport stays pinned.** `kilo/nvidia/nemotron-3.5-super-vl:free` via
   `opencode run`, per `DECISIONS/0040`. No provider change in this work package.
2. **Every scripted call uses `--archived`.** The sessions stay out of session
   lists. No existing sessions are deleted.
3. **A confirmed provider refusal stops the run.** `RateLimited` is raised on an
   error event with status 402 or 429, `error.type == "provider.quota"`, or
   rate/credit/quota text; a non-zero exit without an error event raises
   `TransportError`. Both trip the shared guard.
4. **Frozen secondary threshold: 10 consecutive responses with no token usage.**
   This catches a failure that produced no structured error event. A response
   with token counts resets the streak, so a merely unparseable answer does not
   stop the run. The killed run would have stopped near row 160 instead of 500.
5. **Bounded submission.** At most `--workers` calls are in flight and no new
   task is submitted after the guard trips. Submitting all 19,962 futures at
   once — what the old code did — means stopping the result loop does not stop
   the calls. This is the actual fix for the session flood.
6. **Partial results are kept and the stage exits non-zero.** A resume reads
   only `parsed=True` rows as done, so blank rows are retried rather than
   silently counted as conversations. `manifest()` also excludes unparsed rows
   and reports `n_unparsed`.
7. The threshold in item 4 is frozen before the next run. Changing it needs a
   new DECISIONS entry and a recorded reason.

## Open item

- **[U] Transport switch** (direct OpenRouter HTTP for the same pinned model).
  Measurement [V]: the same model answered over plain HTTP with ~130 output
  tokens instead of ~3,179, and no sessions are created. This would change the
  provider identity recorded in provenance and needs a DECISIONS entry plus a
  parse/grounding parity check. OpenRouter shows `total_credits: 0` [V], so the
  free-tier request cap probably still bounds a 20k run [S].
  Owner: Keylhan Paumard--André.
  Closing step: compare quota recovery on Kilo with a paid or alternative
  transport, then decide before scaling past 20k.

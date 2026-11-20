# M2a annotator quota probe (2026-11-20)

## Question

Can the pinned annotator sustain the 20k run on the current transport, now that
the rate-limit guard stops cleanly instead of spawning sessions?

## Measurements [V]

| run | calls before 429 | outcome |
| --- | --- | --- |
| original 20k annotate (killed) | ~150 good, then 386/500 unparsed | continued, 433 blank rows |
| bounded 500-row resume | 144 attempted, then 429 | stopped cleanly, exit 1 |
| 3-image probe just before it | 3/3 parsed | exit 0 |

The bounded resume log (`m2a_iter20k_resume500.log`):

```
[annotate] 50/500 unparsed=0
[annotate] 100/500 unparsed=2
[annotate] STOPPED: rate limited (HTTP 429): Rate limit exceeded
```

`RateLimited` was raised from the CLI's own error event
(`status: 429`, `provider.quota`), so detection is working as designed. The
visible session count stayed at 100 throughout the guarded runs.

## Corpus state

- `fetched.jsonl`: 19,962 images, unchanged and reusable.
- `conversations.jsonl`: 256 unique parsed conversations after purging the 436
  failed rows left by the killed run and the retry duplicates. All 256 carry
  prompt hash `b6d8d625b4ffc424`.
- Remaining: 19,706 annotations. At ~145 calls per quota window this is about
  136 windows. The reset period is **[U]** — the error body does not state it.

## Interpretation

The fix solves the two reported faults: no blank rows are silently produced, and
no sessions appear in the menu. It does not make the free quota large enough for
20k in one run. Continuing on this transport means roughly 136 stop-and-resume
cycles unless a wait-and-retry policy or a different transport is chosen.

The direct-HTTP option measured earlier used ~130 output tokens per conversation
versus ~3,179 here, but the OpenRouter key has 0 credits [V], so its free-tier
cap is likely to bind as well [S].

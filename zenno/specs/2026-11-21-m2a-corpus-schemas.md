# M2a corpus schemas — what each role writes

Date: 2026-11-21
Status: revised after builder-check; awaiting owner approval
Work package: M2a

## The rule this design enforces

**No model both produces a claim and judges it.**

| role | model | transport | sees the image? |
| --- | --- | --- | --- |
| annotator — writes the conversation | `Qwen/Qwen3.6-27B-FP8` | vLLM, local | yes |
| verifier — judges each claim | `opencode/hy4-flash-free` | opencode CLI | yes |
| blind reader — answers without the image | `opencode/hy4-flash-free` | opencode CLI | **no** |
| human — audits a sample | the owner | review tool | yes |

One verifier, one blind reader, both hy4. They are the same model on purpose: different jobs, no shared state, and neither is the annotator. An earlier draft had two verifiers; that doubled the quota cost for a robustness gain we could not afford, and it is dropped.

There is **no separate "label" stage**. Verifying per claim across the corpus replaced the retired 50-image attribute key (`DECISIONS/0048`).

## Join keys

Two keys, used consistently:

- `id` — the COCO image id. Used by the **image** and **conversation** layers, matching `fetched.jsonl` today. Not renamed.
- `claim_id` — `sha256(f"{image_id}:{turn}:{claim}")[:16]`. Used by **claim**, **verdict**, **blind** and **decision** rows, where `image_id` is a foreign key to `id`.

A sequence would shift if a claim were inserted; a content hash cannot.

## The layers

```
image record      immutable, from COCO        fetched.jsonl
   ↓  annotate
conversation      the training data           conversations.jsonl
   ↓  extract (deterministic)
claim             one checkable statement     claims.jsonl
   ↓  verify
verdict           true / false / can't-tell   verdicts.jsonl   (flat: one row per claim × verifier)
   ↓  blind
blind result      answerable without image?   blind.jsonl
   ↓  review
human decision    verifier right or wrong?    decisions.jsonl
   ↓
manifest          aggregator over all layers manifest.json
```

Every layer is append-only and carries what produced it. Nothing is recomputed in place.

## 1. Image record — `fetched.jsonl`

Written once by `fetch`. **Immutable**: a re-fetch never changes an existing id's bytes.

| field | type | null | writer |
| --- | --- | --- | --- |
| `id` | int | no | index |
| `file_name` | str | no | index |
| `source` | str (`"coco"`) | no | index |
| `license_id` | int | yes | index |
| `license_name` | str | yes | index |
| `license_url` | str | yes | index |
| `flickr_url` | str | yes | index |
| `coco_url` | str | yes | index |
| `url` | str | no | index |
| `sha256` | str (64 hex) | no | fetch |
| `path` | str (repo-relative) | no | fetch |
| `expected_sha256` | str | yes | fetch `--write-checksums` |

This layer is already implemented and matches, including `url` and `license_url`.

## 2. Conversation — `conversations.jsonl`

One row per image. Written by `annotate`.

| field | type | null | notes |
| --- | --- | --- | --- |
| `id` | int | no | the COCO image id |
| `file_name` | str | no | |
| `messages` | list of `{role, content}` | no | `role` is `user` or `assistant` |
| `parsed` | bool | no | `messages` conforms to the schema |
| `model` | str | no | annotator id |
| `revision` | str | no | checkpoint commit |
| `quantization` | str | no | e.g. `fp8` |
| `engine` | str | no | `vllm` |
| `engine_version` | str | no | recorded, never assumed |
| `prompt_sha256` | str | no | the annotator contract |
| `chat_template_sha256` | str | no | template the server applied |
| `seed` | int | no | generation seed |
| `order_seed` | int | no | corpus permutation |
| `temperature` | float | no | |
| `tokens_in` | int | yes | null only when the call failed |
| `tokens_out` | int | yes | |
| `image_sha256` | str | no | must equal the image record's `sha256` |
| `run_id` | str | no | **new**: one per annotate invocation |
| `created_at` | str (ISO 8601) | no | **new**: written at row time |

`turns`, `expected` and `grounded` exist in the pilot output but are **not** part of the corpus schema: turns is derivable, and grounding is not measurable on COCO (`DECISIONS/0049`). They stay pilot-only.

**Invariant:** all rows in one corpus carry the same `model`+`revision`+`quantization`+`prompt_sha256`+`chat_template_sha256`+`run_id`. The manifest counts distinct annotators so a mid-corpus switch is visible, but the writer should refuse to append a row whose annotator differs from the first one in the file.

## 3. Claim — `claims.jsonl`

Deterministic, derived from conversations. Re-running `extract` on the same input produces identical rows except `extracted_at`.

| field | type | null | notes |
| --- | --- | --- | --- |
| `claim_id` | str | no | the join key |
| `image_id` | int | no | foreign key to conversations `id` |
| `turn` | int | no | 1-based |
| `question` | str | no | verbatim |
| `answer` | str | no | verbatim |
| `claim` | str | no | declarative, names its referent |
| `claim_type` | enum | no | `existence`, `count`, `colour`, `spatial`, `text`, `action` |
| `image_path` | str | yes | null when it could not be resolved |
| `source_run` | str | no | `run_id` of the conversation |
| `extracted_at` | str | no | |

`image_path` is the only non-deterministic field, because it depends on where the repo is checked out. It is nullable so a claim is still usable if the image cannot be found.

## 4. Verdict — `verdicts.jsonl`

**Flat: one row per claim per verifier.** Appended as each judgement completes, so a crash loses one row, not a claim's worth.

| field | type | null | notes |
| --- | --- | --- | --- |
| `claim_id` | str | no | |
| `verifier_model` | str | no | e.g. `opencode/hy4-flash-free` |
| `verifier_engine` | str | no | `opencode` |
| `verifier_prompt_sha256` | str | no | **new**: pins the judging prompt |
| `verdict` | enum | no | `true`, `false`, `can't-tell` |
| `evidence` | str | no | one line, may be empty |
| `error` | str | yes | non-null means the call failed, **not** a verdict |
| `judged_at` | str | no | |

`verifier_revision` is **not** a field: the opencode CLI exposes no provider revision, so it would always be null. A column that is always empty is a lie about what is recorded.

**Invariant:** a row with `error != null` is a gap and is excluded from every rate. Counting errors as `can't-tell` is what made a metered provider look like a cautious model.

## 5. Blind result — `blind.jsonl`

One row per claim. The question goes to a text-only model with **no image**.

| field | type | null |
| --- | --- | --- |
| `claim_id` | str | no |
| `question` | str | no |
| `expected` | str | no |
| `blind_model` | str | no |
| `blind_answer` | str | yes |
| `blind_matches` | bool | no |
| `error` | str | yes |
| `judged_at` | str | no |

`blind_matches` is a word-subset test: every word of `expected` appears in `blind_answer`.

## 6. Human decision — `decisions.jsonl`

The only ground truth in the system. Written by the review tool.

| field | type | null | notes |
| --- | --- | --- | --- |
| `claim_id` | str | no | |
| `decision` | enum | no | `verifiers_right`, `verifiers_wrong` |
| `truth` | str | yes | what the reviewer believes, when the verifiers were wrong |
| `reviewer` | str | no |
| `reviewed_at` | str | no |

**The human judges the verifier, not the annotator.** The question is "are the verdicts correct?", because that is what calibrates the instrument. `truth` is optional and exists so a wrong verdict can be corrected without re-annotating.

`truth` stays **optional**, by owner decision: it should not slow the review, but recording it when a verdict is wrong turns that item into a usable correction rather than just a mark against the verifier. The review tool should make it a one-click optional field, not a required one.

## The review session

**50 claims to start**, 100 if the owner wants a tighter interval later.

The 50 are sampled across verifier strata, not at random, because the informative cases are the disagreements:

| stratum | share | why |
| --- | --- | --- |
| verifiers disagree | ~50% | where the instrument is most likely wrong |
| both say true | ~25% | catches a verifier that agrees with everything |
| both say false or can't-tell | ~25% | catches a verifier that cries wolf |

At 50 items the owner's agreement figure carries a wide Wilson interval, and it is reported as calibration evidence, not as an estimate of verifier accuracy (`DECISIONS/0049`). Roughly 15 minutes of review.

## 7. Manifest — `manifest.json`

A **new aggregator** over every layer, not an extension of the existing corpus manifest. The existing `manifest()` keeps its current fields for the corpus stages; this one adds the gate results.

| group | fields |
| --- | --- |
| run | `run_id`, `git` (commit + dirty) |
| counts | `n_images`, `n_conversations`, `n_claims`, `n_errored`, `n_decisions` |
| provenance | `source_count`, `sources`, `annotators`, `licences`, `order_seed`, `seed` |
| quality | `contradiction_rate`, `contradiction_rate_upper_95`, `planted_rejection_rate`, `blind_rate`, `human_agreement` |
| gates | `thresholds`, `passes_contradiction`, `passes_planted`, `passes_human` |

## Invariants that must hold

1. Every conversation's `image_sha256` equals its image record's `sha256`.
2. `claim_id` is a pure function of `(image_id, turn, claim)`.
3. Every verdict and decision references a `claim_id` that exists.
4. A claim with any `error` verdict is excluded from every rate.
5. One annotator identity across the whole corpus, enforced at write time.
6. Verifier model ≠ annotator model.
7. Only `decisions.jsonl` may claim ground truth.

## What this implies for the existing code

| change | scope |
| --- | --- |
| `annotate` writes `run_id` and `created_at`; refuses a foreign annotator | `cmd_annotate` |
| `extract` emits `claim_id`, reads/writes `image_id`, drops `turns` | `extract_claims`, `extract` CLI |
| `judge` writes flat rows keyed by `claim_id`, hashes its prompt, resumes on `claim_id` | `judge_claims`, `_result_to_row`, `judge` CLI |
| `blind` carries `claim_id` and `blind_model` | `judge_blind` |
| new `review` command writes `decisions.jsonl` | new CLI subcommand |
| new manifest aggregator over all layers | new `manifest` path, existing one untouched |

The core — claim classification, planting, summaries, Wilson intervals, thresholds — does not change.

## Out of scope here

- The 20k annotation run itself.
- The verifier sample size (needs a measurement first).
- The blind-reader threshold (deliberately `[U]` until the distribution is seen).
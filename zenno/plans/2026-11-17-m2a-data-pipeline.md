# M2a Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn COCO 2017 into a multi-turn image-QA corpus with a recorded licence for every image, a recorded annotator and prompt for every conversation, and a manifest that makes the whole set reproducible.

**Architecture:** A small set of pure functions, each stage reading the previous stage's JSONL and writing its own. Nothing is held in memory across stages, so any stage can be re-run alone and a crash costs one stage. Images are fetched once into a content-addressed cache and never re-fetched.

**Tech Stack:** Python 3.13, `requests` for COCO range reads, `Pillow` for hashing, `opencode run` as the annotator transport.

**Spec:** `reports/m2a-entry-criteria.md` (items 6, 7, 8), `DECISIONS/0040-m2a-annotator.md` for the annotator contract, `DECISIONS/0041-licence-allow-list.md` for the licence policy, `docs/measurement-pitfalls.md` for the ordered-split and sampling traps.

## Global Constraints

- No source enters without a recorded licence. A record with no licence id is dropped, not defaulted.
- The annotator is `kilo/nvidia/nemotron-3.5-super-vl:free`, prompt hash `b6d8d625b4ffc424`, recorded per conversation. One annotator for both arms of M2b, never switched.
- Measured annotator cost is 3,179 input tokens per conversation; 20k conversations is ~64M tokens, inside the 200M/day cap (`DECISIONS/0040`).
- Every stage writes JSONL and is resumable: re-running skips records already written, keyed by a stable id.
- Every build writes a `RunRecord` (project rule 8). A data build without a run record does not exist.
- Splits are sharded by source and by perceptual hash, never by taking the first N of an ordered file (`docs/measurement-pitfalls.md`).
- MMMU-Pro and BLINK are separate datasets and are simply not used. Since the only source is COCO, no image-level exclusion list is needed; this is recorded so the constraint is not re-opened later.
- No emojis. Type hints. Comments explain intent.

## Review Focus

- **An image whose licence id is absent from the annotations.** Expected: dropped with reason `missing-licence`, counted, never assumed permissive.
- **A licence whose name carries NC or ND.** Expected: dropped with the reason recorded, even if a fixed allow-list would have admitted it.
- **The same image at two COCO ids.** Expected: one survives, chosen deterministically, duplicate counted — and the test must exercise the hash, not just the file name.
- **An annotator call returning prose instead of JSON.** Expected: record marked `parsed: false`, run continues, failure rate reported.
- **A re-run of the build.** Expected: byte-identical manifest and identical split assignment.

---

### Task 1: COCO index reader with licence capture

**Files:**
- Create: `src/sightline/m2a/__init__.py`
- Create: `src/sightline/m2a/coco_index.py`
- Test: `tests/unit/test_m2a_coco_index.py`

**Interfaces:**
- Consumes: a COCO `instances_train2017.json` path.
- Produces: `read_annotations(path: Path) -> tuple[list[dict], dict[int, dict]]` — per-image records `{"file_name", "id", "license_id", "license_name", "license_url"}` and the licence table keyed by id.

- [ ] **Step 1: Write the failing tests**

```python
def test_records_carry_the_licence_id(tmp_path):
    annotations = {"images": [{"id": 1, "file_name": "a.jpg", "license": 3}],
                   "licenses": [{"id": 3, "name": "CC BY 4.0", "url": "u"}]}
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(annotations))

    records, table = read_annotations(path)

    assert records[0]["license_id"] == 3
    assert records[0]["license_name"] == "CC BY 4.0"
    assert table[3]["url"] == "u"
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_coco_index.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement the reader in `src/sightline/m2a/coco_index.py`**

Index `licenses` by id, join onto each image, and emit one record per image. Do not filter here — filtering is Task 2, so the raw counts stay visible.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_coco_index.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a tests
git commit -m "M2a: read COCO annotations with their per-image licence ids"
```

### Task 2: Licence gate

**Files:**
- Create: `src/sightline/m2a/licence.py`
- Create: `DECISIONS/0041-licence-allow-list.md`
- Test: `tests/unit/test_m2a_licence.py`

**Interfaces:**
- Consumes: records from Task 1.
- Produces: `classify(records: list[dict], allow: set[str] | None = None) -> tuple[list[dict], list[dict]]`. When `allow` is `None`, the production policy applies: drop anything whose name carries `nc` or `nd`, and drop a missing name. Dropped records carry `drop_reason` in `{"non-commercial", "no-derivatives", "missing-licence"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_policy_drops_nc_nd_and_missing():
    records = [
        {"file_name": "a.jpg", "license_name": "CC BY 4.0"},
        {"file_name": "b.jpg", "license_name": "CC BY-NC 2.0"},
        {"file_name": "c.jpg", "license_name": "CC BY-ND 2.0"},
        {"file_name": "d.jpg", "license_name": None},
    ]

    kept, dropped = classify(records)

    assert [r["file_name"] for r in kept] == ["a.jpg"]
    assert {r["drop_reason"] for r in dropped} == {"non-commercial", "no-derivatives", "missing-licence"}


def test_an_explicit_allow_list_overrides_the_policy():
    records = [{"file_name": "a.jpg", "license_name": "CC BY-NC 2.0"}]

    kept, dropped = classify(records, allow={"cc by-nc 2.0"})

    assert len(kept) == 1 and dropped == []
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_licence.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write `DECISIONS/0041-licence-allow-list.md`**

Record the policy as **drop NC and ND, drop missing**, with licence names read from the annotations file rather than assumed. Record why: COCO's images are per-image Flickr licences and the consortium does not own them, so the dataset-level "CC BY 4.0" is not permission for the pixels.

- [ ] **Step 4: Implement `classify` in `src/sightline/m2a/licence.py`**

Normalise the name (case-fold, collapse separators). Check `nc` and `nd` as tokens before consulting any allow-list, so the policy cannot be bypassed by a caller.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_licence.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sightline/m2a/licence.py tests DECISIONS
git commit -m "M2a: drop images without a permissive recorded licence"
```

### Task 3: Image cache and duplicate detection

**Files:**
- Create: `src/sightline/m2a/images.py`
- Test: `tests/unit/test_m2a_images.py`

**Interfaces:**
- Consumes: kept records from Task 2, a `get(url) -> bytes` callable.
- Produces: `fetch(record: dict, cache_dir: Path, get) -> Path`; `dhash(path: Path) -> int`; `dedupe(records: list[dict], hashes: dict[int, int]) -> tuple[list[dict], list[dict]]` keyed by image id.

- [ ] **Step 1: Write the failing tests**

```python
def test_identical_images_at_two_ids_keeps_one(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_solid(a, (10, 10), (255, 0, 0))
    _write_solid(b, (10, 10), (255, 0, 0))
    c = tmp_path / "c.png"
    _write_solid(c, (10, 10), (0, 0, 255))

    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    hashes = {1: dhash(a), 2: dhash(b), 3: dhash(c)}

    assert dhash(a) == dhash(b)
    assert dhash(a) != dhash(c)

    kept, dupes = dedupe(records, hashes)

    assert {r["id"] for r in kept} == {1, 3}
    assert [r["id"] for r in dupes] == [2]
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement the cache, dHash and dedupe in `src/sightline/m2a/images.py`**

`dhash` is a 64-bit difference hash via Pillow, implemented locally — no new dependency. `dedupe` walks records in order and keeps the first occurrence of each hash, so the survivor is deterministic. `fetch` writes to `cache_dir / file_name` only if absent.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/images.py tests
git commit -m "M2a: cache images and drop duplicates by difference hash"
```

### Task 4: Conversation generation with provenance

**Files:**
- Create: `src/sightline/m2a/annotate.py`
- Test: `tests/unit/test_m2a_annotate.py`

**Interfaces:**
- Consumes: deduped records with local image paths; `PROMPT` and `parse_messages` imported from `tools/annotate_pilot.py`.
- Produces: `annotate_one(model: str, image_path: Path, call: Callable[[str, Path, str], tuple[str, dict | None]]) -> dict` returning `{"messages", "parsed", "tokens_in", "tokens_out", "model", "prompt_sha256", "image_sha256"}`.

- [ ] **Step 1: Write the failing tests**

```python
def _stub(text: str):
    def call(model, image_path, prompt):
        return text, {"input": 3179, "output": 99}
    return call


def test_provenance_is_recorded():
    good = json.dumps({"messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "Red."}]})

    result = annotate_one("model-x", Path("a.png"), _stub(good))

    assert result["model"] == "model-x"
    assert result["prompt_sha256"]
    assert result["image_sha256"]
    assert result["parsed"] is True
    assert result["tokens_in"] == 3179


def test_prose_output_is_marked_unparsed():
    result = annotate_one("model-x", Path("a.png"), _stub("sorry, I cannot"))

    assert result["parsed"] is False
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `annotate.py`, importing the pilot's prompt and parser**

Do not copy `PROMPT` or `parse_messages`; import them so the prompt hash cannot drift. Hash the image bytes and the prompt, and pass everything through unchanged. The transport callable is injected so the tests need no network.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/annotate.py tests
git commit -m "M2a: annotate with the frozen prompt, recording provenance"
```

### Task 5: Manifest, splits and the run record

**Files:**
- Create: `src/sightline/m2a/manifest.py`
- Create: `tools/build_m2a.py`
- Test: `tests/unit/test_m2a_manifest.py`

**Interfaces:**
- Consumes: annotated JSONL from Task 4.
- Produces: `splits(records: list[dict], seed: int = 0) -> dict[str, list[dict]]` into train/val/test, disjoint and complete; `manifest(records: list[dict], path: Path) -> dict` with the input sha256, per-source and per-licence counts, and the token projection at 3,179 per conversation.

- [ ] **Step 1: Write the failing tests**

```python
def test_splits_are_disjoint_and_complete():
    records = [{"id": i, "source": "coco", "hash": i} for i in range(300)]

    parts = splits(records, seed=0)
    flat = [r["id"] for part in parts.values() for r in part]

    assert len(flat) == 300
    assert len(set(flat)) == 300
    assert set(parts) == {"train", "val", "test"}


def test_split_assignment_is_reproducible():
    records = [{"id": i, "source": "coco", "hash": i} for i in range(300)]

    assert splits(records, seed=0) == splits(records, seed=0)
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write `tools/build_m2a.py`**

Subcommands `index`, `filter`, `fetch`, `annotate`, `manifest`, each reading the previous stage's JSONL and resuming from it. Print the record count and the drop count at every stage so a licence drop is visible as a number, not inferred.

- **Step 4: Implement `splits` and `manifest`, and write the run record**

Split by hash with a seeded RNG so the assignment is reproducible. The manifest carries the input sha256, per-source counts, licence counts, and the projection `n_conversations × 3179` tokens. Write a `RunRecord` for the build per project rule 8.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sightline/m2a/manifest.py tools/build_m2a.py tests
git commit -m "M2a: manifest, disjoint splits, and the run record for the build"
```

---

## Deviations from the plan

Recorded as they happened, so a later reader can see what was built differs from
what was written and why.

### Task 2 — the allow-list narrows, it never widens

- **Planned:** an explicit allow-list acts as an override, tested by admitting a
  `CC BY-NC 2.0` record.
- **Built:** the NC/ND checks run first and cannot be bypassed. An allow-list can
  only narrow further; an empty allow-list admits nothing.
- **Why:** the planned behaviour would let any caller re-admit a non-commercial
  image by passing a permissive list, which defeats the point of the gate. My
  first two tests asserted the unsafe behaviour; they were wrong, not the code.
- **Cost if wrong:** a caller who genuinely needs a non-commercial image for a
  research-only arm cannot get it through this function. That is the intended
  behaviour, and such a source should be a separate decision.

### Task 2 — reasons are reported, not just counted

- **Planned:** dropped records carry a `drop_reason`.
- **Built:** as planned, plus `not-in-allow-list` for records that pass the
  policy but are excluded by a narrowing list. Without it, a narrowing
  allow-list would silently shrink the corpus with no explanation.
- **Cost if wrong:** none; the reason string is additive.

### Task 1 — unknown licence ids stay visible

- **Planned:** records carry the licence fields; filtering is Task 2.
- **Built:** as planned. An id present in the image record but absent from the
  licence table yields `license_name=None` and the record is still returned, so
  the caller counts it rather than losing it.
- **Cost if wrong:** none; the gate treats it as `missing-licence`.

### Tests added beyond the plan's examples

- `test_unknown_licence_id_is_reported_not_dropped`, `test_names_are_matched_case_insensitively`,
  `test_an_empty_allow_list_admits_nothing`. All additive; each covers a case the
  plan's Review Focus named but its examples did not exercise.

## Notes for the executor

- The 20k iteration set is the target, not 131k: at 3,179 tokens each, 20k is ~64M and 131k is ~417M, over the 200M/day cap (`DECISIONS/0040`).
- The licence gate is built before fetching images, because afterwards it is tempting to keep what you already downloaded.
- COCO is the source because it records a licence id per image, which is what makes the gate automatable. Flickr30k would need manual review, PixMo has no per-image provenance, and The Cauldron carries 50 sub-licences.
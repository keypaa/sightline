# M2a Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn COCO 2017 into a multi-turn image-QA corpus with a recorded licence for every image, a recorded annotator and prompt for every conversation, and a manifest that makes the whole set reproducible.

**Architecture:** A small set of pure functions, each stage reading the previous stage's JSONL and writing its own. Nothing is held in memory across stages, so any stage can be re-run alone and a crash costs one stage. Images are fetched once into a content-addressed cache and never re-fetched.

**Tech Stack:** Python 3.13, `requests` for COCO range reads, `Pillow` for hashing, `transformers` not needed, `opencode run` as the annotator transport.

**Spec:** `reports/m2a-entry-criteria.md` (items 6, 7, 8), `DECISIONS/0035-projector-output-scale.md` for style, `DECISIONS/0040-m2a-annotator.md` for the annotator contract, `docs/measurement-pitfalls.md` for the ordered-split and sampling traps.

## Global Constraints

- No source enters without a recorded licence. A record with no licence id is dropped, not defaulted.
- The annotator is `kilo/nvidia/nemotron-3.5-super-vl:free`, prompt hash `b6d8d625b4ffc424`, recorded per conversation. One annotator for both arms of M2b, never switched.
- Measured annotator cost is 3,179 input tokens per conversation; 20k conversations is ~64M tokens, inside the 200M/day cap.
- Every stage writes JSONL and is resumable: re-running skips records already written, keyed by a stable id.
- Splits are sharded by source and by perceptual hash, never by taking the first N of an ordered file (`docs/measurement-pitfalls.md`).
- MMMU-Pro and BLINK images are excluded. Neither enters training.
- No emojis. Type hints. Comments explain intent.

## Review Focus

- **An image whose licence id is absent from the annotations.** Expected: dropped, counted, and reported — never assumed permissive.
- **A licence id whose name is not in the allow-list** (NC or ND variants). Expected: dropped with the reason recorded.
- **The same image reachable at two COCO ids** (duplicates exist in the wild). Expected: one survives, chosen deterministically, and the duplicate is counted.
- **An annotator call that returns prose instead of JSON.** Expected: the record is marked unparsed and the run continues; the failure rate is reported, not hidden.
- **A resharded corpus.** Expected: the same seed and the same inputs give the same split assignment, and no image appears in two splits.

---

### Task 1: COCO index reader with licence capture

**Files:**
- Create: `src/sightline/m2a/coco_index.py`
- Create: `src/sightline/m2a/__init__.py`
- Test: `tests/unit/test_m2a_coco_index.py`

**Interfaces:**
- Consumes: a COCO `instances_train2017.json` path.
- Produces: `read_annotations(path) -> tuple[list[dict], dict[int, dict]]` returning per-image records `{"file_name", "id", "license_id", "license_name", "license_url"}` and the licence table; `iter_licence_ids(records) -> set[int]`.

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

Read the JSON, index `licenses` by id, and emit one record per image with the licence fields joined. Do not filter here — filtering is Task 2, so the raw counts stay visible.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_coco_index.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a tools tests
git commit -m "M2a: read COCO annotations with their per-image licence ids"
```

### Task 2: Licence gate as a recorded decision

**Files:**
- Create: `src/sightline/m2a/licence.py`
- Create: `DECISIONS/0041-licence-allow-list.md`
- Test: `tests/unit/test_m2a_licence.py`

**Interfaces:**
- Consumes: `read_annotations` output from Task 1.
- Produces: `classify(records, allow: set[str]) -> tuple[list[dict], list[dict]]` returning kept and dropped records, each dropped record carrying a `drop_reason`.

- [ ] **Step 1: Write the failing tests**

```python
def test_non_commercial_and_no_derivative_are_dropped():
    records = [
        {"file_name": "a.jpg", "license_name": "CC BY 4.0"},
        {"file_name": "b.jpg", "license_name": "CC BY-NC 2.0"},
        {"file_name": "c.jpg", "license_name": "CC BY-ND 2.0"},
        {"file_name": "d.jpg", "license_name": None},
    ]

    kept, dropped = classify(records, allow={"cc by 4.0", "cc by 2.0", "cc0"})

    assert [r["file_name"] for r in kept] == ["a.jpg"]
    assert {r["drop_reason"] for r in dropped} == {"non-commercial", "no-derivatives", "missing-licence"}
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_licence.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write `DECISIONS/0041-licence-allow-list.md`**

State the allow-list as **any licence without NC or ND**, and require the licence *name* to be recorded from the annotations file rather than assumed. Record the reason: COCO's images are per-image Flickr licences and the consortium does not own them.

- [ ] **Step 4: Implement `classify` in `src/sightline/m2a/licence.py`**

Match on the normalised licence name, case-folded. A missing name is `missing-licence`. Return counts as part of the caller's report.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_licence.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sightline/m2a/licence.py tests DECISIONS
git commit -m "M2a: drop images without a permissive recorded licence"
```

### Task 3: Content-addressed image cache and duplicate detection

**Files:**
- Create: `src/sightline/m2a/images.py`
- Test: `tests/unit/test_m2a_images.py`

**Interfaces:**
- Consumes: kept records from Task 2, a download callable.
- Produces: `fetch(record, cache_dir, get) -> Path` returning the local path; `perceptual_hash(path) -> int`; `dedupe(records, hashes) -> tuple[list, list]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_same_image_at_two_ids_keeps_one(tmp_path):
    records = [{"file_name": "a.jpg", "id": 1}, {"file_name": "a.jpg", "id": 2}]
    hashes = {"a.jpg": 12345, }

    kept, dupes = dedupe(records, hashes)

    assert len(kept) == 1
    assert len(dupes) == 1
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement the cache and hash in `src/sightline/m2a/images.py`**

The cache key is the file name, which is stable in COCO. The perceptual hash is a 64-bit dHash via Pillow, implemented locally — no new dependency.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/images.py tests
git commit -m "M2a: cache images by name and drop duplicates by perceptual hash"
```

### Task 4: Conversation generation with provenance

**Files:**
- Create: `src/sightline/m2a/annotate.py`
- Test: `tests/unit/test_m2a_annotate.py`

**Interfaces:**
- Consumes: deduped records with local image paths; the prompt from `tools/annotate_pilot.py` (import it, do not copy it).
- Produces: `annotate_one(model, image_path, prompt) -> dict` returning `{"messages", "parsed", "tokens_in", "tokens_out", "model", "prompt_sha256"}`; `conversation_id(record, index) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_provenance_is_recorded_on_every_conversation():
    result = annotate_one("model-x", "prompt", _fake_call_returning_valid_json())

    assert result["model"] == "model-x"
    assert result["prompt_sha256"]
    assert result["parsed"] is True
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `annotate.py` reusing the pilot's prompt and parser**

Import `PROMPT` and `parse_messages` from `tools/annotate_pilot.py` so the prompt hash is shared. Record the model id and prompt hash on every row. Reuse the pilot's retry and `RateLimited` behaviour so a rate limit stops the run with partial results on disk.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/annotate.py tests
git commit -m "M2a: annotate with the frozen prompt, recording provenance per row"
```

### Task 5: Manifest, splits and the budget check

**Files:**
- Create: `src/sightline/m2a/manifest.py`
- Create: `tools/build_m2a.py`
- Test: `tests/unit/test_m2a_manifest.py`

**Interfaces:**
- Consumes: annotated JSONL from Task 4.
- Produces: `splits(records, seed=0) -> dict[str, list]` splitting by source and hash into train/val/test with no overlap; `manifest(records, path) -> dict` with sha256 of the file and per-source licence counts.

- [ ] **Step 1: Write the failing tests**

```python
def test_splits_never_overlap():
    records = [{"id": i, "source": "coco", "hash": i} for i in range(300)]

    parts = splits(records, seed=0)

    ids = [set(r["id"] for r in part) for part in parts.values()]
    assert sum(len(part) for part in parts.values()) == 300
    assert not (ids[0] & ids[1])
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Write `tools/build_m2a.py`**

A CLI with subcommands `index`, `filter`, `fetch`, `annotate`, `manifest`, each reading the previous stage's JSONL and resuming from it. Print counts at every stage so a licence drop is visible as a number.

- [ ] **Step 4: Implement `splits` and `manifest`**

Split by hash into three disjoint parts with a recorded seed. The manifest carries the input file sha256, per-source counts, licence counts, and the token projection at 3,179 tokens per conversation.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/sightline/m2a/manifest.py tools/build_m2a.py tests
git commit -m "M2a: manifest, disjoint splits, and the token projection"
```

---

## Notes for the executor

- The 20k iteration set is the target, not 131k. At 3,179 tokens per conversation, 20k is ~64M tokens and 131k is ~417M, which is over the 200M/day cap (`DECISIONS/0040`).
- The licence gate is the part that cannot be retrofitted. Build it before fetching images, because after fetching it is tempting to keep what you already have.
- COCO is the chosen source because it records a licence id per image. That is the property that makes Task 2 automatable; Flickr30k and PixMo would both need manual review.
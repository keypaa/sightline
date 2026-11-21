# M2a clean data build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the M2a 20k corpus end to end from the committed COCO index, with one annotator, one recorded permutation, and full per-example provenance, so that every number in the manifest is reproducible.

**Architecture:** A staged pipeline where each stage reads the previous stage's JSONL and writes its own, so a crash costs one stage. Provenance is a single `provenance.py` module that every annotation row is built from, so a field cannot be forgotten by one call site. The annotator transport stays injected, so the same code path serves the pilot and the full run.

**Tech Stack:** Python 3.14, pytest, ruff, Pillow, requests. Annotator served by vLLM (Qwen3.6-27B-FP8) on the local RTX PRO 6000.

**Spec:** `DECISIONS/0046-m2a-source-coco-only.md` (source decision and the eight required measurements), `DECISIONS/0042-m2a-split-ratios.md` (80/10/10), `DECISIONS/0043-non-commercial-images.md` (licence policy), `DECISIONS/0040-m2a-annotator.md` (annotator contract).

## Global Constraints

- Licence policy is the signed one: non-commercial allowed, no-derivatives and missing dropped (`DECISIONS/0043`, implemented in `src/sightline/m2a/licence.py`).
- Seeds are 0, 1, 2. A literal 42 as a seed is a defect (`AGENTS.md`).
- No inherited data from the previous project (`AGENTS.md` rule 5).
- Never write durable data to `/tmp` (`docs/measurement-pitfalls.md` section 0).
- Every run writes a run record; metrics go to JSONL, not print statements (`AGENTS.md` rules 8 and 9).
- Lint is `ruff check src tests tools`; format is `ruff format --check src tests tools`.
- Tests live in `tests/unit/`, run with `pytest` (pythonpath is `src`).
- The old Kilo annotator (`kilo/nvidia/nemotron-3.5-super-vl:free`) must not appear in any row of the new corpus.

## Review Focus

- **A fetch that returns the wrong bytes.** A 200 response with an HTML error page would be hashed and licensed as if it were the image. Expected: the row is dropped with a reason, never written as kept. → Task 3.
- **A retry that re-annotates a finished image.** Resume is by id; a duplicate row for one id makes the corpus count wrong and splits leak. Expected: each id appears exactly once. → Task 7.
- **A split boundary crossed by a near-duplicate.** dHash dedup happens at fetch, but a recompressed copy can still land in both train and test. Expected: the manifest reports zero cross-split hash collisions. → Task 9.
- **A silent annotator change.** Same prompt text, different model or template, produces rows that look comparable and are not. Expected: the run record names model, revision, quantisation, engine version and both hashes. → Task 6.
- **A decontamination claim nobody measured.** MMMU-Pro and BLINK are absent from FineVision's 66-benchmark dedup, so COCO carries no dedup claim at all. Expected: the dedup report names both benchmarks and the count removed. → Task 10.

---

### Task 1: Delete the Kilo corpus and confirm the tree is clean

**Files:**
- Delete: `experiments/m2a_iter20k/conversations.jsonl`
- Delete: `experiments/m2a_iter20k/conversations.pre_purge.jsonl`
- Delete: `experiments/m2a_iter20k/conversations.pre_stop_purge.jsonl`
- Keep: `experiments/m2a_iter20k/index.jsonl`, `filtered.jsonl`, `fetched.jsonl`
- Test: `tests/unit/test_m2a_provenance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no row in `experiments/m2a_iter20k/` names the Kilo model. `sightline.m2a.provenance.annotator_fields(...) -> dict[str, Any]` (defined in Task 5) is the only writer of model fields from here on.

- [ ] **Step 1: Write the failing test**

```python
def test_no_kilo_rows_remain_in_the_20k_tree():
    from pathlib import Path
    import json

    root = Path("experiments/m2a_iter20k")
    rows = []
    for path in root.glob("*conversations*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    assert rows, "expected the fetched index to still be present"
    assert all("kilo" not in r.get("model", "") for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_provenance.py::test_no_kilo_rows_remain_in_the_20k_tree -v`
Expected: FAIL — 402 rows name `kilo/nvidia/nemotron-3.5-super-vl:free`.

- [ ] **Step 3: Delete the three conversation files**

```bash
rm experiments/m2a_iter20k/conversations.jsonl \
   experiments/m2a_iter20k/conversations.pre_purge.jsonl \
   experiments/m2a_iter20k/conversations.pre_stop_purge.jsonl
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_provenance.py::test_no_kilo_rows_remain_in_the_20k_tree -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A tests/unit/test_m2a_provenance.py
git commit -m "M2a: delete the Kilo annotator rows before the clean rebuild"
```

---

### Task 2: Record the corpus order once, as a file

**Files:**
- Create: `src/sightline/m2a/order.py`
- Test: `tests/unit/test_m2a_order.py`

**Interfaces:**
- Consumes: a list of `fetched.jsonl` records, each with an `id`.
- Produces:
  - `corpus_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]` — the records in corpus order.
  - `write_order(path: Path, records: list[dict[str, Any]], seed: int) -> None` — writes `order.jsonl` with one `{"position": int, "id": Any}` per line.
  - `read_order(path: Path) -> list[dict[str, Any]]` — reads it back.
  - `apply_order(records: list[dict[str, Any]], order: list[dict[str, Any]]) -> list[dict[str, Any]]` — returns records sorted by recorded position; raises `ValueError` if an id is missing from either side.

- [ ] **Step 1: Write the failing test**

```python
def test_order_is_a_permutation_and_is_reproducible():
    records = [{"id": i} for i in range(10)]
    first = corpus_order(records, seed=0)
    second = corpus_order(records, seed=0)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert sorted(r["id"] for r in first) == list(range(10))


def test_order_survives_a_round_trip(tmp_path):
    records = [{"id": i} for i in range(5)]
    write_order(tmp_path / "order.jsonl", corpus_order(records, seed=0), seed=0)
    restored = apply_order(list(reversed(records)), read_order(tmp_path / "order.jsonl"))
    assert [r["id"] for r in restored] == [r["id"] for r in corpus_order(records, seed=0)]


def test_apply_order_rejects_a_missing_id():
    with pytest.raises(ValueError):
        apply_order([{"id": 0}], [{"position": 0, "id": 1}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_order.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.order'`.

- [ ] **Step 3: Implement `src/sightline/m2a/order.py`**

`corpus_order` uses `random.Random(seed).sample(list(records), len(records))` so the permutation is total and reproducible. `write_order` writes `{"position": i, "id": r["id"]}` per line in the given order. `apply_order` builds `{id: position}` from the order file, then returns records sorted by that position, raising `ValueError` when the id sets differ.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_order.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/order.py tests/unit/test_m2a_order.py
git commit -m "M2a: record the corpus permutation once instead of re-deriving it"
```

---

### Task 3: Make fetch verify what it downloaded

**Files:**
- Modify: `src/sightline/m2a/images.py`
- Modify: `tools/build_m2a.py` (`cmd_fetch`, `_get`)
- Test: `tests/unit/test_m2a_images.py`

**Interfaces:**
- Consumes: `fetch(record, cache_dir, get) -> Path` and `content_hash(path) -> str` from `images.py`.
- Produces:
  - `sniff_image(path: Path) -> str | None` — returns the Pillow format name (`"JPEG"`, `"PNG"`, ...) or `None` when the bytes are not an image.
  - `fetch(...)` now raises `ValueError` when `sniff_image` returns `None` or when the downloaded bytes do not match `record["expected_sha256"]` (when that key is present).
  - `cmd_fetch` writes `fetched.jsonl` with an added `order_seed` field on every row, and drops failures into `fetch_dropped.jsonl` with a `drop_reason`.

- [ ] **Step 1: Write the failing test**

```python
def test_sniff_rejects_an_html_error_page(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"<html><body>404 not found</body></html>")
    assert sniff_image(path) is None


def test_sniff_accepts_a_real_jpeg(tmp_path):
    from PIL import Image

    path = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8), "red").save(path)
    assert sniff_image(path) == "JPEG"


def test_fetch_rejects_bytes_that_do_not_match_the_expected_hash(tmp_path):
    from PIL import Image

    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    record = {"file_name": "good.jpg", "url": "http://x/good.jpg",
              "expected_sha256": "0" * 64}
    with pytest.raises(ValueError):
        fetch(record, tmp_path / "cache", lambda url: good.read_bytes())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -v`
Expected: FAIL — `ImportError: cannot import name 'sniff_image'`.

- [ ] **Step 3: Implement `sniff_image` and the checks in `fetch`**

Open with `Image.open(path)` and return `image.format`; return `None` on `UnidentifiedImageError` or any `OSError`. In `fetch`, after writing the file, call `sniff_image`; if `None`, raise `ValueError("not an image")`. If `record.get("expected_sha256")` is set and differs from `content_hash(path)`, raise `ValueError("checksum mismatch")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_images.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/images.py tests/unit/test_m2a_images.py
git commit -m "M2a: verify a fetched image really is an image and matches its checksum"
```

---

### Task 4: Extend the COCO index with `expected_sha256` and an explicit source

**Files:**
- Modify: `src/sightline/m2a/coco_index.py`
- Test: `tests/unit/test_m2a_coco_index.py`

**Interfaces:**
- Consumes: `read_annotations(path: Path) -> tuple[list[dict], dict[int, dict]]`.
- Produces: each record from `read_annotations` also carries `"source": "coco"` and `"expected_sha256": None`. `cmd_index` in `tools/build_m2a.py` writes those fields to `index.jsonl` and adds `"source": "coco"` to every row. `expected_sha256` is `None` here because the COCO annotations file carries no per-image checksum; `cmd_fetch` fills it in after the first successful download, so a later re-fetch can be verified against it (Task 3 reads the key when present).

- [ ] **Step 1: Write the failing test**

```python
def test_index_records_carry_an_explicit_source(tmp_path):
    import json

    path = tmp_path / "ann.json"
    path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "license": 1,
                    "flickr_url": "http://f/a", "coco_url": "http://c/a"}],
        "licenses": [{"id": 1, "name": "Attribution License", "url": "http://l"}],
        "annotations": [],
    }))
    records, _ = read_annotations(path)
    assert records[0]["source"] == "coco"
    assert records[0]["expected_sha256"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_coco_index.py::test_index_records_carry_an_explicit_source -v`
Expected: FAIL — `KeyError: 'source'`.

- [ ] **Step 3: Add the two fields in `read_annotations`**

Set `record["source"] = "coco"` and `record.setdefault("expected_sha256", None)` on each record before returning.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_coco_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/coco_index.py tests/unit/test_m2a_coco_index.py
git commit -m "M2a: give every indexed image an explicit source"
```

---

### Task 5: One module that builds the provenance fields

**Files:**
- Create: `src/sightline/m2a/provenance.py`
- Test: `tests/unit/test_m2a_provenance.py`

**Interfaces:**
- Consumes: `prompt_hash` from `sightline.m2a.annotate`, `file_hash` from the same module.
- Produces:
  - `chat_template_hash(template: str) -> str` — 16-char sha256 prefix, same convention as `prompt_hash`.
  - `annotator_fields(*, model: str, revision: str, quantization: str, engine: str, engine_version: str, prompt_sha256: str, chat_template_sha256: str, seed: int, order_seed: int, temperature: float) -> dict[str, Any]` — returns exactly these keys, plus `"model"` and `"prompt_sha256"` for backward compatibility.
  - `row_provenance(record: dict[str, Any], image_sha256: str, annotator: dict[str, Any]) -> dict[str, Any]` — merges `source, license_id, license_name, license_url, flickr_url, coco_url, url, sha256, image_sha256` from the record with the annotator fields.

- [ ] **Step 1: Write the failing test**

```python
def test_annotator_fields_pin_every_reproducibility_key():
    fields = annotator_fields(
        model="Qwen/Qwen3.6-27B-FP8", revision="abc123", quantization="fp8",
        engine="vllm", engine_version="0.17.0", prompt_sha256="b6d8d625b4ffc424",
        chat_template_sha256="deadbeefdeadbeef", seed=0, order_seed=0, temperature=0.0,
    )
    assert fields["revision"] == "abc123"
    assert fields["quantization"] == "fp8"
    assert fields["engine"] == "vllm"
    assert fields["engine_version"] == "0.17.0"
    assert fields["seed"] == 0
    assert fields["temperature"] == 0.0


def test_row_provenance_keeps_source_licence_and_checksum():
    row = row_provenance(
        {"source": "coco", "license_name": "CC BY 2.0", "sha256": "aa",
         "flickr_url": "http://f", "coco_url": "http://c", "url": "http://u",
         "license_id": 1, "license_url": "http://l"},
        image_sha256="bb",
        annotator={"model": "m"},
    )
    assert row["source"] == "coco"
    assert row["license_name"] == "CC BY 2.0"
    assert row["flickr_url"] == "http://f"
    assert row["sha256"] == "aa"
    assert row["image_sha256"] == "bb"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.provenance'`.

- [ ] **Step 3: Implement `src/sightline/m2a/provenance.py`**

`annotator_fields` returns a dict literal with exactly the pinned keys; no defaults beyond the ones passed. `row_provenance` copies the nine record fields, then `image_sha256`, then `**annotator`, so a record field cannot shadow an annotator field.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_provenance.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/provenance.py tests/unit/test_m2a_provenance.py
git commit -m "M2a: one module that pins every provenance field"
```

---

### Task 6: A local vLLM transport behind the existing `Transport` signature

**Files:**
- Create: `src/sightline/m2a/local_transport.py`
- Test: `tests/unit/test_m2a_local_transport.py`

**Interfaces:**
- Consumes: the `Transport = Callable[[str, Path, str], tuple[str, dict[str, Any] | None]]` alias from `sightline.m2a.annotate`.
- Produces:
  - `LocalVLLM(model: str, *, base_url: str, chat_template_kwargs: dict[str, Any] | None = None, temperature: float = 0.0, max_tokens: int = 512)` with `__call__(self, model: str, image_path: Path, prompt: str) -> tuple[str, dict[str, Any] | None]`.
  - Raises `RateLimited` (from `sightline.m2a.transport`) on HTTP 429 and `TransportError` on any other non-200.
  - Image is sent as a base64 `data:image/...;base64,` URL derived from `sniff_image`.

- [ ] **Step 1: Write the failing test**

```python
def test_local_transport_builds_an_image_message(monkeypatch, tmp_path):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)

    captured = {}

    class Resp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": '{"messages": []}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return Resp()

    monkeypatch.setattr("sightline.m2a.local_transport.requests.post", fake_post)
    client = LocalVLLM("Qwen/Qwen3.6-27B-FP8", base_url="http://localhost:8000/v1")
    text, tokens = client("Qwen/Qwen3.6-27B-FP8", img, "prompt")

    assert text == '{"messages": []}'
    assert tokens == {"input": 10, "output": 5}
    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["type"] == "text"


def test_local_transport_maps_429_to_rate_limited(monkeypatch, tmp_path):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)

    class Resp:
        status_code = 429
        def json(self):
            return {"error": {"message": "Rate limit exceeded"}}

    monkeypatch.setattr("sightline.m2a.local_transport.requests.post",
                        lambda *a, **k: Resp())
    with pytest.raises(RateLimited):
        LocalVLLM("m", base_url="http://x/v1")("m", img, "p")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_local_transport.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.local_transport'`.

- [ ] **Step 3: Implement `src/sightline/m2a/local_transport.py`**

Build `content = [{"type": "image_url", "image_url": {"url": f"data:image/{fmt.lower()};base64,{b64}"}}, {"type": "text", "text": prompt}]` where `fmt = sniff_image(image_path)`. POST to `f"{base_url}/chat/completions"` with `model`, `messages`, `temperature`, `max_tokens`, and `chat_template_kwargs` when given. Map `status_code == 429` to `RateLimited`, any other non-200 to `TransportError`. Return `content` and `{"input": usage["prompt_tokens"], "output": usage["completion_tokens"]}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_local_transport.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/local_transport.py tests/unit/test_m2a_local_transport.py
git commit -m "M2a: local vLLM transport with the same signature as the CLI one"
```

---

### Task 7: Wire annotate to provenance, the recorded order, and the local transport

**Files:**
- Modify: `src/sightline/m2a/annotate.py`
- Modify: `tools/build_m2a.py` (`cmd_annotate`)
- Test: `tests/unit/test_m2a_annotate.py`

**Interfaces:**
- Consumes: `annotator_fields`, `row_provenance` (Task 5); `read_order`, `apply_order` (Task 2); `LocalVLLM` (Task 6); `pending_records` from `annotate.py`.
- Produces:
  - `annotate_one(model, image_path, call, *, annotator: dict[str, Any]) -> dict[str, Any]` — the annotator dict is merged into the returned row, and the row now carries `image_sha256` and the annotator fields. `prompt_sha256` moves into the annotator dict, so a caller that does not pass one records no prompt hash rather than a stale default.
  - `cmd_annotate` reads `order.jsonl` when present and applies it before dispatch; writes `conversations.jsonl` and `annotate_failures.jsonl`. `cmd_fetch` gains `--write-checksums`, which writes `expected_sha256` back into `fetched.jsonl` after a successful download.

- [ ] **Step 1: Write the failing test**

```python
def test_annotate_one_records_the_annotator_fields(tmp_path):
    from PIL import Image
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(img)

    annotator = {"model": "Qwen/Qwen3.6-27B-FP8", "revision": "abc123",
                 "quantization": "fp8", "seed": 0, "order_seed": 0}
    result = annotate_one("Qwen/Qwen3.6-27B-FP8", img, stub(GOOD), annotator=annotator)

    assert result["revision"] == "abc123"
    assert result["quantization"] == "fp8"
    assert result["parsed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py::test_annotate_one_records_the_annotator_fields -v`
Expected: FAIL — `TypeError: annotate_one() got an unexpected keyword argument 'annotator'`.

- [ ] **Step 3: Add the `annotator` parameter and merge it**

In `annotate_one`, add `*, annotator: dict[str, Any]` and build the return dict as `{**annotator, "messages": ..., "parsed": ..., "tokens_in": ..., "tokens_out": ..., "image_sha256": ...}`. In `cmd_annotate`, build the annotator dict from `annotator_fields(...)` using the CLI's `--revision`, `--quantization`, `--engine-version`, `--temperature` arguments, and apply `read_order`/`apply_order` to the fetched records before building `pending`. Existing callers that omit `annotator` keep working but record no model provenance — the pilot and the full run must both pass it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/annotate.py tools/build_m2a.py tests/unit/test_m2a_annotate.py
git commit -m "M2a: every annotation row carries its annotator provenance"
```

---

### Task 8: Annotator pilot against the frozen 0.90 bar

**Files:**
- Create: `tools/m2a_annotator_pilot.py`
- Create: `reports/m2a-annotator-pilot-local.md` (written by the run, not by hand)

**Interfaces:**
- Consumes: `LocalVLLM` (Task 6), `annotate_one` (Task 7), `parse_messages` and `answer_forms` from `sightline.m2a.prompt`. Grounding is computed here, because `annotate_one` does not produce a `grounded` field and `annotate_pilot.py` computes it separately — the pilot must not inherit a second copy of that rule.
- Produces: `experiments/m2a_pilot/pilot.jsonl` with one row per image carrying `grounded`, and a printed summary of parse rate, grounded rate, tokens in/out, and wall-clock per conversation.

- [ ] **Step 1: Write the failing test**

```python
def test_pilot_summary_computes_the_two_rates():
    rows = [
        {"parsed": True, "grounded": True},
        {"parsed": True, "grounded": False},
        {"parsed": False, "grounded": False},
    ]
    summary = summarise(rows)
    assert summary["parse_rate"] == 2 / 3
    assert summary["grounded_rate"] == 1 / 3
    assert summary["n"] == 3
```

The grounded test rows are constructed by hand here; `summarise` only reads the `grounded` key. The rule that *produces* `grounded` is tested in the step below against `answer_forms`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_pilot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.m2a_annotator_pilot'`.

- [ ] **Step 3: Implement `grounded_in(messages, expected) -> bool` and `summarise(rows)`**

`grounded_in` mirrors the existing rule in `tools/annotate_pilot.py`: true when any form from `answer_forms(expected)` appears in any assistant message, lower-cased. Add its test beside the summary test:

```python
def test_grounded_in_uses_answer_forms():
    messages = [{"role": "assistant", "content": "Three."}]
    assert grounded_in(messages, "3") is True
    assert grounded_in(messages, "4") is False
```

`summarise` returns `{"n", "parse_rate", "grounded_rate", "tokens_in_mean", "tokens_out_mean"}`. The CLI takes `--images N`, `--host`, `--model`, `--revision`, `--quantization`, `--engine-version`, samples with `sample_subset(..., seed=0)`, calls `annotate_one` with `LocalVLLM`, writes `pilot.jsonl`, and prints the summary plus `conversations/hour`. Freeze the thresholds in the script docstring: parse >= 0.95, grounded >= 0.90 (the pre-declared bar from `DECISIONS/0040`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_pilot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/m2a_annotator_pilot.py tests/unit/test_m2a_pilot.py
git commit -m "M2a: annotator pilot that reports the two frozen rates"
```

---

### Task 9: Manifest records provenance, source count and coverage

**Files:**
- Modify: `src/sightline/m2a/manifest.py`
- Test: `tests/unit/test_m2a_manifest.py`

**Interfaces:**
- Consumes: `manifest(path, tokens_per_conversation=...) -> dict`, `splits(records, seed=0)`, `usable_records(records)`.
- Produces: the manifest dict gains `"sources": dict[str, int]`, `"source_count": int`, `"annotators": dict[str, int]`, `"order_seed": int | None`, `"n_cross_split_hash_collisions": int`, and `"category_coverage": dict[str, int]` when rows carry a `category` field.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_reports_sources_and_collisions(tmp_path):
    path = tmp_path / "conversations.jsonl"
    rows = [
        {"id": 0, "source": "coco", "license_name": "CC BY 2.0", "parsed": True,
         "model": "Qwen/Qwen3.6-27B-FP8", "sha256": "aa"},
        {"id": 1, "source": "coco", "license_name": "CC BY 2.0", "parsed": True,
         "model": "Qwen/Qwen3.6-27B-FP8", "sha256": "bb"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)
    assert result["source_count"] == 1
    assert result["sources"] == {"coco": 2}
    assert result["annotators"] == {"Qwen/Qwen3.6-27B-FP8": 2}
    assert result["n_cross_split_hash_collisions"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py::test_manifest_reports_sources_and_collisions -v`
Expected: FAIL — `KeyError: 'source_count'`.

- [ ] **Step 3: Extend `manifest`**

Count `sources` and `annotators` with the same `dict[str, int]` pattern already used for `licences`. Compute cross-split collisions by splitting on `usable_records(rows)` with `splits(..., seed=args.seed)`, collecting `sha256` per split, and counting ids whose hash appears in more than one split. Add `order_seed` from the first row that carries one. Add `category_coverage` only when any row has `category`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/manifest.py tests/unit/test_m2a_manifest.py
git commit -m "M2a: manifest reports sources, annotators and cross-split collisions"
```

---

### Task 10: SSCD decontamination against MMMU-Pro and BLINK

**Files:**
- Create: `src/sightline/m2a/decontaminate.py`
- Modify: `tools/build_m2a.py` (new `decontaminate` subcommand)
- Test: `tests/unit/test_m2a_decontaminate.py`

**Interfaces:**
- Consumes: `dhash` and `distance` from `sightline.m2a.images`; a benchmark manifest of `{name: [image_sha256, ...]}`.
- Produces:
  - `load_benchmark_hashes(path: Path) -> dict[str, set[str]]`
  - `find_contamination(records: list[dict], benchmark: dict[str, set[str]], *, sscd_similarity: Callable[[Path, Path], float] | None = None) -> list[dict]` — returns one row per contaminated record with `{"id", "benchmark", "sha256", "similarity"}`.
  - `write_report(path: Path, rows: list[dict], n_records: int) -> dict` — writes JSONL and returns `{"n_records", "n_contaminated", "per_benchmark": {name: count}}`.
  - The `decontaminate` subcommand writes `decontamination.jsonl` and `decontamination_report.json` into the run root and exits 0.

- [ ] **Step 1: Write the failing test**

```python
def test_exact_hash_match_is_contamination():
    records = [{"id": 0, "sha256": "aa", "path": "/x/0.jpg"}]
    hits = find_contamination(records, {"MMMU-Pro": {"aa"}})
    assert hits == [{"id": 0, "benchmark": "MMMU-Pro", "sha256": "aa", "similarity": 1.0}]


def test_report_counts_per_benchmark(tmp_path):
    rows = [{"id": 0, "benchmark": "MMMU-Pro", "sha256": "aa", "similarity": 1.0},
            {"id": 1, "benchmark": "BLINK", "sha256": "bb", "similarity": 1.0}]
    summary = write_report(tmp_path / "r.jsonl", rows, n_records=10)
    assert summary["n_contaminated"] == 2
    assert summary["per_benchmark"] == {"MMMU-Pro": 1, "BLINK": 1}


def test_report_names_both_benchmarks_even_when_clean(tmp_path):
    summary = write_report(tmp_path / "r.jsonl", [], n_records=5)
    assert summary["per_benchmark"] == {}
    assert summary["n_contaminated"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_decontaminate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.decontaminate'`.

- [ ] **Step 3: Implement `src/sightline/m2a/decontaminate.py`**

`find_contamination` checks exact `sha256` against every benchmark set, reporting `similarity: 1.0`. When `sscd_similarity` is supplied, it also compares each record image against that benchmark's images and reports the max similarity, keeping rows at or above 0.95. `write_report` always emits `per_benchmark` keyed by the benchmarks that were checked — pass the checked names in so a clean run still proves both were tested.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_decontaminate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/decontaminate.py tools/build_m2a.py tests/unit/test_m2a_decontaminate.py
git commit -m "M2a: SSCD decontamination against MMMU-Pro and BLINK"
```

---

## Execution order and dependencies

Tasks 1–5 are independent of the GPU and land first. Task 6 needs vLLM installed and the model served. Task 8 needs Task 6. Tasks 7 and 9 can land before or after the pilot, but Task 8 must pass its frozen bar before Task 7 runs at 20k scale. Task 10 needs the MMMU-Pro and BLINK image sets, which is the recorded `[U]` in `DECISIONS/0046`; it can land last without blocking the iteration.

## Definition of done

- `python3 -m pytest tests/unit -q` passes.
- `python3 -m ruff check src tests tools` and `python3 -m ruff format --check src tests tools` pass.
- `experiments/m2a_iter20k/` contains no row naming a Kilo model.
- `experiments/m2a_iter20k/order.jsonl` exists and `manifest.json` reports `order_seed`, `source_count: 1`, and the annotator model.
- A pilot report exists with parse rate, grounded rate and conversations/hour.
- A decontamination report exists naming MMMU-Pro and BLINK, even if the count is zero.
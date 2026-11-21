# M2a corpus pipeline: schemas, ids and the review loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the M2a corpus self-describing — every row carries the identity of what produced it, every claim has a stable id, and verdicts, blind results and human decisions join on that id.

**Architecture:** Add one small identity module (`ids.py`) and thread `run_id`, `created_at` and `claim_id` through the existing stages. Verdicts become flat rows so they can be appended, resumed and joined per verifier. A new corpus-manifest aggregator reads every layer; the existing corpus manifest is untouched.

**Tech Stack:** Python 3.14, pytest, ruff. No new dependencies.

**Spec:** `zenno/specs/2026-11-21-m2a-corpus-schemas.md`

## Global Constraints

- Seeds are 0, 1, 2. A literal 42 as a seed is a defect.
- No emojis. Type hints required. Comments explain intent, not mechanics.
- Tests live in `tests/unit/`, run with `python3 -m pytest` (pythonpath is `src`).
- Lint: `python3 -m ruff check src tests tools`. Format: `python3 -m ruff format --check src tests tools`.
- Metrics go to JSONL, not print statements.
- The existing `manifest()` in `src/sightline/m2a/manifest.py` must keep working unchanged.
- Frozen thresholds from `DECISIONS/0049`: contradicted rate upper 95% bound ≤ 0.15, planted rejection ≥ 0.90, human agreement ≥ 0.85.
- Verifier is `opencode/hy4-flash-free`; annotator is `Qwen/Qwen3.6-27B-FP8`. They must never be equal.

## Review Focus

- **A re-run with a different annotator** — appending rows from a second model would silently make the corpus two-annotator. Expected: the writer refuses. → Task 2.
- **An error verdict counted as "can't-tell"** — 4,139 rate limits once looked like a cautious model. Expected: errors are excluded from every rate. → Task 4.
- **`claim_id` shifting between extracts** — a sequence or an unstable hash would break every join. Expected: identical claims yield identical ids across runs. → Task 1, Task 3.
- **A claim whose image is missing** — `image_path` null must not drop the claim. Expected: the claim survives and verdicts still run. → Task 3.
- **A human decision pointing at a claim that does not exist** — an orphan decision would inflate the agreement figure. Expected: import rejects it. → Task 6.

---

### Task 1: Identity helpers

**Files:**
- Create: `src/sightline/m2a/ids.py`
- Test: `tests/unit/test_m2a_ids.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `claim_id(image_id: int, turn: int, claim: str) -> str` — 16 lowercase hex chars.
  - `new_run_id() -> str` — `"<UTC compact>-<8 hex>"`, e.g. `"20261121T174500Z-1a2b3c4d"`.
  - `now_iso() -> str` — timezone-aware ISO 8601, seconds precision.

- [ ] **Step 1: Write the failing test**

```python
def test_claim_id_is_stable_and_short():
    first = claim_id(12, 1, "Red (car)")
    assert first == claim_id(12, 1, "Red (car)")
    assert len(first) == 16
    assert first != claim_id(12, 2, "Red (car)")
    assert first != claim_id(13, 1, "Red (car)")


def test_run_id_has_a_timestamp_and_a_random_tail():
    run = new_run_id()
    assert run[8] == "T" and run.endswith("Z") is False
    assert "-" in run and len(run.split("-")[-1]) == 8
    assert new_run_id() != new_run_id()


def test_now_iso_is_timezone_aware():
    value = now_iso()
    assert value.endswith("+00:00")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_ids.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.ids'`

- [ ] **Step 3: Implement `src/sightline/m2a/ids.py`**

`claim_id` hashes `f"{image_id}:{turn}:{claim}"` with `hashlib.sha256` and returns the first 16 hex chars. `new_run_id` formats `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")` plus `uuid4().hex[:8]`. `now_iso` is `datetime.now(timezone.utc).isoformat(timespec="seconds")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_ids.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/ids.py tests/unit/test_m2a_ids.py
git commit -m "M2a: stable claim ids and run identity"
```

---

### Task 2: Annotate writes run identity and refuses a foreign annotator

**Files:**
- Modify: `src/sightline/m2a/annotate.py`
- Modify: `tools/build_m2a.py` (`cmd_annotate`)
- Modify: `tools/m2a_annotator_pilot.py` (the row it writes)
- Test: `tests/unit/test_m2a_annotate.py`

**Interfaces:**
- Consumes: `new_run_id`, `now_iso` from Task 1.
- Produces:
  - `annotate_row(record: dict, result: dict, *, run_id: str, created_at: str) -> dict` in `annotate.py` — the conversation row, dropping `path` and adding the two identity fields.
  - `check_single_annotator(existing: list[dict], row: dict) -> None` in `annotate.py` — raises `ValueError` when `row` disagrees with the first existing row on `model`, `revision`, `quantization`, `prompt_sha256` or `chat_template_sha256`.
  - `cmd_annotate` generates one `run_id` per invocation and passes the same value to every row.

- [ ] **Step 1: Write the failing tests**

```python
def test_annotate_row_drops_path_and_adds_identity():
    row = annotate_row(
        {"id": 7, "file_name": "a.jpg", "path": "/tmp/a.jpg"},
        {"messages": [], "parsed": True, "model": "m", "image_sha256": "ab"},
        run_id="run-1",
        created_at="2026-11-21T00:00:00+00:00",
    )
    assert "path" not in row
    assert row["run_id"] == "run-1"
    assert row["created_at"].startswith("2026-11-21")
    assert row["id"] == 7


def test_check_single_annotator_allows_a_matching_row():
    existing = [{"model": "m", "revision": "r", "quantization": "fp8",
                 "prompt_sha256": "p", "chat_template_sha256": "c"}]
    check_single_annotator(existing, dict(existing[0]))


def test_check_single_annotator_refuses_a_different_model():
    existing = [{"model": "m", "revision": "r", "quantization": "fp8",
                 "prompt_sha256": "p", "chat_template_sha256": "c"}]
    with pytest.raises(ValueError, match="annotator"):
        check_single_annotator(existing, {**existing[0], "model": "other"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -v`
Expected: FAIL — `ImportError: cannot import name 'annotate_row'`

- [ ] **Step 3: Implement in `annotate.py` and wire `cmd_annotate`**

`annotate_row` copies every key of `record` except `path`, updates with `result`, then sets `run_id` and `created_at`. `check_single_annotator` returns immediately on an empty list, otherwise compares the five fields against `existing[0]` and raises `ValueError("annotator identity differs from the first row in this corpus")` on a mismatch.

In `cmd_annotate`, generate `run_id = new_run_id()` before the pool starts, and call `check_single_annotator(read_jsonl(out), row)` inside `on_row` before writing. In `m2a_annotator_pilot.py`, build the row through `annotate_row` with a `run_id` generated once per invocation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_annotate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/annotate.py tools/build_m2a.py tools/m2a_annotator_pilot.py tests/unit/test_m2a_annotate.py
git commit -m "M2a: every conversation row carries its run and refuses a second annotator"
```

---

### Task 3: Extract emits claim ids and an explicit image foreign key

**Files:**
- Modify: `src/sightline/m2a/verify.py` (`Claim`, `extract_claims`, `extract` CLI)
- Test: `tests/unit/test_m2a_verify.py`

**Interfaces:**
- Consumes: `claim_id` from Task 1.
- Produces:
  - `Claim` gains `claim_id: str = ""` and keeps `image_path: str | None`.
  - `extract_claims(rows, images_dir=None, project_root=None) -> list[Claim]` sets `claim_id` and leaves `image_path` `None` when it cannot be resolved.
  - The `extract` CLI writes `image_id` (not `image_id`+`id`) and `claim_id` on every row.

- [ ] **Step 1: Write the failing tests**

```python
def test_claim_id_is_set_and_stable_across_extracts():
    rows = [{"id": 5, "file_name": "a.jpg",
             "messages": [{"role": "user", "content": "What colour is the car?"},
                          {"role": "assistant", "content": "Red."}]}]
    first = extract_claims(rows)
    second = extract_claims(rows)
    assert first[0].claim_id
    assert first[0].claim_id == second[0].claim_id
    assert first[0].claim_id == claim_id(5, 1, "Red (car)")


def test_a_missing_image_does_not_drop_the_claim(tmp_path):
    rows = [{"id": 5, "file_name": "gone.jpg",
             "messages": [{"role": "user", "content": "What colour is the car?"},
                          {"role": "assistant", "content": "Red."}]}]
    claim = extract_claims(rows, images_dir=tmp_path)[0]
    assert claim.image_path is None
    assert claim.claim_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_verify.py -v`
Expected: FAIL — `AttributeError: 'Claim' object has no attribute 'claim_id'`

- [ ] **Step 3: Implement**

Add `claim_id: str = ""` to the `Claim` dataclass. In `extract_claims`, after building the claim text, set `claim_id=claim_id(row["id"], turn, claim_text)` — computing the text once into a local so the id and the stored `claim` cannot differ. `resolve_image_path` already returns `None` on a miss; keep that. In the `extract` CLI, write `"image_id": row.get("image_id", row.get("id"))` and include `claim_id`; stop writing a separate `id` key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_verify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/verify.py tests/unit/test_m2a_verify.py
git commit -m "M2a: claims get a stable id, and a missing image does not drop them"
```

---

### Task 4: Verdicts become flat rows with a prompt hash

**Files:**
- Modify: `src/sightline/m2a/verify.py` (`judge_claims`, `_result_to_row`, `judge` CLI)
- Test: `tests/unit/test_m2a_verify.py`, `tests/unit/test_m2a_verifier_transport.py`

**Interfaces:**
- Consumes: `claim_id` from Task 3, `VERDICT_*` constants.
- Produces:
  - `verdict_row(claim: Claim, verdict: Verdict, *, prompt_sha256: str, judged_at: str) -> dict` — one flat row.
  - `judge_claims(claims, callers, *, existing=None, on_verdict=None, workers=1, max_consecutive_failures=10) -> list[ClaimResult]` — `on_verdict(claim, verdict)` fires once per pair; `existing` is the flat row list.
  - `judge_verdict_prompt_sha256() -> str` — hash of `build_verifier_prompt` output for the plain variant.

- [ ] **Step 1: Write the failing tests**

```python
def test_verdict_row_is_flat():
    claim = Claim(image_id=5, turn=1, question="q", answer="Red.",
                  claim="Red (car)", claim_type="colour", claim_id="abc")
    row = verdict_row(claim, Verdict("hy4", "true", "because"),
                      prompt_sha256="p", judged_at="2026-11-21T00:00:00+00:00")
    assert row["claim_id"] == "abc"
    assert row["verifier_model"] == "hy4"
    assert row["verdict"] == "true"
    assert "verdicts" not in row


def test_judge_resumes_from_flat_rows():
    callers = [Stub("v1", "true"), Stub("v2", "false")]
    existing = [{"claim_id": "abc", "verifier_model": "v1", "verdict": "false",
                 "evidence": "prior", "error": None}]
    claim = Claim(image_id=0, turn=1, question="q", answer="a",
                  claim="c", claim_type="colour", claim_id="abc")
    judge_claims([claim], callers, existing=existing)
    assert callers[0].calls == 0
    assert callers[1].calls == 1


def test_an_error_row_is_never_carded_as_cant_tell():
    results = [ClaimResult(Claim(0, 1, "q", "a", "c", "colour", claim_id="x"),
                           verdicts=[Verdict("v1", "can't-tell", error="429")])]
    summary = summarise(results)
    assert summary["n_errored"] == 1
    assert summary["n_judged"] == 0
    assert summary["contradiction_rate"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_verify.py -v`
Expected: FAIL — `ImportError: cannot import name 'verdict_row'`

- [ ] **Step 3: Implement**

Add `verdict_row`. Change `_result_to_row` to return `list[dict]` — one row per verdict — or replace its uses with `verdict_row`. In `judge_claims`, change `existing` to a flat list and build `prior` keyed by `(claim_id, verifier_model)`, skipping rows with an `error`. Call `on_verdict(claim, verdict)` as each verdict is produced, so the CLI appends immediately. The `judge` CLI writes one JSON line per verdict and computes `judge_verdict_prompt_sha256()` once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_verify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/verify.py tests/unit/test_m2a_verify.py
git commit -m "M2a: verdicts are flat, resumable, and pin the judging prompt"
```

---

### Task 5: Blind rows carry the claim id and the model

**Files:**
- Modify: `src/sightline/m2a/verify.py` (`judge_blind`, `blind` CLI)
- Test: `tests/unit/test_m2a_verifier_transport.py`

**Interfaces:**
- Consumes: `claim_id` from Task 3.
- Produces: `judge_blind(claims, call, *, model: str, on_row=None) -> list[dict]` — each row carries `claim_id`, `blind_model`, `judged_at`.

- [ ] **Step 1: Write the failing test**

```python
def test_blind_row_carries_the_claim_id_and_model():
    claim = Claim(0, 1, "What colour?", "Red.", "Red (car)", "colour", claim_id="abc")
    rows = judge_blind([claim], Stub("v", "true"), model="hy4")
    assert rows[0]["claim_id"] == "abc"
    assert rows[0]["blind_model"] == "hy4"
    assert rows[0]["judged_at"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_verifier_transport.py::test_blind_row_carries_the_claim_id_and_model -v`
Expected: FAIL — `TypeError: judge_blind() got an unexpected keyword argument 'model'`

- [ ] **Step 3: Implement**

Add the `model` keyword, set `claim_id`, `blind_model` and `judged_at` on each row. The `blind` CLI passes `args.model`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_verifier_transport.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/verify.py tests/unit/test_m2a_verifier_transport.py
git commit -m "M2a: blind results join on the claim id and name their model"
```

---

### Task 6: Review export becomes `decisions.jsonl`

**Files:**
- Modify: `tools/m2a_review.py`
- Test: `tests/unit/test_m2a_review.py`

**Interfaces:**
- Consumes: `claim_id` from Task 3.
- Produces:
  - `merge_decisions(export: dict, out: Path, *, reviewer: str, reviewed_at: str, claims_path: Path | None = None) -> tuple[int, int]` — returns `(added, rejected)`; rejects any decision whose `claim_id` is not in the claims file. When `claims_path` is omitted the merge accepts every row and rejects nothing.
  - A `merge` subcommand: `python tools/m2a_review.py merge --export FILE --claims PATH --out PATH --reviewer NAME`.

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_writes_one_row_per_decision(tmp_path):
    export = {"decisions": [{"image_id": 1, "turn": 1, "decision": "right"}]}
    added, rejected = merge_decisions(
        export, tmp_path / "d.jsonl", reviewer="k", reviewed_at="2026-11-21T00:00:00+00:00"
    )
    assert (added, rejected) == (1, 0)


def test_merge_rejects_a_decision_with_no_matching_claim(tmp_path):
    claims = tmp_path / "claims.jsonl"
    claims.write_text(json.dumps({"claim_id": "abc", "image_id": 1, "turn": 1}) + "\n")
    export = {"decisions": [{"image_id": 2, "turn": 9, "decision": "right"}]}
    added, rejected = merge_decisions(
        export, tmp_path / "d.jsonl", reviewer="k",
        reviewed_at="2026-11-21T00:00:00+00:00", claims_path=claims,
    )
    assert (added, rejected) == (0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_m2a_review.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge_decisions'`

- [ ] **Step 3: Implement**

`merge_decisions` maps each exported `{image_id, turn}` back to its `claim_id` by recomputing `claim_id(image_id, turn, claim)` over the claims file; a pair with no matching claim is counted as rejected, never written. `truth` is copied when present. Rows append to `out`. Add the `merge` subparser and dispatch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_m2a_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/m2a_review.py tests/unit/test_m2a_review.py
git commit -m "M2a: review exports merge into decisions.jsonl and reject orphans"
```

---

### Task 7: Corpus manifest aggregator

**Files:**
- Create: `src/sightline/m2a/corpus_manifest.py`
- Create: `tests/unit/test_m2a_corpus_manifest.py`
- Modify: `tools/build_m2a.py` (new `corpus-manifest` subcommand)

**Interfaces:**
- Consumes: the layer files from Tasks 2–6.
- Produces:
  - `build(root: Path) -> dict` — reads `conversations.jsonl`, `claims.jsonl`, `verdicts.jsonl`, `blind.jsonl`, `decisions.jsonl` when present, and returns the manifest dict from the spec.
  - `corpus-manifest` subcommand writing `corpus-manifest.json` into the root.

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_reports_error_free_rates_only(tmp_path):
    claims = [
        {"claim_id": "a", "image_id": 1, "turn": 1, "claim": "c", "claim_type": "colour"},
        {"claim_id": "b", "image_id": 2, "turn": 1, "claim": "c", "claim_type": "count"},
    ]
    verdicts = [
        {"claim_id": "a", "verifier_model": "v", "verdict": "false", "error": None},
        {"claim_id": "b", "verifier_model": "v", "verdict": "can't-tell", "error": "429"},
    ]
    write_jsonl(tmp_path / "claims.jsonl", claims)
    write_jsonl(tmp_path / "verdicts.jsonl", verdicts)

    manifest = build(tmp_path)

    assert manifest["n_claims"] == 2
    assert manifest["n_errored"] == 1
    assert manifest["contradiction_rate"] == 1.0
    assert manifest["thresholds"]["contradiction_ceiling"] == 0.15


def test_manifest_flags_a_checksum_mismatch(tmp_path):
    write_jsonl(tmp_path / "index.jsonl", [{"id": 1, "sha256": "aa"}])
    write_jsonl(
        tmp_path / "conversations.jsonl",
        [{"id": 1, "image_sha256": "bb", "model": "m", "parsed": True}],
    )

    manifest = build(tmp_path)

    assert manifest["n_sha_mismatch"] == 1


def test_manifest_flags_an_annotator_that_is_also_the_verifier(tmp_path):
    write_jsonl(
        tmp_path / "conversations.jsonl",
        [{"id": 1, "image_sha256": "aa", "model": "opencode/hy4-flash-free", "parsed": True}],
    )
    write_jsonl(
        tmp_path / "verdicts.jsonl",
        [{"claim_id": "x", "verifier_model": "opencode/hy4-flash-free",
          "verdict": "true", "error": None}],
    )

    manifest = build(tmp_path)

    assert manifest["annotator_verifier_distinct"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_m2a_corpus_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sightline.m2a.corpus_manifest'`

- [ ] **Step 3: Implement**

Reuse `summarise`, `summarise_planted` and `summarise_blind` from `verify.py` by rebuilding `Claim`/`ClaimResult` from the rows, so the aggregator and the CLI cannot drift. Add the counts, provenance groups and `passes_*` fields from the spec. Add `n_sha_mismatch` (conversations whose `image_sha256` differs from the fetched record's `sha256`) and `annotator_verifier_distinct` (whether any annotator model also appears as a verifier). The existing `manifest()` in `manifest.py` is not touched.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_m2a_corpus_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/m2a/corpus_manifest.py tests/unit/test_m2a_corpus_manifest.py tools/build_m2a.py
git commit -m "M2a: one manifest over every layer, with the gates computed"
```

---

## Execution order

Tasks 1–3 are the foundation: every later task joins on `claim_id`. Task 4 depends on Task 3, Tasks 5 and 6 on Task 3, and Task 7 on all of them. No task needs a GPU; the annotator run itself is the owner's.
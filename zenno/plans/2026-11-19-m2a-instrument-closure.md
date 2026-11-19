# M2a Instrument Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two measurement defects that make the signed G1 numbers understate the method, so M2a is built on an instrument that measures what it claims.

**Architecture:** The exact-match parser stays the default and the recorded G1 result; a second `first-line` parser is added behind a flag and used only to produce a comparison report. A `--filter-answer` option on the debug path isolates one answer class. No training code changes, so the three checkpoints remain valid and no re-training is needed.

**Tech Stack:** Python 3.13, torch 2.13, transformers 5.12, pytest, ruff.

**Spec:** `DECISIONS/0039-g1-signoff.md` (open items 1 and 2), `DECISIONS/0038-frozen-parser-for-g1.md`, `reports/m1b-g1.md` (caveats).

## Global Constraints

- Do not change the default parser, the prompts, or the G1 thresholds. `DECISIONS/0038` freezes them for the recorded runs; this plan adds an alternative and reports both.
- The three checkpoints in `artifacts/m1b/seed{0,1,2}/projector.pt` are the only trained artifacts. No task may require retraining.
- Evaluation runs need the GPU; unit tests must not.
- Every non-trivial claim in a report carries `[V]`, `[S]`, `[I]` or `[U]`.
- Commits say why. No emojis. Type hints. Metrics as JSONL, not prints, except the console progress and report output the runner already prints.
- `ruff check src tests tools` and `ruff format --check` must pass on touched files.

## Review Focus

- **A model that answers correctly but rambles.** `'diamond\n```json\n{\n  "'` is correct and currently scored wrong. Expected: `first-line` scores it correct, `exact` still scores it wrong, and both numbers appear side by side.
- **An answer that merely contains the target string.** `'not red'` contains `red`. Expected: the first-line parser must not score it correct, and a test pins that.
- **A single-class debug run on an ordered split.** Filtering to magenta from an answer-ordered split must sample evenly, not take the first N.
- **A checkpoint that predates a parser change.** Re-scoring must not depend on retraining or on the parser used during training.
- **An empty or whitespace-only generation.** Expected: scored wrong, never an exception.

---

### Task 1: First-line parser

**Files:**
- Modify: `src/sightline/evaluate.py`
- Test: `tests/unit/test_evaluate_stats.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `first_line_answer(text: str) -> str` and `normalise_answer(text: str) -> str` in `sightline.evaluate`; the existing exact-match normalisation becomes `normalise_answer` so both parsers share one rule.

- [ ] **Step 1: Write the failing tests**

```python
def test_first_line_drops_trailing_markdown() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer('diamond\n\`\`\`json\n{\n  "') == "diamond"


def test_first_line_rejects_a_negated_answer() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer("not red") == "not red"


def test_first_line_handles_an_empty_generation() -> None:
    from sightline.evaluate import first_line_answer

    assert first_line_answer("") == ""
    assert first_line_answer("   \n\n") == ""
```

- [ ] **Step 2: Run them and verify they fail**

Run: `python3 -m pytest tests/unit/test_evaluate_stats.py -q`
Expected: FAIL, `ImportError: cannot import name 'first_line_answer'`

- [ ] **Step 3: Implement both functions in `src/sightline/evaluate.py`**

`normalise_answer` takes the current `.strip().lower().strip(" .!")` rule used by the runner. `first_line_answer` takes the first line, applies `normalise_answer`, and does not attempt to extract a substring — so `"not red"` stays `"not red"` and scores wrong, which is the intended conservative behaviour.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_evaluate_stats.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sightline/evaluate.py tests/unit/test_evaluate_stats.py
git commit -m "M1c: add a first-line answer parser for trailing-markdown answers"
```

### Task 2: Parser switch in the runner

**Files:**
- Modify: `tools/run_m1.py`
- Test: `tests/unit/test_run_m1_eval.py`

**Interfaces:**
- Consumes: `normalise_answer`, `first_line_answer` from `sightline.evaluate`.
- Produces: `--parser {exact,first-line}` on the runner, default `exact`; the chosen parser is recorded in the `g1` metric event and the run summary.

- [ ] **Step 1: Write the failing test**

```python
def test_parser_choice_selects_the_normaliser() -> None:
    runner = load_runner()

    exact = runner.answer_parser("exact")
    first_line = runner.answer_parser("first-line")

    assert exact("diamond\n```json") == "diamond\n\`\`\`json"
    assert first_line("diamond\n```json") == "diamond"
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_run_m1_eval.py -q`
Expected: FAIL, `AttributeError: module ... has no attribute 'answer_parser'`

- [ ] **Step 3: Implement `answer_parser(name)` in `tools/run_m1.py` and thread it through**

`generate_answer` and the no-tokens branch both normalise with the selected function. `evaluate` takes the parser as an argument. `build_prompt` is untouched. Record `"parser": args.parser` in the `g1` event and in the summary.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_run_m1_eval.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/run_m1.py tests/unit/test_run_m1_eval.py
git commit -m "M1c: select the answer parser with --parser"
```

### Task 3: Answer-class filter for debug runs

**Files:**
- Modify: `tools/run_m1.py`
- Test: `tests/unit/test_run_m1_eval.py`

**Interfaces:**
- Consumes: `sample_subset` from `tools/run_m1.py`.
- Produces: `--filter-answer TEXT`, applied to `--debug-logits` and `--debug-generations` only, sampling evenly from the matching examples.

- [ ] **Step 1: Write the failing test**

```python
def test_filter_answer_keeps_only_matching_examples() -> None:
    runner = load_runner()
    examples = [
        SimpleNamespace(task="color", answer="magenta"),
        SimpleNamespace(task="color", answer="red"),
        SimpleNamespace(task="color", answer="magenta"),
    ]

    kept = runner.filter_examples(examples, "magenta")

    assert [e.answer for e in kept] == ["magenta", "magenta"]
```

- [ ] **Step 2: Run it and verify it fails**

Run: `python3 -m pytest tests/unit/test_run_m1_eval.py -q`
Expected: FAIL, `AttributeError: module ... has no attribute 'filter_examples'`

- [ ] **Step 3: Implement `filter_examples` and wire `--filter-answer`**

The filter applies after the split is sampled and before the debug function runs. A filter that matches nothing exits with a message rather than running an empty loop.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `python3 -m pytest tests/unit/test_run_m1_eval.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/run_m1.py tests/unit/test_run_m1_eval.py
git commit -m "M1c: isolate one answer class in debug runs"
```

### Task 4: Re-score all three seeds under both parsers

**Files:**
- Create: `reports/g1-parser-comparison.md`
- Modify: `DECISIONS/0038-frozen-parser-for-g1.md` (record the outcome)

**Interfaces:**
- Consumes: the three checkpoints, `tools/run_m1.py --parser`.
- Produces: a table of per-seed, per-task accuracy under both parsers, and a revised shape estimate.

- [ ] **Step 1: Re-score seed 0 under both parsers**

Run, on the GPU box:
```bash
PYTHONPATH=src python tools/run_m1.py --seed 0 --eval-only --eval-split test --parser exact
PYTHONPATH=src python tools/run_m1.py --seed 0 --eval-only --eval-split test --parser first-line
```
Expected: two JSON reports; the `first-line` shape score is greater than or equal to the `exact` one.

- [ ] **Step 2: Repeat for seeds 1 and 2**

Same two commands with `--seed 1` and `--seed 2`. Expected: the same direction of change on each seed.

- [ ] **Step 3: Write `reports/g1-parser-comparison.md`**

Include per-seed, per-task accuracy under both parsers, the delta, and a `[V]` line for each number. State whether the G1 verdict changes under either parser.

- [ ] **Step 4: Update `DECISIONS/0038` with the outcome**

Record the measured delta and whether the recorded G1 verdict stands. If the `first-line` parser changes any seed's verdict, say so explicitly and do not silently switch the recorded result.

- [ ] **Step 5: Commit**

```bash
git add reports/g1-parser-comparison.md DECISIONS/0038-frozen-parser-for-g1.md
git commit -m "M1c: report both parsers side by side"
```

### Task 5: Explain the magenta gap

**Files:**
- Create: `reports/m1b-magenta.md`

**Interfaces:**
- Consumes: `--filter-answer magenta` on seed 0's checkpoint.
- Produces: a short report on what the model answers when shown magenta, and whether the encoder separates magenta at all.

- [ ] **Step 1: Run the magenta-only debug on the test split**

```bash
PYTHONPATH=src python tools/run_m1.py --seed 0 --eval-only --eval-split test \
  --debug-logits 12 --filter-answer magenta
```
Expected: 12 lines of `expected='magenta' got=... top5=[...]`.

- [ ] **Step 2: Run the same on the train split as a control**

Same command with `--eval-split train`. Expected: if the model answers magenta correctly on train and not on test, the gap is generalisation; if it fails on both, it is a class-level mapping problem.

- [ ] **Step 3: Write `reports/m1b-magenta.md`**

Record the per-split counts, the most common wrong answer, and a `[U]` line if the cause is not determinable from these runs, with the closing step named.

- [ ] **Step 4: Commit**

```bash
git add reports/m1b-magenta.md
git commit -m "M1c: report what the model answers for magenta"
```

---

## Notes for the executor

- Tasks 4 and 5 need the GPU box. Tasks 1–3 do not, and must be finished and merged first so the GPU runs use the final code.
- `--parser first-line` is diagnostic. Do not make it the default and do not change the recorded G1 verdict without a DECISIONS entry and the owner's approval.
- The `n_supervised` / `logits_to_keep` path is untouched by these tasks; if a re-score produces a different number from the recorded one under `--parser exact`, stop and report it rather than adjusting the parser.
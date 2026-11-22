"""Tests for the calibration review tool."""

from __future__ import annotations

import json
from pathlib import Path

from tools.m2a_review import _verdict_rows, render, stratify


def result(
    image_id: int, claim: str, verdicts: list[str], claim_type: str = "colour"
) -> list[dict]:
    """Flat verdict rows, the shape judge actually writes."""

    return [
        {
            "claim_id": f"c{image_id}",
            "image_id": image_id,
            "turn": 1,
            "claim": claim,
            "claim_type": claim_type,
            "planted": False,
            "verifier_model": "a/b",
            "verdict": v,
            "evidence": f"evidence for {v}",
            "error": None,
            "image_path": None,
        }
        for v in verdicts
    ]


def test_verdict_rows_flatten_results():
    rows = _verdict_rows(result(1, "Red (car)", ["true", "false"]))

    assert len(rows) == 1
    assert rows[0]["claim"] == "Red (car)"
    assert len(rows[0]["verdicts"]) == 2


def test_stratify_over_samples_disagreement():
    flat = [r for i in range(10) for r in result(i, f"claim {i}", ["false"])]
    flat += [r for i in range(90) for r in result(100 + i, f"claim {i}", ["true"])]
    rows = _verdict_rows(flat)

    picked = stratify(rows, sample=20, seed=0)

    assert len(picked) == 20
    assert sum(1 for r in picked if r["contradicted"]) >= 10


def test_stratify_returns_everything_when_sample_exceeds_population():
    rows = _verdict_rows(result(0, "a", ["true"]))

    assert stratify(rows, sample=50, seed=0) == rows


def test_render_names_every_claim_and_skips_a_missing_image(tmp_path: Path):
    rows = _verdict_rows(result(1, "Red (car)", ["false"]))
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    html = out.read_text()
    assert "Red (car)" in html
    assert "image not found" in html
    assert "Test review" in html


def test_render_emits_an_absolute_file_uri_for_an_existing_image(tmp_path: Path):
    image = tmp_path / "000000000001.jpg"
    image.write_bytes(b"x")
    rows = _verdict_rows(
        [{**r, "image_path": str(image)} for r in result(1, "Red (car)", ["true"])]
    )
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    assert 'src="file:///' in out.read_text()


def test_render_marks_a_planted_claim(tmp_path: Path):
    rows = _verdict_rows([{**r, "planted": True} for r in result(1, "Blue (car)", ["false"])])
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    assert "tag-planted" in out.read_text()


def test_cards_are_inside_main(tmp_path: Path):
    """Cards appended after </main> lose the page padding and clip at the edge."""

    rows = _verdict_rows(result(1, "Red (car)", ["true"]))
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    html = out.read_text()
    main_start = html.index("<main>")
    main_end = html.index("</main>")
    card_at = html.index('<article class="card"')
    assert main_start < card_at < main_end


def test_merge_writes_one_row_per_decision(tmp_path: Path):
    from tools.m2a_review import merge_decisions

    export = {"decisions": [{"image_id": 1, "turn": 1, "decision": "right"}]}
    added, rejected = merge_decisions(
        export,
        tmp_path / "d.jsonl",
        reviewer="k",
        reviewed_at="2026-11-21T00:00:00+00:00",
    )
    assert (added, rejected) == (1, 0)


def test_merge_rejects_a_decision_with_no_matching_claim(tmp_path: Path):
    from tools.m2a_review import merge_decisions

    claims = tmp_path / "claims.jsonl"
    claims.write_text(json.dumps({"claim_id": "abc", "image_id": 1, "turn": 1}) + "\n")
    export = {"decisions": [{"image_id": 2, "turn": 9, "decision": "right"}]}
    added, rejected = merge_decisions(
        export,
        tmp_path / "d.jsonl",
        reviewer="k",
        reviewed_at="2026-11-21T00:00:00+00:00",
        claims_path=claims,
    )
    assert (added, rejected) == (0, 1)


def test_merge_prefers_a_claim_id_from_the_export(tmp_path: Path):
    """The page can carry claim_id directly; (image_id, turn) is the legacy fallback."""

    from tools.m2a_review import merge_decisions

    export = {"decisions": [{"claim_id": "abc", "image_id": 99, "turn": 9, "decision": "right"}]}
    out = tmp_path / "d.jsonl"
    added, rejected = merge_decisions(
        export, out, reviewer="k", reviewed_at="2026-11-21T00:00:00+00:00"
    )
    assert (added, rejected) == (1, 0)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["claim_id"] == "abc"


def test_review_main_runs_without_a_conflicting_flag(tmp_path: Path, monkeypatch):
    """The tool crashed at parser construction: --out was registered twice."""

    import sys

    from tools import m2a_review

    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "image_id": 1,
                "turn": 1,
                "claim": "Red (car)",
                "claim_type": "colour",
                "planted": False,
                "contradicted": False,
                "path": None,
                "verdicts": [{"model": "v", "verdict": "true", "evidence": "yes"}],
            }
        )
        + "\n"
    )
    out = tmp_path / "review.html"

    monkeypatch.setattr(
        sys,
        "argv",
        ["m2a_review.py", "--claims", str(results), "--out", str(out), "--sample", "5"],
    )
    assert m2a_review.main() == 0
    assert out.exists()


def test_verdict_rows_carry_turn_and_claim_id():
    from tools.m2a_review import _verdict_rows

    rows = _verdict_rows(
        [
            {
                "claim_id": "abc",
                "image_id": 1,
                "turn": 2,
                "claim": "c",
                "claim_type": "colour",
                "planted": False,
                "contradicted": False,
                "path": None,
                "verdicts": [],
            }
        ]
    )

    assert rows[0]["turn"] == 2
    assert rows[0]["claim_id"] == "abc"


def test_render_page_carries_the_claim_id(tmp_path: Path):
    from tools.m2a_review import _verdict_rows, render

    rows = _verdict_rows(
        [
            {
                "claim_id": "abc",
                "image_id": 1,
                "turn": 2,
                "claim": "Red (car)",
                "claim_type": "colour",
                "planted": False,
                "contradicted": False,
                "path": None,
                "verdicts": [],
            }
        ]
    )
    out = tmp_path / "r.html"

    render(rows, out, tmp_path, "test")

    html = out.read_text()
    assert 'data-claim-id="abc"' in html
    assert 'data-turn="2"' in html

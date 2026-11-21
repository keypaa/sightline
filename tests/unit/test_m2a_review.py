"""Tests for the calibration review tool."""

from __future__ import annotations

from pathlib import Path

from tools.m2a_review import _verdict_rows, render, stratify


def result(image_id: int, claim: str, verdicts: list[str], claim_type: str = "colour") -> dict:
    return {
        "image_id": image_id,
        "claim": claim,
        "claim_type": claim_type,
        "planted": False,
        "contradicted": any(v == "false" for v in verdicts),
        "path": None,
        "verdicts": [
            {"model": "a/b", "verdict": v, "evidence": f"evidence for {v}"} for v in verdicts
        ],
    }


def test_verdict_rows_flatten_results():
    rows = _verdict_rows([result(1, "Red (car)", ["true", "false"])])

    assert len(rows) == 1
    assert rows[0]["claim"] == "Red (car)"
    assert len(rows[0]["verdicts"]) == 2


def test_stratify_over_samples_disagreement():
    rows = [result(i, f"claim {i}", ["false"]) for i in range(10)]
    rows += [result(100 + i, f"claim {i}", ["true"]) for i in range(90)]

    picked = stratify(rows, sample=20, seed=0)

    assert len(picked) == 20
    assert sum(1 for r in picked if r["contradicted"]) >= 10


def test_stratify_returns_everything_when_sample_exceeds_population():
    rows = [result(0, "a", ["true"])]

    assert stratify(rows, sample=50, seed=0) == rows


def test_render_names_every_claim_and_skips_a_missing_image(tmp_path: Path):
    rows = _verdict_rows([result(1, "Red (car)", ["false"])])
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    html = out.read_text()
    assert "Red (car)" in html
    assert "image not found" in html
    assert "Test review" in html


def test_render_emits_an_absolute_file_uri_for_an_existing_image(tmp_path: Path):
    image = tmp_path / "000000000001.jpg"
    image.write_bytes(b"x")
    rows = _verdict_rows([{**result(1, "Red (car)", ["true"]), "path": str(image)}])
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    assert 'src="file:///' in out.read_text()


def test_render_marks_a_planted_claim(tmp_path: Path):
    rows = _verdict_rows([{**result(1, "Blue (car)", ["false"]), "planted": True}])
    out = tmp_path / "review.html"

    render(rows, out, tmp_path, "Test review")

    assert "tag-planted" in out.read_text()

"""Tests for the M2a grounding set builder and scorer."""

from __future__ import annotations

from pathlib import Path

from tools.m2a_grounding import (
    ATTRIBUTES,
    blank_labels,
    label_agreement,
    render_review_html,
    sample_ids,
    summarise,
)


def records(n: int) -> list[dict]:
    return [
        {
            "id": i,
            "file_name": f"{i}.jpg",
            "path": f"/tmp/{i}.jpg",
            "sha256": f"{i:064x}",
            "license_name": "CC BY 2.0",
            "flickr_url": f"http://f/{i}",
        }
        for i in range(n)
    ]


def test_sample_ids_is_reproducible_and_a_permutation():
    first = sample_ids(records(20), 5, seed=0)
    second = sample_ids(records(20), 5, seed=0)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_blank_labels_covers_every_attribute_as_unlabelled():
    rows = blank_labels(records(3), [0, 1])

    assert len(rows) == 2
    assert rows[0]["labels"] == {attribute: None for attribute in ATTRIBUTES}
    assert rows[0]["human_confirmed"] is False


def test_unanswered_attribute_is_not_scored_as_a_miss():
    row = {"id": 0, "labels": {"colour": "red", "shape": None}, "human_confirmed": True}

    result = label_agreement(row, {"colour": "red"})

    assert result["n_scored"] == 1
    assert result["n_correct"] == 1


def test_agreement_is_case_insensitive_and_rejects_a_wrong_answer():
    row = {"id": 0, "labels": {"colour": "Red"}, "human_confirmed": False}

    assert label_agreement(row, {"colour": "red"})["n_correct"] == 1
    assert label_agreement(row, {"colour": "blue"})["n_correct"] == 0
    # An absent answer is a miss, not an unscored attribute: the label exists, so
    # the annotator failing to answer it is exactly what grounding should catch.
    assert label_agreement(row, {})["n_correct"] == 0
    assert label_agreement(row, {})["n_scored"] == 1


def test_summary_separates_labeller_and_human_confirmed_rates():
    rows = [
        {"id": 0, "labels": {"colour": "red"}, "human_confirmed": True},
        {"id": 1, "labels": {"colour": "blue"}, "human_confirmed": False},
    ]
    predictions = {0: {"colour": "red"}, 1: {"colour": "green"}}

    summary = summarise(rows, predictions)

    assert summary["labeller_agreement"] == 0.5
    assert summary["human_confirmed_agreement"] == 1.0
    assert summary["n_human_confirmed"] == 1


def test_summary_reports_zero_when_nothing_is_confirmed():
    rows = [{"id": 0, "labels": {"colour": "red"}, "human_confirmed": False}]

    summary = summarise(rows, {0: {"colour": "red"}})

    assert summary["human_confirmed_agreement"] is None
    assert summary["labeller_agreement"] == 1.0


def test_review_page_names_every_image_and_carries_the_licence(tmp_path: Path):
    rows = blank_labels(records(2), [0, 1])
    out = tmp_path / "review.html"

    render_review_html(rows, out)

    html = out.read_text()
    assert "0.jpg" in html
    assert "1.jpg" in html
    assert "CC BY 2.0" in html
    assert "human_confirmed" not in html or "I checked this image" in html


def test_review_page_uses_absolute_file_urls(tmp_path: Path):
    """A relative src is read as a host name, not a path, so images never load."""

    rows = blank_labels(records(1), [0])
    out = tmp_path / "review.html"

    render_review_html(rows, out)

    html = out.read_text()
    assert 'src="file:///' in html
    assert "file://artifacts" not in html
    assert "file://tmp" not in html

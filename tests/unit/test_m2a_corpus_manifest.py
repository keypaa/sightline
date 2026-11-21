"""Tests for the corpus manifest aggregator."""

from __future__ import annotations

from pathlib import Path

from sightline.m2a.corpus_manifest import build
from sightline.m2a.verify import write_jsonl


def test_manifest_reports_error_free_rates_only(tmp_path: Path):
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


def test_manifest_flags_a_checksum_mismatch(tmp_path: Path):
    write_jsonl(tmp_path / "fetched.jsonl", [{"id": 1, "sha256": "aa"}])
    write_jsonl(
        tmp_path / "conversations.jsonl",
        [{"id": 1, "image_sha256": "bb", "model": "m", "parsed": True}],
    )

    manifest = build(tmp_path)

    assert manifest["n_sha_mismatch"] == 1


def test_manifest_flags_an_annotator_that_is_also_the_verifier(tmp_path: Path):
    write_jsonl(
        tmp_path / "conversations.jsonl",
        [{"id": 1, "image_sha256": "aa", "model": "opencode/hy4-flash-free", "parsed": True}],
    )
    write_jsonl(
        tmp_path / "verdicts.jsonl",
        [
            {
                "claim_id": "x",
                "verifier_model": "opencode/hy4-flash-free",
                "verdict": "true",
                "error": None,
            }
        ],
    )

    manifest = build(tmp_path)

    assert manifest["annotator_verifier_distinct"] is False


def test_manifest_on_an_empty_root_does_not_invent_numbers(tmp_path: Path):
    manifest = build(tmp_path)

    assert manifest["n_images"] == 0
    assert manifest["n_claims"] == 0
    assert manifest["contradiction_rate"] is None


def test_manifest_reads_sources_and_licences(tmp_path: Path):
    write_jsonl(
        tmp_path / "conversations.jsonl",
        [
            {"id": 1, "source": "coco", "license_name": "CC BY 2.0", "parsed": True},
            {"id": 2, "source": "coco", "license_name": "CC BY-NC 2.0", "parsed": True},
        ],
    )

    manifest = build(tmp_path)

    assert manifest["source_count"] == 1
    assert manifest["sources"] == {"coco": 2}
    assert manifest["licences"]["CC BY 2.0"] == 1

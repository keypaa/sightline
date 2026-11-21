from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sightline.m2a.manifest import manifest, splits, usable_records

TOKENS_PER_CONVERSATION = 3179


def records(n: int) -> list[dict]:
    return [
        {"id": i, "source": "coco", "hash": i * 7, "license_name": "CC BY 4.0"} for i in range(n)
    ]


def test_splits_are_disjoint_and_complete() -> None:
    parts = splits(records(300), seed=0)
    flat = [r["id"] for part in parts.values() for r in part]

    assert set(parts) == {"train", "val", "test"}
    assert len(flat) == 300
    assert len(set(flat)) == 300


def test_split_assignment_is_reproducible() -> None:
    assert splits(records(300), seed=0) == splits(records(300), seed=0)


def test_a_different_seed_changes_the_assignment() -> None:
    assert splits(records(300), seed=0) != splits(records(300), seed=1)


def test_split_sizes_follow_the_documented_ratio() -> None:
    parts = splits(records(3000), seed=0)

    assert abs(len(parts["train"]) - 2400) <= 2
    assert abs(len(parts["val"]) - 300) <= 2
    assert abs(len(parts["test"]) - 300) <= 2


def test_manifest_records_the_hash_and_the_budget(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    rows = [
        {"id": r["id"], "source": r["source"], "license_name": r["license_name"]}
        for r in records(40)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)

    assert result["n_conversations"] == 40
    assert result["input_sha256"] == _sha256(path)
    assert result["tokens_projected"] == 40 * 3179
    assert result["licences"] == {"CC BY 4.0": 40}
    assert result["sources"] == {"coco": 40}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_does_not_count_unparsed_rows(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    rows = [
        {"id": 0, "source": "coco", "license_name": "CC BY 4.0", "parsed": True},
        {"id": 1, "source": "coco", "license_name": "CC BY 4.0", "parsed": False},
        {"id": 2, "source": "coco", "license_name": "CC BY 4.0", "parsed": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)

    assert result["n_conversations"] == 2
    assert result["tokens_projected"] == 2 * TOKENS_PER_CONVERSATION


def test_usable_records_drops_only_explicit_failure() -> None:
    rows = [{"id": 0}, {"id": 1, "parsed": True}, {"id": 2, "parsed": False}]

    assert [r["id"] for r in usable_records(rows)] == [0, 1]


def test_manifest_reports_sources_and_collisions(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    rows = [
        {
            "id": 0,
            "source": "coco",
            "license_name": "CC BY 2.0",
            "parsed": True,
            "model": "Qwen/Qwen3.6-27B-FP8",
            "sha256": "aa",
        },
        {
            "id": 1,
            "source": "coco",
            "license_name": "CC BY 2.0",
            "parsed": True,
            "model": "Qwen/Qwen3.6-27B-FP8",
            "sha256": "bb",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)
    assert result["source_count"] == 1
    assert result["sources"] == {"coco": 2}
    assert result["annotators"] == {"Qwen/Qwen3.6-27B-FP8": 2}
    assert result["n_cross_split_hash_collisions"] == 0


def test_manifest_records_order_seed_and_category_coverage(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    rows = [
        {"id": 0, "parsed": True, "order_seed": 7, "category": "counting", "sha256": "aa"},
        {"id": 1, "parsed": True, "order_seed": 7, "category": "counting", "sha256": "bb"},
        {"id": 2, "parsed": True, "order_seed": 7, "category": "spatial", "sha256": "cc"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)

    assert result["order_seed"] == 7
    assert result["category_coverage"] == {"counting": 2, "spatial": 1}


def test_manifest_omits_category_coverage_when_unrecorded(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    path.write_text(json.dumps({"id": 0, "source": "coco", "parsed": True}) + "\n")

    result = manifest(path)

    assert "category_coverage" not in result
    assert result["order_seed"] is None


def test_manifest_counts_colliding_ids_not_duplicate_pairs(tmp_path: Path) -> None:
    path = tmp_path / "conversations.jsonl"
    rows = [{"id": i, "parsed": True, "sha256": "shared"} for i in range(10)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = manifest(path)

    assert result["n_cross_split_hash_collisions"] == 10

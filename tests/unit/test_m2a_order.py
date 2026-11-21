from __future__ import annotations

from pathlib import Path

import pytest

from sightline.m2a.order import apply_order, corpus_order, read_order, write_order


def test_order_is_a_permutation_and_is_reproducible() -> None:
    records = [{"id": i} for i in range(10)]
    first = corpus_order(records, seed=0)
    second = corpus_order(records, seed=0)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert sorted(r["id"] for r in first) == list(range(10))


def test_order_survives_a_round_trip(tmp_path: Path) -> None:
    records = [{"id": i} for i in range(5)]
    write_order(tmp_path / "order.jsonl", corpus_order(records, seed=0), seed=0)
    restored = apply_order(list(reversed(records)), read_order(tmp_path / "order.jsonl"))
    assert [r["id"] for r in restored] == [r["id"] for r in corpus_order(records, seed=0)]


def test_apply_order_rejects_a_missing_id() -> None:
    with pytest.raises(ValueError):
        apply_order([{"id": 0}], [{"position": 0, "id": 1}])

"""Identity fields: stable claim ids, run ids, timestamps."""

from __future__ import annotations

from sightline.m2a.ids import claim_id, new_run_id, now_iso


def test_claim_id_is_stable_and_short():
    first = claim_id(12, 1, "Red (car)")
    assert first == claim_id(12, 1, "Red (car)")
    assert len(first) == 16
    assert first != claim_id(12, 2, "Red (car)")
    assert first != claim_id(13, 1, "Red (car)")


def test_claim_id_distinguishes_different_claims():
    assert claim_id(1, 1, "Red (car)") != claim_id(1, 1, "Blue (car)")


def test_run_id_has_a_timestamp_and_a_random_tail():
    run = new_run_id()
    assert run[8] == "T"
    assert "-" in run and len(run.split("-")[-1]) == 8
    assert new_run_id() != new_run_id()


def test_now_iso_is_timezone_aware():
    assert now_iso().endswith("+00:00")

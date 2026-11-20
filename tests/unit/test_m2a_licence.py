from __future__ import annotations

from sightline.m2a.licence import classify


def test_policy_drops_nc_nd_and_missing() -> None:
    records = [
        {"file_name": "a.jpg", "license_name": "CC BY 4.0"},
        {"file_name": "b.jpg", "license_name": "CC BY-NC 2.0"},
        {"file_name": "c.jpg", "license_name": "CC BY-ND 2.0"},
        {"file_name": "d.jpg", "license_name": None},
    ]

    kept, dropped = classify(records)

    assert [r["file_name"] for r in kept] == ["a.jpg"]
    assert {r["drop_reason"] for r in dropped} == {
        "non-commercial",
        "no-derivatives",
        "missing-licence",
    }


def test_an_explicit_allow_list_narrows_further() -> None:
    """An allow-list restricts; it never re-admits something the policy drops."""

    records = [
        {"file_name": "a.jpg", "license_name": "CC BY 4.0"},
        {"file_name": "b.jpg", "license_name": "CC BY 2.0"},
    ]

    kept, dropped = classify(records, allow={"cc by 4.0"})

    assert [r["file_name"] for r in kept] == ["a.jpg"]
    assert dropped[0]["drop_reason"] == "not-in-allow-list"


def test_the_policy_cannot_be_bypassed_by_an_allow_list() -> None:
    """Asking for a non-commercial licence explicitly still drops it."""

    records = [{"file_name": "a.jpg", "license_name": "CC BY-NC 2.0"}]

    kept, dropped = classify(records, allow={"cc by-nc 2.0"})

    assert kept == []
    assert dropped[0]["drop_reason"] == "non-commercial"


def test_an_empty_allow_list_admits_nothing() -> None:
    records = [{"file_name": "a.jpg", "license_name": "CC BY 4.0"}]

    kept, dropped = classify(records, allow=set())

    assert kept == []
    assert dropped[0]["drop_reason"] == "not-in-allow-list"


def test_names_are_matched_case_insensitively() -> None:
    records = [
        {"file_name": "a.jpg", "license_name": "cc by 4.0"},
        {"file_name": "b.jpg", "license_name": "CC-BY 4.0"},
        {"file_name": "c.jpg", "license_name": "CC BY-NC-SA 2.0"},
    ]

    kept, dropped = classify(records)

    assert len(kept) == 2
    assert dropped[0]["drop_reason"] == "non-commercial"

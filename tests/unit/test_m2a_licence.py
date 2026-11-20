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


def test_the_real_coco_licence_names_are_handled() -> None:
    """COCO's own names, from instances_train2017.json."""

    records = [
        {"file_name": "1.jpg", "license_name": "Attribution-NonCommercial-ShareAlike License"},
        {"file_name": "2.jpg", "license_name": "Attribution-NonCommercial License"},
        {"file_name": "3.jpg", "license_name": "Attribution-NonCommercial-NoDerivs License"},
        {"file_name": "4.jpg", "license_name": "Attribution License"},
        {"file_name": "5.jpg", "license_name": "Attribution-ShareAlike License"},
        {"file_name": "6.jpg", "license_name": "Attribution-NoDerivs License"},
        {"file_name": "7.jpg", "license_name": "No known copyright restrictions"},
        {"file_name": "8.jpg", "license_name": "United States Government Work"},
    ]

    kept, dropped = classify(records)

    assert {r["file_name"] for r in kept} == {"4.jpg", "5.jpg", "7.jpg", "8.jpg"}
    reasons = {r["file_name"]: r["drop_reason"] for r in dropped}
    assert reasons["1.jpg"] == "non-commercial"
    assert reasons["2.jpg"] == "non-commercial"
    assert reasons["3.jpg"] == "non-commercial"
    assert reasons["6.jpg"] == "no-derivatives"

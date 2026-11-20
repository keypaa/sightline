from __future__ import annotations

import json
from pathlib import Path

from sightline.m2a.coco_index import read_annotations


def write_annotations(path: Path, images: list[dict], licenses: list[dict]) -> Path:
    path.write_text(json.dumps({"images": images, "licenses": licenses}))
    return path


def test_records_carry_the_licence_id(tmp_path: Path) -> None:
    path = write_annotations(
        tmp_path / "instances.json",
        [{"id": 1, "file_name": "a.jpg", "license": 3}],
        [{"id": 3, "name": "CC BY 4.0", "url": "u"}],
    )

    records, table = read_annotations(path)

    assert records[0]["license_id"] == 3
    assert records[0]["license_name"] == "CC BY 4.0"
    assert table[3]["url"] == "u"


def test_a_missing_licence_id_becomes_none(tmp_path: Path) -> None:
    path = write_annotations(
        tmp_path / "instances.json",
        [{"id": 1, "file_name": "a.jpg"}],
        [],
    )

    records, _ = read_annotations(path)

    assert records[0]["license_id"] is None
    assert records[0]["license_name"] is None


def test_unknown_licence_id_is_reported_not_dropped(tmp_path: Path) -> None:
    path = write_annotations(
        tmp_path / "instances.json",
        [{"id": 1, "file_name": "a.jpg", "license": 99}],
        [{"id": 3, "name": "CC BY 4.0", "url": "u"}],
    )

    records, table = read_annotations(path)

    assert len(records) == 1
    assert records[0]["license_name"] is None
    assert 99 not in table
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


def test_attribution_fields_are_kept(tmp_path: Path) -> None:
    """s.4(a) requires crediting the author; COCO records no name, only a URL."""

    path = write_annotations(
        tmp_path / "instances.json",
        [
            {
                "id": 1,
                "file_name": "a.jpg",
                "license": 3,
                "flickr_url": "http://f/p1",
                "coco_url": "http://c/p1",
            }
        ],
        [{"id": 3, "name": "CC BY 4.0", "url": "u"}],
    )

    records, _ = read_annotations(path)

    assert records[0]["flickr_url"] == "http://f/p1"
    assert records[0]["coco_url"] == "http://c/p1"


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


def test_index_records_carry_an_explicit_source(tmp_path: Path) -> None:
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "a.jpg",
                        "license": 1,
                        "flickr_url": "http://f/a",
                        "coco_url": "http://c/a",
                    }
                ],
                "licenses": [{"id": 1, "name": "Attribution License", "url": "http://l"}],
                "annotations": [],
            }
        )
    )
    records, _ = read_annotations(path)
    assert records[0]["source"] == "coco"
    assert records[0]["expected_sha256"] is None


def test_cmd_index_writes_source_and_expected_sha256(tmp_path: Path) -> None:
    """The index stage must not strip provenance on its way to index.jsonl."""
    import argparse

    from tools import build_m2a

    annotations = write_annotations(
        tmp_path / "instances.json",
        [{"id": 2, "file_name": "b.jpg", "license": 3}],
        [{"id": 3, "name": "CC BY 4.0", "url": "u"}],
    )
    root = tmp_path / "build"
    args = argparse.Namespace(root=root, annotations=annotations)

    assert build_m2a.cmd_index(args) == 0

    rows = build_m2a.read_jsonl(root / "index.jsonl")
    assert len(rows) == 1
    assert rows[0]["source"] == "coco"
    assert rows[0]["expected_sha256"] is None
    assert rows[0]["url"] == build_m2a.COCO_IMAGE_URL + "b.jpg"

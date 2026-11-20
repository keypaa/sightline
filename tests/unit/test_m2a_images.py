from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from sightline.m2a.images import content_hash, dedupe, dhash, distance, fetch


def write_pattern(path: Path, kind: str) -> Path:
    """Write an image with a recognisable gradient structure."""

    image = Image.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    if kind == "left-bright":
        draw.rectangle([0, 0, 31, 63], fill=(255, 255, 255))
    elif kind == "right-bright":
        draw.rectangle([32, 0, 63, 63], fill=(255, 255, 255))
    elif kind == "top-bright":
        draw.rectangle([0, 0, 63, 31], fill=(255, 255, 255))
    elif kind == "checker":
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    draw.rectangle(
                        [col * 8, row * 8, col * 8 + 7, row * 8 + 7], fill=(255, 255, 255)
                    )
    image.save(path)
    return path


def test_identical_files_hash_the_same(tmp_path: Path) -> None:
    a = write_pattern(tmp_path / "a.png", "checker")
    b = write_pattern(tmp_path / "b.png", "checker")
    c = write_pattern(tmp_path / "c.png", "left-bright")

    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)
    assert dhash(a) == dhash(b)
    assert dhash(a) != dhash(c)


def test_identical_images_at_two_ids_keep_one(tmp_path: Path) -> None:
    a = write_pattern(tmp_path / "a.png", "left-bright")
    b = write_pattern(tmp_path / "b.png", "left-bright")
    c = write_pattern(tmp_path / "c.png", "right-bright")

    records = [{"id": 1}, {"id": 2}, {"id": 3}]
    hashes = {
        1: (content_hash(a), dhash(a)),
        2: (content_hash(b), dhash(b)),
        3: (content_hash(c), dhash(c)),
    }

    kept, dupes = dedupe(records, hashes)

    assert {r["id"] for r in kept} == {1, 3}
    assert dupes[0]["id"] == 2
    assert dupes[0]["reason"] == "exact"


def test_structurally_different_images_are_kept(tmp_path: Path) -> None:
    a = write_pattern(tmp_path / "a.png", "left-bright")
    b = write_pattern(tmp_path / "b.png", "right-bright")

    assert distance(dhash(a), dhash(b)) > 5


def test_dedupe_keeps_the_first_occurrence(tmp_path: Path) -> None:
    a = write_pattern(tmp_path / "a.png", "top-bright")
    records = [{"id": i} for i in (5, 9, 12)]
    hashes = {i: (content_hash(a), dhash(a)) for i in (5, 9, 12)}

    kept, dupes = dedupe(records, hashes)

    assert [r["id"] for r in kept] == [5]
    assert [r["id"] for r in dupes] == [9, 12]


def test_fetch_writes_once_and_reuses_the_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def get(url: str) -> bytes:
        calls.append(url)
        return b"payload"

    record = {"file_name": "a.jpg", "url": "http://example/a.jpg"}

    first = fetch(record, tmp_path, get)
    second = fetch(record, tmp_path, get)

    assert first == second
    assert first.read_bytes() == b"payload"
    assert calls == ["http://example/a.jpg"]


def test_fetch_creates_the_cache_directory(tmp_path: Path) -> None:
    path = fetch(
        {"file_name": "a.jpg", "url": "u"}, tmp_path / "nested" / "cache", lambda url: b"x"
    )

    assert path.exists()

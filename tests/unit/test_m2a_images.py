from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sightline.m2a.images import content_hash, dedupe, dhash, distance, fetch, sniff_image


def image_bytes() -> bytes:
    """Return real JPEG bytes so fetch's image check passes."""

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


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
    payload = image_bytes()

    def get(url: str) -> bytes:
        calls.append(url)
        return payload

    record = {"file_name": "a.jpg", "url": "http://example/a.jpg"}

    first = fetch(record, tmp_path, get)
    second = fetch(record, tmp_path, get)

    assert first == second
    assert first.read_bytes() == payload
    assert calls == ["http://example/a.jpg"]


def test_fetch_creates_the_cache_directory(tmp_path: Path) -> None:
    payload = image_bytes()
    path = fetch(
        {"file_name": "a.jpg", "url": "u"}, tmp_path / "nested" / "cache", lambda url: payload
    )

    assert path.exists()


def test_sniff_rejects_an_html_error_page(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"<html><body>404 not found</body></html>")

    assert sniff_image(path) is None


def test_sniff_accepts_a_real_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8), "red").save(path)

    assert sniff_image(path) == "JPEG"


def test_fetch_rejects_an_html_error_page(tmp_path: Path) -> None:
    record = {"file_name": "bad.jpg", "url": "http://x/bad.jpg"}

    with pytest.raises(ValueError, match="not an image"):
        fetch(record, tmp_path / "cache", lambda url: b"<html>404</html>")


def test_fetch_rejects_bytes_that_do_not_match_the_expected_hash(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    record = {"file_name": "good.jpg", "url": "http://x/good.jpg", "expected_sha256": "0" * 64}

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch(record, tmp_path / "cache", lambda url: good.read_bytes())


def test_fetch_evicts_non_image_bytes_from_the_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    path = cache / "bad.jpg"
    record = {"file_name": "bad.jpg", "url": "http://x/bad.jpg"}

    with pytest.raises(ValueError, match="not an image"):
        fetch(record, cache, lambda url: b"<html>404</html>")

    assert not path.exists()


def test_fetch_evicts_checksum_mismatch_bytes_from_the_cache(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    cache = tmp_path / "cache"
    path = cache / "good.jpg"
    record = {"file_name": "good.jpg", "url": "http://x/good.jpg", "expected_sha256": "0" * 64}

    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch(record, cache, lambda url: good.read_bytes())

    assert not path.exists()


def test_fetch_accepts_a_matching_checksum(tmp_path: Path) -> None:
    payload = image_bytes()
    record = {
        "file_name": "good.jpg",
        "url": "http://x/good.jpg",
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
    }

    path = fetch(record, tmp_path / "cache", lambda url: payload)

    assert content_hash(path) == record["expected_sha256"]


def test_cmd_fetch_records_order_seed_and_drops_bad_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from tools import build_m2a

    root = tmp_path / "build"
    payload = image_bytes()
    build_m2a.write_jsonl(
        root / "filtered.jsonl",
        [
            {"id": 1, "file_name": "a.jpg", "url": "http://x/a.jpg"},
            {"id": 2, "file_name": "bad.jpg", "url": "http://x/bad.jpg"},
        ],
    )

    def fake_get(url: str) -> bytes:
        return b"<html>404</html>" if "bad" in url else payload

    monkeypatch.setattr(build_m2a, "_get", fake_get)
    args = argparse.Namespace(
        root=root,
        cache=tmp_path / "cache",
        workers=1,
        limit=0,
        seed=7,
        write_checksums=True,
    )

    assert build_m2a.cmd_fetch(args) == 0

    fetched = build_m2a.read_jsonl(root / "fetched.jsonl")
    dropped = build_m2a.read_jsonl(root / "fetch_dropped.jsonl")
    assert [row["id"] for row in fetched] == [0]
    assert fetched[0]["order_seed"] == 7
    assert fetched[0]["expected_sha256"] == hashlib.sha256(payload).hexdigest()
    assert [row["id"] for row in dropped] == [2]
    assert "not an image" in dropped[0]["drop_reason"]


def test_fetch_subparser_seed_reaches_the_recorded_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from tools import build_m2a

    root = tmp_path / "build"
    payload = image_bytes()
    build_m2a.write_jsonl(
        root / "filtered.jsonl",
        [{"id": 1, "file_name": "a.jpg", "url": "http://x/a.jpg"}],
    )
    monkeypatch.setattr(build_m2a, "_get", lambda url: payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_m2a.py",
            "--root",
            str(root),
            "fetch",
            "--cache",
            str(tmp_path / "cache"),
            "--seed",
            "7",
        ],
    )

    # main() is the only place the subparser's value is resolved via getattr, so
    # going through it proves both that --seed is exposed and that it is not lost.
    assert build_m2a.main() == 0

    fetched = build_m2a.read_jsonl(root / "fetched.jsonl")
    assert fetched[0]["order_seed"] == 7


def test_stage_images_downloads_only_missing_files(tmp_path: Path):
    """Re-staging a complete cache must be a no-op, not a re-download."""

    from PIL import Image

    from sightline.m2a.images import stage_images

    good = tmp_path / "present.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    digest = content_hash(good)

    calls = []

    def get(url):
        calls.append(url)
        return good.read_bytes()

    records = [
        {"id": 1, "file_name": "present.jpg", "url": "http://x/a", "sha256": digest},
        {"id": 2, "file_name": "absent.jpg", "url": "http://x/b", "sha256": "0" * 64},
    ]

    # Second record has a deliberately wrong expected hash so it fails verification;
    # the first must still be skipped without a download.
    result = stage_images(records, tmp_path, get, workers=1)

    assert calls == ["http://x/b"]
    assert result["n_present"] == 1
    assert result["n_failed"] == 1


def test_stage_images_is_idempotent(tmp_path: Path):
    from PIL import Image

    from sightline.m2a.images import stage_images

    image = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "blue").save(image)
    record = {"id": 1, "file_name": "a.jpg", "url": "http://x/a", "sha256": content_hash(image)}

    def boom(url):
        raise AssertionError("must not download an already-correct file")

    result = stage_images([record], tmp_path, boom, workers=1)

    assert result["n_present"] == 1
    assert result["n_downloaded"] == 0

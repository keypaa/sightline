"""Fetch images once into a cache, and drop duplicates by hash.

Duplicates matter because aggregation artefacts (mirrors, re-uploads, the same
photo at two ids) would otherwise put one image in two splits and leak across the
train/test boundary.

Two signals, because each has a blind spot:

- **sha256** of the file bytes is authoritative for exact duplicates.
- **dHash**, a 64-bit difference hash, catches near-duplicates such as a
  recompressed or resized copy. It is compared by Hamming distance, not equality,
  because exact equality is brittle.

dHash is **colour-blind**: it compares neighbouring luminance, so two images that
differ only in colour hash identically. That is why sha256 runs first and why the
threshold is a distance rather than equality. For photographic COCO data this is
acceptable; for flat synthetic images it would not be.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

_DHASH_SIDE = 8


def content_hash(path: Path) -> str:
    """sha256 of the file bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, side: int = _DHASH_SIDE) -> int:
    """Return a 64-bit difference hash of an image's luminance."""

    with Image.open(path) as image:
        small = image.convert("L").resize((side + 1, side), Image.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for row in range(side):
        offset = row * (side + 1)
        for col in range(side):
            bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return bits


def distance(a: int, b: int) -> int:
    """Hamming distance between two hashes."""

    return bin(a ^ b).count("1")


def fetch(record: dict[str, Any], cache_dir: Path, get: Callable[[str], bytes]) -> Path:
    """Download once into ``cache_dir`` and return the local path."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / record["file_name"]
    if not path.exists():
        path.write_bytes(get(record["url"]))
    return path


def dedupe(
    records: list[dict[str, Any]],
    hashes: dict[int, tuple[str, int]],
    max_distance: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first record for each distinct image; return the rest.

    ``hashes`` maps image id to ``(sha256, dhash)``. A record is dropped when its
    sha256 has been seen, or when its dHash is within ``max_distance`` of a kept
    image's. Order decides the survivor, so the result is deterministic.
    """

    seen_sha: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_hashes: list[int] = []
    dupes: list[dict[str, Any]] = []

    for record in records:
        sha, digest = hashes[record["id"]]
        if sha in seen_sha:
            dupes.append({**record, "duplicate_of": "sha256", "reason": "exact"})
            continue
        near = next(
            (i for i, other in enumerate(kept_hashes) if distance(digest, other) <= max_distance),
            None,
        )
        if near is not None:
            dupes.append(
                {
                    **record,
                    "duplicate_of": kept[near]["id"],
                    "reason": "perceptual",
                    "distance": distance(digest, kept_hashes[near]),
                }
            )
            continue
        seen_sha.add(sha)
        kept_hashes.append(digest)
        kept.append(record)
    return kept, dupes

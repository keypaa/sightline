"""Generate one conversation per image, with the provenance G1 would need.

The prompt and parser come from ``sightline.m2a.prompt`` — the same module the
pilot uses — so there is only one prompt to hash and one parser to satisfy. The
caller builds the annotator identity once per run (``provenance.annotator_fields``)
and passes it here, so every row carries the same provenance and a caller that
omits it records none rather than a stale default. The transport is injected,
which keeps the tests off the network and makes a rate limit the caller's problem
to handle.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sightline.m2a.prompt import PROMPT, parse_messages

Transport = Callable[[str, Path, str], tuple[str, dict[str, Any] | None]]


def prompt_hash(prompt: str = PROMPT) -> str:
    """Short sha256 of the prompt, for the run record."""

    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pending_records(
    fetched: list[dict[str, Any]], conversations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Records still needing annotation on a resume.

    Only parsed rows count as done. A rate-limited run writes blank rows with
    ``parsed=False``; treating those as done would silently drop them from the
    corpus and leave a hole instead of retrying them.
    """

    done = {row["id"] for row in conversations if row.get("parsed")}
    return [row for row in fetched if row["id"] not in done]


def annotate_one(
    model: str,
    image_path: Path,
    call: Transport,
    *,
    annotator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate one image and merge the run's annotator identity into the row.

    ``annotator`` is built once per run by the caller. Omitting it still produces
    a usable row, but one with no model or prompt hash: provenance is never
    guessed here.
    """

    raw, tokens = call(model, image_path, PROMPT)
    messages = parse_messages(raw)
    return {
        **(annotator or {}),
        "messages": messages,
        "parsed": messages is not None,
        "tokens_in": (tokens or {}).get("input"),
        "tokens_out": (tokens or {}).get("output"),
        "image_sha256": file_hash(image_path),
    }

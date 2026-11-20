"""Generate one conversation per image, with the provenance G1 would need.

The prompt and parser come from ``sightline.m2a.prompt`` — the same module the
pilot uses — so the prompt hash recorded here is the annotator's identity rather
than one of two drifting copies. The transport is injected, which keeps the tests
off the network and makes a rate limit the caller's problem to handle.
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


def annotate_one(model: str, image_path: Path, call: Transport) -> dict[str, Any]:
    """Annotate one image and return the conversation with its provenance."""

    raw, tokens = call(model, image_path, PROMPT)
    messages = parse_messages(raw)
    return {
        "messages": messages,
        "parsed": messages is not None,
        "tokens_in": (tokens or {}).get("input"),
        "tokens_out": (tokens or {}).get("output"),
        "model": model,
        "prompt_sha256": prompt_hash(),
        "image_sha256": file_hash(image_path),
    }

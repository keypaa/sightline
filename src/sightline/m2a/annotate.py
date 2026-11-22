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
from sightline.m2a.transport import RateLimited, TransportError

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


def annotate_row(
    record: dict[str, Any], result: dict[str, Any], *, run_id: str, created_at: str
) -> dict[str, Any]:
    """Build the conversation row: the fetch record plus the annotator result.

    ``path`` is dropped because it points into a local cache and means nothing on
    another machine; ``sha256`` and ``image_sha256`` carry the identity instead.
    """

    row = {key: value for key, value in record.items() if key != "path"}
    row.update(result)
    row["run_id"] = run_id
    row["created_at"] = created_at
    return row


_ANNOTATOR_KEYS = (
    "model",
    "revision",
    "quantization",
    "prompt_sha256",
    "chat_template_sha256",
)


def check_single_annotator(existing: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Refuse a row written by a different annotator than the corpus already has.

    A corpus with two annotators cannot be compared to itself, and the difference
    is invisible in a loss curve. Refusing at write time is cheaper than detecting
    it in the manifest afterwards.
    """

    if not existing:
        return
    # A row with no annotator recorded predates this schema. It cannot be compared,
    # so it is skipped rather than blocking the run; rows that do declare an
    # annotator must all agree with each other.
    declared = [row for row in existing if all(row.get(key) is not None for key in _ANNOTATOR_KEYS)]
    if not declared:
        return
    first = declared[0]
    for key in _ANNOTATOR_KEYS:
        if row.get(key) != first.get(key):
            raise ValueError(
                f"annotator identity differs from the first row in this corpus ({key})"
            )


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


def annotate_with_retry(
    model: str,
    image_path: Path,
    call: Transport,
    *,
    annotator: dict[str, Any] | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    """Retry an empty completion from the local server; never retry a quota error.

    Under load the server occasionally returns HTTP 200 with no content. That is a
    transient glitch, not a reason to abort a corpus run. A confirmed rate limit is
    different: it must reach the guard so the whole pool stops and waits.
    """

    for attempt in range(attempts):
        try:
            return annotate_one(model, image_path, call, annotator=annotator)
        except RateLimited:
            raise
        except TransportError as exc:
            if "no content" not in str(exc) or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def failed_row(
    record: dict[str, Any], error: str, *, run_id: str, created_at: str
) -> dict[str, Any]:
    """A row for an image that could not be annotated, so the run continues.

    It stays in conversations.jsonl with parsed=False, which makes resume retry it.
    The error is recorded rather than swallowed.
    """

    row = {key: value for key, value in record.items() if key != "path"}
    row.update(
        {
            "messages": None,
            "parsed": False,
            "tokens_in": None,
            "tokens_out": None,
            "error": error,
            "run_id": run_id,
            "created_at": created_at,
        }
    )
    return row

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sightline.m2a.annotate import annotate_one, pending_records


class StubTok:
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "red"


def stub(text: str, tokens: dict | None = None):
    def call(model, image_path, prompt):
        return text, tokens if tokens is not None else {"input": 3179, "output": 99}

    return call


GOOD = json.dumps(
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Red."}]}
)


def test_provenance_is_recorded(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"payload")

    result = annotate_one("model-x", image, stub(GOOD))

    assert result["model"] == "model-x"
    assert result["prompt_sha256"]
    assert result["image_sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert result["parsed"] is True
    assert result["tokens_in"] == 3179
    assert result["tokens_out"] == 99


def test_prose_output_is_marked_unparsed(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"x")

    result = annotate_one("model-x", image, stub("sorry, I cannot"))

    assert result["parsed"] is False
    assert result["messages"] is None


def test_fenced_json_still_parses(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"x")

    result = annotate_one("model-x", image, stub("```json\n" + GOOD + "\n```"))

    assert result["parsed"] is True


def test_prompt_hash_is_stable(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"x")

    first = annotate_one("m", image, stub(GOOD))
    second = annotate_one("m", image, stub(GOOD))

    assert first["prompt_sha256"] == second["prompt_sha256"]
    assert len(first["prompt_sha256"]) == 16


def test_resume_retries_unparsed_rows() -> None:
    """A rate-limited run left blank rows; they must not count as done."""

    fetched = [{"id": i} for i in range(5)]
    conversations = [
        {"id": 0, "parsed": True},
        {"id": 1, "parsed": False, "tokens_in": None},
        {"id": 2, "parsed": True},
    ]

    assert [r["id"] for r in pending_records(fetched, conversations)] == [1, 3, 4]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sightline.m2a.annotate import annotate_one, pending_records, prompt_hash


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

ANNOTATOR = {
    "model": "Qwen/Qwen3.6-27B-FP8",
    "revision": "abc123",
    "quantization": "fp8",
    "engine": "vllm",
    "engine_version": "0.17.0",
    "prompt_sha256": prompt_hash(),
    "chat_template_sha256": "deadbeefdeadbeef",
    "seed": 0,
    "order_seed": 0,
    "temperature": 0.0,
}


def test_provenance_is_recorded(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"payload")

    result = annotate_one(ANNOTATOR["model"], image, stub(GOOD), annotator=ANNOTATOR)

    assert result["model"] == ANNOTATOR["model"]
    assert result["revision"] == "abc123"
    assert result["prompt_sha256"] == prompt_hash()
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

    first = annotate_one("m", image, stub(GOOD), annotator=ANNOTATOR)
    second = annotate_one("m", image, stub(GOOD), annotator=ANNOTATOR)

    assert first["prompt_sha256"] == second["prompt_sha256"] == prompt_hash()
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


def test_annotate_one_records_the_annotator_fields(tmp_path):
    from PIL import Image

    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(img)

    annotator = {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "revision": "abc123",
        "quantization": "fp8",
        "seed": 0,
        "order_seed": 0,
    }
    result = annotate_one("Qwen/Qwen3.6-27B-FP8", img, stub(GOOD), annotator=annotator)

    assert result["revision"] == "abc123"
    assert result["quantization"] == "fp8"
    assert result["parsed"] is True


def test_caller_without_annotator_records_no_model_provenance(tmp_path: Path) -> None:
    """Omitting the annotator must not fall back to a stale model or prompt hash."""

    image = tmp_path / "a.png"
    image.write_bytes(b"x")

    result = annotate_one("model-x", image, stub(GOOD))

    assert "model" not in result
    assert "prompt_sha256" not in result
    assert result["image_sha256"] == hashlib.sha256(b"x").hexdigest()
    assert result["parsed"] is True


def test_cmd_annotate_uses_recorded_order_and_one_annotator(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from PIL import Image

    from sightline.m2a.order import corpus_order, write_order
    from tools import build_m2a

    root = tmp_path / "build"
    records = []
    for i in range(3):
        image = root / "images" / f"{i}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(image)
        records.append(
            {"id": i, "path": str(image), "source": "coco", "order_seed": 0, "sha256": f"h{i}"}
        )
    build_m2a.write_jsonl(root / "fetched.jsonl", records)
    ordered = corpus_order(records, seed=0)
    write_order(root / "order.jsonl", ordered, seed=0)

    done_id = ordered[2]["id"]
    fail_id = ordered[1]["id"]
    build_m2a.write_jsonl(root / "conversations.jsonl", [{"id": done_id, "parsed": True}])

    seen: list[tuple] = []

    class FakeVLLM:
        def __init__(self, model: str, *, base_url: str, temperature: float = 0.0) -> None:
            seen.append(("init", model, base_url, temperature))

        def __call__(self, model: str, image_path: Path, prompt: str):
            seen.append(("call", model))
            text = "no json here" if Path(image_path).stem == str(fail_id) else GOOD
            return text, {"input": 10, "output": 5}

    def sync_retry_loop(pending, work, on_row, workers, **kwargs):
        for record in pending:
            on_row(work(record))
        return len(pending), 0

    monkeypatch.setattr("sightline.m2a.local_transport.LocalVLLM", FakeVLLM)
    monkeypatch.setattr("sightline.m2a.transport.retry_loop", sync_retry_loop)

    args = argparse.Namespace(
        root=root,
        model="Qwen/Qwen3.6-27B-FP8",
        revision="abc123",
        quantization="fp8",
        engine_version="0.17.0",
        chat_template_sha256="deadbeefdeadbeef",
        temperature=0.0,
        base_url="http://localhost:8000/v1",
        workers=1,
        limit=0,
        empty_limit=10,
        wait_seconds=0,
        max_waits=0,
        seed=0,
    )

    assert build_m2a.cmd_annotate(args) == 0

    rows = build_m2a.read_jsonl(root / "conversations.jsonl")
    new_rows = rows[1:]
    assert [row["id"] for row in new_rows] == [ordered[0]["id"], fail_id]
    for row in new_rows:
        assert row["model"] == args.model
        assert row["revision"] == "abc123"
        assert row["quantization"] == "fp8"
        assert row["engine"] == "vllm"
        assert row["engine_version"] == "0.17.0"
        assert row["prompt_sha256"] == prompt_hash()
        assert row["chat_template_sha256"] == "deadbeefdeadbeef"
        assert row["order_seed"] == 0
        assert row["temperature"] == 0.0
        assert row["image_sha256"]

    failures = build_m2a.read_jsonl(root / "annotate_failures.jsonl")
    assert [row["id"] for row in failures] == [fail_id]
    assert failures[0]["parsed"] is False

    init_models = [event[1] for event in seen if event[0] == "init"]
    call_models = [event[1] for event in seen if event[0] == "call"]
    assert init_models == [args.model]
    assert call_models == [args.model, args.model]

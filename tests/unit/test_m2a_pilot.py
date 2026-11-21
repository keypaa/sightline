"""Tests for the local annotator pilot, all with a stubbed transport."""

from __future__ import annotations

import json
from pathlib import Path

GOOD = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "How many?"},
            {"role": "assistant", "content": "Three."},
        ]
    }
)


def test_pilot_summary_computes_the_two_rates() -> None:
    from tools.m2a_annotator_pilot import summarise

    rows = [
        {"parsed": True, "grounded": True},
        {"parsed": True, "grounded": False},
        {"parsed": False, "grounded": False},
    ]
    summary = summarise(rows)
    assert summary["parse_rate"] == 2 / 3
    assert summary["grounded_rate"] == 1 / 3
    assert summary["n"] == 3


def test_summary_means_ignore_missing_token_counts() -> None:
    from tools.m2a_annotator_pilot import summarise

    rows = [
        {"parsed": True, "grounded": True, "tokens_in": 10, "tokens_out": 2},
        {"parsed": True, "grounded": True, "tokens_in": 20, "tokens_out": 4},
        {"parsed": False, "grounded": False, "tokens_in": None, "tokens_out": None},
    ]
    summary = summarise(rows)
    assert summary["tokens_in_mean"] == 15
    assert summary["tokens_out_mean"] == 3


def test_grounded_in_uses_answer_forms() -> None:
    from tools.m2a_annotator_pilot import grounded_in

    messages = [{"role": "assistant", "content": "Three."}]
    assert grounded_in(messages, "3") is True
    assert grounded_in(messages, "4") is False


def test_grounded_in_ignores_user_turns_and_handles_unparsed() -> None:
    from tools.m2a_annotator_pilot import grounded_in

    messages = [{"role": "user", "content": "It is three."}]
    assert grounded_in(messages, "3") is False
    assert grounded_in(None, "3") is False


def test_missing_fetched_exits_with_a_clear_message(tmp_path: Path, capsys) -> None:
    from tools import m2a_annotator_pilot

    code = m2a_annotator_pilot.main(
        [
            "--fetched",
            str(tmp_path / "fetched.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--revision",
            "abc",
            "--quantization",
            "fp8",
            "--engine-version",
            "0.17.0",
            "--chat-template-sha256",
            "deadbeefdeadbeef",
        ]
    )

    assert code == 2
    assert "refusing to invent pilot images" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_pilot_writes_grounded_rows_with_a_stubbed_transport(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from PIL import Image

    from tools import m2a_annotator_pilot

    images = []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (8, 8), "red").save(path)
        images.append(path)
    fetched = tmp_path / "fetched.jsonl"
    records = [
        {"id": 0, "file_name": "0.jpg", "path": str(images[0]), "expected": "3"},
        {"id": 1, "file_name": "1.jpg", "path": str(images[1]), "expected": "4"},
        {"id": 2, "file_name": "2.jpg", "path": str(images[2])},
    ]
    fetched.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    seen: list[tuple[str, str]] = []

    class FakeVLLM:
        def __init__(
            self, model: str, *, base_url: str, temperature: float = 0.0, **kwargs
        ) -> None:
            seen.append(("init", base_url))

        def __call__(self, model: str, image_path: Path, prompt: str):
            seen.append(("call", model))
            text = "no json here" if Path(image_path).stem == "1" else GOOD
            return text, {"input": 10, "output": 2}

    monkeypatch.setattr(m2a_annotator_pilot, "LocalVLLM", FakeVLLM)
    out = tmp_path / "out"
    report = tmp_path / "report.md"

    code = m2a_annotator_pilot.main(
        [
            "--images",
            "3",
            "--fetched",
            str(fetched),
            "--out",
            str(out),
            "--report",
            str(report),
            "--host",
            "http://localhost:8000/v1",
            "--revision",
            "abc123",
            "--quantization",
            "fp8",
            "--engine-version",
            "0.17.0",
            "--chat-template-sha256",
            "deadbeefdeadbeef",
        ]
    )

    rows = [json.loads(line) for line in (out / "pilot.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == [0, 1, 2]
    assert [row["parsed"] for row in rows] == [True, False, True]
    assert [row["grounded"] for row in rows] == [True, False, False]
    assert rows[0]["model"] == "Qwen/Qwen3.6-27B-FP8"
    assert rows[0]["revision"] == "abc123"
    assert rows[0]["prompt_sha256"]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["parse_rate"] == 2 / 3
    assert summary["grounded_rate"] == 1 / 3
    assert summary["tokens_in_mean"] == 10
    assert summary["n_with_expected"] == 2
    assert summary["conversations_per_hour"] > 0
    assert summary["passes_parse_bar"] is False  # 2/3 < 0.95
    assert summary["grounding_measurable"] is True
    assert code == 1  # the gate failed, so the run must not be treated as a pass

    assert (out / "run_record.json").exists()
    assert "grounded rate" in report.read_text()
    assert "http://localhost:8000/v1" in [event[1] for event in seen if event[0] == "init"]
    assert '"parse_rate"' in capsys.readouterr().out  # the summary is printed


def test_pilot_disables_thinking_when_building_the_transport(monkeypatch, tmp_path):
    """Qwen3.6 thinks by default; with a small cap that leaves no answer to parse."""

    from tools import m2a_annotator_pilot as pilot

    captured = {}

    class FakeLocal:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)

        def __call__(self, model, image_path, prompt):
            return "ok", {"input": 1, "output": 1}

    class FakeAnnotator:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(pilot, "LocalVLLM", FakeLocal)
    monkeypatch.setattr(
        pilot,
        "annotate_one",
        lambda *a, **k: {
            "messages": None,
            "parsed": False,
            "tokens_in": None,
            "tokens_out": None,
            "model": "m",
            "prompt_sha256": "x",
        },
    )
    monkeypatch.setattr(pilot, "prompt_hash", lambda: "abc")
    monkeypatch.setattr(pilot, "summarise", lambda rows: {"parse_rate": 0.0, "grounded_rate": 0.0})
    monkeypatch.setattr(pilot, "write_report", lambda *a, **k: None)
    monkeypatch.setattr(pilot, "sample_subset", lambda records, n, seed: records[:1])

    fetched = tmp_path / "fetched.jsonl"
    fetched.write_text(json.dumps({"id": 0, "file_name": "a.jpg", "path": "a.jpg"}) + "\n")

    pilot.main(
        [
            "--fetched",
            str(fetched),
            "--out",
            str(tmp_path / "out"),
            "--report",
            str(tmp_path / "r.md"),
            "--revision",
            "r",
            "--quantization",
            "fp8",
            "--engine-version",
            "0.17",
            "--chat-template-sha256",
            "d",
        ]
    )

    assert captured.get("chat_template_kwargs") == {"enable_thinking": False}


def test_pilot_excludes_listed_ids(tmp_path):
    from tools import m2a_annotator_pilot as pilot

    records = [{"id": i, "file_name": f"{i}.jpg", "path": f"{i}.jpg"} for i in range(10)]
    excluded = tmp_path / "exclude.json"
    excluded.write_text(json.dumps({"ids": [0, 1, 2]}))

    kept = pilot.exclude_ids(records, pilot.load_excluded(excluded))

    assert [r["id"] for r in kept] == list(range(3, 10))


def test_unmeasurable_grounding_does_not_fail_the_run():
    """COCO has no expected answers; that is the instrument, not a bad annotator."""

    from tools import m2a_annotator_pilot as pilot
    import inspect

    source = inspect.getsource(pilot.main)
    assert "grounding_skipped" in source
    assert "grounding_skipped" in source.split('summary["passes"]')[0]

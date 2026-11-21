"""Tests for the verifier transport and the judge/blind orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sightline.m2a.transport import RateLimited, TransportError
from sightline.m2a.verifier_transport import OpencodeCaller
from sightline.m2a.verify import (
    Claim,
    blind_matches,
    judge_blind,
    judge_claims,
    normalise_tokens,
)


class Resp:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def text_event(text: str) -> str:
    return json.dumps({"type": "text", "part": {"text": text}})


def error_event(status: int, message: str) -> str:
    return json.dumps(
        {"type": "error", "error": {"type": "provider.quota", "status": status, "message": message}}
    )


def test_caller_passes_archived_and_the_image(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return Resp(text_event("true\nthe car is red"))

    monkeypatch.setattr("sightline.m2a.verifier_transport.subprocess.run", fake_run)
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")

    text = OpencodeCaller("m")(image, "prompt")

    assert "true" in text
    assert captured["args"][:3] == ["opencode", "run", "--archived"]
    assert "--file" in captured["args"]


def test_caller_omits_the_image_for_a_blind_call(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        return Resp(text_event("red"))

    monkeypatch.setattr("sightline.m2a.verifier_transport.subprocess.run", fake_run)

    OpencodeCaller("m")(None, "prompt")

    assert "--file" not in captured["args"]


def test_caller_maps_429_to_rate_limited(monkeypatch):
    monkeypatch.setattr(
        "sightline.m2a.verifier_transport.subprocess.run",
        lambda *a, **k: Resp(error_event(429, "Rate limit exceeded"), returncode=1),
    )

    with pytest.raises(RateLimited):
        OpencodeCaller("m")(None, "prompt")


def test_caller_maps_other_errors_to_transport_error(monkeypatch):
    monkeypatch.setattr(
        "sightline.m2a.verifier_transport.subprocess.run",
        lambda *a, **k: Resp("", returncode=2),
    )

    with pytest.raises(TransportError):
        OpencodeCaller("m")(None, "prompt")


class Stub:
    def __init__(self, model: str, verdict: str, evidence: str = "because") -> None:
        self.model = model
        self.verdict = verdict
        self.evidence = evidence
        self.calls = 0

    def __call__(self, image, prompt):
        self.calls += 1
        return f"{self.verdict}\n{self.evidence}"


def claim(image_id: int = 0, planted: bool = False) -> Claim:
    return Claim(
        image_id=image_id,
        turn=1,
        question="What colour is the car?",
        answer="Red.",
        claim="Red (car)",
        claim_type="colour",
        planted=planted,
        image_path=None,
    )


def test_judge_collects_a_verdict_from_every_caller():
    callers = [Stub("v1", "true"), Stub("v2", "false")]

    results = judge_claims([claim()], callers)

    assert len(results) == 1
    assert {v.verdict for v in results[0].verdicts} == {"true", "false"}
    assert results[0].contradicted is True


def test_judge_resumes_without_re_asking_a_recorded_model():
    callers = [Stub("v1", "true"), Stub("v2", "false")]
    existing = [
        {
            "image_id": 0,
            "turn": 1,
            "planted": False,
            "verdicts": [{"model": "v1", "verdict": "false", "evidence": "prior"}],
        }
    ]

    results = judge_claims([claim()], callers, existing=existing)

    assert callers[0].calls == 0
    assert callers[1].calls == 1
    assert any(v.model == "v1" and v.verdict == "false" for v in results[0].verdicts)


def test_judge_records_a_failed_call_as_an_error_not_a_verdict():
    class Boom:
        model = "v1"

        def __call__(self, image, prompt):
            raise TransportError("opencode exited with code 1")

    results = judge_claims([claim()], [Boom()])

    assert results[0].verdicts[0].error is not None
    assert results[0].contradicted is False


def test_blind_matches_uses_word_subset():
    assert blind_matches("Red.", "The car is red.") is True
    assert blind_matches("Three.", "three") is True
    assert blind_matches("Red.", "I do not know") is False
    assert blind_matches("", "anything") is False


def test_normalise_tokens_drops_punctuation():
    assert normalise_tokens("Red.") == {"red"}
    assert normalise_tokens("Three!") == {"three"}


def test_judge_blind_asks_without_an_image_and_records_the_answer():
    seen = {}

    class Blind:
        def __call__(self, image, prompt):
            seen["image"] = image
            seen["prompt"] = prompt
            return "Red."

    rows = judge_blind([claim()], Blind())

    assert seen["image"] is None
    assert len(rows) == 1
    assert rows[0]["blind_matches"] is True
    assert rows[0]["expected"] == "Red."


def test_judge_blind_marks_a_failed_call_instead_of_guessing():
    class Boom:
        def __call__(self, image, prompt):
            raise TransportError("no quota")

    rows = judge_blind([claim()], Boom())

    assert rows[0]["blind_answer"] is None
    assert rows[0]["blind_matches"] is False
    assert "no quota" in rows[0]["error"]

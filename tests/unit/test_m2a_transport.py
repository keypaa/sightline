"""The rate-limit guard, tested against the error event the CLI really emits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sightline.m2a.transport import (
    RateLimited,
    RateLimitGuard,
    TransportError,
    cli_call,
    parse_cli_events,
    run_pool,
)

# Captured from a real `opencode run` during the killed 20k annotation run.
RATE_LIMIT_EVENT = json.dumps(
    {
        "type": "error",
        "sessionID": "ses_e0609dde7ffe05FcBR0QpkiQc7",
        "error": {
            "type": "provider.quota",
            "message": "Rate limit exceeded",
            "status": 429,
            "response": {
                "body": (
                    '{"error":"Rate limit exceeded","error_type":"rate_limit_exceeded",'
                    '"message":"Free model usage limit reached. Please try again later, '
                    'or switch to a paid model or your own API key."}'
                )
            },
        },
    }
)

TEXT_EVENT = json.dumps({"type": "text", "part": {"text": "Red."}})
TOKENS_EVENT = json.dumps(
    {"type": "step_finish", "part": {"tokens": {"input": 16231, "output": 110}}}
)


def test_the_real_429_event_is_detected() -> None:
    with pytest.raises(RateLimited) as caught:
        parse_cli_events(TEXT_EVENT + "\n" + RATE_LIMIT_EVENT, returncode=1)

    assert caught.value.status == 429
    assert "Rate limit exceeded" in caught.value.message


def test_a_credit_or_payment_error_is_also_rate_limited() -> None:
    event = json.dumps(
        {"type": "error", "error": {"status": 402, "message": "Insufficient credits"}}
    )

    with pytest.raises(RateLimited) as caught:
        parse_cli_events(event, returncode=1)

    assert caught.value.status == 402


def test_a_clean_run_returns_text_and_tokens() -> None:
    text, tokens = parse_cli_events(TEXT_EVENT + "\n" + TOKENS_EVENT, returncode=0)

    assert text == "Red."
    assert tokens == {"input": 16231, "output": 110}


def test_a_nonzero_exit_without_an_error_event_is_a_transport_error() -> None:
    with pytest.raises(TransportError):
        parse_cli_events("", returncode=1)


def test_cli_call_creates_an_archived_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The flood was one visible session per call; --archived is the fix."""

    captured: dict = {}

    class Result:
        returncode = 0
        stdout = TEXT_EVENT + "\n" + TOKENS_EVENT
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return Result()

    monkeypatch.setattr("sightline.m2a.transport.subprocess.run", fake_run)
    image = tmp_path / "a.png"
    image.write_bytes(b"x")

    cli_call("model-x", image, "prompt")

    assert captured["args"][:4] == ["opencode", "run", "--archived", "--model"]


def test_guard_trips_on_the_first_confirmed_rate_limit() -> None:
    guard = RateLimitGuard(empty_limit=100)

    guard.note_rate_limit(RateLimited(status=429, message="Rate limit exceeded"))

    assert guard.tripped
    assert "429" in (guard.reason or "")


def test_guard_trips_after_a_streak_of_tokenless_responses() -> None:
    guard = RateLimitGuard(empty_limit=3)
    blank = {"parsed": False, "tokens_in": None, "tokens_out": None}

    guard.observe(blank)
    guard.observe(blank)
    assert not guard.tripped

    guard.observe(blank)
    assert guard.tripped
    assert "3 consecutive" in (guard.reason or "")


def test_a_good_response_resets_the_tokenless_streak() -> None:
    guard = RateLimitGuard(empty_limit=3)
    blank = {"parsed": False, "tokens_in": None, "tokens_out": None}
    good = {"parsed": True, "tokens_in": 100, "tokens_out": 10}

    guard.observe(blank)
    guard.observe(blank)
    guard.observe(good)
    guard.observe(blank)
    guard.observe(blank)

    assert not guard.tripped


def test_run_pool_stops_submitting_once_the_guard_trips() -> None:
    """The core of the flood fix: no new calls after a rate limit."""

    calls = 0

    def work(record):
        nonlocal calls
        calls += 1
        raise RateLimited(status=429, message="Rate limit exceeded")

    guard = RateLimitGuard(empty_limit=100)
    written = []
    run_pool(
        list(range(1000)),
        work,
        written.append,
        workers=3,
        guard=guard,
    )

    assert guard.tripped
    assert calls <= 3, f"spawned {calls} calls after the limit was hit"
    assert written == []


def test_run_pool_drains_in_flight_work_and_keeps_completed_rows() -> None:
    def work(record):
        return {"id": record, "parsed": True, "tokens_in": 10, "tokens_out": 2}

    guard = RateLimitGuard(empty_limit=100)
    written = []
    count = run_pool(list(range(10)), work, written.append, workers=3, guard=guard)

    assert count == 10
    assert {row["id"] for row in written} == set(range(10))
    assert not guard.tripped


def test_guard_also_trips_on_a_non_rate_transport_error() -> None:
    from sightline.m2a.transport import RateLimitGuard

    guard = RateLimitGuard(empty_limit=100)
    guard.note_transport_error(TransportError("opencode exited with code 1"))

    assert guard.tripped
    assert "transport error" in (guard.reason or "")


def test_guard_distinguishes_rate_limits_from_other_transport_errors() -> None:
    rate = RateLimitGuard(empty_limit=100)
    rate.note_rate_limit(RateLimited(status=429, message="Rate limit exceeded"))

    other = RateLimitGuard(empty_limit=100)
    other.note_transport_error(TransportError("opencode exited with code 1"))

    assert rate.rate_limited
    assert not other.rate_limited


def test_retry_loop_waits_for_the_quota_window_and_finishes() -> None:
    from sightline.m2a.transport import retry_loop

    attempts: dict[int, int] = {}
    sleeps: list[int] = []

    def work(record):
        rid = record["id"]
        attempts[rid] = attempts.get(rid, 0) + 1
        if rid == 1 and attempts[rid] == 1:
            raise RateLimited(status=429, message="Rate limit exceeded")
        return {"id": rid, "parsed": True, "tokens_in": 10, "tokens_out": 2}

    rows = []
    completed, waits = retry_loop(
        [{"id": i} for i in range(3)],
        work,
        rows.append,
        workers=2,
        wait_seconds=3600,
        max_waits=3,
        sleeper=sleeps.append,
        reporter=lambda *a: None,
    )

    assert completed == 3
    assert waits == 1
    assert sleeps == [3600]
    assert {r["id"] for r in rows} == {0, 1, 2}


def test_retry_loop_stops_at_the_wait_cap() -> None:
    from sightline.m2a.transport import retry_loop

    def work(record):
        raise RateLimited(status=429, message="Rate limit exceeded")

    sleeps: list[int] = []
    completed, waits = retry_loop(
        [{"id": i} for i in range(5)],
        work,
        lambda row: None,
        workers=2,
        wait_seconds=60,
        max_waits=2,
        sleeper=sleeps.append,
        reporter=lambda *a: None,
    )

    assert completed == 0
    assert waits == 2
    assert sleeps == [60, 60]


def test_retry_loop_does_not_wait_for_a_non_rate_transport_error() -> None:
    from sightline.m2a.transport import retry_loop

    def work(record):
        raise TransportError("opencode exited with code 1")

    sleeps: list[int] = []
    completed, waits = retry_loop(
        [{"id": 0}],
        work,
        lambda row: None,
        workers=1,
        wait_seconds=60,
        max_waits=3,
        sleeper=sleeps.append,
        reporter=lambda *a: None,
    )

    assert completed == 0
    assert waits == 0
    assert sleeps == []

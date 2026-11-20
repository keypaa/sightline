"""Transport and circuit breaker for scripted annotation calls.

`opencode run` is the only verified transport for the pinned annotator, and each
call creates a session. Two guards matter in a long run:

1. every call is archived, so scripted sessions stay out of the session list;
2. a rate limit stops the pool *before* another call is submitted, rather than
   turning into a blank row that looks like an annotation failure.

The 429 event in the tests was captured from a real run, not invented.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

RATE_LIMIT_STATUSES = frozenset({402, 429})
_RATE_TEXT = re.compile(r"rate limit|quota|credit|usage limit|insufficient", re.IGNORECASE)


class TransportError(RuntimeError):
    """A call failed for a reason other than rate limiting."""


class RateLimited(TransportError):
    """The provider refused the call; stop submitting more work."""

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"rate limited (HTTP {status}): {message}")


def parse_cli_events(stdout: str, returncode: int) -> tuple[str, dict[str, Any] | None]:
    """Read the JSON event stream, raising on provider errors.

    Events are parsed before the return code is checked: a 429 arrives as both an
    ``error`` event and exit code 1, and the event carries the useful status.
    """

    texts: list[str] = []
    tokens: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "error":
            error = event.get("error") or {}
            status = error.get("status")
            try:
                status = int(status) if status is not None else None
            except (TypeError, ValueError):
                status = None
            message = str(error.get("message") or "")
            body = json.dumps(error)
            if (
                status in RATE_LIMIT_STATUSES
                or error.get("type") == "provider.quota"
                or _RATE_TEXT.search(message)
                or _RATE_TEXT.search(body)
            ):
                raise RateLimited(status=status, message=message or body[:200])
            raise TransportError(message or body[:200])
        elif kind == "text":
            texts.append(event["part"]["text"])
        elif kind == "step_finish":
            tokens = event["part"].get("tokens")

    if returncode != 0:
        raise TransportError(f"opencode exited with code {returncode}")
    return "\n".join(texts).strip(), tokens


def cli_call(model: str, image_path: Path, prompt: str) -> tuple[str, dict[str, Any] | None]:
    """Annotate one image through an archived opencode session."""

    result = subprocess.run(
        [
            "opencode",
            "run",
            "--archived",
            "--model",
            model,
            "--file",
            str(image_path),
            "--format",
            "json",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return parse_cli_events(result.stdout, result.returncode)


class RateLimitGuard:
    """A stop signal shared by every worker in a pool.

    Two trips: a confirmed provider rate limit, or a run of responses with no
    token usage. The second catches a call that failed without emitting a
    structured error event — the signature of the 433 blank rows in the killed
    run. A response with tokens resets the streak, so genuinely unparseable
    answers do not stop the corpus build.
    """

    def __init__(self, empty_limit: int = 10) -> None:
        self.empty_limit = empty_limit
        self.reason: str | None = None
        self.rate_limited = False
        self._consecutive_empty = 0
        self._lock = threading.Lock()

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self.reason is not None

    def note_rate_limit(self, error: RateLimited) -> None:
        with self._lock:
            self.rate_limited = True
            if self.reason is None:
                self.reason = f"rate limited (HTTP {error.status}): {error.message}"

    def note_transport_error(self, error: TransportError) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = f"transport error: {error}"

    def observe(self, row: dict[str, Any]) -> None:
        with self._lock:
            if self.reason is not None:
                return
            if row.get("tokens_in") is None and row.get("tokens_out") is None:
                self._consecutive_empty += 1
                if self._consecutive_empty >= self.empty_limit:
                    self.reason = f"{self.empty_limit} consecutive responses with no token usage"
            else:
                self._consecutive_empty = 0


def run_pool(
    pending: list[dict[str, Any]],
    work: Callable[[dict[str, Any]], dict[str, Any]],
    on_row: Callable[[dict[str, Any]], None],
    workers: int,
    guard: RateLimitGuard,
) -> int:
    """Run at most `workers` calls in flight and submit no more once guarded.

    Submitting all futures up front — what the killed run did — means stopping
    the result loop does not stop the calls. Here a new task is submitted only
    when an in-flight one finishes and the guard has not tripped.
    """

    inflight: dict = {}
    completed = 0
    remaining = iter(pending)

    def submit() -> bool:
        try:
            record = next(remaining)
        except StopIteration:
            return False
        inflight[pool.submit(work, record)] = record
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            if not submit():
                break

        while inflight:
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                inflight.pop(future)
                try:
                    row = future.result()
                    on_row(row)
                    guard.observe(row)
                    completed += 1
                except RateLimited as error:
                    guard.note_rate_limit(error)
                except TransportError as error:
                    guard.note_transport_error(error)
            if not guard.tripped:
                submit()

    return completed


def retry_loop(
    pending: list[dict[str, Any]],
    work: Callable[[dict[str, Any]], dict[str, Any]],
    on_row: Callable[[dict[str, Any]], None],
    workers: int,
    *,
    empty_limit: int = 10,
    wait_seconds: int = 3600,
    max_waits: int = 120,
    sleeper: Callable[[int], None] = time.sleep,
    reporter: Callable[..., None] = print,
) -> tuple[int, int]:
    """Resume after each quota window, waiting only on a confirmed rate limit.

    Returns the number of distinct rows completed and the number of waits used.
    A record whose call failed is left in ``remaining`` and retried next window;
    the rows that did complete are never re-annotated.
    """

    remaining = list(pending)
    completed = 0
    waits = 0

    while remaining:
        guard = RateLimitGuard(empty_limit=empty_limit)
        finished_ids: set[Any] = set()

        def tracked(row: dict[str, Any], finished_ids: set[Any] = finished_ids) -> None:
            on_row(row)
            finished_ids.add(row.get("id"))

        run_pool(remaining, work, tracked, workers, guard)
        completed += len(finished_ids)
        remaining = [row for row in remaining if row.get("id") not in finished_ids]

        if not guard.tripped or not remaining:
            break
        if not guard.rate_limited:
            reporter(f"[retry] stopped: {guard.reason}")
            break
        if waits >= max_waits:
            reporter(f"[retry] stopped: {max_waits} waits exhausted")
            break

        waits += 1
        reporter(
            f"[retry] quota window {waits}/{max_waits}: "
            f"{completed} done, {len(remaining)} left; waiting {wait_seconds}s"
        )
        sleeper(wait_seconds)

    return completed, waits

"""Console progress for long training runs.

A 2048-step run prints one line every ``every`` steps with the step count, the
loss, the median step time, the elapsed time and an ETA. The median is used
rather than the mean because the first steps are slower while the allocator and
cudnn settle.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any


def format_duration(seconds: float) -> str:
    """Format a duration as H:MM:SS or M:SS."""

    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class ProgressPrinter:
    """Callable progress sink for ``Trainer(on_step=...)``.

    Prints on every ``every``-th step and always on the final step, so a run
    never ends without a closing line. ``every=0`` silences it.
    """

    def __init__(
        self,
        total: int,
        every: int = 10,
        stream: Callable[[str], None] | None = None,
        label: str = "",
    ) -> None:
        self.total = total
        self.every = every
        self.stream = stream if stream is not None else _print_flushed
        self.label = label

    def __call__(self, info: dict[str, Any]) -> None:
        step = int(info["step"])
        if self.every <= 0:
            return
        if step % self.every != 0 and step != self.total:
            return
        median_step_s = float(info.get("median_step_s", info["step_s"]))
        remaining = max(0, self.total - step)
        percent = 100.0 * step / self.total if self.total else 100.0
        parts = [
            f"step {step}/{self.total}",
            f"{percent:.0f}%",
            f"loss {float(info['loss']):.4f}",
            f"{median_step_s:.3f} s/step",
            f"elapsed {format_duration(info.get('elapsed_s', median_step_s * step))}",
            f"eta {format_duration(median_step_s * remaining)}",
        ]
        if info.get("grad_norm") is not None:
            parts.append(f"gnorm {float(info['grad_norm']):.3e}")
        if self.label:
            parts.insert(0, self.label)
        self.stream("  ".join(parts))


def _print_flushed(line: str) -> None:
    print(line, flush=True)


def stderr_printer(total: int, every: int = 10, label: str = "") -> ProgressPrinter:
    """Progress printer that writes to stderr so stdout stays parseable."""

    return ProgressPrinter(
        total, every, stream=lambda line: print(line, file=sys.stderr, flush=True), label=label
    )

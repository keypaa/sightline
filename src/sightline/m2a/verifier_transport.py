"""Call an opencode-hosted model, for claim verification and blind reading.

The verifiers are opencode providers (``hy4-flash-free`` and
``particle-ai/particle-1:free``), so the CLI is the common transport for both. Each
call is archived, and a provider error event is mapped onto the same exception types
the rest of the pipeline uses, so the rate-limit guard needs no special case.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sightline.m2a.transport import RateLimited, TransportError

RATE_LIMIT_STATUSES = frozenset({402, 429})


class OpencodeCaller:
    """Call one opencode model with an optional image, returning its text."""

    def __init__(self, model: str, *, timeout: int = 300) -> None:
        self.model = model
        self.timeout = timeout

    def __call__(self, image_path: Path | None, prompt: str) -> str:
        args = [
            "opencode",
            "run",
            "--archived",
            "--format",
            "json",
            "--model",
            self.model,
        ]
        if image_path is not None:
            args += ["--file", str(image_path)]
        args.append(prompt)

        result = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout)

        texts: list[str] = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "error":
                error = event.get("error") or {}
                status = error.get("status")
                message = str(error.get("message") or "opencode error")
                if status in RATE_LIMIT_STATUSES or error.get("type") == "provider.quota":
                    raise RateLimited(status=status, message=message)
                raise TransportError(message)
            if kind == "text":
                texts.append(event["part"]["text"])

        if result.returncode != 0:
            raise TransportError(f"opencode exited with code {result.returncode}")
        return "\n".join(texts).strip()

"""A local vLLM transport with the same signature as the CLI one.

This lets the annotate path stay identical between the CLI-era pilot and a run
against a locally served model: both are just ``Transport`` callables. Errors are
mapped onto the same exception types as the CLI transport so the existing pool
and retry guard need no changes.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from sightline.m2a.images import sniff_image
from sightline.m2a.transport import RateLimited, TransportError

# Mirrors the per-call budget of the CLI transport; a local server still deserves
# a ceiling so a hung request cannot stall the whole pool.
_TIMEOUT_S = 900


def _error_message(response: requests.Response) -> str:
    """Best-effort text from an error body, for the exception message."""

    try:
        body = response.json()
    except ValueError:
        return str(getattr(response, "text", ""))[:200] or f"HTTP {response.status_code}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or body.get("message") or body)[:200]
    return str(body)[:200]


class LocalVLLM:
    """Annotate images through a local OpenAI-compatible vLLM chat endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        chat_template_kwargs: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_template_kwargs = chat_template_kwargs
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(
        self, model: str, image_path: Path, prompt: str
    ) -> tuple[str, dict[str, Any] | None]:
        image_path = Path(image_path)
        fmt = sniff_image(image_path)
        if fmt is None:
            raise TransportError(f"not an image: {image_path}")
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt.lower()};base64,{b64}"},
            },
            {"type": "text", "text": prompt},
        ]
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=_TIMEOUT_S,
        )

        if response.status_code == 429:
            raise RateLimited(status=response.status_code, message=_error_message(response))
        if response.status_code != 200:
            raise TransportError(f"vLLM HTTP {response.status_code}: {_error_message(response)}")

        body = response.json()
        text = body["choices"][0]["message"]["content"]
        usage = body["usage"]
        tokens = {"input": usage["prompt_tokens"], "output": usage["completion_tokens"]}
        # An HTTP 200 with content=None happens when reasoning consumes the whole
        # budget. That is a failed annotation, not a blank answer to parse.
        if not text:
            raise TransportError("vLLM returned no content")
        return text, tokens

"""The local vLLM transport, exercised with a mocked HTTP response."""

from __future__ import annotations

from pathlib import Path

import pytest

from sightline.m2a.local_transport import LocalVLLM
from sightline.m2a.transport import RateLimited, TransportError


def test_local_transport_builds_an_image_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)

    captured: dict = {}

    class Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": '{"messages": []}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return Resp()

    monkeypatch.setattr("sightline.m2a.local_transport.requests.post", fake_post)
    client = LocalVLLM("Qwen/Qwen3.6-27B-FP8", base_url="http://localhost:8000/v1")
    text, tokens = client("Qwen/Qwen3.6-27B-FP8", img, "prompt")

    assert text == '{"messages": []}'
    assert tokens == {"input": 10, "output": 5}
    content = captured["payload"]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["type"] == "text"


def test_local_transport_maps_429_to_rate_limited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)

    class Resp:
        status_code = 429

        def json(self):
            return {"error": {"message": "Rate limit exceeded"}}

    monkeypatch.setattr("sightline.m2a.local_transport.requests.post", lambda *a, **k: Resp())
    with pytest.raises(RateLimited):
        LocalVLLM("m", base_url="http://x/v1")("m", img, "p")


def test_local_transport_maps_other_errors_to_transport_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(img)

    class Resp:
        status_code = 500

        def json(self):
            return {"error": {"message": "server error"}}

    monkeypatch.setattr("sightline.m2a.local_transport.requests.post", lambda *a, **k: Resp())
    with pytest.raises(TransportError):
        LocalVLLM("m", base_url="http://x/v1")("m", img, "p")

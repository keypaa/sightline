"""Modal conformance harness (test-only).

Runs the heavy K3/K2.6 MoonViT tower parity tests on a T4.

    python3 modal/conformance.py
"""

from __future__ import annotations

import modal

app = modal.App("sightline-conformance")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "safetensors==0.8.0",
        "huggingface-hub==1.25.1",
        "numpy==2.5.1",
        "pillow==12.3.0",
        "einops",
        "pytest==9.1.1",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("tests", "/root/tests")
    .add_local_file(
        "artifacts/k3_vision_f831ab66.safetensors",
        "/root/artifacts/k3_vision_f831ab66.safetensors",
    )
    .add_local_file(
        "artifacts/k26_vision_7eb5002f.safetensors",
        "/root/artifacts/k26_vision_7eb5002f.safetensors",
    )
)


@app.function(image=image, gpu="T4", cpu=4.0, memory=8192, timeout=1800)
def tower_conformance() -> dict:
    import sys

    sys.path.insert(0, "/root/src")
    import pytest

    code = pytest.main(
        [
            "-q",
            "-m",
            "slow",
            "/root/tests/conformance/test_vision_k3_tower.py",
            "/root/tests/conformance/test_vision_k26_tower.py",
        ]
    )
    return {"returncode": int(code)}


if __name__ == "__main__":
    import asyncio

    async def run() -> dict:
        async with app.run():
            return await tower_conformance.remote.aio()

    print(asyncio.run(run()))
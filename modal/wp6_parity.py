"""Modal parity harness for the K2.6 MoonViT tower.

This is a test/execution harness, not production code. `src/sightline` never
imports it. It compares the vendored official K2.6 tower against the community
standalone packaging on 20 images at several resolutions.

Run from the repository root:

    modal run modal/wp6_parity.py -- --resolutions 448 672 896 --count 20

Modal is authenticated with the `modal` CLI profile; no token is stored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("sightline-wp6")

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
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir("src", "/root/src")
    .add_local_file(
        "artifacts/k26_vision_7eb5002f.safetensors",
        "/root/k26_vision_7eb5002f.safetensors",
    )
)


def _make_image(resolution: int, index: int, rng):
    import numpy as np
    from PIL import Image

    if index % 3 == 0:
        array = np.full(
            (resolution, resolution, 3),
            (index * 10 % 256, 255 - index * 5 % 256, index * 20 % 256),
            dtype=np.uint8,
        )
    elif index % 3 == 1:
        ramp = np.linspace(0, 255, resolution, dtype=np.uint8)
        array = np.stack([np.tile(ramp, (resolution, 1))] * 3, axis=-1)
    else:
        array = rng.integers(0, 256, (resolution, resolution, 3), dtype=np.uint8)
    return Image.fromarray(array)


@app.function(
    image=image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=1800,
)
def parity(resolutions=(448, 672, 896), count: int = 20) -> dict:
    import resource
    import sys
    import time

    sys.path.insert(0, "/root/src")

    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    from sightline.vision_k25_encoder import K26VisionEncoder

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reference = AutoModel.from_pretrained(
        "immanuelpeter/MoonViT-K2.6",
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    reference.eval()
    reference_processor = AutoImageProcessor.from_pretrained(
        "immanuelpeter/MoonViT-K2.6",
        trust_remote_code=True,
    )

    ours = K26VisionEncoder(
        weights="/root/k26_vision_7eb5002f.safetensors",
        device=device,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    rng = np.random.default_rng(0)
    results = []
    for resolution in resolutions:
        for index in range(count):
            image = _make_image(resolution, index, rng)
            reference_inputs = reference_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                reference_merged = reference(
                    reference_inputs["pixel_values"].to(device),
                    reference_inputs["grid_thws"].to(device),
                )
            reference_features = torch.stack(reference_merged).flatten(2)
            our_features = ours.encode([image])

            difference = (reference_features.float() - our_features.float()).abs()
            relative = difference / (reference_features.float().abs() + 1e-12)
            cosine = (
                torch.nn.functional.cosine_similarity(
                    reference_features.float().flatten(1),
                    our_features.float().flatten(1),
                    dim=1,
                )
                .min()
                .item()
            )

            results.append(
                {
                    "resolution": resolution,
                    "index": index,
                    "tokens": int(reference_features.shape[1]),
                    "max_abs": difference.max().item(),
                    "max_rel": relative.max().item(),
                    "cosine": cosine,
                    "allclose": bool(
                        torch.allclose(
                            reference_features.float(),
                            our_features.float(),
                            atol=1e-5,
                            rtol=0,
                        )
                    ),
                }
            )

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1),
        "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if torch.cuda.is_available()
        else None,
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=int, nargs="+", default=[448, 672, 896])
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    async def run() -> dict:
        async with app.run():
            return await parity.remote.aio(resolutions=tuple(args.resolutions), count=args.count)

    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    Path("/tmp/opencode/wp6_parity.json").write_text(json.dumps(result, indent=2) + "\n")

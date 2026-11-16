"""Modal wrapper that runs the local runner on an A10G.

This is the test harness: it ships ``tools/run_m1.py`` and ``src`` unchanged and
executes the same entry point, so a Modal number and a local number come from one
code path rather than two.

    python3 modal/run_m1.py -- --seed 0 --max-steps 200 --log-every 1 --track-grad-norm

Everything after ``--`` is passed to ``tools/run_m1.py``.
"""

from __future__ import annotations

import json
import sys

import modal

app = modal.App("sightline-run")
volume = modal.Volume.from_name("sightline-m1", create_if_missing=True)

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
    .add_local_file("tools/run_m1.py", "/root/run_m1.py")
    .add_local_file(
        "artifacts/k3_vision_f831ab66.safetensors",
        "/root/artifacts/k3_vision_f831ab66.safetensors",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=21600,
    volumes={"/ckpt": volume},
)
def run(args: list[str], timeout_steps: int = 0) -> dict:
    import io
    import os
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import run_m1

    buffer = io.StringIO()
    sys.argv = ["run_m1.py", *args]
    with redirect_stdout(buffer):
        code = run_m1.main()
    return {"exit_code": code, "stdout": buffer.getvalue()}


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: python3 modal/run_m1.py -- <run_m1.py args>")

    with app.run():
        result = run.remote(argv)
    print(result["stdout"])
    print(json.dumps({"exit_code": result["exit_code"]}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

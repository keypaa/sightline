"""Modal wrapper that runs one of the local tools on an A10G.

The tool file is uploaded unchanged and its ``main()`` is called, so a Modal
number and a local number come from one code path rather than two.

    python3 modal/run_tool.py -- probe_features --per-task 512
    python3 modal/run_tool.py -- profile_m1 --n-examples 32 --micro-batch 4
    python3 modal/run_tool.py -- run_m1 --seed 0 --max-steps 50 --no-eval

Everything after ``--`` is ``<tool> [args...]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

TOOLS = {"probe_features", "profile_m1", "run_m1"}

app = modal.App("sightline-tool")
volume = modal.Volume.from_name("sightline-m1", create_if_missing=True)


def _build_image():
    """Base image plus any checkpoints that exist locally.

    The checkpoints are 302 MB each and gitignored, so requiring them would break
    the other tools this wrapper runs (probe_features, profile_m1) on any clone
    without them, and cost ~906 MB per image for nothing.
    """

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
        .add_local_file("tools/run_m1.py", "/root/local_runner.py")
        .add_local_file(
            "artifacts/k3_vision_f831ab66.safetensors",
            "/root/artifacts/k3_vision_f831ab66.safetensors",
        )
    )
    for seed in (0, 1, 2):
        checkpoint = Path(f"artifacts/m1b/seed{seed}/projector.pt")
        if checkpoint.exists():
            image = image.add_local_file(
                str(checkpoint), f"/root/artifacts/m1b/seed{seed}/projector.pt"
            )
    return image


image = _build_image()


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=21600,
    volumes={"/ckpt": volume},
)
def run(tool: str, source: str, args: list[str]) -> dict:
    import io
    import os
    import sys
    import types
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/src")
    os.chdir("/root")

    module = types.ModuleType(tool)
    module.__file__ = f"/root/{tool}.py"
    sys.modules[tool] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)

    buffer = io.StringIO()
    sys.argv = [f"{tool}.py", *args]
    with redirect_stdout(buffer):
        code = module.main()
    return {"exit_code": code, "stdout": buffer.getvalue()}


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: python3 modal/run_tool.py -- <tool> [args...]")

    tool, args = argv[0], argv[1:]
    if tool not in TOOLS:
        raise SystemExit(f"unknown tool {tool!r}; choose from {sorted(TOOLS)}")

    source = (Path(__file__).resolve().parents[1] / "tools" / f"{tool}.py").read_text()
    with app.run():
        result = run.remote(tool, source, args)
    print(result["stdout"])
    print(json.dumps({"exit_code": result["exit_code"]}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

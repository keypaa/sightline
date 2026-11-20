"""Throwaway: re-score saved checkpoints under both parsers (use only, no tests).

Uploads the three local checkpoints into the image, then runs the real runner
once per (seed, parser) and prints each summary. Results are printed as they
land so a reclaimed container still leaves the earlier ones in the log.

    python3 modal/eval_recheck.py --seeds 0 --parsers exact first-line
"""

from __future__ import annotations

import json
import sys

import modal

app = modal.App("sightline-recheck")
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
    .add_local_file("tools/run_m1.py", "/root/local_runner.py")
    .add_local_file(
        "artifacts/k3_vision_f831ab66.safetensors",
        "/root/artifacts/k3_vision_f831ab66.safetensors",
    )
    .add_local_file(
        "artifacts/m1b/seed0/projector.pt",
        "/root/artifacts/m1b/seed0/projector.pt",
    )
    .add_local_file(
        "artifacts/m1b/seed1/projector.pt",
        "/root/artifacts/m1b/seed1/projector.pt",
    )
    .add_local_file(
        "artifacts/m1b/seed2/projector.pt",
        "/root/artifacts/m1b/seed2/projector.pt",
    )
)


def _last_json(text: str) -> dict:
    """The runner prints progress and a scale line before the summary JSON."""

    index = text.rfind("\n{")
    if index < 0:
        raise ValueError(f"no JSON object in output:\n{text[-500:]}")
    return json.loads(text[index + 1 :])


@app.function(image=image, gpu="A10G", cpu=4.0, memory=16384, timeout=14400)
def run(seeds: list[int], parsers: list[str], split: str = "test") -> dict:
    import io
    import os
    import sys
    from contextlib import redirect_stdout

    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root")
    os.chdir("/root")

    import local_runner

    out = {}
    for seed in seeds:
        for parser in parsers:
            buffer = io.StringIO()
            sys.argv = [
                "local_runner.py",
                "--seed", str(seed),
                "--eval-only",
                "--eval-split", split,
                "--parser", parser,
            ]
            try:
                with redirect_stdout(buffer):
                    code = local_runner.main()
                text = buffer.getvalue()
                out[f"seed{seed}-{parser}"] = {
                    "exit_code": code,
                    "summary": _last_json(text),
                }
            except Exception as exc:  # keep going; report the failure
                out[f"seed{seed}-{parser}"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "tail": buffer.getvalue()[-800:],
                }
            print(f"DONE seed{seed} {parser}", flush=True)
    return out


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]

    seeds = [0]
    parsers = ["exact", "first-line"]
    split = "test"
    if "--seeds" in argv:
        seeds = [int(x) for x in argv[argv.index("--seeds") + 1].split(",")]
    if "--parsers" in argv:
        parsers = argv[argv.index("--parsers") + 1].split(",")
    if "--split" in argv:
        split = argv[argv.index("--split") + 1]

    with app.run():
        result = run.remote(seeds, parsers, split)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Run records.

Every run writes a ``run.json`` and a ``metrics.jsonl``. A result without a run
record does not exist, so the record is written before training starts and the
metrics file is appended to as the run progresses.

The dry run writes a complete record without loading a model: checkpoint-derived
fields are present and explicitly ``None`` rather than omitted.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sightline.config import Config, resolve_config

SCHEMA_VERSION = "0.1"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def find_repo(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def git_state(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"commit": commit, "branch": branch, "dirty": bool(status.strip())}


def library_versions() -> dict[str, str]:
    import torch
    import transformers

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "missing"

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": version("tokenizers"),
        "safetensors": version("safetensors"),
        "numpy": version("numpy"),
        "huggingface_hub": version("huggingface-hub"),
    }


def cuda_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "arch_list": torch.cuda.get_arch_list(),
    }
    if info["cuda_available"] and info["device_count"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def make_run_id(repo: Path) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{_git(repo, 'rev-parse', 'HEAD')[:7]}"


class RunRecord:
    def __init__(
        self,
        root: str | Path,
        *,
        dry_run: bool,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        repo: str | Path | None = None,
        runtime: dict[str, Any] | None = None,
        chat_template: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.dry_run = dry_run
        self.config = config if config is not None else resolve_config(Config())
        self.runtime = runtime
        self.chat_template = chat_template
        self.repo = find_repo(Path(repo) if repo is not None else Path.cwd())
        self.run_id = run_id or make_run_id(self.repo)
        self.run_path = self.root / "run.json"
        self.metrics_path = self.root / "metrics.jsonl"

    def payload(self) -> dict[str, Any]:
        reader = self.config["reader"]
        encoder = self.config["encoder"]
        training = self.config["training"]
        config = dict(self.config)
        if self.runtime is not None:
            config["runtime"] = self.runtime
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "dry_run": self.dry_run,
            "code": git_state(self.repo),
            "config": config,
            "runtime": config["runtime"],
            "models": {
                "reader": {
                    "model_id": reader["model_id"],
                    "revision": reader["revision"],
                    "tokenizer_revision": reader["revision"],
                    "dtype": reader["dtype"],
                },
                "encoder": {
                    "model_id": encoder["model_id"],
                    "revision": encoder["revision"],
                    "dtype": encoder["dtype"],
                },
            },
            "chat_template": self.chat_template
            if self.chat_template is not None
            else {
                "tokenizer_sha256": None,
                "rendered_sha256": None,
                "enable_thinking": False,
            },
            "seeds": list(training["seeds"]),
            "environment": {**library_versions(), **cuda_info()},
            "attention_backend": reader["attn_backend"],
            "dtypes": self.config["precision"],
            "metrics_path": self.metrics_path.name,
            "gpu_hours": 0.0,
            "outcome": "dry run: no model loaded" if self.dry_run else "pending",
        }

    def write(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.metrics_path.touch(exist_ok=True)
        self.run_path.write_text(json.dumps(self.payload(), indent=2, sort_keys=True) + "\n")
        return self.run_path

    def append_metric(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a Sightline run record.")
    parser.add_argument(
        "--dry-run", action="store_true", help="write a record without loading a model"
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/_dryrun"))
    parser.add_argument("--config", type=Path, default=None, help="JSON config to resolve")
    args = parser.parse_args(argv)

    repo = find_repo(Path.cwd())
    run_id = make_run_id(repo)
    root = args.out / "runs" / run_id
    config = None
    if args.config is not None:
        config = resolve_config(Config.from_dict(json.loads(args.config.read_text())))
    record = RunRecord(root, dry_run=True, config=config, run_id=run_id, repo=repo)
    path = record.write()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

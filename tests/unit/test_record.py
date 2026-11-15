from __future__ import annotations

import json
from pathlib import Path

from sightline.record import RunRecord, git_state, library_versions


def test_run_record_has_required_sections(tmp_path: Path) -> None:
    rec = RunRecord(root=tmp_path, dry_run=True)
    path = rec.write()
    data = json.loads(Path(path).read_text())
    for key in (
        "schema_version",
        "run_id",
        "created_utc",
        "dry_run",
        "code",
        "config",
        "runtime",
        "models",
        "chat_template",
        "seeds",
        "environment",
        "attention_backend",
        "dtypes",
        "metrics_path",
        "gpu_hours",
        "outcome",
    ):
        assert key in data, key
    assert data["dry_run"] is True
    assert data["code"]["commit"]
    assert data["seeds"] == [0, 1, 2]
    assert data["attention_backend"] == "sdpa"


def test_metrics_jsonl_roundtrip(tmp_path: Path) -> None:
    rec = RunRecord(root=tmp_path, dry_run=True)
    rec.write()
    rec.append_metric({"step": 0, "loss": 1.0})
    rec.append_metric({"step": 1, "loss": 0.5})
    lines = rec.metrics_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 0
    assert json.loads(lines[1])["loss"] == 0.5


def test_runtime_and_chat_template_overrides(tmp_path: Path) -> None:
    rec = RunRecord(
        root=tmp_path,
        dry_run=True,
        runtime={"reader_hidden_size": 2048, "projector_hidden_width": 2048},
        chat_template={
            "tokenizer_sha256": "t",
            "rendered_sha256": "r",
            "enable_thinking": False,
        },
    )
    data = json.loads(rec.write().read_text())
    assert data["runtime"]["reader_hidden_size"] == 2048
    assert data["config"]["runtime"]["reader_hidden_size"] == 2048
    assert data["chat_template"]["rendered_sha256"] == "r"


def test_git_state_has_commit_branch_and_dirty() -> None:
    root = Path(__file__).resolve().parents[2]
    state = git_state(root)
    assert len(state["commit"]) == 40
    assert state["branch"]
    assert isinstance(state["dirty"], bool)


def test_library_versions_has_core_packages() -> None:
    versions = library_versions()
    assert versions["python"]
    assert versions["torch"]
    assert versions["transformers"]

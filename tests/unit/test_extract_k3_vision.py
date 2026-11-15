"""The extraction tool must create its own output directory.

`artifacts/` is gitignored, so a fresh clone has no `artifacts/` and the
extractor used to die with FileNotFoundError after downloading the shards.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "extract_k3_vision.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("extract_k3_vision", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_safetensors_creates_parent_directories(tmp_path: Path) -> None:
    tool = load_tool()
    tensors = {
        "vision_tower.a": {"dtype": "BF16", "shape": [2], "raw": b"\x00\x01\x02\x03"},
    }
    out = tmp_path / "artifacts" / "nested" / "k3_vision.safetensors"

    tool.write_safetensors(out, tensors, {"format": "pt"})

    assert out.exists()
    assert tool.load_safetensors_raw(out)["vision_tower.a"]["raw"] == b"\x00\x01\x02\x03"
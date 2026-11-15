from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "manifests" / "k3_vision_f831ab66.manifest.json"
WEIGHTS = REPO_ROOT / "artifacts" / "k3_vision_f831ab66.safetensors"

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="extracted tower not present; run WP5 extraction"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_matches_extracted_file() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert sha256_file(WEIGHTS) == manifest["file_sha256"]

    with safe_open(WEIGHTS, framework="pt") as handle:
        keys = set(handle.keys())
        assert keys == set(manifest["tensors"])
        for name, meta in manifest["tensors"].items():
            tensor = handle.get_tensor(name)
            assert list(tensor.shape) == meta["shape"], name
            assert tensor.dtype == DTYPES[meta["dtype"]], name
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            assert hashlib.sha256(raw).hexdigest() == meta["sha256"], name


def test_tensor_counts_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    tower = [n for n in manifest["tensors"] if n.startswith("vision_tower.")]
    projector = [n for n in manifest["tensors"] if n.startswith("mm_projector.")]
    assert len(tower) == 165
    assert len(projector) == 3
    assert manifest["parameters"]["vision_tower."] == 401214464
    assert manifest["parameters"]["mm_projector."] == 46144512
    assert manifest["revision"] == "f831ab66814297da540d832a5235f8e904f29d06"
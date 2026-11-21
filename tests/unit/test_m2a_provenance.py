from __future__ import annotations

from sightline.m2a.annotate import prompt_hash
from sightline.m2a.provenance import (
    annotator_fields,
    chat_template_hash,
    row_provenance,
)


def test_annotator_fields_pin_every_reproducibility_key():
    fields = annotator_fields(
        model="Qwen/Qwen3.6-27B-FP8",
        revision="abc123",
        quantization="fp8",
        engine="vllm",
        engine_version="0.17.0",
        prompt_sha256="b6d8d625b4ffc424",
        chat_template_sha256="deadbeefdeadbeef",
        seed=0,
        order_seed=0,
        temperature=0.0,
    )
    assert fields["revision"] == "abc123"
    assert fields["quantization"] == "fp8"
    assert fields["engine"] == "vllm"
    assert fields["engine_version"] == "0.17.0"
    assert fields["seed"] == 0
    assert fields["temperature"] == 0.0


def test_annotator_fields_has_exactly_the_pinned_keys():
    fields = annotator_fields(
        model="m",
        revision="r",
        quantization="q",
        engine="e",
        engine_version="v",
        prompt_sha256="p",
        chat_template_sha256="c",
        seed=0,
        order_seed=1,
        temperature=0.0,
    )
    assert set(fields) == {
        "model",
        "revision",
        "quantization",
        "engine",
        "engine_version",
        "prompt_sha256",
        "chat_template_sha256",
        "seed",
        "order_seed",
        "temperature",
    }


def test_chat_template_hash_uses_the_prompt_hash_convention():
    template = "a chat template"
    assert chat_template_hash(template) == prompt_hash(template)
    assert len(chat_template_hash(template)) == 16


def test_row_provenance_keeps_source_licence_and_checksum():
    row = row_provenance(
        {
            "source": "coco",
            "license_name": "CC BY 2.0",
            "sha256": "aa",
            "flickr_url": "http://f",
            "coco_url": "http://c",
            "url": "http://u",
            "license_id": 1,
            "license_url": "http://l",
        },
        image_sha256="bb",
        annotator={"model": "m"},
    )
    assert row["source"] == "coco"
    assert row["license_name"] == "CC BY 2.0"
    assert row["flickr_url"] == "http://f"
    assert row["sha256"] == "aa"
    assert row["image_sha256"] == "bb"


def test_row_provenance_annotator_wins_over_a_record_field():
    row = row_provenance(
        {"source": "coco", "model": "stale"},
        image_sha256="bb",
        annotator={"model": "current"},
    )
    assert row["model"] == "current"


def test_no_kilo_rows_remain_in_the_20k_tree():
    import json
    from pathlib import Path

    import pytest

    root = Path("experiments/m2a_iter20k")
    if not root.exists():
        pytest.skip("20k tree is not present in this checkout")
    assert (root / "fetched.jsonl").exists(), "fetched index must survive the rebuild"

    rows = []
    for path in root.glob("*conversations*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    assert all("kilo" not in r.get("model", "") for r in rows)

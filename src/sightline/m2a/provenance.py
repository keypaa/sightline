"""Build every provenance field from one module.

Each annotation row is assembled here so a call site cannot forget a field. The
annotator keys are the ones that pin reproducibility: a row that looks the same
but was produced by a different model, engine or template must be detectable.
"""

from __future__ import annotations

from typing import Any

from sightline.m2a.annotate import prompt_hash

_RECORD_FIELDS = (
    "source",
    "license_id",
    "license_name",
    "license_url",
    "flickr_url",
    "coco_url",
    "url",
    "sha256",
)


def chat_template_hash(template: str) -> str:
    """Hash a chat template with the same convention as the prompt hash."""

    return prompt_hash(template)


def annotator_fields(
    *,
    model: str,
    revision: str,
    quantization: str,
    engine: str,
    engine_version: str,
    prompt_sha256: str,
    chat_template_sha256: str,
    seed: int,
    order_seed: int,
    temperature: float,
) -> dict[str, Any]:
    """Pin the full identity of the annotator that produced a row.

    ``model`` and ``prompt_sha256`` are kept for backward compatibility with
    rows written before the other keys were recorded.
    """

    return {
        "model": model,
        "revision": revision,
        "quantization": quantization,
        "engine": engine,
        "engine_version": engine_version,
        "prompt_sha256": prompt_sha256,
        "chat_template_sha256": chat_template_sha256,
        "seed": seed,
        "order_seed": order_seed,
        "temperature": temperature,
    }


def row_provenance(
    record: dict[str, Any], image_sha256: str, annotator: dict[str, Any]
) -> dict[str, Any]:
    """Merge source, licence and checksum fields with the annotator identity.

    The annotator fields are applied last so a record cannot shadow them: the
    identity of the writer must survive whatever the fetch stage recorded.
    """

    row = {field: record.get(field) for field in _RECORD_FIELDS}
    row["image_sha256"] = image_sha256
    row.update(annotator)
    return row

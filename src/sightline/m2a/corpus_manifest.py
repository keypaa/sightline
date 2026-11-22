"""One manifest over every corpus layer, with the gates computed.

The corpus manifest (``manifest.py``) describes the conversations for the
trainer. This one describes the *build*: what was fetched, how many claims came
out of it, what the verifiers said, and whether the frozen gates passed. Reading
both from the same rows keeps the rates here identical to the ones the verify CLI
prints, because the summaries are reused rather than reimplemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sightline.m2a.ids import now_iso
from sightline.m2a.verify import (
    CONTRADICTION_CEILING,
    HUMAN_AGREEMENT_FLOOR,
    PLANTED_REJECTION_FLOOR,
    Claim,
    ClaimResult,
    Verdict,
    read_jsonl,
    summarise,
    summarise_blind,
    summarise_planted,
)
from sightline.record import git_state

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LAYERS = (
    "fetched.jsonl",
    "conversations.jsonl",
    "claims.jsonl",
    "planted.jsonl",
    "verdicts.jsonl",
    "blind.jsonl",
    "decisions.jsonl",
)


def _claim_from_row(row: dict[str, Any]) -> Claim:
    return Claim(
        image_id=row.get("image_id", row.get("id")),
        turn=row.get("turn", 0),
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        claim=row.get("claim", ""),
        claim_type=row.get("claim_type", "existence"),
        planted=bool(row.get("planted")),
        image_path=row.get("image_path"),
        claim_id=row.get("claim_id", ""),
    )


def _results(claims: list[dict], verdicts: list[dict]) -> list[ClaimResult]:
    """Rebuild ClaimResult objects from flat verdict rows."""

    by_claim: dict[str, list[Verdict]] = {}
    for row in verdicts:
        by_claim.setdefault(row.get("claim_id"), []).append(
            Verdict(
                model=row.get("verifier_model", ""),
                verdict=row.get("verdict", "can't-tell"),
                evidence=row.get("evidence", ""),
                error=row.get("error"),
            )
        )
    return [
        ClaimResult(claim=_claim_from_row(row), verdicts=by_claim.get(row.get("claim_id"), []))
        for row in claims
    ]


def build(root: Path) -> dict[str, Any]:
    """Assemble the manifest from whatever layers exist under ``root``."""

    root = Path(root)
    layers = {name: read_jsonl(root / name) for name in LAYERS}
    fetched = layers["fetched.jsonl"]
    conversations = [r for r in layers["conversations.jsonl"] if r.get("parsed") is not False]
    # Planted claims live in their own file; the gate needs them alongside the real
    # ones, and their `planted` flag keeps them out of the contradiction rate.
    claims = layers["claims.jsonl"] + layers["planted.jsonl"]
    results = _results(claims, layers["verdicts.jsonl"])

    real = [r for r in results if not r.claim.planted]
    planted = [r for r in results if r.claim.planted]
    judged = [r for r in results if r.verdicts and all(not v.error for v in r.verdicts)]

    summary = summarise(real)
    planted_summary = summarise_planted(planted)
    blind_summary = summarise_blind(layers["blind.jsonl"])

    decisions = layers["decisions.jsonl"]
    right = sum(1 for d in decisions if d.get("decision") == "verifiers_right")

    fetched_by_id = {row.get("id"): row.get("sha256") for row in fetched}
    sha_mismatch = sum(
        1
        for row in conversations
        if fetched_by_id.get(row.get("id")) not in (None, row.get("image_sha256"))
    )

    annotators = sorted(
        {
            f"{row.get('model')}/{row.get('revision')}/{row.get('quantization')}"
            for row in conversations
            if row.get("model")
        }
    )
    verifier_models = {row.get("verifier_model") for row in layers["verdicts.jsonl"]}
    annotator_models = {row.get("model") for row in conversations if row.get("model")}

    licences: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in conversations:
        key = row.get("license_name") or "missing"
        licences[key] = licences.get(key, 0) + 1
        source = row.get("source") or "unknown"
        sources[source] = sources.get(source, 0) + 1

    order_seeds = {
        row.get("order_seed") for row in conversations if row.get("order_seed") is not None
    }
    run_ids = {row.get("run_id") for row in conversations if row.get("run_id")}

    return {
        "built_at": now_iso(),
        # Which code produced this manifest. A number that cannot be tied to a
        # commit cannot be reproduced.
        "git": git_state(PROJECT_ROOT),
        "n_images": len(fetched),
        "n_conversations": len(conversations),
        "n_claims": len(claims),
        "n_planted": len(planted),
        "n_judged": len(judged),
        "n_errored": summary["n_errored"],
        "n_decisions": len(decisions),
        "source_count": len(sources),
        "sources": dict(sorted(sources.items())),
        "annotators": annotators,
        "licences": dict(sorted(licences.items())),
        "order_seed": next(iter(order_seeds)) if len(order_seeds) == 1 else None,
        "n_sha_mismatch": sha_mismatch,
        "annotator_verifier_distinct": annotator_models.isdisjoint(verifier_models),
        "run_ids": sorted(run_ids),
        "contradiction_rate": summary["contradiction_rate"],
        "contradiction_rate_upper_95": summary["contradiction_rate_upper_95"],
        "planted_rejection_rate": planted_summary["rejection_rate"],
        "blind_rate": blind_summary["blind_rate"],
        "human_agreement": (right / len(decisions)) if decisions else None,
        "thresholds": {
            "contradiction_ceiling": CONTRADICTION_CEILING,
            "planted_rejection_floor": PLANTED_REJECTION_FLOOR,
            "human_agreement_floor": HUMAN_AGREEMENT_FLOOR,
        },
        "passes_contradiction": (
            summary["contradiction_rate_upper_95"] is not None
            and summary["contradiction_rate_upper_95"] <= CONTRADICTION_CEILING
        ),
        "passes_planted": planted_summary["passes"],
        "passes_human": (len(decisions) > 0 and (right / len(decisions)) >= HUMAN_AGREEMENT_FLOOR),
    }

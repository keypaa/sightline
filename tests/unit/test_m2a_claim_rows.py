"""The claim and verdict row shapes are the join contract; these pin them."""

from __future__ import annotations

import json
from pathlib import Path

from sightline.m2a.verify import (
    Claim,
    ClaimResult,
    Verdict,
    _claim_from_row,
    claim_to_row,
    judge_verdict_prompt_sha256,
    summarise_planted,
    verdict_row,
)

CONVERSATIONS = [
    {
        "id": 5,
        "file_name": "a.jpg",
        "run_id": "run-1",
        "messages": [
            {"role": "user", "content": "What colour is the car?"},
            {"role": "assistant", "content": "Red."},
        ],
    }
]


def test_claim_to_row_writes_the_spec_schema():
    claim = Claim(
        image_id=5,
        turn=1,
        question="q",
        answer="Red.",
        claim="Red (car)",
        claim_type="colour",
        image_path=None,
        claim_id="abc123",
    )

    row = claim_to_row(claim, source_run="run-1", extracted_at="2026-11-21T00:00:00+00:00")

    assert row["claim_id"] == "abc123"
    assert row["image_id"] == 5
    assert row["source_run"] == "run-1"
    assert row["extracted_at"].startswith("2026-11-21")
    # Verdict-shaped fields belong on verdict rows, not claims.
    assert "contradicted" not in row
    assert "verdicts" not in row


def test_claim_survives_a_round_trip():
    claim = Claim(
        image_id=5,
        turn=1,
        question="q",
        answer="Red.",
        claim="Red (car)",
        claim_type="colour",
        image_path=None,
        claim_id="abc123",
    )
    row = claim_to_row(claim, source_run="run-1", extracted_at="2026-11-21T00:00:00+00:00")

    restored = _claim_from_row(row)

    assert restored.claim_id == "abc123"
    assert restored.claim == claim.claim
    assert restored.claim_type == claim.claim_type


def test_prompt_hash_tracks_the_instruction_not_the_claim(monkeypatch):
    """The hash pins the judging instructions, so a prompt change is detectable."""

    import sightline.m2a.verify as verify

    first = judge_verdict_prompt_sha256()
    assert first == judge_verdict_prompt_sha256()

    monkeypatch.setattr(verify, "JUDGE_INSTRUCTION", "different instructions")
    assert judge_verdict_prompt_sha256() != first


def test_extract_cli_writes_claim_ids(monkeypatch, tmp_path: Path):
    import sightline.m2a.verify as verify
    from tools import m2a_annotator_pilot  # noqa: F401  (ensures tools import path)

    conversations = tmp_path / "pilot.jsonl"
    conversations.write_text("\n".join(json.dumps(r) for r in CONVERSATIONS) + "\n")
    out = tmp_path / "claims.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "verify",
            "extract",
            "--conversations",
            str(conversations),
            "--out",
            str(out),
            "--images-dir",
            str(tmp_path),
        ],
    )
    assert verify.main() == 0

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["claim_id"]
    assert rows[0]["source_run"] == "run-1"
    assert "verdicts" not in rows[0]


def test_plant_cli_preserves_claim_ids(monkeypatch, tmp_path: Path):
    import sightline.m2a.verify as verify

    claims = tmp_path / "claims.jsonl"
    claims.write_text(
        json.dumps(
            {
                "claim_id": "abc123",
                "image_id": 5,
                "turn": 1,
                "question": "q",
                "answer": "Red.",
                "claim": "Red (car)",
                "claim_type": "colour",
            }
        )
        + "\n"
    )
    out = tmp_path / "planted.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        ["verify", "plant", "--claims", str(claims), "--out", str(out)],
    )
    assert verify.main() == 0

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert rows
    assert all(r["claim_id"] for r in rows)
    assert all(r["planted"] for r in rows)


def test_verdict_row_is_self_describing():
    """A verdict row must name its claim and whether it was planted."""

    claim = Claim(
        image_id=5,
        turn=1,
        question="What colour?",
        answer="Red.",
        claim="Red (car)",
        claim_type="colour",
        claim_id="abc",
        planted=True,
    )
    row = verdict_row(
        claim, Verdict("v", "false", "no"), prompt_sha256="p", judged_at="2026-11-21T00:00:00+00:00"
    )

    assert row["planted"] is True
    assert row["claim"] == "Red (car)"
    assert row["claim_type"] == "colour"
    assert row["question"] == "What colour?"


def test_planted_rejection_excludes_errored_rows():

    rows = [
        ClaimResult(
            Claim(0, 1, "q", "a", "c", "colour", claim_id="a", planted=True),
            verdicts=[Verdict("v", "false")],
        ),
        ClaimResult(
            Claim(1, 1, "q", "a", "c", "colour", claim_id="b", planted=True),
            verdicts=[Verdict("v", "can't-tell", error="429")],
        ),
    ]

    summary = summarise_planted(rows)

    assert summary["n_planted"] == 1
    assert summary["rejection_rate"] == 1.0
    assert summary["passes"] is True

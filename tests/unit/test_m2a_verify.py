"""Tests for claim extraction, planted errors and the frozen thresholds."""

from __future__ import annotations

from pathlib import Path

import pytest

from sightline.m2a.ids import claim_id
from sightline.m2a.verify import (
    CLAIM_TYPES,
    CONTRADICTION_CEILING,
    HUMAN_AGREEMENT_FLOOR,
    PLANTED_REJECTION_FLOOR,
    VERIFIERS,
    Claim,
    ClaimResult,
    Verdict,
    classify_claim,
    extract_claims,
    judge_verdict,
    plant_errors,
    summarise,
    summarise_blind,
    summarise_planted,
    wilson_upper,
)


def conversation(turns: list[tuple[str, str]], image_id: int = 0) -> dict:
    messages: list[dict] = []
    for question, answer in turns:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    return {"id": image_id, "messages": messages}


def test_extract_one_claim_per_qa_turn():
    rows = [
        conversation(
            [
                ("What colour is the car?", "Red."),
                ("How many cars are there?", "Three."),
            ]
        )
    ]

    claims = extract_claims(rows)

    assert len(claims) == 2
    assert claims[0].claim_type == "colour"
    assert claims[1].claim_type == "count"


def test_extract_skips_a_turn_without_an_answer():
    rows = [{"id": 0, "messages": [{"role": "user", "content": "What colour?"}]}]

    assert extract_claims(rows) == []


def test_claim_names_its_referent():
    rows = [conversation([("What colour is the red motorbike?", "Red.")])]

    claim = extract_claims(rows)[0]

    # The answer is kept verbatim and the referent appended, so a verifier checks
    # the claim without guessing which object is meant.
    assert "red motorbike" in claim.claim.lower()
    assert claim.claim.startswith("Red")
    assert claim.claim.endswith("(red motorbike)")


def test_classify_covers_every_documented_type():
    for question, answer, expected in [
        ("Is there a dog?", "Yes.", "existence"),
        ("How many people?", "Two.", "count"),
        ("What colour is it?", "Blue.", "colour"),
        ("Where is the cat?", "Left.", "spatial"),
        ("What does the sign say?", "Stop.", "text"),
        ("What is the man doing?", "Running.", "action"),
    ]:
        assert classify_claim(question, answer) == expected


def test_classify_returns_only_documented_types():
    for question, answer in [("What is this?", "A thing."), ("Describe it.", "A scene.")]:
        assert classify_claim(question, answer) in CLAIM_TYPES


def test_planted_errors_alter_the_claim_and_record_the_original():
    claims = [Claim(0, 1, "What colour is it?", "Red.", "Red (car)", "colour")]

    planted = plant_errors(claims, seed=0)

    assert planted
    assert planted[0].planted is True
    assert planted[0].original == "Red (car)"
    assert planted[0].claim != "Red (car)"


def test_planted_errors_are_reproducible():
    claims = [
        Claim(0, 1, "What colour is it?", "Red.", "Red (car)", "colour"),
        Claim(1, 1, "How many?", "Three.", "Three (dogs)", "count"),
    ]

    assert plant_errors(claims, seed=0) == plant_errors(claims, seed=0)


def test_judge_verdict_reads_each_of_the_three_answers():
    assert judge_verdict("true\nbecause it is red")[0] == "true"
    assert judge_verdict("false\nno car is visible")[0] == "false"
    assert judge_verdict("can't-tell\ntoo blurry")[0] == "can't-tell"
    assert judge_verdict("")[0] == "can't-tell"


def test_judge_verdict_keeps_the_evidence_line():
    _, evidence = judge_verdict("false\nThe image shows a bicycle, not a car.")

    assert "bicycle" in evidence


def test_contradiction_requires_a_false_verdict():
    claim = Claim(0, 1, "q", "a", "Red (car)", "colour")
    result = ClaimResult(
        claim=claim,
        verdicts=[Verdict("v1", "true"), Verdict("v2", "false")],
    )

    assert result.contradicted is True
    assert result.all_sure is True


def test_a_verifier_error_is_not_a_contradiction():
    claim = Claim(0, 1, "q", "a", "Red (car)", "colour")
    result = ClaimResult(claim=claim, verdicts=[Verdict("v1", "false", error="timeout")])

    assert result.contradicted is False
    assert result.all_sure is False


def test_cant_tell_is_not_a_contradiction():
    claim = Claim(0, 1, "q", "a", "Red (car)", "colour")
    result = ClaimResult(claim=claim, verdicts=[Verdict("v1", "can't-tell")])

    assert result.contradicted is False


def test_summary_reports_rate_and_upper_bound():
    results = [
        ClaimResult(
            Claim(0, 1, "q", "a", "c", "colour"),
            verdicts=[Verdict("v1", "false")],
        )
    ] + [
        ClaimResult(Claim(i, 1, "q", "a", "c", "colour"), verdicts=[Verdict("v1", "true")])
        for i in range(1, 10)
    ]

    summary = summarise(results)

    assert summary["n_claims"] == 10
    assert summary["contradicted"] == 1
    assert summary["contradiction_rate"] == pytest.approx(0.1)
    assert summary["contradiction_rate_upper_95"] is not None
    assert summary["contradiction_rate_upper_95"] > summary["contradiction_rate"]


def test_summary_breaks_down_by_claim_type():
    results = [
        ClaimResult(Claim(0, 1, "q", "a", "c", "count"), verdicts=[Verdict("v1", "true")]),
        ClaimResult(Claim(1, 1, "q", "a", "c", "colour"), verdicts=[Verdict("v1", "false")]),
    ]

    summary = summarise(results)

    assert summary["by_type"]["count"]["n"] == 1
    assert summary["by_type"]["colour"]["contradicted"] == 1


def test_planted_rejection_is_measured_against_the_frozen_floor():
    caught = [
        ClaimResult(Claim(i, 1, "q", "a", "c", "colour"), verdicts=[Verdict("v1", "false")])
        for i in range(9)
    ]
    missed = [ClaimResult(Claim(99, 1, "q", "a", "c", "colour"), verdicts=[Verdict("v1", "true")])]

    summary = summarise_planted(caught + missed)

    assert summary["n_planted"] == 10
    assert summary["rejection_rate"] == pytest.approx(0.9)
    assert summary["passes"] is True
    assert summary["floor"] == PLANTED_REJECTION_FLOOR


def test_frozen_thresholds_have_their_decided_values():
    assert CONTRADICTION_CEILING == 0.15
    assert PLANTED_REJECTION_FLOOR == 0.90
    assert HUMAN_AGREEMENT_FLOOR == 0.85


def test_two_verifiers_are_configured_and_neither_is_the_annotator():
    assert len(VERIFIERS) == 2
    assert all("qwen3.6" not in model.lower() for model in VERIFIERS)


def test_blind_summary_reports_a_rate_and_warns_it_is_unthresholded():
    rows = [
        {"blind_answer": "red", "blind_matches": True},
        {"blind_answer": "blue", "blind_matches": False},
    ]

    summary = summarise_blind(rows)

    assert summary["n_questions"] == 2
    assert summary["blind_rate"] == pytest.approx(0.5)
    assert "[U]" in summary["note"]


def test_wilson_upper_is_wider_than_the_point_estimate():
    upper = wilson_upper(45, 50)

    assert upper is not None
    assert upper > 0.9
    assert upper < 1.0


def test_count_referent_is_the_thing_counted_not_the_sentence_tail():
    from sightline.m2a.verify import make_declarative

    assert make_declarative("How many tiers does the cake have?", "Five.") == (
        "Five (tiers of the cake)"
    )
    # The preposition from the question is kept, so the referent reads naturally.
    assert make_declarative("How many wheels are visible on the skateboard?", "Four.") == (
        "Four (wheels on the skateboard)"
    )
    assert make_declarative("How many people are there?", "Two.") == "Two (people)"


def test_colour_referent_is_unchanged_by_the_count_fix():
    from sightline.m2a.verify import make_declarative

    assert make_declarative("What color is the woman's skirt?", "Red.") == ("Red (woman's skirt)")


def test_claim_id_is_set_and_stable_across_extracts():
    rows = [
        {
            "id": 5,
            "file_name": "a.jpg",
            "messages": [
                {"role": "user", "content": "What colour is the car?"},
                {"role": "assistant", "content": "Red."},
            ],
        }
    ]
    first = extract_claims(rows)
    second = extract_claims(rows)
    assert first[0].claim_id
    assert first[0].claim_id == second[0].claim_id
    assert first[0].claim_id == claim_id(5, 1, "Red (car)")


def test_a_missing_image_does_not_drop_the_claim(tmp_path: Path):
    rows = [
        {
            "id": 5,
            "file_name": "gone.jpg",
            "messages": [
                {"role": "user", "content": "What colour is the car?"},
                {"role": "assistant", "content": "Red."},
            ],
        }
    ]
    claim = extract_claims(rows, images_dir=tmp_path)[0]
    assert claim.image_path is None
    assert claim.claim_id


def test_verdict_row_is_flat():
    from sightline.m2a.verify import Verdict, verdict_row

    claim = Claim(
        image_id=5,
        turn=1,
        question="q",
        answer="Red.",
        claim="Red (car)",
        claim_type="colour",
        claim_id="abc",
    )
    row = verdict_row(
        claim,
        Verdict("hy4", "true", "because"),
        prompt_sha256="p",
        judged_at="2026-11-21T00:00:00+00:00",
    )
    assert row["claim_id"] == "abc"
    assert row["verifier_model"] == "hy4"
    assert row["verdict"] == "true"
    assert "verdicts" not in row


def test_judge_resumes_from_flat_rows():
    from sightline.m2a.verify import Verdict, judge_claims

    class Stub:
        def __init__(self, model, verdict):
            self.model = model
            self.verdict = verdict
            self.calls = 0

        def __call__(self, image, prompt):
            self.calls += 1
            return f"{self.verdict}\nbecause"

    callers = [Stub("v1", "true"), Stub("v2", "false")]
    existing = [
        {
            "claim_id": "abc",
            "verifier_model": "v1",
            "verdict": "false",
            "evidence": "prior",
            "error": None,
        }
    ]
    claim = Claim(
        image_id=0, turn=1, question="q", answer="a", claim="c", claim_type="colour", claim_id="abc"
    )
    judge_claims([claim], callers, existing=existing)
    assert callers[0].calls == 0
    assert callers[1].calls == 1


def test_an_error_row_is_never_counted_as_cant_tell():
    from sightline.m2a.verify import ClaimResult, Verdict, summarise

    results = [
        ClaimResult(
            Claim(0, 1, "q", "a", "c", "colour", claim_id="x"),
            verdicts=[Verdict("v1", "can't-tell", error="429")],
        )
    ]
    summary = summarise(results)
    assert summary["n_errored"] == 1
    assert summary["n_judged"] == 0
    assert summary["contradiction_rate"] is None

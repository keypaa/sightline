"""Tests for claim extraction, planted errors and the frozen thresholds."""

from __future__ import annotations

import pytest

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

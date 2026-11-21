"""Verify annotator claims against images, and measure blind-answerability.

Grounding is two separate questions (``DECISIONS/0049``):

1. **Does the annotator state falsehoods?** Each Q/A turn becomes one declarative
   claim naming its own referent, judged by two verifiers from families other than the
   annotator. Nothing is wrong unless a verifier says ``false``.
2. **Do the answers need the image at all?** Each question goes to a text-only model
   without the image. A question a blind model answers is a question that teaches the
  projector nothing visual.

A third check, **planted errors**, perturbs true claims and measures how often the
verifier catches them. The contradiction rate alone cannot detect a verifier that
acquiesces to everything.

    extract   conversation rows -> one claim per Q/A turn
    verify    claims -> verdicts from both verifiers, plus planted errors
    blind     questions -> text-only answers
    report    fold the three into a calibration summary
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Two verifier families, neither of which is the annotator (Qwen3.6-27B-FP8).
VERIFIERS = ("opencode/hy4-flash-free", "kilo/particle-ai/particle-1:free")

# The blind reader is text-only and must not be one of the verifiers.
BLIND_MODEL = "kilo/z-ai/glm-5-turbo:free"

# Claim types reported separately; a single rate would hide which capability failed.
CLAIM_TYPES = ("existence", "count", "colour", "spatial", "text", "action")

VERDICT_TRUE = "true"
VERDICT_FALSE = "false"
VERDICT_UNSURE = "can't-tell"

# Frozen in DECISIONS 0049 before the pilot run.
CONTRADICTION_CEILING = 0.15
PLANTED_REJECTION_FLOOR = 0.90
HUMAN_AGREEMENT_FLOOR = 0.85

_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_COLOURS = (
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "grey",
    "gray",
    "brown",
    "orange",
    "purple",
    "pink",
    "silver",
)

_SPATIAL = (
    "left",
    "right",
    "centre",
    "center",
    "top",
    "bottom",
    "above",
    "below",
    "behind",
    "front",
    "middle",
)


@dataclass
class Claim:
    """One declarative statement extracted from a Q/A turn."""

    image_id: Any
    turn: int
    question: str
    answer: str
    claim: str
    claim_type: str
    referent: str | None = None
    planted: bool = False
    original: str | None = None


@dataclass
class Verdict:
    """One verifier's judgement of one claim."""

    model: str
    verdict: str
    evidence: str = ""
    error: str | None = None


@dataclass
class ClaimResult:
    """A claim plus every verifier's verdict on it."""

    claim: Claim
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def contradicted(self) -> bool:
        return any(v.verdict == VERDICT_FALSE and not v.error for v in self.verdicts)

    @property
    def all_sure(self) -> bool:
        return bool(self.verdicts) and all(
            v.verdict in (VERDICT_TRUE, VERDICT_FALSE) and not v.error for v in self.verdicts
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def classify_claim(question: str, answer: str) -> str:
    """Pick the claim type from the question and answer wording."""

    text = f"{question} {answer}".lower()
    if re.search(r"\b(how many|number of|count)\b", text) or any(
        re.search(rf"\b{word}\b", text) for word in _COUNT_WORDS
    ):
        return "count"
    if any(re.search(rf"\b{colour}\b", text) for colour in _COLOURS):
        return "colour"
    if any(re.search(rf"\b{word}\b", text) for word in _SPATIAL):
        return "spatial"
    if re.search(r"\b(what does .* say|text|word|sign|label)\b", text):
        return "text"
    if re.search(r"\b(doing|action|playing|holding|wearing)\b", text):
        return "action"
    return "existence"


def extract_claims(rows: list[dict[str, Any]]) -> list[Claim]:
    """One claim per Q/A turn, with the referent named explicitly.

    A turn whose assistant message does not state an answer is skipped and counted
    by the caller, not silently dropped.
    """

    claims: list[Claim] = []
    for row in rows:
        messages = row.get("messages") or []
        question = None
        turn = 0
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role == "user":
                question = content
            elif role == "assistant" and question and content:
                turn += 1
                claims.append(
                    Claim(
                        image_id=row.get("id"),
                        turn=turn,
                        question=question,
                        answer=content,
                        claim=make_declarative(question, content),
                        claim_type=classify_claim(question, content),
                    )
                )
                question = None
    return claims


def make_declarative(question: str, answer: str) -> str:
    """Turn a Q/A pair into a single statement that names its referent.

    The referent is whatever noun phrase the question asks about, kept verbatim so a
    verifier can check the claim without guessing which object is meant.
    """

    match = re.search(
        r"\b(?:about|of|is|are|does|do|for)\s+(?:the\s+)?(.+?)\s*\?",
        question,
        flags=re.IGNORECASE,
    )
    referent = match.group(1).strip(" .") if match else None
    if referent:
        return f"{answer.rstrip('. ')} ({referent})"
    return answer.rstrip(". ")


def _swap_colour(claim: str) -> str | None:
    for colour in _COLOURS:
        if re.search(rf"\b{colour}\b", claim, flags=re.IGNORECASE):
            other = "blue" if colour.lower() != "blue" else "red"
            return re.sub(rf"\b{colour}\b", other, claim, count=1, flags=re.IGNORECASE)
    return None


def _swap_count(claim: str) -> str | None:
    for word, value in _COUNT_WORDS.items():
        if re.search(rf"\b{word}\b", claim, flags=re.IGNORECASE):
            return re.sub(rf"\b{word}\b", str(value + 1), claim, count=1, flags=re.IGNORECASE)
    return re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), claim, count=1)


def _swap_noun(claim: str) -> str | None:
    nouns = ("car", "person", "dog", "cat", "bike", "motorcycle", "table", "chair")
    for noun in nouns:
        if re.search(rf"\b{noun}\b", claim, flags=re.IGNORECASE):
            other = "bicycle" if noun.lower() != "bicycle" else "car"
            return re.sub(rf"\b{noun}\b", other, claim, count=1, flags=re.IGNORECASE)
    return None


def plant_errors(claims: list[Claim], seed: int = 0) -> list[Claim]:
    """Return planted variants of true claims, for measuring verifier rejection."""

    planted: list[Claim] = []
    rng = random.Random(seed)
    for claim in claims:
        perturbed = _swap_colour(claim.claim) or _swap_count(claim.claim) or _swap_noun(claim.claim)
        if perturbed and perturbed != claim.claim:
            planted.append(
                Claim(
                    image_id=claim.image_id,
                    turn=claim.turn,
                    question=claim.question,
                    answer=claim.answer,
                    claim=perturbed,
                    claim_type=claim.claim_type,
                    referent=claim.referent,
                    planted=True,
                    original=claim.claim,
                )
            )
    rng.shuffle(planted)
    return planted


def judge_verdict(raw: str) -> tuple[str, str]:
    """Parse a verifier reply into (verdict, evidence)."""

    text = (raw or "").strip()
    lowered = text.lower()
    if VERDICT_FALSE in lowered or "false" in lowered.split("\n")[0]:
        verdict = VERDICT_FALSE
    elif VERDICT_UNSURE in lowered or "can't tell" in lowered or "cannot tell" in lowered:
        verdict = VERDICT_UNSURE
    elif VERDICT_TRUE in lowered or "true" in lowered.split("\n")[0]:
        verdict = VERDICT_TRUE
    else:
        verdict = VERDICT_UNSURE
    evidence = text.split("\n", 1)[1].strip() if "\n" in text else ""
    return verdict, evidence


def build_verifier_prompt(claim: Claim, *, adversarial: bool = False) -> str:
    """The judging prompt. The adversarial variant is used for planted errors."""

    instruction = (
        "You are checking whether a claim about an image is true. "
        "Look at the image and answer with one word on the first line: true, false, "
        "or can't-tell. On the second line, give a short evidence note."
    )
    if adversarial:
        instruction += (
            " Assume the claim may be deliberately wrong; check it carefully rather "
            "than agreeing by default."
        )
    return f"{instruction}\n\nClaim: {claim.claim}\nQuestion it came from: {claim.question}"


def summarise(results: list[ClaimResult]) -> dict[str, Any]:
    """Report contradiction, can't-tell and per-type rates, with a Wilson interval."""

    n = len(results)
    contradicted = sum(1 for r in results if r.contradicted)
    unsure = sum(
        1 for r in results if r.verdicts and all(v.verdict == VERDICT_UNSURE for v in r.verdicts)
    )

    by_type: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_type.setdefault(
            result.claim.claim_type, {"n": 0, "contradicted": 0, "unsure": 0}
        )
        bucket["n"] += 1
        bucket["contradicted"] += int(result.contradicted)
        bucket["unsure"] += int(
            bool(result.verdicts) and all(v.verdict == VERDICT_UNSURE for v in result.verdicts)
        )

    return {
        "n_claims": n,
        "contradicted": contradicted,
        "contradiction_rate": (contradicted / n) if n else None,
        "contradiction_rate_upper_95": wilson_upper(contradicted, n),
        "can_tell_share": (unsure / n) if n else None,
        "by_type": dict(sorted(by_type.items())),
        "thresholds": {
            "contradiction_ceiling": CONTRADICTION_CEILING,
            "planted_rejection_floor": PLANTED_REJECTION_FLOOR,
            "human_agreement_floor": HUMAN_AGREEMENT_FLOOR,
        },
    }


def wilson_upper(successes: int, n: int, z: float = 1.959964) -> float | None:
    """Upper bound of the Wilson interval, for a frozen ceiling comparison."""

    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denominator
    return centre + margin


def summarise_planted(results: list[ClaimResult]) -> dict[str, Any]:
    """Rejection rate on planted errors — the guard against an acquiescent verifier."""

    n = len(results)
    caught = sum(1 for r in results if r.contradicted)
    return {
        "n_planted": n,
        "rejected": caught,
        "rejection_rate": (caught / n) if n else None,
        "floor": PLANTED_REJECTION_FLOOR,
        "passes": (caught / n) >= PLANTED_REJECTION_FLOOR if n else False,
    }


def summarise_blind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often a text-only model reproduces the answer without the image."""

    scored = [r for r in rows if r.get("blind_answer") is not None]
    hits = sum(1 for r in scored if r.get("blind_matches"))
    return {
        "n_questions": len(scored),
        "blind_reproduced": hits,
        "blind_rate": (hits / len(scored)) if scored else None,
        "note": (
            "A high rate means the question is answerable without the image and "
            "teaches the projector nothing visual. Threshold is [U] until the "
            "distribution is seen (DECISIONS 0049)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="conversation rows -> claims")
    extract.add_argument("--conversations", type=Path, required=True)
    extract.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl"))

    plant = sub.add_parser("plant", help="claims -> planted-error variants")
    plant.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    plant.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/planted.jsonl"))
    plant.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    args.seed = getattr(args, "seed", 0)

    if args.command == "extract":
        rows = read_jsonl(args.conversations)
        claims = extract_claims(rows)
        write_jsonl(
            args.out,
            [
                {
                    "image_id": c.image_id,
                    "turn": c.turn,
                    "question": c.question,
                    "answer": c.answer,
                    "claim": c.claim,
                    "claim_type": c.claim_type,
                    "planted": False,
                }
                for c in claims
            ],
        )
        counts: dict[str, int] = {}
        for claim in claims:
            counts[claim.claim_type] = counts.get(claim.claim_type, 0) + 1
        print(f"[extract] {len(claims)} claims from {len(rows)} conversations")
        print(f"[extract] by type: {dict(sorted(counts.items()))}")
        print(f"[extract] -> {args.out}")
        return 0

    claims = [
        Claim(
            image_id=row.get("image_id"),
            turn=row.get("turn", 0),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            claim=row.get("claim", ""),
            claim_type=row.get("claim_type", "existence"),
            planted=bool(row.get("planted")),
            original=row.get("original"),
        )
        for row in read_jsonl(args.claims)
    ]
    planted = plant_errors(claims, seed=args.seed)
    write_jsonl(
        args.out,
        [
            {
                "image_id": c.image_id,
                "turn": c.turn,
                "question": c.question,
                "answer": c.answer,
                "claim": c.claim,
                "claim_type": c.claim_type,
                "planted": True,
                "original": c.original,
            }
            for c in planted
        ],
    )
    print(f"[plant] {len(planted)} planted errors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

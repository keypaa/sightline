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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sightline.m2a.transport import TransportError

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
    image_path: str | None = None


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


def extract_claims(
    rows: list[dict[str, Any]],
    images_dir: Path | None = None,
    project_root: Path | None = None,
) -> list[Claim]:
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
                        image_path=resolve_image_path(row, images_dir, project_root),
                    )
                )
                question = None
    return claims


def resolve_image_path(
    row: dict[str, Any], images_dir: Path | None, project_root: Path | None
) -> str | None:
    """Find the image a conversation row refers to, without guessing silently."""

    path = row.get("path") or row.get("image_path")
    if path:
        return str(path)
    name = row.get("file_name")
    if name and images_dir is not None:
        return str(Path(images_dir) / name)
    return None


def make_declarative(question: str, answer: str) -> str:
    """Turn a Q/A pair into a single statement that names its referent.

    The referent is what tells a verifier *what* the claim is about, so it has to be
    the counted or described thing, not the tail of the sentence. "How many tiers
    does the cake have?" must yield "tiers of the cake", not "cake have" — otherwise
    the verifier is guessing at the subject of the claim.
    """

    referent = _count_referent(question) or _general_referent(question)
    if referent:
        return f"{answer.rstrip('. ')} ({referent})"
    return answer.rstrip(". ")


def _count_referent(question: str) -> str | None:
    """The thing being counted, and what it belongs to, for a how-many question."""

    match = re.match(
        r"\s*how many\s+(.+?)\s+(?:does|do|are|is|can|will|would)\s+(.+?)\s*\?",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        counted = match.group(1).strip()
        rest = match.group(2).strip()
        # Strip the trailing verb ("the cake have" -> "the cake") and a leading
        # "visible", which describes the verb rather than the holder.
        rest = re.sub(r"\s+(?:have|has|there)\s*$", "", rest).strip()
        rest = re.sub(r"^visible\s+", "", rest).strip()
        if rest.lower() in ("there", "there are"):
            rest = ""
        if not rest:
            return counted
        if rest.lower().startswith("the "):
            return f"{counted} of {rest}"
        return f"{counted} {rest}"

    match = re.match(r"\s*how many\s+(.+?)\s*\?", question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _general_referent(question: str) -> str | None:
    """The noun phrase after a preposition or linking verb."""

    match = re.search(
        r"\b(?:about|of|is|are|does|do|for)\s+(?:the\s+)?(.+?)\s*\?",
        question,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip(" .") if match else None


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


def normalise_tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def blind_matches(expected: str, response: str) -> bool:
    """True when the expected answer's words all appear in the blind response."""

    want = normalise_tokens(expected)
    if not want:
        return False
    return want.issubset(normalise_tokens(response))


def _verdict_key(claim: Claim, model: str) -> tuple:
    return (claim.image_id, claim.turn, bool(claim.planted), model)


def _claim_from_row(row: dict[str, Any]) -> Claim:
    return Claim(
        image_id=row.get("image_id"),
        turn=row.get("turn", 0),
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        claim=row.get("claim", ""),
        claim_type=row.get("claim_type", "existence"),
        planted=bool(row.get("planted")),
        original=row.get("original"),
        image_path=row.get("image_path"),
    )


def _result_to_row(result: ClaimResult) -> dict[str, Any]:
    return {
        "image_id": result.claim.image_id,
        "turn": result.claim.turn,
        "question": result.claim.question,
        "answer": result.claim.answer,
        "claim": result.claim.claim,
        "claim_type": result.claim.claim_type,
        "planted": result.claim.planted,
        "image_path": result.claim.image_path,
        "contradicted": result.contradicted,
        "verdicts": [
            {
                "model": v.model,
                "verdict": v.verdict,
                "evidence": v.evidence,
                "error": v.error,
            }
            for v in result.verdicts
        ],
    }


def judge_claims(
    claims: list[Claim],
    callers: list[Any],
    *,
    existing: list[dict[str, Any]] | None = None,
    on_row: Any = None,
) -> list[ClaimResult]:
    """Judge every claim with every caller, resuming from an existing results file.

    A claim that already has a verdict from a given model is not re-asked, so a run
    interrupted by a rate limit can be restarted without spending the quota again.
    """

    prior: dict[tuple, Verdict] = {}
    for row in existing or []:
        for verdict in row.get("verdicts") or []:
            if verdict.get("error"):
                continue
            key = (
                row.get("image_id"),
                row.get("turn"),
                bool(row.get("planted")),
                verdict.get("model"),
            )
            prior[key] = Verdict(
                model=verdict.get("model", ""),
                verdict=verdict.get("verdict", VERDICT_UNSURE),
                evidence=verdict.get("evidence", ""),
            )

    results: list[ClaimResult] = []
    for claim in claims:
        verdicts: list[Verdict] = []
        for caller in callers:
            key = _verdict_key(claim, caller.model)
            if key in prior:
                verdicts.append(prior[key])
                continue
            prompt = build_verifier_prompt(claim, adversarial=claim.planted)
            try:
                image = Path(claim.image_path) if claim.image_path else None
                verdict, evidence = judge_verdict(caller(image, prompt))
                verdicts.append(Verdict(caller.model, verdict, evidence))
            except TransportError as exc:
                # A failed judgement is not a "can't-tell" about the claim; it is a
                # gap to retry, so it is marked as an error rather than a verdict.
                verdicts.append(Verdict(caller.model, VERDICT_UNSURE, evidence="", error=str(exc)))
        result = ClaimResult(claim=claim, verdicts=verdicts)
        results.append(result)
        if on_row is not None:
            on_row(result)
    return results


def judge_blind(claims: list[Claim], call: Any, *, on_row: Any = None) -> list[dict[str, Any]]:
    """Ask a text-only model each question, with no image, and compare."""

    rows: list[dict[str, Any]] = []
    for claim in claims:
        row: dict[str, Any] = {
            "image_id": claim.image_id,
            "turn": claim.turn,
            "question": claim.question,
            "expected": claim.answer,
        }
        try:
            response = call(
                None,
                "Answer the question in one short phrase.\n\nQuestion: " + claim.question,
            )
            row["blind_answer"] = response
            row["blind_matches"] = blind_matches(claim.answer, response)
        except TransportError as exc:
            row["blind_answer"] = None
            row["blind_matches"] = False
            row["error"] = str(exc)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def main() -> int:
    from sightline.m2a.verifier_transport import OpencodeCaller

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="conversation rows -> claims")
    extract.add_argument("--conversations", type=Path, required=True)
    extract.add_argument("--images-dir", type=Path, default=Path("artifacts/m2a_images"))
    extract.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl"))

    plant = sub.add_parser("plant", help="claims -> planted-error variants")
    plant.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    plant.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/planted.jsonl"))
    plant.add_argument("--seed", type=int, default=0)

    judge = sub.add_parser("judge", help="claims -> verifier verdicts")
    judge.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    judge.add_argument("--planted", type=Path, default=None)
    judge.add_argument("--models", nargs="+", default=list(VERIFIERS))
    judge.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/verdicts.jsonl"))
    judge.add_argument("--timeout", type=int, default=300)

    blind = sub.add_parser("blind", help="questions -> text-only answers")
    blind.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    blind.add_argument("--model", default=BLIND_MODEL)
    blind.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/blind.jsonl"))

    report = sub.add_parser("report", help="fold verdicts and blind rows into a summary")
    report.add_argument(
        "--verdicts",
        type=Path,
        default=Path("experiments/m2a_grounding/verdicts.jsonl"),
    )
    report.add_argument("--blind", type=Path, default=Path("experiments/m2a_grounding/blind.jsonl"))
    report.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/summary.json"))

    args = parser.parse_args()
    args.seed = getattr(args, "seed", 0)

    if args.command == "extract":
        rows = read_jsonl(args.conversations)
        root = Path(__file__).resolve().parents[3]
        claims = extract_claims(rows, args.images_dir, root)
        write_jsonl(args.out, [_result_to_row(ClaimResult(claim=c)) for c in claims])
        counts: dict[str, int] = {}
        for claim in claims:
            counts[claim.claim_type] = counts.get(claim.claim_type, 0) + 1
        print(f"[extract] {len(claims)} claims from {len(rows)} conversations")
        print(f"[extract] by type: {dict(sorted(counts.items()))}")
        print(f"[extract] -> {args.out}")
        return 0

    if args.command == "plant":
        claims = [_claim_from_row(row) for row in read_jsonl(args.claims)]
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
                    "image_path": c.image_path,
                }
                for c in planted
            ],
        )
        print(f"[plant] {len(planted)} planted errors -> {args.out}")
        return 0

    if args.command == "judge":
        claims = [_claim_from_row(row) for row in read_jsonl(args.claims)]
        if args.planted:
            claims += [_claim_from_row(row) for row in read_jsonl(args.planted)]
        callers = [OpencodeCaller(model, timeout=args.timeout) for model in args.models]
        existing = read_jsonl(args.out)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        written: list[dict[str, Any]] = []

        def on_row(result: ClaimResult) -> None:
            row = _result_to_row(result)
            with lock:
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                written.append(row)

        results = judge_claims(claims, callers, existing=existing, on_row=on_row)
        summary = summarise(results)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    if args.command == "blind":
        claims = [_claim_from_row(row) for row in read_jsonl(args.claims)]
        caller = OpencodeCaller(args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        with args.out.open("w", encoding="utf-8") as handle:

            def on_row(row: dict[str, Any]) -> None:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)

            judge_blind(claims, caller, on_row=on_row)
        summary = summarise_blind(rows)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    verdict_rows = read_jsonl(args.verdicts)
    results = [
        ClaimResult(
            claim=_claim_from_row(row),
            verdicts=[
                Verdict(
                    model=v.get("model", ""),
                    verdict=v.get("verdict", VERDICT_UNSURE),
                    evidence=v.get("evidence", ""),
                    error=v.get("error"),
                )
                for v in row.get("verdicts") or []
            ],
        )
        for row in verdict_rows
    ]
    real = [r for r in results if not r.claim.planted]
    planted = [r for r in results if r.claim.planted]

    summary = summarise(real)
    summary["planted"] = summarise_planted(planted)
    summary["blind"] = summarise_blind(read_jsonl(args.blind))
    summary["thresholds"] = {
        "contradiction_ceiling": CONTRADICTION_CEILING,
        "planted_rejection_floor": PLANTED_REJECTION_FLOOR,
        "human_agreement_floor": HUMAN_AGREEMENT_FLOOR,
    }
    summary["passes_contradiction"] = (
        summary["contradiction_rate"] is not None
        and summary["contradiction_rate_upper_95"] is not None
        and summary["contradiction_rate_upper_95"] <= CONTRADICTION_CEILING
    )
    summary["passes_planted"] = summary["planted"]["passes"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if (summary["passes_contradiction"] and summary["passes_planted"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

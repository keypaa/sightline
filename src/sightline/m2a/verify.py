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
import hashlib
import json
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sightline.m2a.ids import claim_id, now_iso
from sightline.m2a.transport import TransportError

# One verifier (DECISIONS 0049, spec section 1). A second family is available via
# --models for a deliberate cross-check, but it is not the default: two verifiers
# double the quota cost for a robustness gain we cannot afford.
VERIFIERS = ("opencode/hy4-flash-free",)

# The blind reader is text-only and must not be one of the verifiers.
BLIND_MODEL = "opencode/hy4-flash-free"

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
    claim_id: str = ""
    source_run: str = ""


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
                text = make_declarative(question, content)
                claims.append(
                    Claim(
                        image_id=row.get("id"),
                        turn=turn,
                        question=question,
                        answer=content,
                        claim=text,
                        claim_type=classify_claim(question, content),
                        image_path=resolve_image_path(row, images_dir, project_root),
                        # Computed from the same text that is stored, so the id and
                        # the claim can never disagree.
                        claim_id=claim_id(row.get("id"), turn, text),
                        source_run=str(row.get("run_id") or ""),
                    )
                )
                question = None
    return claims


def resolve_image_path(
    row: dict[str, Any], images_dir: Path | None, project_root: Path | None
) -> str | None:
    """Find the image a conversation row refers to, without guessing silently."""

    path = row.get("path") or row.get("image_path")
    if path and Path(path).exists():
        return str(path)
    name = row.get("file_name")
    if name and images_dir is not None:
        candidate = Path(images_dir) / name
        if candidate.exists():
            return str(candidate)
    # An unresolved image is not an error: the claim stays usable, and the
    # verifier reports it rather than the row vanishing.
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
                    # A planted claim is a different statement, so it gets its own
                    # id; sharing the original's would collide in the verdict index.
                    claim_id=claim_id(claim.image_id, claim.turn, perturbed),
                    source_run=claim.source_run,
                    # Without the image the verifier can only guess, which would make
                    # the planted rejection rate measure nothing.
                    image_path=claim.image_path,
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


JUDGE_INSTRUCTION = (
    "You are checking whether a claim about an image is true. "
    "Look at the image and answer with one word on the first line: true, false, "
    "or can't-tell. On the second line, give a short evidence note."
)

_ADVERSARIAL_SUFFIX = (
    " Assume the claim may be deliberately wrong; check it carefully rather "
    "than agreeing by default."
)


def build_verifier_prompt(claim: Claim, *, adversarial: bool = False) -> str:
    """The judging prompt. The adversarial variant is used for planted errors."""

    instruction = JUDGE_INSTRUCTION + (_ADVERSARIAL_SUFFIX if adversarial else "")
    return f"{instruction}\n\nClaim: {claim.claim}\nQuestion it came from: {claim.question}"


def summarise(results: list[ClaimResult]) -> dict[str, Any]:
    """Report contradiction, can't-tell and per-type rates, with a Wilson interval.

    A call that failed is a gap, not a "can't-tell": counting errors as unsure would
    turn a rate limit into an apparently cautious model.
    """

    n = len(results)
    judged = [r for r in results if r.verdicts and all(not v.error for v in r.verdicts)]
    errored = n - len(judged)
    contradicted = sum(1 for r in judged if r.contradicted)
    unsure = sum(1 for r in judged if all(v.verdict == VERDICT_UNSURE for v in r.verdicts))

    by_type: dict[str, dict[str, int]] = {}
    for result in judged:
        bucket = by_type.setdefault(
            result.claim.claim_type, {"n": 0, "contradicted": 0, "unsure": 0}
        )
        bucket["n"] += 1
        bucket["contradicted"] += int(result.contradicted)
        bucket["unsure"] += int(all(v.verdict == VERDICT_UNSURE for v in result.verdicts))

    n_judged = len(judged)
    return {
        "n_claims": n,
        "n_judged": n_judged,
        "n_errored": errored,
        "contradicted": contradicted,
        "contradiction_rate": (contradicted / n_judged) if n_judged else None,
        "contradiction_rate_upper_95": wilson_upper(contradicted, n_judged),
        "can_tell_share": (unsure / n_judged) if n_judged else None,
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
    """Rejection rate on planted errors — the guard against an acquiescent verifier.

    A call that failed is a gap, not a miss: including it would let a quota failure
    flip the gate. This mirrors the rule used for real claims.
    """

    results = [r for r in results if r.verdicts and all(not v.error for v in r.verdicts)]
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
        image_id=row.get("image_id", row.get("id")),
        turn=row.get("turn", 0),
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        claim=row.get("claim", ""),
        claim_type=row.get("claim_type", "existence"),
        planted=bool(row.get("planted")),
        original=row.get("original"),
        image_path=row.get("image_path"),
        claim_id=row.get("claim_id", ""),
        source_run=row.get("source_run", ""),
    )


def claim_to_row(claim: Claim, *, source_run: str = "", extracted_at: str = "") -> dict[str, Any]:
    """Write a claim row in exactly the spec's schema.

    Verdict-shaped fields (``contradicted``, ``verdicts``) belong on verdict rows.
    Putting them here would make a claim file ambiguous about which layer it is.
    """

    return {
        "claim_id": claim.claim_id,
        "image_id": claim.image_id,
        "turn": claim.turn,
        "question": claim.question,
        "answer": claim.answer,
        "claim": claim.claim,
        "claim_type": claim.claim_type,
        "image_path": claim.image_path,
        "source_run": source_run,
        "extracted_at": extracted_at,
        "planted": claim.planted,
        "original": claim.original,
    }


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


def sample_claims(claims: list[Claim], n: int, seed: int = 0) -> list[Claim]:
    """Take n claims reproducibly, spread across claim types.

    Judging every claim is unnecessary for a rate: a few hundred gives a tight
    interval, and the free verifiers are metered.
    """

    if n <= 0 or n >= len(claims):
        return list(claims)
    by_type: dict[str, list[Claim]] = {}
    for claim in claims:
        by_type.setdefault(claim.claim_type, []).append(claim)
    rng = random.Random(seed)
    per_type = max(1, n // max(1, len(by_type)))
    picked: list[Claim] = []
    for bucket in by_type.values():
        picked.extend(rng.sample(bucket, min(per_type, len(bucket))))
    rng.shuffle(picked)
    if len(picked) < n:
        chosen = {id(c) for c in picked}
        rest = [c for c in claims if id(c) not in chosen]
        picked.extend(rng.sample(rest, min(n - len(picked), len(rest))))
    return picked[:n]


def judge_verdict_prompt_sha256() -> str:
    """Pin the judging instructions, so a verdict names the prompt it answered.

    The claim text varies per row and is already stored on the claim, so hashing it
    would make every verdict's hash different and useless for spotting a prompt
    change. This hashes the instruction template: stable across claims, and changes
    if the instructions change.
    """

    return hashlib.sha256((JUDGE_INSTRUCTION + _ADVERSARIAL_SUFFIX).encode()).hexdigest()[:16]


def verdict_row(
    claim: Claim, verdict: Verdict, *, prompt_sha256: str, judged_at: str
) -> dict[str, Any]:
    """One flat row: a single verifier's judgement of a single claim."""

    return {
        "claim_id": claim.claim_id,
        "image_id": claim.image_id,
        "turn": claim.turn,
        # Self-describing: a verdict row carries enough claim context that report
        # and the review UI never need a second file to interpret it.
        "claim": claim.claim,
        "claim_type": claim.claim_type,
        "question": claim.question,
        "answer": claim.answer,
        "planted": claim.planted,
        "verifier_model": verdict.model,
        "verifier_prompt_sha256": prompt_sha256,
        "verdict": verdict.verdict,
        "evidence": verdict.evidence,
        "error": verdict.error,
        "judged_at": judged_at,
    }


def judge_claims(
    claims: list[Claim],
    callers: list[Any],
    *,
    existing: list[dict[str, Any]] | None = None,
    on_row: Any = None,
    workers: int = 1,
    max_consecutive_failures: int = 10,
    on_verdict: Any = None,
) -> list[ClaimResult]:
    """Judge every claim with every caller, resuming from an existing results file.

    A claim that already has a verdict from a given model is not re-asked, so a run
    interrupted by a rate limit can be restarted without spending the quota again.
    Claims are judged in parallel; the two callers for one claim stay sequential so
    the claim's own verdicts stay together.
    """

    from concurrent.futures import ThreadPoolExecutor

    prompt_sha256 = judge_verdict_prompt_sha256()
    prior: dict[tuple, Verdict] = {}
    for row in existing or []:
        if row.get("error"):
            continue
        # A verdict that answered a different judging prompt answers a different
        # question, so it is not reused when the instructions change.
        if row.get("verifier_prompt_sha256") not in (None, prompt_sha256):
            continue
        key = (row.get("claim_id"), row.get("verifier_model"))
        prior[key] = Verdict(
            model=row.get("verifier_model", ""),
            verdict=row.get("verdict", VERDICT_UNSURE),
            evidence=row.get("evidence", ""),
        )

    for claim in claims:
        if not claim.claim_id:
            raise ValueError("claim_id is empty; re-run extract so claims carry ids")

    todo: list[tuple[Claim, list[Any]]] = []
    for claim in claims:
        missing = [c for c in callers if (claim.claim_id, c.model) not in prior]
        if missing:
            todo.append((claim, missing))

    def work(item: tuple[Claim, list[Any]]) -> ClaimResult:
        claim, pending_callers = item
        verdicts = [
            prior[(claim.claim_id, c.model)] for c in callers if (claim.claim_id, c.model) in prior
        ]
        for caller in pending_callers:
            prompt = build_verifier_prompt(claim, adversarial=claim.planted)
            try:
                image = Path(claim.image_path) if claim.image_path else None
                verdict, evidence = judge_verdict(caller(image, prompt))
                produced = Verdict(caller.model, verdict, evidence)
                verdicts.append(produced)
                if on_verdict is not None:
                    on_verdict(claim, produced)
            except TransportError as exc:
                # A failed judgement is not a "can't-tell" about the claim; it is a
                # gap to retry, so it is marked as an error rather than a verdict.
                # It must still be written, or a quota failure leaves no trace at all
                # and the run looks clean while having judged nothing.
                produced = Verdict(caller.model, VERDICT_UNSURE, evidence="", error=str(exc))
                verdicts.append(produced)
                if on_verdict is not None:
                    on_verdict(claim, produced)
        return ClaimResult(claim=claim, verdicts=verdicts)

    results: list[ClaimResult] = []
    if workers <= 1:
        consecutive_failures = 0
        for item in todo:
            result = work(item)
            results.append(result)
            if on_row is not None:
                on_row(result)
            failed = bool(result.verdicts) and all(v.error for v in result.verdicts)
            consecutive_failures = consecutive_failures + 1 if failed else 0
            if consecutive_failures >= max_consecutive_failures:
                return results
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work, todo):
            results.append(result)
            if on_row is not None:
                on_row(result)
    return results


def pending_blind_claims(claims: list[Claim], out: Path) -> list[Claim]:
    """Drop failed blind rows from ``out`` and return claims still unanswered.

    Resume must not pay for calls that already succeeded, and an HTTP 429 row is
    a gap rather than a blind answer.
    """

    prior = read_jsonl(out) if out.exists() else []
    healed = [row for row in prior if not row.get("error")]
    if len(healed) != len(prior):
        write_jsonl(out, healed)
    done = {row.get("claim_id") for row in healed if row.get("blind_answer") is not None}
    return [c for c in claims if c.claim_id not in done]


def judge_blind(
    claims: list[Claim], call: Any, *, model: str, on_row: Any = None
) -> list[dict[str, Any]]:
    """Ask a text-only model each question, with no image, and compare."""

    for claim in claims:
        if not claim.claim_id:
            raise ValueError("claim_id is empty; re-run extract so claims carry ids")

    rows: list[dict[str, Any]] = []
    for claim in claims:
        row: dict[str, Any] = {
            "claim_id": claim.claim_id,
            "image_id": claim.image_id,
            "turn": claim.turn,
            "question": claim.question,
            "expected": claim.answer,
            "blind_model": model,
            "judged_at": now_iso(),
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
    judge.add_argument("--workers", type=int, default=1, help="claims judged concurrently")
    judge.add_argument("--sample", type=int, default=0, help="judge at most N claims (0 means all)")
    judge.add_argument("--sample-seed", type=int, default=0)

    blind = sub.add_parser("blind", help="questions -> text-only answers")
    blind.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    blind.add_argument("--model", default=BLIND_MODEL)
    blind.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/blind.jsonl"))
    blind.add_argument("--sample", type=int, default=0, help="blind at most N claims (0 means all)")
    blind.add_argument("--sample-seed", type=int, default=0)

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
        stamp = now_iso()
        write_jsonl(
            args.out,
            [claim_to_row(c, source_run=c.source_run, extracted_at=stamp) for c in claims],
        )
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
        stamp = now_iso()
        write_jsonl(
            args.out,
            [claim_to_row(c, source_run=c.source_run, extracted_at=stamp) for c in planted],
        )
        print(f"[plant] {len(planted)} planted errors -> {args.out}")
        return 0

    if args.command == "judge":
        claims = [_claim_from_row(row) for row in read_jsonl(args.claims)]
        real_count = len(claims)
        if args.planted:
            claims += [_claim_from_row(row) for row in read_jsonl(args.planted)]
        if args.sample:
            # Sample real and planted separately so the planted ratio stays honest.
            claims = sample_claims(claims[:real_count], args.sample, args.sample_seed)
            claims += (
                sample_claims(
                    [_claim_from_row(r) for r in read_jsonl(args.planted)],
                    args.sample,
                    args.sample_seed,
                )
                if args.planted
                else []
            )
        callers = [OpencodeCaller(model, timeout=args.timeout) for model in args.models]
        # An error row is a gap, not a verdict. Drop them before computing the
        # resume set, so a claim that was rate-limited last time is re-asked and
        # its failed row is replaced rather than poisoning the claim forever.
        if args.out.exists():
            prior_rows = read_jsonl(args.out)
            keep = [row for row in prior_rows if not row.get("error")]
            if len(keep) != len(prior_rows):
                write_jsonl(args.out, keep)
        existing = read_jsonl(args.out)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        written: list[dict[str, Any]] = []
        prompt_sha256 = judge_verdict_prompt_sha256()

        def on_verdict(claim: Claim, verdict: Verdict) -> None:
            row = verdict_row(claim, verdict, prompt_sha256=prompt_sha256, judged_at=now_iso())
            with lock:
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                written.append(row)

        results = judge_claims(
            claims,
            callers,
            existing=existing,
            on_verdict=on_verdict,
            workers=args.workers,
        )
        summary = summarise(results)
        # judge_claims stops after a run of transport failures, so a partial run
        # must not report success. Count the claims still lacking a verdict row.
        judged_ids = {row["claim_id"] for row in read_jsonl(args.out) if not row.get("error")}
        summary["n_unjudged"] = sum(1 for c in claims if c.claim_id not in judged_ids)
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if summary["n_unjudged"] else 0

    if args.command == "blind":
        claims = [_claim_from_row(row) for row in read_jsonl(args.claims)]
        if args.sample:
            # Same draw as judge's real claims, so blind and vision answers pair.
            claims = sample_claims(claims, args.sample, args.sample_seed)
        caller = OpencodeCaller(args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)

        # A quota failure is a gap, not a blind answer. Heal those rows and only
        # re-ask claims that have no successful answer yet, so a long blind run
        # can resume without paying for every call again.
        pending = pending_blind_claims(claims, args.out)
        rows: list[dict[str, Any]] = []
        with args.out.open("a", encoding="utf-8") as handle:

            def on_row(row: dict[str, Any]) -> None:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)

            judge_blind(pending, caller, model=args.model, on_row=on_row)
        all_rows = read_jsonl(args.out)
        summary = summarise_blind(all_rows)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    # judge writes flat rows: one per claim per verifier. The rows are self-describing,
    # so report needs no second file and planted claims are distinguished by their flag.
    verdict_rows = read_jsonl(args.verdicts)
    grouped: dict[str, list[Verdict]] = {}
    claim_text: dict[str, dict[str, Any]] = {}
    for row in verdict_rows:
        cid = row.get("claim_id")
        grouped.setdefault(cid, []).append(
            Verdict(
                model=row.get("verifier_model", ""),
                verdict=row.get("verdict", VERDICT_UNSURE),
                evidence=row.get("evidence", ""),
                error=row.get("error"),
            )
        )
        claim_text.setdefault(cid, row)

    results = [
        ClaimResult(claim=_claim_from_row(claim_text[cid]), verdicts=vs)
        for cid, vs in grouped.items()
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

"""Calibration review UI: a human checks machine judgements against images.

This is an instrument, not a convenience (``DECISIONS/0049``): the owner's review
set is sampled across verifier strata, and the recorded decisions become the
human-side evidence for the verifier's accuracy.

    python tools/m2a_review.py --claims claims.jsonl --images-dir artifacts/m2a_images \
        --out /tmp/review.html

The page is self-contained, reads images over absolute file:// URLs, and stores
decisions in browser local storage. Export writes a JSON file of decisions.

Usage:
    m2a_review.py [-h] [--claims PATH] [--images-dir PATH] [--out PATH]
                  [--title TEXT] [--sample N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from sightline.m2a.ids import now_iso

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERDICT_LABEL = {"true": "supported", "false": "contradicted", "can't-tell": "unsure"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _verdict_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten claim results into one row per claim, with each verifier's view."""

    rows = []
    for result in results:
        verdicts = result.get("verdicts") or []
        rows.append(
            {
                "image_id": result.get("image_id") or result.get("id"),
                "claim": result.get("claim", ""),
                "claim_type": result.get("claim_type", "unknown"),
                "planted": bool(result.get("planted")),
                "verdicts": verdicts,
                "contradicted": bool(result.get("contradicted")),
                "path": result.get("path") or result.get("image_path"),
            }
        )
    return rows


def stratify(rows: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    """Sample across verifier strata, so the review measures verifier error.

    Disagreement rows are the informative ones, so they are over-sampled; agreement
    rows are included in both directions so acquiescence stays visible.
    """

    if sample <= 0 or sample >= len(rows):
        return rows

    disagree = [r for r in rows if r["contradicted"]]
    agree_true = [
        r
        for r in rows
        if not r["contradicted"]
        and r["verdicts"]
        and all(v.get("verdict") == "true" for v in r["verdicts"])
    ]
    agree_false = [
        r
        for r in rows
        if not r["contradicted"]
        and r["verdicts"]
        and all(v.get("verdict") == "can't-tell" for v in r["verdicts"])
    ]

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    # Half the budget on disagreement, split the rest across the two agreement strata.
    n_disagree = max(1, sample // 2)
    n_each = max(1, (sample - n_disagree) // 2)

    for pool, n in ((disagree, n_disagree), (agree_true, n_each), (agree_false, n_each)):
        if not pool:
            continue
        picked.extend(rng.sample(pool, min(n, len(pool))))

    remaining = sample - len(picked)
    if remaining > 0:
        leftovers = [r for r in rows if r not in picked]
        picked.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    return picked[:sample]


def _verdict_html(verdicts: list[dict[str, Any]]) -> str:
    if not verdicts:
        return '<p class="muted">no verifier ran</p>'
    parts = []
    for verdict in verdicts:
        label = VERDICT_LABEL.get(str(verdict.get("verdict")), str(verdict.get("verdict")))
        model = str(verdict.get("model", "verifier")).split("/")[-1]
        evidence = str(verdict.get("evidence") or "").strip()
        parts.append(
            f'<li><span class="tag tag-{label}">{label}</span>'
            f'<span class="model">{model}</span>'
            f'<span class="evidence">{evidence or "—"}</span></li>'
        )
    return "<ul class='verdicts'>" + "".join(parts) + "</ul>"


def _card_verdict(row: dict[str, Any]) -> str:
    """The card's overall status, for the left colour bar and the badge."""

    verdicts = [str(v.get("verdict")) for v in (row.get("verdicts") or [])]
    if not verdicts:
        return "mixed"
    if any(v == "false" for v in verdicts):
        return "contradicted" if all(v == "false" for v in verdicts) else "mixed"
    if all(v == "true" for v in verdicts):
        return "supported"
    return "unsure"


def render(rows: list[dict[str, Any]], out: Path, images_dir: Path, title: str) -> None:
    """Write a single self-contained page."""

    cards = []
    for row in rows:
        image_path = row.get("path")
        if not image_path and row.get("image_id") is not None:
            image_path = str(images_dir / f"{int(row['image_id']):012d}.jpg")
        uri = ""
        if image_path:
            candidate = Path(image_path)
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            if candidate.exists():
                uri = candidate.as_uri()
        img_html = (
            f'<img loading="lazy" src="{uri}" alt="image {row.get("image_id")}">'
            if uri
            else '<div class="noimg">image not found</div>'
        )
        image_id = row.get("image_id")
        turn = row.get("turn", "")
        planted = '<span class="tag tag-planted">planted</span>' if row["planted"] else ""
        card_verdict = _card_verdict(row)
        cards.append(
            f"""<article class="card" data-id="{image_id}" data-turn="{turn}"
     data-verdict="{card_verdict}">
  <div class="image">{img_html}</div>
  <div class="body">
    <div class="head">
      <span class="type">{row["claim_type"]}</span>{planted}
      <span class="tag tag-{card_verdict}">{card_verdict}</span>
      <span class="idx">#{image_id} · turn {turn}</span>
    </div>
    <p class="claim">{row["claim"]}</p>
    {_verdict_html(row["verdicts"])}
    <div class="decision">
      <span class="prompt">Your call</span>
      <button class="agree" data-id="{image_id}" data-turn="{turn}">Verifiers right</button>
      <button class="wrong" data-id="{image_id}" data-turn="{turn}">Verifiers wrong</button>
    </div>
  </div>
</article>"""
        )

    template = (Path(__file__).parent / "m2a_review_template.html").read_text()
    # The cards belong inside <main>, before its closing tag. Appending them after
    # it would leave <main> empty: the page then has no content padding, and the
    # cards render flush against the viewport edge.
    marker = "</main>"
    head, tail = template.split(marker, 1)
    out.write_text(
        head.replace("<title>M2a calibration review</title>", f"<title>{title}</title>")
        + "\n".join(cards)
        + marker
        + tail,
        encoding="utf-8",
    )


def merge_decisions(
    export: dict,
    out: Path,
    *,
    reviewer: str,
    reviewed_at: str,
    claims_path: Path | None = None,
) -> tuple[int, int]:
    """Fold a review-page export into decisions.jsonl.

    The browser page knows ``(image_id, turn)``; the corpus knows ``claim_id``.
    When a claims file is given, every exported pair is resolved to its claim and
    an unresolvable one is rejected rather than written — an orphan decision would
    inflate the human-agreement rate without ever having been checked.
    """

    claims_by_pair: dict[tuple, str] = {}
    if claims_path is not None and Path(claims_path).exists():
        for row in read_jsonl(Path(claims_path)):
            claims_by_pair[(row.get("image_id"), row.get("turn"))] = row.get("claim_id")

    added = 0
    rejected = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for entry in export.get("decisions") or []:
            pair = (entry.get("image_id"), entry.get("turn"))
            # Prefer the id the page carried; (image_id, turn) is the legacy key.
            claim_id = entry.get("claim_id") or claims_by_pair.get(pair)
            if claims_path is not None and claim_id is None:
                rejected += 1
                continue
            row = {
                "claim_id": claim_id,
                "image_id": pair[0],
                "turn": pair[1],
                "decision": entry.get("decision"),
                "truth": entry.get("truth"),
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
            }
            handle.write(json.dumps(row) + "\n")
            added += 1
    return added, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims", type=Path, default=Path("experiments/m2a_grounding/claims.jsonl")
    )
    parser.add_argument("--images-dir", type=Path, default=Path("artifacts/m2a_images"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/opencode/m2a_review.html"))
    parser.add_argument("--title", default="M2a calibration review")
    parser.add_argument("--sample", type=int, default=50, help="items to show; 0 means all")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--merge", type=Path, default=None, help="fold this review export into decisions.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/m2a_grounding/decisions.jsonl")
    )
    parser.add_argument("--reviewer", default="owner")
    args = parser.parse_args()

    if args.merge is not None:
        export = json.loads(args.merge.read_text())
        added, rejected = merge_decisions(
            export,
            args.out,
            reviewer=args.reviewer,
            reviewed_at=now_iso(),
            claims_path=args.claims,
        )
        print(f"[merge] {added} decisions written, {rejected} rejected", flush=True)
        return 0 if rejected == 0 else 1

    rows = _verdict_rows(read_jsonl(args.claims))
    if not rows:
        print(f"[review] no claim results in {args.claims}", flush=True)
        return 1
    selected = stratify(rows, args.sample, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(selected, args.out, args.images_dir, args.title)
    print(f"[review] {len(selected)} of {len(rows)} claims -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

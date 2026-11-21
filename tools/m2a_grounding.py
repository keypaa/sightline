"""Build and score the M2a COCO grounding set.

Grounding asks whether the annotator's answers are true of the image, which needs an
independent answer key. COCO has captions, not question/answer labels, so the key is
built here: a vision model labels a fixed sample, the owner spot-checks it, and the
pilot scores against the confirmed subset.

The labeller is deliberately NOT the annotator (``DECISIONS/0048``). Agreement between
two different families is informative; agreement with itself would be circular.

Stages, each independently re-runnable:

    build   sample the images and write the review page
    score   score an annotation file against the labels
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ATTRIBUTES = ("colour", "shape", "count", "position")

# The question the labeller answers for each attribute. Kept next to the attribute so
# the label and the question that produced it cannot drift apart.
ATTRIBUTE_QUESTIONS = {
    "colour": "What is the main colour of the most prominent object?",
    "shape": "What is the shape of the most prominent object?",
    "count": "How many distinct objects of the same kind are visible?",
    "position": "Where is the most prominent object in the frame?",
}

DEFAULT_SAMPLE = 50
DEFAULT_SEED = 0

# Repo root, so a relative image path becomes a real file:// URL.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sample_ids(records: list[dict[str, Any]], n: int, seed: int) -> list[int]:
    """Pick n records reproducibly, by the same seeded sample the fetch uses."""

    if n <= 0 or n >= len(records):
        return [r["id"] for r in records]
    return [r["id"] for r in random.Random(seed).sample(list(records), n)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def blank_labels(records: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    """One unlabelled row per sampled image, ready for the labeller to fill."""

    by_id = {r["id"]: r for r in records}
    rows = []
    for image_id in ids:
        record = by_id[image_id]
        rows.append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "path": record["path"],
                "sha256": record["sha256"],
                "license_name": record.get("license_name"),
                "flickr_url": record.get("flickr_url"),
                "labels": {attribute: None for attribute in ATTRIBUTES},
                "human_confirmed": False,
            }
        )
    return rows


def label_agreement(row: dict[str, Any], predicted: dict[str, Any]) -> dict[str, Any]:
    """Compare one annotator answer map against a label row.

    Only attributes the label row actually has are scored, so an unanswered
    attribute is not counted as a miss. Returns per-attribute booleans plus the
    fraction over the attributes that both sides answered.
    """

    labels = row.get("labels") or {}
    scored = {}
    for attribute in ATTRIBUTES:
        expected = labels.get(attribute)
        if expected is None:
            continue
        got = predicted.get(attribute)
        scored[attribute] = (
            got is not None and str(got).strip().lower() == str(expected).strip().lower()
        )
    return {
        "id": row["id"],
        "human_confirmed": bool(row.get("human_confirmed")),
        "scored": scored,
        "n_scored": len(scored),
        "n_correct": sum(scored.values()),
    }


def summarise(rows: list[dict[str, Any]], predictions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Report labeller agreement and human-confirmed agreement separately.

    The two numbers are never merged: the 0.90 bar applies only to the
    human-confirmed subset, and labeller agreement is context.
    """

    per_row = [label_agreement(row, predictions.get(row["id"], {})) for row in rows]
    scored = [r for r in per_row if r["n_scored"] > 0]

    def rate(subset: list[dict[str, Any]]) -> float | None:
        total = sum(r["n_scored"] for r in subset)
        return (sum(r["n_correct"] for r in subset) / total) if total else None

    confirmed = [r for r in scored if r["human_confirmed"]]
    return {
        "n_images": len(rows),
        "n_labelled": sum(1 for row in rows if any((row.get("labels") or {}).values())),
        "n_scored_images": len(scored),
        "n_human_confirmed": len(confirmed),
        "labeller_agreement": rate(scored),
        "human_confirmed_agreement": rate(confirmed),
        "per_attribute": {
            attribute: {
                "n": sum(1 for r in scored if attribute in r["scored"]),
                "correct": sum(1 for r in scored if r["scored"].get(attribute)),
            }
            for attribute in ATTRIBUTES
        },
    }


def render_review_html(rows: list[dict[str, Any]], out: Path) -> None:
    """A single self-contained page for the owner's spot-check."""

    cards = []
    for row in rows:
        labels = row.get("labels") or {}
        rows_html = "".join(
            f"<tr><th>{attribute}</th><td>{labels.get(attribute) or '<i>unlabelled</i>'}</td></tr>"
            for attribute in ATTRIBUTES
        )
        checked_attr = "checked" if row.get("human_confirmed") else ""
        # A relative path is not a file URL: browsers read "file://artifacts/..." as
        # a host named "artifacts". Resolve against the repo root so the page can
        # load images from wherever it is opened.
        image_uri = (PROJECT_ROOT / row["path"]).resolve().as_uri()
        cards.append(
            f"""<article class="card" data-id="{row["id"]}">
  <img loading="lazy" src="{image_uri}" alt="{row["file_name"]}">
  <div class="body">
    <h2>{row["file_name"]}</h2>
    <p class="meta">id {row["id"]} &middot; sha256 {row["sha256"][:16]}&hellip;</p>
    <table>{rows_html}</table>
    <p class="lic">{row.get("license_name") or "unknown licence"}</p>
    <label class="check"><input type="checkbox" data-id="{row["id"]}"
      {checked_attr}> I checked this image and the labels are right</label>
  </div>
</article>"""
        )

    # The page template is a separate file so ruff never parses embedded CSS/JS.
    template = (Path(__file__).parent / "m2a_review_template.html").read_text()
    marker = "</main>"
    head, tail = template.split(marker, 1)
    out.write_text(head + marker + "\n".join(cards) + tail, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="sample images and write the review page")
    build.add_argument(
        "--fetched", type=Path, default=Path("experiments/m2a_iter20k/fetched.jsonl")
    )
    build.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding"))
    build.add_argument("--images", type=int, default=DEFAULT_SAMPLE)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--html", type=Path, default=Path("/tmp/opencode/m2a_grounding_review.html"))
    build.add_argument(
        "--force",
        action="store_true",
        help="overwrite a labels.jsonl that already holds labels (destroys them)",
    )

    score = sub.add_parser("score", help="score annotations against the labels")
    score.add_argument(
        "--labels", type=Path, default=Path("experiments/m2a_grounding/labels.jsonl")
    )
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--out", type=Path, default=Path("experiments/m2a_grounding/score.json"))

    args = parser.parse_args()
    args.seed = getattr(args, "seed", DEFAULT_SEED)

    if args.command == "build":
        records = read_jsonl(args.fetched)
        if not records:
            print(f"[build] no records in {args.fetched}", flush=True)
            return 1
        labels_path = args.out / "labels.jsonl"
        if labels_path.exists() and not args.force:
            existing = read_jsonl(labels_path)
            filled = sum(1 for r in existing if any((r.get("labels") or {}).values()))
            if filled:
                # Re-running build would silently erase real labels. Refuse, and
                # make the owner say --force.
                print(
                    f"[build] {labels_path} already holds {filled} labelled rows; "
                    "refusing to overwrite (use --force to discard them)",
                    flush=True,
                )
                return 1
        ids = sample_ids(records, args.images, args.seed)
        rows = blank_labels(records, ids)
        write_jsonl(labels_path, rows)
        (args.out / "sample.json").write_text(
            json.dumps({"seed": args.seed, "n": len(ids), "ids": ids}, indent=2)
        )
        args.html.parent.mkdir(parents=True, exist_ok=True)
        render_review_html(rows, args.html)
        print(f"[build] {len(rows)} images -> {args.out / 'labels.jsonl'}", flush=True)
        print(f"[build] review page -> {args.html}", flush=True)
        return 0

    labels = read_jsonl(args.labels)
    annotations = read_jsonl(args.annotations)
    predictions: dict[int, dict[str, Any]] = {}
    for row in annotations:
        # The annotator writes a conversation; the labeller writes an attribute map.
        # A row that already carries an attribute map is accepted directly, which is
        # what the pilot produces once it is wired to the labels.
        if isinstance(row.get("labels"), dict):
            predictions[row["id"]] = row["labels"]
    summary = summarise(labels, predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

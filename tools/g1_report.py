#!/usr/bin/env python3
"""Render reports/g1.md from the M1b run records.

Reads the ``g1`` event in ``experiments/m1b_train/seed<seed>/metrics.jsonl`` for
each seed and applies the frozen G1 thresholds from ``DECISIONS/0032``. Exit
code 0 when G1 passes, 1 otherwise (including when a seed is missing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sightline.report_g1 import build_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/m1b_train"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path("reports/g1.md"))
    parser.add_argument(
        "--parser",
        default="exact",
        help="which answer parser's events to report; exact is the frozen G1 rule",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="which evaluation split to report; a seed without one is marked missing",
    )
    args = parser.parse_args()

    markdown, summary = build_report(args.root, args.seeds, args.split, args.parser)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {args.out}")
    return 0 if summary["g1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

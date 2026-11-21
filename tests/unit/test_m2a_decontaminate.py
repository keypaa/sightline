"""Tests for SSCD decontamination against MMMU-Pro and BLINK.

The SSCD model is never loaded here: similarity is always a stubbed callable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from sightline.m2a.decontaminate import (
    REQUIRED_BENCHMARKS,
    SSCD_THRESHOLD,
    find_contamination,
    load_benchmark_hashes,
    write_report,
)


def test_exact_hash_match_is_contamination():
    records = [{"id": 0, "sha256": "aa", "path": "/x/0.jpg"}]
    hits = find_contamination(records, {"MMMU-Pro": {"aa"}})
    assert hits == [{"id": 0, "benchmark": "MMMU-Pro", "sha256": "aa", "similarity": 1.0}]


def test_report_counts_per_benchmark(tmp_path: Path) -> None:
    rows = [
        {"id": 0, "benchmark": "MMMU-Pro", "sha256": "aa", "similarity": 1.0},
        {"id": 1, "benchmark": "BLINK", "sha256": "bb", "similarity": 1.0},
    ]
    summary = write_report(tmp_path / "r.jsonl", rows, n_records=10)
    assert summary["n_contaminated"] == 2
    assert summary["per_benchmark"] == {"MMMU-Pro": 1, "BLINK": 1}


def test_report_names_both_benchmarks_even_when_clean(tmp_path: Path) -> None:
    summary = write_report(tmp_path / "r.jsonl", [], n_records=5)
    assert summary["per_benchmark"] == {}
    assert summary["n_contaminated"] == 0


def test_report_names_every_checked_benchmark_even_when_clean(tmp_path: Path) -> None:
    """A zero count must still show that both benchmarks were checked."""

    summary = write_report(
        tmp_path / "r.jsonl",
        [],
        n_records=5,
        checked_benchmarks=list(REQUIRED_BENCHMARKS),
    )
    assert summary["per_benchmark"] == {"MMMU-Pro": 0, "BLINK": 0}
    assert summary["n_contaminated"] == 0


def test_write_report_writes_the_rows_as_jsonl(tmp_path: Path) -> None:
    rows = [{"id": 0, "benchmark": "BLINK", "sha256": "bb", "similarity": 0.97}]
    write_report(tmp_path / "r.jsonl", rows, n_records=3)
    written = [json.loads(line) for line in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert written == rows


def test_load_benchmark_hashes_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"MMMU-Pro": ["aa", "bb"], "BLINK": ["cc"]}))
    assert load_benchmark_hashes(path) == {
        "MMMU-Pro": {"aa", "bb"},
        "BLINK": {"cc"},
    }


def test_load_benchmark_hashes_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "b.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"MMMU-Pro": ["aa"]}),
                json.dumps({"BLINK": ["bb", "cc"]}),
            ]
        )
        + "\n"
    )
    assert load_benchmark_hashes(path) == {
        "MMMU-Pro": {"aa"},
        "BLINK": {"bb", "cc"},
    }


def test_similarity_match_is_contamination(tmp_path: Path) -> None:
    record_path = tmp_path / "rec.jpg"
    bench_path = tmp_path / "bench.jpg"
    seen: list[tuple[Path, Path]] = []

    def stub(a: Path, b: Path) -> float:
        seen.append((a, b))
        return 0.97

    hits = find_contamination(
        [{"id": 1, "sha256": "aa", "path": str(record_path)}],
        {"BLINK": {"bb"}},
        sscd_similarity=stub,
        benchmark_images={"bb": bench_path},
    )
    assert hits == [{"id": 1, "benchmark": "BLINK", "sha256": "aa", "similarity": 0.97}]
    assert seen == [(record_path, bench_path)]


def test_similarity_below_threshold_is_not_contamination(tmp_path: Path) -> None:
    hits = find_contamination(
        [{"id": 1, "sha256": "aa", "path": str(tmp_path / "rec.jpg")}],
        {"BLINK": {"bb"}},
        sscd_similarity=lambda a, b: SSCD_THRESHOLD - 0.01,
        benchmark_images={"bb": tmp_path / "bench.jpg"},
    )
    assert hits == []


def test_similarity_without_benchmark_images_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="benchmark_images"):
        find_contamination(
            [{"id": 1, "sha256": "aa", "path": str(tmp_path / "rec.jpg")}],
            {"BLINK": {"bb"}},
            sscd_similarity=lambda a, b: 0.99,
        )


def _benchmark_manifest(tmp_path: Path, table: dict[str, list[str]]) -> Path:
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(table))
    return path


def test_cmd_decontaminate_reports_exact_hits_without_deleting_rows(tmp_path: Path) -> None:
    """Contamination is a report, not an edit to the corpus."""

    from tools import build_m2a

    root = tmp_path / "run"
    root.mkdir()
    records = [
        {"id": 0, "sha256": "aa", "path": "/x/0.jpg"},
        {"id": 1, "sha256": "bb", "path": "/x/1.jpg"},
    ]
    build_m2a.write_jsonl(root / "fetched.jsonl", records)
    manifest = _benchmark_manifest(tmp_path, {"MMMU-Pro": ["aa"], "BLINK": ["zz"]})
    args = argparse.Namespace(
        root=root,
        benchmark_manifest=str(manifest),
        sscd_callable=None,
        similarity=False,
        benchmark_images_dir=None,
        sscd_threshold=SSCD_THRESHOLD,
    )

    assert build_m2a.cmd_decontaminate(args) == 0
    assert build_m2a.read_jsonl(root / "decontamination.jsonl") == [
        {"id": 0, "benchmark": "MMMU-Pro", "sha256": "aa", "similarity": 1.0}
    ]
    summary = json.loads((root / "decontamination_report.json").read_text())
    assert summary == {
        "n_records": 2,
        "n_contaminated": 1,
        "per_benchmark": {"MMMU-Pro": 1, "BLINK": 0},
    }
    assert not (root / "conversations.jsonl").exists()


def test_cmd_decontaminate_names_both_benchmarks_when_clean(tmp_path: Path) -> None:
    from tools import build_m2a

    root = tmp_path / "run"
    root.mkdir()
    build_m2a.write_jsonl(root / "fetched.jsonl", [{"id": 0, "sha256": "aa", "path": "/x/0.jpg"}])
    manifest = _benchmark_manifest(tmp_path, {"MMMU-Pro": ["mm"], "BLINK": ["bl"]})
    args = argparse.Namespace(
        root=root,
        benchmark_manifest=str(manifest),
        sscd_callable=None,
        similarity=False,
        benchmark_images_dir=None,
        sscd_threshold=SSCD_THRESHOLD,
    )

    assert build_m2a.cmd_decontaminate(args) == 0
    summary = json.loads((root / "decontamination_report.json").read_text())
    assert summary["n_contaminated"] == 0
    assert summary["per_benchmark"] == {"MMMU-Pro": 0, "BLINK": 0}
    assert (root / "decontamination.jsonl").read_text() == ""


def test_cmd_decontaminate_refuses_manifest_missing_a_benchmark(tmp_path: Path) -> None:
    from tools import build_m2a

    root = tmp_path / "run"
    root.mkdir()
    manifest = _benchmark_manifest(tmp_path, {"MMMU-Pro": ["mm"]})
    args = argparse.Namespace(
        root=root,
        benchmark_manifest=str(manifest),
        sscd_callable=None,
        similarity=False,
        benchmark_images_dir=None,
        sscd_threshold=SSCD_THRESHOLD,
    )

    with pytest.raises(ValueError, match="BLINK"):
        build_m2a.cmd_decontaminate(args)


def test_similarity_without_callable_fails_loudly(tmp_path: Path, monkeypatch, capsys) -> None:
    """A requested similarity mode must never silently fall back to exact-only."""

    from tools import build_m2a

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_m2a.py",
            "--root",
            str(tmp_path),
            "decontaminate",
            "--benchmark-manifest",
            str(tmp_path / "benchmarks.json"),
            "--similarity",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        build_m2a.main()
    assert excinfo.value.code == 2
    assert "--sscd-callable" in capsys.readouterr().err

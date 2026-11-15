from __future__ import annotations

from sightline.progress import ProgressPrinter, format_duration


def test_format_duration_is_clock_shaped() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(9) == "0:09"
    assert format_duration(90) == "1:30"
    assert format_duration(3661) == "1:01:01"


def test_printer_runs_every_n_steps_and_on_the_last_one() -> None:
    lines: list[str] = []

    printer = ProgressPrinter(total=10, every=4, stream=lines.append)
    for step in range(1, 11):
        printer({"step": step, "loss": 2.0, "step_s": 0.5, "median_step_s": 0.5})

    assert len(lines) == 3
    assert "4/10" in lines[0]
    assert "8/10" in lines[1]
    assert "10/10" in lines[2]


def test_printer_reports_eta_from_the_median_step_time() -> None:
    lines: list[str] = []

    printer = ProgressPrinter(total=10, every=2, stream=lines.append)
    printer({"step": 2, "loss": 1.0, "step_s": 0.4, "median_step_s": 0.5})

    line = lines[0]
    assert "0:04" in line  # 8 remaining x 0.5 s
    assert "0:01" in line  # 2 elapsed x 0.5 s
    assert "20%" in line
    assert "loss 1.0000" in line


def test_printer_never_prints_when_every_is_zero() -> None:
    lines: list[str] = []

    printer = ProgressPrinter(total=10, every=0, stream=lines.append)
    for step in range(1, 11):
        printer({"step": step, "loss": 1.0, "step_s": 0.1, "median_step_s": 0.1})

    assert lines == []


def test_printer_marks_the_run_as_complete_on_the_final_step() -> None:
    lines: list[str] = []

    printer = ProgressPrinter(total=3, every=10, stream=lines.append)
    for step in range(1, 4):
        printer({"step": step, "loss": 1.0, "step_s": 0.1, "median_step_s": 0.1})

    assert len(lines) == 1
    assert "3/3" in lines[0]


def test_printer_prefixes_the_label() -> None:
    lines: list[str] = []

    printer = ProgressPrinter(total=2, every=1, stream=lines.append, label="[seed 0]")
    printer({"step": 1, "loss": 1.0, "step_s": 0.1, "median_step_s": 0.1})

    assert lines[0].startswith("[seed 0]")

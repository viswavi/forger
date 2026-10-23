"""Saved pipeline runs: inspect.runs."""

from __future__ import annotations

import pytest

from pyforger import _tracking, filter, inspect
from pyforger.steps import pipeline, saved_runs, step

SEEDS = [{"id": f"p{i:03d}", "text": " ".join(["w"] * (i % 9 + 1))} for i in range(60)]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FORGER_SPEND_LEDGER", "0")
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()


@step
def drop_short(rows: list[dict]) -> list[dict]:
    """Rows of 2+ words."""
    return filter.by_length(rows, instruction_field="text", min_len=2)


@step
def keep_long(rows: list[dict], min_len: int) -> list[dict]:
    """Rows of min_len+ words."""
    return filter.by_length(rows, instruction_field="text", min_len=min_len)


@pipeline
def main_a():
    """Variant A: 5+ words."""
    return keep_long(drop_short(SEEDS), 5)


@pipeline
def main_b():
    """Variant B: 3+ words."""
    return keep_long(drop_short(SEEDS), 3)


def test_each_run_is_saved(capsys):
    """Runs never overwrite each other."""
    main_a()
    main_b()
    saved = saved_runs()
    assert [r.name for r in saved] == ["main_a", "main_b"]
    assert saved[0].run_id != saved[1].run_id
    assert [s.name for s in saved[0].steps] == ["drop_short", "keep_long"]
    assert [r.name for r in inspect.runs()] == ["main_a", "main_b"]
    assert "main_a-" in capsys.readouterr().out


def test_runs_lists_step_yields_and_the_last_score(capsys):
    """Each run's second line: yields of steps that dropped rows, the score mean."""
    main_a()
    capsys.readouterr()
    inspect.runs()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("  main_a-")
    assert "drop_short 0.88 (0.78–0.94) · keep_long 0.60 (" in lines[1]
    assert "mean filter.by_length words" in lines[1]

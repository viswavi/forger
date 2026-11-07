"""Saved pipeline runs: inspect.runs and inspect.compare_pipelines."""

from __future__ import annotations

import json

import pytest

from pyforger import _tracking, filter, inspect
from pyforger.op import _cache_dir
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
    """Runs never overwrite each other; shared steps have the same key."""
    main_a()
    main_b()
    saved = saved_runs()
    assert [r.name for r in saved] == ["main_a", "main_b"]
    assert saved[0].run_id != saved[1].run_id
    assert [s.name for s in saved[0].steps] == ["drop_short", "keep_long"]
    first_a, first_b = saved[0].steps[0], saved[1].steps[0]
    assert first_a.cache_key and first_a.cache_key == first_b.cache_key
    assert "argument min_len" in saved[0].steps[1].key_parts
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


def test_compare_names_the_first_difference_and_matches_rows(capsys):
    """The step whose argument changed is the first difference; rows match by id."""
    main_a()
    main_b()
    result = inspect.compare_pipelines("main_a", "main_b")
    assert result["first_difference"] == "keep_long"
    same, differs = result["steps"]
    assert same["same"] and not differs["same"]
    assert differs["why"] == ["argument min_len differs"]
    rows = differs["rows"]
    assert rows["matched_by"] == "id"
    assert len(rows["only_a"]) == 0 and len(rows["only_b"]) > 0
    printed = capsys.readouterr().out
    assert "first difference: keep_long" in printed
    assert "only in B" in printed


def test_compare_says_whether_a_difference_is_noise(capsys):
    """Op yields get a difference interval and the rows only one run kept a sign test.

    keep_long keeps 32 of 53 rows in A and 46 of 53 in B; the 14 rows only B
    kept against 0 only in A are clearly not a coin's split.
    """
    main_a()
    main_b()
    result = inspect.compare_pipelines("main_a", "main_b")
    differs = result["steps"][1]
    op = differs["ops"]["filter.by_length"]
    assert op["a"] == (53, 32) and op["b"] == (53, 46)
    assert op["b_minus_a"]["clear"] and op["b_minus_a"]["ci"][0] > 0
    assert differs["rows"]["sign_test_p"] < 0.001
    assert differs["scores"]["filter.by_length words"]["clearly_different"] is False
    printed = capsys.readouterr().out
    assert "B − A +0.26 (95% CI" in printed and "clearly different" in printed
    assert "0 vs 14: clearly different (B keeps more of them; sign test p < 0.001)" in (
        printed
    )


def test_compare_defaults_to_the_last_two_runs():
    """With no arguments, B is the latest run and A the run before it."""
    main_a()
    main_b()
    result = inspect.compare_pipelines()
    assert result["a"].startswith("main_a-") and result["b"].startswith("main_b-")
    with pytest.raises(KeyError, match="no saved run"):
        inspect.compare_pipelines("main_a", "nope")


@step
def shout(rows: list[dict], suffix: str) -> list[dict]:
    """Append suffix to every text."""
    rows = [{**r, "text": r["text"] + suffix} for r in rows]
    return filter.by_length(rows, instruction_field="text", min_len=0)


def test_compare_shows_rows_whose_values_changed(capsys):
    """Rows kept by both runs but rewritten differently are counted by field."""

    @pipeline
    def quiet():
        """Suffix '.'."""
        return shout(SEEDS, ".")

    @pipeline
    def loud():
        """Suffix '!'."""
        return shout(SEEDS, "!")

    quiet()
    loud()
    result = inspect.compare_pipelines("quiet", "loud")
    rows = result["steps"][0]["rows"]
    assert rows["changed"] == {"text": len(SEEDS)}
    assert "rows in both whose values differ: text in 60" in capsys.readouterr().out


def test_runs_saved_without_keys_still_load_and_compare(capsys):
    """A run file from before key parts were recorded loads; its steps say so.

    No step is named the first difference, and rows are not compared.
    """
    main_a()
    main_b()
    for path in (_cache_dir() / "pipeline_runs").glob("main_a-*.json"):
        saved = json.loads(path.read_text())
        for s in saved["steps"]:
            del s["cache_key"], s["key_parts"]
        path.write_text(json.dumps(saved))
    old = [r for r in saved_runs() if r.name == "main_a"][0]
    assert old.steps[0].cache_key is None and old.steps[0].key_parts == {}
    result = inspect.compare_pipelines("main_a", "main_b")
    assert result["first_difference"] is None
    assert [s["why"] for s in result["steps"]] == [
        ["no key recorded (an older run)"]
    ] * 2
    assert result["steps"][1]["rows"] is None
    printed = capsys.readouterr().out
    assert "filter.by_length: A 32/53 (0.60)  B 46/53 (0.87)" in printed
    assert "rows: not in the step cache" in printed

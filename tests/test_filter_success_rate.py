"""Passive difficulty selection over already evaluated samples."""
from __future__ import annotations

import pytest

from pyforger.filter import by_success_rate
from pyforger.inspect import success_rates


def test_inclusive_bounds_and_reports_for_rejected_rows(tmp_path, monkeypatch):
    """Compute reports without executing work or losing rejected evidence."""
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    rows = [
        {"id": "mixed", "evaluations": {"target": [True, False, False, False, False]}},
        {"id": "easy", "evaluations": {"target": [True] * 5}},
    ]
    assert by_success_rate(rows) == rows[:1]
    reports = success_rates(rows)
    assert reports[0]["target"]["success_rate"] == 0.2
    assert reports[1]["target"]["success_rate"] == 1
    assert rows[1]["id"] == "easy"


def test_exclusions_are_explicit():
    """Missing grades do not become failures or silently shrink the panel."""
    rows = [{"evaluations": {"target": [True, False, None]}}]
    assert by_success_rate(rows) == []
    assert by_success_rate(rows, min_valid=2) == rows
    assert success_rates(rows)[0]["target"]["excluded"] == 1


@pytest.mark.parametrize("outcomes", [{}, {"a": []}, {"a": [None]}])
def test_empty_measurements_do_not_pass(outcomes):
    """An undefined rate never passes selection."""
    assert by_success_rate([{"evaluations": outcomes}]) == []


def test_groups_are_not_pooled():
    """One population cannot hide another population's failure rate."""
    rows = [{"grades": {"a": [True], "b": [False]}}]
    assert by_success_rate(rows, outcomes_field="grades", max_rate=1) == []
    assert success_rates(rows, outcomes_field="grades")[0]["a"]["passed"] == 1


@pytest.mark.parametrize("value", [1, "true", 0.0, {}, []])
def test_only_boolean_or_missing_grades(value):
    """Malformed grader output must not enter the denominator."""
    with pytest.raises(ValueError):
        by_success_rate([{"evaluations": {"a": [value]}}])


@pytest.mark.parametrize(
    "bounds",
    [
        {"min_rate": -1.0},
        {"max_rate": 2.0},
        {"min_rate": 0.9, "max_rate": 0.1},
        {"min_valid": 0},
        {"min_rate": float("nan")},
    ],
)
def test_invalid_bounds(bounds):
    """Selection validates policy independently of execution."""
    with pytest.raises(ValueError):
        by_success_rate([], **bounds)

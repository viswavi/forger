"""A full run of pipeline code refuses to start until a limited run of it finished."""

from __future__ import annotations

from typing import Any

import pytest

from pyforger import _tracking, filter, op
from pyforger.steps import UntestedFullRunError, pipeline, step

PROMPT = "first prompt"


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch):
    """Turn the check on (conftest turns it off) and silence reports."""
    monkeypatch.delenv("FORGER_FULL_RUN", raising=False)
    monkeypatch.delenv("FORGER_LIMIT", raising=False)
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


@op(name="alter.test_full_run_tag")
def _tag(rows: list[Any]) -> list[Any]:
    return [{**r, "prompt": PROMPT} for r in rows]


@step
def tag(rows: list[dict]) -> list[dict]:
    """Tag rows."""
    return filter.by_length(_tag(rows), instruction_field="text", min_len=1)


@pipeline
def main(rows: list[dict]) -> list[dict]:
    """One step."""
    return tag(rows)


ROWS = [{"text": f"row {i}"} for i in range(5)]


def test_full_run_refused_until_a_limited_run_finished(monkeypatch):
    """The first full run raises; after a finished limited run it starts."""
    with pytest.raises(UntestedFullRunError, match="FORGER_LIMIT=3"):
        main(ROWS)
    monkeypatch.setenv("FORGER_LIMIT", "3")
    assert len(main(ROWS)) == 3
    monkeypatch.delenv("FORGER_LIMIT")
    assert len(main(ROWS)) == 5


def test_an_edit_to_a_prompt_needs_a_new_limited_run(monkeypatch):
    """Changing a global a step reads makes the code new."""
    monkeypatch.setenv("FORGER_LIMIT", "3")
    main(ROWS)
    monkeypatch.delenv("FORGER_LIMIT")
    main(ROWS)
    monkeypatch.setitem(globals(), "PROMPT", "second prompt")
    with pytest.raises(UntestedFullRunError):
        main(ROWS)


def test_a_failed_limited_run_does_not_count(monkeypatch):
    """A limited run that raises leaves the full run refused."""

    @pipeline(name="breaks")
    def breaks(rows: list[dict]) -> list[dict]:
        tag(rows)
        raise ValueError("bug")

    monkeypatch.setenv("FORGER_LIMIT", "3")
    with pytest.raises(ValueError):
        breaks(ROWS)
    monkeypatch.delenv("FORGER_LIMIT")
    with pytest.raises(UntestedFullRunError):
        breaks(ROWS)


def test_full_run_override(monkeypatch):
    """FORGER_FULL_RUN=1 skips the check."""
    monkeypatch.setenv("FORGER_FULL_RUN", "1")
    assert len(main(ROWS)) == 5

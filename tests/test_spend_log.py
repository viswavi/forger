"""Live spend lines: "Spend so far" at $1, $2, $5, ..., and batch spend."""

import logging

import pytest

from pyforger import _tracking
from pyforger.acquire import model as model_mod


@pytest.fixture
def spend_lines(caplog):
    """The messages logged to the Spend logger so far."""
    caplog.set_level(logging.INFO, logger="Spend")
    return lambda: [r.getMessage() for r in caplog.records if r.name == "Spend"]


def _scope(kind, name):
    usage, token = _tracking._open_usage(kind, name)
    return usage, token


def test_next_mark():
    """Marks run 1, 2, 5, 10, 20, 50, ..."""
    assert [_tracking._next_mark(x) for x in (0, 1, 1.5, 2, 4.9, 5, 9.99, 10, 37)] == [
        1,
        2,
        2,
        5,
        5,
        10,
        10,
        20,
        50,
    ]


def test_step_logs_at_marks_with_op_and_pipeline(spend_lines, monkeypatch):
    """A step logs at $1, $2, $5 with its op and the pipeline total."""
    monkeypatch.delenv("FORGER_SPEND_LOG", raising=False)
    tokens = []
    for kind, name in (
        ("pipeline", "main"),
        ("step", "balance"),
        ("op", "filter.by_topic_frequency"),
    ):
        tokens.append(_scope(kind, name)[1])
    try:
        for _ in range(12):  # $0.60 a call: passes $1, $2, $5 (and not $10)
            _tracking.add_usage(100, 10, 0.6, 1000.0)
    finally:
        for token in reversed(tokens):
            _tracking._ACCUMULATORS.reset(token)
    lines = spend_lines()
    assert len(lines) == 3
    assert lines[0].startswith(
        "Spend so far: step balance $1.20 (2,000 token units, 2 calls, "
    )
    assert "; now in filter.by_topic_frequency, $1.20 (" in lines[0]
    assert lines[0].endswith("; pipeline main $1.20 in all. Next line at $2.")
    assert "$2.40" in lines[1] and lines[1].endswith("Next line at $5.")
    assert "$5.40" in lines[2] and lines[2].endswith("Next line at $10.")


def test_op_outside_step_and_unpriced_units(spend_lines, monkeypatch):
    """An op outside a step is watched; unpriced spend counts token units."""
    monkeypatch.delenv("FORGER_SPEND_LOG", raising=False)
    _, token = _scope("op", "alter.generate_from_context")
    try:
        for _ in range(3):
            _tracking.add_usage(100, 10, 0.0, 600_000.0)
    finally:
        _tracking._ACCUMULATORS.reset(token)
    lines = spend_lines()
    assert len(lines) == 1
    assert lines[0].startswith(
        "Spend so far: alter.generate_from_context $0.00 "
        "(1,200,000 token units, 2 calls"
    )
    assert lines[0].endswith("Next line at 2,000,000 token units.")


def test_no_line_without_scope_or_when_off(spend_lines, monkeypatch):
    """No line without a step or op, or with FORGER_SPEND_LOG=0."""
    _tracking.add_usage(100, 10, 5.0, 0.0)  # no step or op open
    monkeypatch.setenv("FORGER_SPEND_LOG", "0")
    _, token = _scope("step", "s")
    try:
        _tracking.add_usage(100, 10, 5.0, 0.0)
    finally:
        _tracking._ACCUMULATORS.reset(token)
    assert spend_lines() == []


def test_batch_spend_text():
    """Progress lines get the batch's spend so far and projected total."""

    class FakeModel:
        _token_budget = None

    projection = model_mod._SpendProjection(FakeModel(), total=400)
    token = model_mod._PROJECTION.set(projection)
    try:
        assert model_mod._batch_spend() == ""
        projection.add(100, 0.42, 0.0)
        assert model_mod._batch_spend() == "; $0.42 so far, ~$1.68 for the batch"
        projection.add(300, 1.0, 0.0)
        assert model_mod._batch_spend() == "; $1.42 for the batch"
    finally:
        model_mod._PROJECTION.reset(token)
    assert model_mod._batch_spend() == ""

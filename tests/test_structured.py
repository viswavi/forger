"""Real callback paths for bounded validated generation; no mocks required."""

import pytest

from pyforger.measurements import Measurement
from pyforger.structured import (
    GenerationInconclusive,
    ValidationExhausted,
    generate_validated,
)


def test_revision_and_history():
    """Revision receives the failed output and produces the next request."""
    result = generate_validated(
        1,
        generate=lambda request: request * 2,
        validate=lambda output: Measurement(passed=output == 4, reason="need four"),
        revise=lambda request, attempt: request + (attempt.output // 2),
    )
    assert result.output == 4
    assert [a.output for a in result.attempts] == [2, 4]
    assert result.attempts[0].measurement.reason == "need four"


def test_exhaustion_retains_outputs():
    """Unrevised requests retry up to the budget and retain every verdict."""
    with pytest.raises(ValidationExhausted) as caught:
        generate_validated(
            "x", generate=str, validate=lambda _: Measurement(passed=False)
        )
    error = caught.value
    assert len(error.attempts) == 3
    assert error.output == error.failure.output == "x"
    assert error.failure.measurement.passed is False


def test_inconclusive_stops():
    """An ungraded result stops immediately rather than consuming repairs."""
    with pytest.raises(GenerationInconclusive) as caught:
        generate_validated(
            None, generate=lambda _: None, validate=lambda _: Measurement()
        )
    assert len(caught.value.attempts) == 1
    assert caught.value.output is None


@pytest.mark.parametrize("stage", ["generate", "validate", "revise"])
def test_callback_errors_propagate(stage):
    """Execution and validation callback exceptions are never reclassified."""

    def fail(*args):
        raise OSError("infrastructure")

    callbacks = dict(
        generate=str, validate=lambda _: Measurement(passed=False), revise=fail
    )
    callbacks[stage] = fail
    with pytest.raises(OSError, match="infrastructure"):
        generate_validated("x", **callbacks)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_bad_budget(budget):
    """Invalid budgets fail before generation."""
    with pytest.raises(ValueError):
        generate_validated(
            "x", generate=str, validate=lambda _: Measurement(), max_attempts=budget
        )

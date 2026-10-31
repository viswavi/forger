"""Real callback paths for bounded validated generation; no mocks required."""

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from pyforger.measurements import Measurement
from pyforger.structured import (
    GenerationInconclusive,
    GenerationResult,
    ValidationExhausted,
    generate_result,
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


def test_generate_result_returns_outcomes_as_data():
    """Accepted, exhausted, and inconclusive outcomes return without raising."""
    accepted = generate_result(
        1,
        generate=lambda request: request * 2,
        validate=lambda output: Measurement(passed=output == 4, reason="need four"),
        revise=lambda request, attempt: request + (attempt.output // 2),
    )
    assert isinstance(accepted, GenerationResult)
    assert accepted.status == "accepted" and accepted.accepted is True
    assert accepted.output == 4
    assert [a.output for a in accepted.attempts] == [2, 4]

    exhausted = generate_result(
        "x", generate=str, validate=lambda _: Measurement(passed=False)
    )
    assert exhausted.status == "exhausted" and exhausted.accepted is False
    assert exhausted.output == "x" and len(exhausted.attempts) == 3
    assert exhausted.measurement.passed is False

    inconclusive = generate_result(
        None, generate=lambda _: None, validate=lambda _: Measurement()
    )
    assert inconclusive.status == "inconclusive" and inconclusive.accepted is False
    assert len(inconclusive.attempts) == 1


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_generate_result_bad_budget(budget):
    """Invalid budgets fail before generation, matching generate_validated."""
    with pytest.raises(ValueError):
        generate_result(
            "x", generate=str, validate=lambda _: Measurement(), max_attempts=budget
        )


class _TrialRow(BaseModel):
    """A typed panel-adjacent row carrying graded outcomes."""

    name: str
    evaluations: dict[str, list[bool | None]] = {}
    samples: list[dict] = []


@dataclass
class _DataclassRow:
    """A dataclass row carrying graded outcomes."""

    name: str
    evaluations: dict[str, list[bool | None]] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)


def test_model_rows_flow_through_sample_inspect_and_filter():
    """Dict-shaped primitives accept pydantic and dataclass rows unchanged."""
    from pyforger.alter import sample
    from pyforger.filter import by_success_rate
    from pyforger.inspect import success_rates

    for row in (
        _TrialRow(name="pydantic"),
        _DataclassRow(name="dataclass"),
    ):
        sampled = sample.no_cache(
            [row],
            lambda item, identity: item.name + "-" + identity,
            sample_ids=["s1"],
        )
        assert sampled[0].samples[0]["value"] == row.name + "-s1"
        graded = type(row)(
            name=row.name, evaluations={"m": [True, False, None]}, samples=[]
        )
        assert success_rates.no_cache([graded])[0]["m"]["valid"] == 2
        assert by_success_rate.no_cache([graded], min_valid=2) == [graded]
        assert by_success_rate.no_cache([graded]) == []

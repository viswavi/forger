"""Evaluation attribution tests use real callback and artifact code paths."""
from __future__ import annotations

import pytest

from pyforger.evaluation import (
    AttemptOutcome,
    InfrastructureFailure,
    InvalidSubmission,
    TaskDefect,
    apply_checks,
    evaluate_submission,
)
from pyforger.trials import EvaluationResult


def test_success_failure_and_roundtrip():
    """Grade model answers and persist explicit attribution."""
    for passed in (True, False):
        outcome = evaluate_submission(
            lambda: "answer", lambda x: None, lambda x: passed
        )
        assert outcome.counted and outcome.passed is passed
        result = EvaluationResult(
            passed=passed, attribution=outcome.attribution, reason=outcome.reason
        )
        assert EvaluationResult.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    "error,attribution,passed",
    [
        (InvalidSubmission, "model", False),
        (TaskDefect, "task", None),
        (InfrastructureFailure, "infrastructure", None),
        (RuntimeError, "unknown", None),
    ],
)
def test_stage_owned_failure_attribution(error, attribution, passed):
    """Only declared submission errors become model failures."""

    def extract():
        raise error("fixture")

    outcome = evaluate_submission(extract, lambda x: None, lambda x: True)
    assert outcome.attribution == attribution and outcome.passed is passed


def test_broken_grader_does_not_count_as_wrong_answer():
    """Unexpected grading errors and non-boolean grades remain unknown."""
    assert (
        evaluate_submission(
            lambda: "answer", lambda x: None, lambda x: "yes"
        ).attribution
        == "unknown"
    )

    def grade(value):
        raise ValueError("grader bug")

    assert (
        evaluate_submission(lambda: "answer", lambda x: None, grade).attribution
        == "unknown"
    )


def test_checks_preserve_task_and_infrastructure_failures():
    """Failed audit/replay services take precedence over model-check failures."""
    passed = AttemptOutcome(True, "model", "correct")
    cheating = AttemptOutcome(False, "model", "reward_hacking")
    broken = AttemptOutcome(None, "infrastructure", "replay_start_failed")
    task = AttemptOutcome(None, "task", "underspecified")
    assert apply_checks(passed, [cheating]) == cheating
    assert apply_checks(passed, [cheating, broken]) == broken
    assert apply_checks(passed, [broken, task]) == task
    assert apply_checks(cheating, [broken]) == broken


@pytest.mark.parametrize(
    "passed,attribution",
    [(None, "model"), (False, "infrastructure"), (True, "task"), (None, "bogus")],
)
def test_invalid_contract_rejected(passed, attribution):
    """Ambiguous grades cannot enter success-rate measurements."""
    with pytest.raises(ValueError):
        AttemptOutcome(passed, attribution, "fixture")

"""Explicit attempt attribution and composable submission evaluation."""
from __future__ import annotations

from dataclasses import dataclass


class InvalidSubmission(ValueError):
    """The model did not satisfy the declared submission contract."""


class TaskDefect(ValueError):
    """The task contract is invalid or cannot be graded fairly."""


class InfrastructureFailure(RuntimeError):
    """A service or artifact transport failed independently of the model."""


@dataclass(frozen=True)
class AttemptOutcome:
    """An explicit grade and attribution, independent of execution status."""

    passed: bool | None
    attribution: str
    reason: str

    def __post_init__(self):
        """Reject ambiguous grades and unsupported attribution values."""
        if self.attribution not in {"model", "infrastructure", "task", "unknown"}:
            raise ValueError("Unsupported attempt attribution")
        if self.attribution == "model" and type(self.passed) is not bool:
            raise ValueError("Model outcomes require a boolean grade")
        if self.attribution != "model" and self.passed is not None:
            raise ValueError("Ungradable outcomes cannot carry a model grade")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("An attribution reason is required")

    @property
    def counted(self):
        """Only attributable model outcomes enter empirical accuracy."""
        return self.attribution == "model"

    def to_dict(self):
        """Serialize the explicit outcome contract."""
        return {
            "passed": self.passed,
            "attribution": self.attribution,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value):
        """Validate a retained outcome without guessing attribution."""
        return cls(value["passed"], value["attribution"], value["reason"])


def apply_checks(outcome, checks):
    """Refine a claimed success while preserving the most serious attribution."""
    if not outcome.counted:
        return outcome
    checks = list(checks)
    for attribution in ("task", "infrastructure", "unknown"):
        for check in checks:
            if check.attribution == attribution:
                return check
    if outcome.passed is False:
        return outcome
    for check in checks:
        if check.passed is False:
            return check
    return outcome


def evaluate_submission(extract, validate, grade, *, checks=()):
    """Extract, validate and grade with explicit stage-owned failure attribution.

    Callbacks raise InvalidSubmission for missing/schema-invalid model output,
    TaskDefect for defective task contracts, or InfrastructureFailure for broken
    transport/services. Unexpected callback exceptions are unknown evaluation
    errors, not model failures. A grader must return bool; optional checks return
    AttemptOutcome and refine successes (e.g. audit and replay).
    """
    try:
        submission = extract()
        validate(submission)
        passed = grade(submission)
        if type(passed) is not bool:
            raise TypeError("Grader must return bool")
        outcome = AttemptOutcome(
            passed, "model", "correct" if passed else "incorrect_answer"
        )
        if passed:
            outcomes = [check(submission) for check in checks]
            if any(not isinstance(check, AttemptOutcome) for check in outcomes):
                raise TypeError("Checks must return AttemptOutcome")
            outcome = apply_checks(outcome, outcomes)
        return outcome
    except InvalidSubmission:
        return AttemptOutcome(False, "model", "invalid_submission")
    except TaskDefect:
        return AttemptOutcome(None, "task", "invalid_task")
    except InfrastructureFailure:
        return AttemptOutcome(None, "infrastructure", "evaluation_infrastructure")
    except Exception:
        return AttemptOutcome(None, "unknown", "evaluation_error")

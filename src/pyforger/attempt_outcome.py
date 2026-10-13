"""Explicit attempt outcomes shared by runtime adapters and filters."""
from __future__ import annotations

from dataclasses import dataclass


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

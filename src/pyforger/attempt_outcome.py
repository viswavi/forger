"""Explicit attempt outcomes shared by runtime adapters and filters."""
from __future__ import annotations


from pydantic import BaseModel, ConfigDict, StrictBool, model_validator


class AttemptOutcome(BaseModel):
    """An explicit grade and attribution, independent of execution status."""

    model_config = ConfigDict(frozen=True)

    passed: StrictBool | None
    attribution: str
    reason: str

    @model_validator(mode="after")
    def validate_outcome(self) -> AttemptOutcome:
        """Reject ambiguous grades and unsupported attribution values."""
        if self.attribution not in {"model", "infrastructure", "task", "unknown"}:
            raise ValueError("Unsupported attempt attribution")
        if self.attribution == "model" and type(self.passed) is not bool:
            raise ValueError("Model outcomes require a boolean grade")
        if self.attribution != "model" and self.passed is not None:
            raise ValueError("Ungradable outcomes cannot carry a model grade")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("An attribution reason is required")
        return self

    @property
    def counted(self):
        """Only attributable model outcomes enter empirical accuracy."""
        return self.attribution == "model"

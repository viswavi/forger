"""Explicit attempt outcomes shared by runtime adapters and filters."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class AttemptOutcome(BaseModel):
    """An explicit grade and attribution, independent of execution status."""

    model_config = ConfigDict(frozen=True)

    passed: StrictBool | None
    attribution: Literal["model", "infrastructure", "task", "unknown"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_grade(self) -> AttemptOutcome:
        if (self.attribution == "model") != (self.passed is not None):
            raise ValueError("Only model outcomes carry a boolean grade")
        return self

    @property
    def counted(self) -> bool:
        """Only attributable model outcomes enter empirical accuracy."""
        return self.attribution == "model"

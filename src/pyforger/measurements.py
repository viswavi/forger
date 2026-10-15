"""Producer-independent evaluation measurements."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictFloat


class Measurement(BaseModel):
    """A measurement of existing data, not an execution or admission policy.

    A missing grade is excluded from binary accuracy; reasons and evidence
    explain why. Infrastructure attribution belongs to the caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    passed: StrictBool | None = None
    score: StrictFloat | None = None
    reason: str = ""
    evidence: dict[str, JsonValue] = Field(default_factory=dict)

"""Select rows by measured success rates, independently of sample production."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyforger.inspect.success_rate import _success_rates
from pyforger.op import op


class _Bounds(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    min_rate: float = Field(ge=0, le=1)
    max_rate: float = Field(ge=0, le=1)
    min_valid: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def ordered(self) -> _Bounds:
        """Reject inverted success-rate intervals."""
        if self.min_rate > self.max_rate:
            raise ValueError("min_rate must not exceed max_rate")
        return self


@op(name="filter.by_success_rate")
def by_success_rate(
    items: list[dict[str, Any]],
    *,
    outcomes_field: str = "evaluations",
    min_rate: float = 0.2,
    max_rate: float = 0.8,
    min_valid: int | None = None,
) -> list[dict[str, Any]]:
    """Keep rows whose existing measurements meet bounds in every group.

    This function never generates samples, evaluates data, or writes files.
    By default all grades must be present; ``min_valid`` explicitly permits
    exclusions. Empty groups and rows without groups never pass. Inspect the
    same inputs with ``inspect.success_rates`` for complete rejected-row reports.
    """
    bounds = _Bounds(min_rate=min_rate, max_rate=max_rate, min_valid=min_valid)
    return [
        item
        for item, groups in zip(items, _success_rates(items, outcomes_field))
        if groups
        and all(
            counts["success_rate"] is not None
            and counts["valid"] >= (bounds.min_valid or counts["attempted"])
            and bounds.min_rate <= counts["success_rate"] <= bounds.max_rate
            for counts in groups.values()
        )
    ]

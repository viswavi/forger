"""Summarize existing binary measurements without running generators or graders."""
from __future__ import annotations

from typing import Any

from pydantic import StrictBool, TypeAdapter

from pyforger.op import op
from pyforger.utils import get_field

_OUTCOMES = TypeAdapter(dict[str, list[StrictBool | None]])


@op(name="inspect.success_rates")
def success_rates(
    items: list[dict[str, Any]], *, outcomes_field: str = "evaluations"
) -> list[dict[str, Any]]:
    """Return per-group counts and rates for every row, including rejected rows.

    Each row contains a mapping from caller-chosen group labels to a list of
    boolean or missing grades. Missing grades are excluded, not failures.
    Groups may identify models, generators, versions, or any other population.
    """
    return _success_rates(items, outcomes_field)


def _success_rates(
    items: list[dict[str, Any]], outcomes_field: str
) -> list[dict[str, Any]]:
    reports = []
    for item in items:
        outcomes = _OUTCOMES.validate_python(
            get_field(item, outcomes_field, "inspect.success_rates")
        )
        reports.append(
            {
                group: {
                    "attempted": len(grades),
                    "valid": sum(value is not None for value in grades),
                    "passed": sum(value is True for value in grades),
                    "excluded": sum(value is None for value in grades),
                    "success_rate": (
                        sum(value is True for value in grades)
                        / sum(value is not None for value in grades)
                        if any(value is not None for value in grades)
                        else None
                    ),
                }
                for group, grades in outcomes.items()
            }
        )
    return reports

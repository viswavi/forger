"""Select existing rows using producer-independent measurements."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pyforger.inspect.measure import _measure
from pyforger.measurements import Measurement
from pyforger.op import op


@op(name="filter.by_evaluation")
def by_evaluation(
    items: list[dict[str, Any]],
    evaluate: Callable[[dict[str, Any]], Measurement],
    *,
    accept: Callable[[Measurement], bool] | None = None,
) -> list[dict[str, Any]]:
    """Keep rows accepted by an evaluator, defaulting to explicit pass grades.

    ``inspect.measure`` exposes the same complete measurements, including
    rejected rows, without running a different evaluation implementation.
    """
    results = _measure(items, evaluate)
    return [
        item
        for item, result in zip(items, results)
        if (result.passed is True if accept is None else accept(result))
    ]

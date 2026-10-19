"""Measure existing data independently of its producer."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pyforger.measurements import Measurement
from pyforger.op import op


@op(name="inspect.measure")
def measure(
    items: list[dict[str, Any]],
    evaluate: Callable[[dict[str, Any]], Measurement],
) -> list[Measurement]:
    """Return one measurement per row without generating or selecting data.

    Evaluators may use rules, LLMs, agents, or external tools. Exceptions
    propagate rather than being converted into failed grades.
    """
    return _measure(items, evaluate)


def _measure(
    items: list[dict[str, Any]], evaluate: Callable[[dict[str, Any]], Measurement]
) -> list[Measurement]:
    return [Measurement.model_validate(evaluate(deepcopy(item))) for item in items]

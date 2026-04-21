"""Wrap an in-memory list as an ACQUIRE DataSpec."""
from __future__ import annotations

from pyforger.core import DataSpec, NodeType


def from_list(
    items: list[dict],
    *,
    name: str = "from_list",
) -> DataSpec:
    """Wrap an in-memory list as an ACQUIRE DataSpec.

    Args:
        items: List of items captured by reference; mutating after
            construction affects later runs.
        name: Label for the DataSpec node.

    Returns:
        An ACQUIRE DataSpec whose execute_fn returns ``items``.
    """

    def _execute() -> list[dict]:
        return items

    return DataSpec(NodeType.ACQUIRE, [], _execute, name)

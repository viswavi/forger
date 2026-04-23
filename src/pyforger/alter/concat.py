"""Concatenate multiple DataSpec outputs into a single list."""
from __future__ import annotations

from pyforger.core import DataSpec, NodeType


def concat(
    inputs: list[DataSpec],
    *,
    name: str = "concat",
) -> DataSpec:
    """Merge the outputs of several DataSpecs into one flat list.

    Each upstream's execute result is expected to be a list; they are
    concatenated in order. Empty input list yields an empty result.

    Args:
        inputs: DataSpec nodes whose results should be merged.
        name: Debug label.

    Returns:
        ALTER DataSpec whose execute_fn returns the concatenation.
    """
    if not inputs:
        return DataSpec(NodeType.ALTER, [], lambda: [], name)

    def _execute(*results: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in results:
            out.extend(r)
        return out

    return DataSpec(NodeType.ALTER, inputs, _execute, name)

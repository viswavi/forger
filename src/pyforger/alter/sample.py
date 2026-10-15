"""Generate data with caller-supplied implementations and explicit sample IDs."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from pyforger.op import op


@op(name="alter.sample")
def sample(
    items: list[dict[str, Any]],
    generate: Callable[[dict[str, Any], str], Any],
    *,
    sample_ids: list[str],
    output_field: str = "samples",
) -> list[dict[str, Any]]:
    """Attach generated values without prescribing a model, runtime, or storage.

    The callable receives a copy of the source row and an explicit sample ID.
    The same request is memoized normally; use distinct IDs or ``no_cache``
    for fresh stochastic samples. Output rows retain their input association.
    """
    if not sample_ids or any(not isinstance(i, str) or not i for i in sample_ids):
        raise ValueError("Require nonempty sample IDs")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique")
    return [
        {
            **item,
            output_field: [
                {"sample_id": identity, "value": generate(deepcopy(item), identity)}
                for identity in sample_ids
            ],
        }
        for item in items
    ]

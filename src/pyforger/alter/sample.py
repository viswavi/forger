"""Generate data with caller-supplied implementations and explicit sample IDs."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Attach generated values without prescribing a model, runtime, or storage.

    The callable receives a copy of the source row and an explicit sample ID.
    The same request is memoized normally; use distinct IDs or ``no_cache``
    for fresh stochastic samples. Output rows retain their input association.
    ``concurrency`` bounds threaded calls across all rows and sample IDs; results
    preserve input order and exceptions propagate to the caller.
    """
    if not sample_ids or any(not isinstance(i, str) or not i for i in sample_ids):
        raise ValueError("Require nonempty sample IDs")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    jobs = [(item, identity) for item in items for identity in sample_ids]

    def invoke(job: tuple[dict[str, Any], str]) -> dict[str, Any]:
        item, identity = job
        return {"sample_id": identity, "value": generate(deepcopy(item), identity)}

    if concurrency == 1:
        values = [invoke(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            values = list(executor.map(invoke, jobs))
    count = len(sample_ids)
    return [
        {**item, output_field: values[index * count : (index + 1) * count]}
        for index, item in enumerate(items)
    ]

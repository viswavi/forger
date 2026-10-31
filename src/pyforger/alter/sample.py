"""Attach samples to dict rows or data-model rows without prescribing storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel

from pyforger.op import op

Row = TypeVar("Row")


@op(name="alter.sample")
def sample(
    items: list[Row],
    generate: Callable[[Row, str], Any],
    *,
    sample_ids: list[str],
    output_field: str = "samples",
    concurrency: int = 1,
) -> list[Row]:
    """Attach generated values without prescribing a model, runtime, or storage.

    Rows may be dicts or data models. The callable receives a copy of the
    source row and an explicit sample ID. Generated values are attached on a
    copied row: as ``output_field`` for dicts, or by setting the same-named
    attribute for models (pydantic models via ``model_copy(update=...)``,
    dataclasses via ``dataclasses.replace``, other objects via a shallow copy).
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

    def invoke(job: tuple[Row, str]) -> dict[str, Any]:
        item, identity = job
        return {"sample_id": identity, "value": generate(deepcopy(item), identity)}

    if concurrency == 1:
        values = [invoke(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            values = list(executor.map(invoke, jobs))
    count = len(sample_ids)
    return [
        _attach(item, output_field, values[index * count : (index + 1) * count])
        for index, item in enumerate(items)
    ]


def _attach(item: Row, output_field: str, values: list[dict[str, Any]]) -> Row:
    """Return a copy of ``item`` with ``values`` attached under ``output_field``."""
    if isinstance(item, Mapping):
        return {**item, output_field: values}  # type: ignore[return-value]
    if isinstance(item, BaseModel):
        return item.model_copy(update={output_field: values})
    import dataclasses

    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return dataclasses.replace(item, **{output_field: values})
    copied = deepcopy(item)
    setattr(copied, output_field, values)
    return copied

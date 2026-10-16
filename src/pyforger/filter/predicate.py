"""Keep rows that pass a plain Python check, and show the check in step reports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from pyforger import _tracking
from pyforger.op import tracked


@tracked("filter.by_predicate")
def by_predicate(
    items: list[Any], fn: Callable[[Any], Any], *, label: str | None = None
) -> list[Any]:
    """Keep rows where ``fn(row)`` is truthy.

    The Python version of a filter, e.g. ``[r for r in rows if len(r["papers"])
    >= 3]``, made visible: the step report lists it with rows in/out and a
    kept / dropped bar. It is cheap, so it is not cached.

    Example::

        rows = filter.by_predicate(
            rows, lambda r: len(r["papers"]) >= 3, label="has ≥3 papers"
        )

    Args:
        items: Rows to filter.
        fn: The check, called once per row.
        label: Names the check in the report; defaults to ``fn.__name__``, or
            ``"predicate"`` for a lambda.

    Raises:
        Exception: Whatever ``fn`` raises, with the row index in the message.
    """
    if label is None:
        name = getattr(fn, "__name__", "<lambda>")
        label = "predicate" if name == "<lambda>" else name
    keep: list[bool] = []
    for i, item in enumerate(items):
        try:
            keep.append(bool(fn(item)))
        except Exception as exc:
            raise _with_row(exc, f"filter.by_predicate ({label}) failed on row {i}")
    _tracking.record_scores(label, keep, keep="true")
    return [item for item, kept in zip(items, keep) if kept]


def _with_row(exc: Exception, where: str) -> Exception:
    """``exc`` again, its message prefixed with ``where``.

    The same exception type, so ``except KeyError`` still catches it; a type
    that cannot be built from one message keeps ``exc`` with a note instead.
    """
    try:
        new = type(exc)(f"{where}: {exc}")
    except Exception:
        cast(Any, exc).add_note(where)  # mypy's target predates add_note
        return exc
    new.__cause__ = exc
    return new

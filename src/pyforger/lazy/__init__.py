"""Lazy runtime for Forger — Dask-style delayed ops with implicit materialization.

Import surface::

    from pyforger.lazy import Lazy, op, run, trace
    from pyforger.lazy import acquire, alter, filter

Ops return ``Lazy[T]`` and chain lazily. Python control flow (``for``,
``if``, list comprehensions) materializes automatically at the boundary.
"""
from pyforger.lazy import acquire, alter, filter  # noqa: A001
from pyforger.lazy.core import CallRecord, Lazy, clear_trace, op, run, trace

__all__ = [
    "Lazy",
    "op",
    "run",
    "trace",
    "clear_trace",
    "CallRecord",
    "acquire",
    "alter",
    "filter",
]

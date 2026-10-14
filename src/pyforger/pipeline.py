"""Pipeline-as-function API (approach #6).

A pipeline is a regular Python function. Two things you can do with it:

  run(pipeline, *args, **kwargs) -> Any
      Equivalent to pipeline(*args, **kwargs). The wrapper exists as a
      hook for trace setup and instrumentation.

  optimize(pipeline, transforms) -> pipeline
      Apply each transform in order. A transform is
      Callable[[PipelineFn], PipelineFn] — it gets a function and
      returns a function. The new function may wrap, decorate, or
      AST-rewrite the original; whatever it likes.

There is no compilation, no graph IR, no shadow Python, no symbolic
tracer. Optimization is "inexact" because there's no low-level
representation to manipulate, but most useful transforms (cost capping,
logging, fixture substitution, kwarg rewriting) are straightforward
function rewriters anyway.

This intentionally collides with no other pyforger.* module; users
import explicitly: ``from pyforger.pipeline import run, optimize``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

P = TypeVar("P", bound=Callable[..., Any])
Transform = Callable[[P], P]


def run(pipeline: P, *args: Any, **kwargs: Any) -> Any:
    """Execute a pipeline. Equivalent to pipeline(*args, **kwargs).

    Exists as a hook so future versions can inject trace context, error
    boundaries, or cost telemetry without changing call sites.
    """
    return pipeline(*args, **kwargs)


def optimize(pipeline: P, transforms: list[Transform[P]]) -> P:
    """Apply transforms in order. Each is a function → function rewriter.

    Returns a new pipeline. The original is not mutated.
    """
    p = pipeline
    for t in transforms:
        p = t(p)
    return p

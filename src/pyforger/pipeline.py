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

import math
import multiprocessing
import pickle
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from typing import Any, TypeVar

P = TypeVar("P", bound=Callable[..., Any])
Transform = Callable[[P], P]


def run(pipeline: P, *args: Any, **kwargs: Any) -> Any:
    """Execute a pipeline. Equivalent to pipeline(*args, **kwargs).

    Exists as a hook so future versions can inject trace context, error
    boundaries, or cost telemetry without changing call sites.
    """
    return pipeline(*args, **kwargs)


def optimize(pipeline: P, transforms: list[Transform]) -> P:
    """Apply transforms in order. Each is a function → function rewriter.

    Returns a new pipeline. The original is not mutated.
    """
    p = pipeline
    for t in transforms:
        p = t(p)
    return p


def _map_child(fn: Callable[[Any], Any], item: Any, queue: Any) -> None:
    """Send one pickled record without exposing exception messages."""
    try:
        payload = pickle.dumps({"status": "completed", "value": fn(item)})
    except BaseException as exc:
        payload = pickle.dumps({"status": "failed", "error": type(exc).__name__})
    queue.put(payload)


def map_pipeline(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    concurrency: int = 4,
    timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    """Map a trusted callable in bounded spawned processes, preserving order.

    Args:
        fn: Importable, picklable callable invoked as ``fn(item)``.
        items: Iterable of picklable inputs; tuples are not unpacked.
        concurrency: Positive maximum number of simultaneous child processes.
        timeout_s: Optional finite positive per-item budget, including startup
            and shutdown, measured once a worker slot becomes available.

    Returns:
        Ordered records with status ``completed`` and an arbitrary picklable
        ``value``, or status ``failed``/``timed_out`` and an ``error`` type name.
        Exception messages are omitted; values and callable logs are not redacted.
        Existing filesystem side effects are retained, including on timeout.
        Only the direct child is killed, not processes started by the callable.

    Raises:
        ValueError: If concurrency or timeout_s is invalid.
    """
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("concurrency must be a positive integer")
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be finite and positive")
    context = multiprocessing.get_context("spawn")

    def execute(item: Any) -> dict[str, Any]:
        queue = context.Queue(maxsize=1)
        process = context.Process(target=_map_child, args=(fn, item, queue))
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        try:
            process.start()
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return {"status": "timed_out", "error": "TimeoutError"}
                try:
                    payload = queue.get(
                        timeout=0.05 if remaining is None else min(0.05, remaining)
                    )
                    process.join(
                        None
                        if deadline is None
                        else max(0, deadline - time.monotonic())
                    )
                    if process.is_alive():
                        return {"status": "timed_out", "error": "TimeoutError"}
                    record: dict[str, Any] = pickle.loads(payload)
                    return record
                except Empty:
                    if not process.is_alive():
                        try:
                            record = pickle.loads(queue.get_nowait())
                            return record
                        except Empty:
                            return {"status": "failed", "error": "ChildProcessError"}
        except Exception as exc:
            return {"status": "failed", "error": type(exc).__name__}
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join()
            process.close()
            queue.close()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(execute, items))

"""Content-addressed disk cache + tracing for Forger ops.

Wrap a plain Python function with ``@op`` and it gains two things: its
output is memoized on disk keyed by a hash of its inputs, and every call
is recorded in a trace that can be inspected post-hoc.

Used by approach #6 (pipeline-as-function): pipelines are plain Python,
control flow is plain Python, but the expensive boundaries (LLM calls,
filtering passes) get cached at the @op layer.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pyforger.utils import get_formatted_logger

logger = get_formatted_logger("forger.op")

T = TypeVar("T")


def _cache_dir() -> Path:
    """Resolve the on-disk cache root, honoring ``$FORGER_CACHE`` per call.

    Read at call time (not import time) so test fixtures can redirect the
    cache via ``monkeypatch.setenv`` without reloading the module.
    """
    return Path(os.environ.get("FORGER_CACHE", Path.home() / ".cache" / "forger"))


@dataclass
class CallRecord:
    """CallRecord contains metadata about each @op call, for tracing and debugging."""

    op_name: str
    input_hash: str
    output_hash: str | None
    cache_hit: bool
    duration_s: float


_TRACE: list[CallRecord] = []
_TRACE_LOCK = threading.Lock()


def trace() -> list[CallRecord]:
    """Return a copy of the in-memory trace of @op calls."""
    with _TRACE_LOCK:
        return list(_TRACE)


def clear_trace() -> None:
    """Drop all trace records. Does not affect the disk cache."""
    with _TRACE_LOCK:
        _TRACE.clear()


_CODE_T = type((lambda: None).__code__)


def _code_key(code: Any) -> list[Any]:
    """Behavior-only key for a code object (to avoid repeating code execution)."""
    consts = [
        _code_key(c) if isinstance(c, _CODE_T) else _canonical(c)
        for c in code.co_consts
    ]
    return [
        code.co_code,
        consts,
        list(code.co_names),
        list(code.co_varnames),
        list(code.co_freevars),
        code.co_argcount,
    ]


def _canonical(obj: Any) -> Any:
    """Convert obj to a stable, picklable structure for hashing."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, type):
        return f"<type {obj.__module__}.{obj.__qualname__}>"
    if hasattr(obj, "cache_key") and callable(obj.cache_key):
        return ["__cache_key__", _canonical(obj.cache_key())]
    if callable(obj):
        ident = (
            f"{getattr(obj, '__module__', '?')}."
            f"{getattr(obj, '__qualname__', repr(obj))}"
        )
        code = getattr(obj, "__code__", None)
        if code is not None:
            closure = getattr(obj, "__closure__", None) or ()
            closure_vals = [_canonical(cell.cell_contents) for cell in closure]
            return ["__callable__", ident, _code_key(code), closure_vals]
        return ["__callable__", ident]
    return repr(obj)


def _hash(obj: Any) -> str:
    payload = pickle.dumps(_canonical(obj), protocol=4)
    return hashlib.sha256(payload).hexdigest()


def op(
    _fn: Callable[..., T] | None = None,
    *,
    name: str | None = None,
    cache: bool = True,
) -> Any:
    """Decorator: cache output on disk, content-addressed by inputs.

    Usage::

        @op
        def by_rouge_l(items, *, field, threshold=0.7): ...

        @op(name="custom", cache=False)
        def debug_op(x): ...

    Pass ``_no_cache=True`` at call time to bypass the cache for one call.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        # Functions decorated with @op cannot use the reserved argument '_no_cache'.
        if "_no_cache" in inspect.signature(fn).parameters:
            raise TypeError(
                "Forger functions decorated with @op cannot use the reserved "
                + "argument name '_no_cache'."
            )

        op_name = name or f"{fn.__module__}.{fn.__qualname__}"
        body_key = _canonical(fn)

        @wraps(fn)
        def wrapper(*args: Any, _no_cache: bool = False, **kwargs: Any) -> T:
            start = time.time()
            input_hash = _hash([op_name, body_key, list(args), kwargs])
            cache_path = _cache_dir() / op_name / f"{input_hash}.pkl"

            use_cache = cache and not _no_cache
            cache_hit = False

            if use_cache and cache_path.exists():
                try:
                    with cache_path.open("rb") as f:
                        result = pickle.load(f)
                    cache_hit = True
                except (EOFError, pickle.UnpicklingError):
                    # Cache file is corrupted; fall through to executing op directly.
                    # The cache will be overwritten with a new result.
                    pass

            if not cache_hit:
                result = fn(*args, **kwargs)

                if use_cache:
                    # Atomic write to prevent corruption from concurrent threads
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = cache_path.with_suffix(f".tmp.{threading.get_ident()}")
                    with tmp_path.open("wb") as f:
                        pickle.dump(result, f, protocol=4)
                    os.replace(tmp_path, cache_path)

            duration = time.time() - start
            with _TRACE_LOCK:
                _TRACE.append(
                    CallRecord(
                        op_name=op_name,
                        input_hash=input_hash,
                        output_hash=None if cache_hit else _hash(result),
                        cache_hit=cache_hit,
                        duration_s=duration,
                    )
                )

            if cache_hit:
                logger.debug("cache HIT %s %s", op_name, input_hash[:8])
            else:
                logger.debug(
                    "cache MISS %s %s (%.2fs)",
                    op_name,
                    input_hash[:8],
                    duration,
                )

            return result

        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate

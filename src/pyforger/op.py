"""Content-addressed disk cache + tracing for Forger ops.

Wrap a plain Python function with ``@op`` and it gains two things: its
output is memoized on disk keyed by a hash of its inputs, and every call
is recorded in a trace that can be inspected post-hoc.

.. warning::
    This module uses ``pickle`` for serialization. Only use this with a cache
    directory that is trusted and secured (e.g., restricted to the current user).
    Loading malicious pickle data can lead to arbitrary code execution.

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
import types
from collections import deque
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pyforger.utils import canonical, get_formatted_logger

logger = get_formatted_logger("forger.op")

T = TypeVar("T")


def _cache_dir() -> Path:
    """Resolve the on-disk cache root, honoring ``$FORGER_CACHE`` per call.

    Read at call time (not import time) so test fixtures can redirect the
    cache via ``monkeypatch.setenv`` without reloading the module.
    """
    return Path(os.environ.get("FORGER_CACHE", Path.home() / ".cache" / "forger"))


class CallRecord(BaseModel):
    """CallRecord contains metadata about each @op call, for tracing and debugging."""

    op_name: str
    input_hash: str
    output_hash: str | None
    cache_hit: bool
    duration_s: float


_TRACE_DEFAULT_MAX = 10_000


def _trace_max() -> int:
    """We cap the number of trace records to observe -- 0 disables tracing."""
    # This environment variable must be set before importing Forger to take effect.
    raw = os.environ.get("PYFORGER_TRACE_MAX")
    if raw is None:
        return _TRACE_DEFAULT_MAX
    try:
        return max(0, int(raw))
    except ValueError:
        return _TRACE_DEFAULT_MAX


_TRACE: deque[CallRecord] = deque(maxlen=_trace_max())
_TRACE_LOCK = threading.Lock()


def trace() -> list[CallRecord]:
    """Return a copy of the in-memory trace of @op calls."""
    with _TRACE_LOCK:
        return list(_TRACE)


def clear_trace() -> None:
    """Drop all trace records. Does not affect the disk cache."""
    with _TRACE_LOCK:
        _TRACE.clear()


_HASH_WARN_BYTES = int(os.environ.get("PYFORGER_HASH_WARN_BYTES", 50 * 1024 * 1024))
_HASH_WARN_SECONDS = float(os.environ.get("PYFORGER_HASH_WARN_SECONDS", 1.0))
_HASH_WARNED: set[str] = set()
_HASH_WARN_LOCK = threading.Lock()


def _hash(obj: Any) -> tuple[str, int]:
    payload = pickle.dumps(canonical(obj), protocol=4)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _maybe_warn_hash_cost(op_name: str, duration: float, size: int) -> None:
    if size <= _HASH_WARN_BYTES and duration <= _HASH_WARN_SECONDS:
        return
    with _HASH_WARN_LOCK:
        if op_name in _HASH_WARNED:
            return
        _HASH_WARNED.add(op_name)
    logger.warning(
        "@op(%s) cache-key hash took %.2fs to hash input and resulted in a %d MiB"
        " payload; consider adding cache_key() on large inputs for a cheaper key.",
        op_name,
        duration,
        size // (1024 * 1024),
    )


def _warn_unstable_refs(fn: Callable[..., Any], op_name: str) -> None:
    """Warn if fn reads module globals, env vars, or closes over enclosing scope."""
    code = fn.__code__
    names = set(code.co_names)
    g = fn.__globals__
    # We look for referents that can quietly change without affecting the op's cache
    # key. This will lead to stale cache results and cause problems. We check for:
    # 1) Module globals that are not modules or callables (e.g. configuration dicts).
    # 2) Environment variables (detected by usage of os.environ or os.getenv).
    # 3) Closure variables
    data_globals = sorted(
        n
        for n in names
        if n in g and not isinstance(g[n], types.ModuleType) and not callable(g[n])
    )
    env_used = bool({"environ", "getenv"} & names)
    # We look for closure variables, e.g. `config["multiplier"]`` in
    # def new_op(x):
    #     return x * config["multiplier"]
    closure_vars = sorted(code.co_freevars) if fn.__closure__ else []
    if not (data_globals or env_used or closure_vars):
        return
    parts = []
    if data_globals:
        parts.append(f"reads globals {data_globals}")
    if env_used:
        parts.append("reads os.environ/os.getenv")
    if closure_vars:
        parts.append(f"closes over {closure_vars}")
    logger.warning(
        "@op(%s) %s; changes to these may not invalidate the cache key, "
        "so cached results may be stale.",
        op_name,
        " and ".join(parts),
    )


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
    Pass ``_cache_only=True`` to return only an existing cached result, raising
    ``FileNotFoundError`` on a miss without executing the function.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            # Built-in functions and some callables don't have inspectable signatures.
            logger.debug("inspect.signature failed for %s: %s", fn, exc)
            sig = None
        else:
            # These names are reserved for per-call cache controls.
            assert sig is not None  # mypy hint
            for reserved in ("_no_cache", "_cache_only"):
                if reserved in sig.parameters:
                    raise TypeError(
                        "Forger functions decorated with @op cannot use the reserved "
                        + f"argument name '{reserved}'."
                    )

        op_name = name or f"{fn.__module__}.{fn.__qualname__}"
        # Sanitize op_name to be a safe directory name
        safe_op_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in op_name)
        body_key = canonical(fn)
        if cache:
            _warn_unstable_refs(fn, op_name)

        @wraps(fn)
        def wrapper(
            *args: Any,
            _no_cache: bool = False,
            _cache_only: bool = False,
            **kwargs: Any,
        ) -> T:
            if _cache_only and (_no_cache or not cache):
                raise ValueError("_cache_only requires an enabled cache")
            start = time.time()
            if sig is not None:
                try:
                    # Attempt to normalize the op function's arguments before hashing.
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    # Normalize the different ways of stating the same arguments.
                    args = bound.args
                    kwargs = bound.kwargs
                    normalized = dict(bound.arguments)
                except TypeError:
                    if _cache_only:
                        raise
                    # If the arguments don't match the signature, use them directly.
                    normalized = {"args": list(args), "kwargs": dict(kwargs)}
            else:
                normalized = {"args": list(args), "kwargs": dict(kwargs)}
            t0 = time.perf_counter()
            input_hash, hash_size = _hash([op_name, body_key, normalized])
            _maybe_warn_hash_cost(op_name, time.perf_counter() - t0, hash_size)

            cache_root = _cache_dir()
            cache_path = (cache_root / safe_op_name / f"{input_hash}.pkl").resolve()

            # Ensure the cache path is within the cache root. This prevents users from
            # supplying cache paths that point outside Forger's directories.
            if cache_root.resolve() not in cache_path.parents:
                raise ValueError(f"Invalid cache path provided: {cache_path}")

            use_cache = cache and not _no_cache
            cache_hit = False

            if use_cache and cache_path.exists():
                try:
                    # Security: verify file ownership matches current process owner
                    if (
                        hasattr(os, "getuid")
                        and cache_path.stat().st_uid != os.getuid()
                    ):
                        raise PermissionError(
                            f"Cache file {cache_path} is owned by another user."
                        )

                    with cache_path.open("rb") as f:
                        result = pickle.load(f)
                    cache_hit = True
                except Exception as exc:
                    if _cache_only:
                        raise RuntimeError(
                            f"Cannot read cached result for {op_name}"
                        ) from exc
                    # Cache file is unreadable or incompatible with the current
                    # code version; fall through to executing the op directly.
                    # The cache will be overwritten with a new result.
                    logger.debug(
                        "cache read failed for %s: %s: %s",
                        cache_path,
                        type(exc).__name__,
                        exc,
                    )

            if not cache_hit:
                if _cache_only:
                    raise FileNotFoundError(
                        f"No cached result for {op_name} with these arguments"
                    )
                result = fn(*args, **kwargs)

                if use_cache:
                    # Atomic write to prevent corruption from concurrent threads
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = (
                        cache_path.parent
                        / f"{cache_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
                    )
                    with tmp_path.open("wb") as f:
                        pickle.dump(result, f, protocol=4)
                    os.replace(tmp_path, cache_path)

            duration = time.time() - start
            # If tracing is enabled, record this call on the in-memory trace.
            if _TRACE.maxlen is not None and _TRACE.maxlen > 0:
                with _TRACE_LOCK:
                    if cache_hit:
                        output_hash = None
                    else:
                        output_hash, _ = _hash(result)
                    _TRACE.append(
                        CallRecord(
                            op_name=op_name,
                            input_hash=input_hash,
                            output_hash=output_hash,
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

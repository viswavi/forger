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
import io
import os
import pickle
import shutil
import threading
import time
import types
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pyforger import _tracking
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
    """Hash ``canonical(obj)``; equal values always hash equally.

    The pickler runs without its memo (``fast``), so the bytes do not depend on
    which equal strings happen to be the same object. With the memo, a field
    name typed in the source and the same name read back from a cached result
    hash differently. ``canonical`` output is acyclic, which fast mode needs.
    """
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=4)
    pickler.fast = True
    pickler.dump(canonical(obj))
    payload = buf.getvalue()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _legacy_hash(obj: Any) -> str:
    """Return the cache key older Forger versions used (memoized pickle)."""
    return hashlib.sha256(pickle.dumps(canonical(obj), protocol=4)).hexdigest()


# True while an op called with ``_no_cache=True`` runs, so the ops it calls
# also skip the cache (e.g. by_llm_score's inner judge-score op).
_NO_CACHE_SCOPE: ContextVar[bool] = ContextVar("forger_no_cache_scope", default=False)

_STEP_META_SUFFIX = ".step.json"


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

    Pass ``_no_cache=True`` at call time to bypass the cache for one call,
    including the ops that call makes. Pass ``_cache_only=True`` to return only
    an existing cached result, raising ``FileNotFoundError`` on a miss without
    executing the function.

    Pass ``min_yield=`` / ``max_yield=`` (fractions in [0, 1]) to check the
    ratio of output rows to input rows, e.g. ``filter.by_length(rows, ...,
    min_yield=0.5)``. A violation is a warning in the ``@step`` report (or a
    logged warning outside steps). Neither keyword is part of the cache key.
    ``acquire.*`` ops take no input rows and reject both keywords.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            # Built-in functions and some callables don't have inspectable signatures.
            logger.debug("inspect.signature failed for %s: %s", fn, exc)
            sig = None
        else:
            # These names are reserved for per-call cache controls and yield checks.
            assert sig is not None  # mypy hint
            for reserved in ("_no_cache", "_cache_only", "min_yield", "max_yield"):
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
            min_yield: float | None = None,
            max_yield: float | None = None,
            **kwargs: Any,
        ) -> T:
            if _cache_only and (_no_cache or not cache):
                raise ValueError("_cache_only requires an enabled cache")
            if not _cache_only and _NO_CACHE_SCOPE.get():
                _no_cache = True
            _check_yield_bounds(min_yield, max_yield)
            if (min_yield is not None or max_yield is not None) and (
                _tracking.category_of(op_name) == "acquire"
            ):
                raise ValueError(
                    f"{op_name} takes no input rows, so min_yield / max_yield do "
                    "not apply; put them on the alter or filter op that follows"
                )
            token = _tracking.enter_op(op_name)
            scope_token = _NO_CACHE_SCOPE.set(True) if _no_cache else None
            call: _tracking.OpCall | None = None
            try:
                result, cache_hit, first_arg, duration = _call(
                    args, kwargs, _no_cache, _cache_only
                )
                call = _op_call(
                    op_name,
                    first_arg,
                    result,
                    cache_hit=cache_hit,
                    duration=duration,
                    usage=token.usage_totals(),
                    scores=_tracking.current_scores(),
                    min_yield=min_yield,
                    max_yield=max_yield,
                )
            finally:
                if scope_token is not None:
                    _NO_CACHE_SCOPE.reset(scope_token)
                _tracking.exit_op(token, call)
            return result

        def _call(
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _no_cache: bool,
            _cache_only: bool,
        ) -> tuple[Any, bool, Any, float]:
            """Run or load the op; return (result, cache_hit, first arg, seconds)."""
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
            key = [op_name, body_key, normalized]
            t0 = time.perf_counter()
            input_hash, hash_size = _hash(key)
            _maybe_warn_hash_cost(op_name, time.perf_counter() - t0, hash_size)

            cache_root = _cache_dir()
            cache_path = (cache_root / safe_op_name / f"{input_hash}.pkl").resolve()

            # Ensure the cache path is within the cache root. This prevents users from
            # supplying cache paths that point outside Forger's directories.
            if cache_root.resolve() not in cache_path.parents:
                raise ValueError(f"Invalid cache path provided: {cache_path}")

            use_cache = cache and not _no_cache
            cache_hit = False
            is_step = op_name.startswith(_tracking.STEP_OP_PREFIX)
            meta_path = cache_path.with_name(input_hash + _STEP_META_SUFFIX)

            if use_cache and not cache_path.exists():
                _adopt_legacy_entry(cache_root / safe_op_name, key, cache_path)

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
                    if is_step and meta_path.exists():
                        _tracking.restore_step_meta(meta_path.read_text())
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
                meta = None
                if is_step:
                    # Raises before caching when a @step breaks its contract.
                    meta = _tracking.check_step_result(result)

                if use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if meta is not None:
                        # Written first, so a cached step always has its ops.
                        _atomic_write(meta_path, meta.encode())
                    _atomic_write(cache_path, pickle.dumps(result, protocol=4))

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

            return result, cache_hit, (args[0] if args else None), duration

        return wrapper

    if _fn is not None:
        return decorate(_fn)
    return decorate


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so concurrent readers never see a partial file."""
    tmp_path = path.parent / f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def _adopt_legacy_entry(op_dir: Path, key: Any, cache_path: Path) -> None:
    """Copy a result cached under the old cache key to ``cache_path``, if any.

    Caches written before keys were hashed without the pickle memo are found
    under their old key once and copied to the new one, so they stay valid.
    """
    if not op_dir.is_dir():
        return
    try:
        legacy = op_dir / f"{_legacy_hash(key)}.pkl"
        if not legacy.exists() or (
            hasattr(os, "getuid") and legacy.stat().st_uid != os.getuid()
        ):
            return
        for suffix in (_STEP_META_SUFFIX, ".pkl"):
            src = legacy.with_name(legacy.stem + suffix)
            if src.exists():
                tmp = cache_path.with_name(f"{cache_path.name}.legacy.{os.getpid()}")
                shutil.copyfile(src, tmp)
                os.replace(tmp, cache_path.with_name(cache_path.stem + suffix))
    except Exception as exc:  # the old entry is only a shortcut; never fail on it
        logger.debug("legacy cache lookup failed in %s: %s", op_dir, exc)


def tracked(op_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Report calls in ``@step`` reports like an op, without caching or hashing.

    For functions whose arguments cannot be hashed into a cache key, e.g.
    ``acquire.retrieve`` (a retrieval source holds a client and credentials).
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            token = _tracking.enter_op(op_name)
            start = time.perf_counter()
            call: _tracking.OpCall | None = None
            try:
                result = fn(*args, **kwargs)
                call = _op_call(
                    op_name,
                    args[0] if args else None,
                    result,
                    cache_hit=False,
                    duration=time.perf_counter() - start,
                    usage=token.usage_totals(),
                    scores=_tracking.current_scores(),
                    min_yield=None,
                    max_yield=None,
                )
            finally:
                _tracking.exit_op(token, call)
            return result

        return wrapper

    return decorate


def _check_yield_bounds(min_yield: float | None, max_yield: float | None) -> None:
    """Raise ValueError unless 0 <= min_yield <= max_yield <= 1 (either optional)."""
    for label, bound in (("min_yield", min_yield), ("max_yield", max_yield)):
        if bound is not None and not 0.0 <= bound <= 1.0:
            raise ValueError(f"{label} must be in [0, 1], got {bound}")
    if min_yield is not None and max_yield is not None and min_yield > max_yield:
        raise ValueError(
            f"min_yield ({min_yield}) must not exceed max_yield ({max_yield})"
        )


def _op_call(
    op_name: str,
    first_arg: Any,
    result: Any,
    *,
    cache_hit: bool,
    duration: float,
    usage: tuple[int, int, float, float],
    scores: list[_tracking.ScoreRecord],
    min_yield: float | None,
    max_yield: float | None,
) -> _tracking.OpCall:
    """Build the step-report entry for one op call and apply its yield bounds."""
    category = _tracking.category_of(op_name)
    # An acquire op's first argument is a prompt or query, not input rows.
    n_in = (
        len(first_arg)
        if isinstance(first_arg, list) and category != "acquire"
        else None
    )
    n_out = len(result) if isinstance(result, list) else None
    yield_ratio = n_out / n_in if n_in and n_out is not None else None
    call = _tracking.OpCall(
        op_name=op_name,
        category=category,
        n_in=n_in,
        n_out=n_out,
        yield_ratio=yield_ratio,
        min_yield=min_yield,
        max_yield=max_yield,
        cache_hit=cache_hit,
        duration_s=duration,
        prompt_tokens=usage[0],
        completion_tokens=usage[1],
        cost_usd=usage[2],
        token_units=usage[3],
        scores=scores[0] if scores else None,
    )
    if (min_yield is not None or max_yield is not None) and yield_ratio is None:
        if n_in == 0 and n_out is not None:
            if min_yield:
                call.warnings.append(
                    f"{op_name}: no input rows, so yield is 0 and below min_yield "
                    f"{min_yield:.2f}"
                )
        else:
            call.warnings.append(
                f"{op_name}: yield unchecked; min_yield / max_yield need a list of "
                "rows in and a list out"
            )
    if yield_ratio is not None:
        if min_yield is not None and yield_ratio < min_yield:
            call.warnings.append(
                f"{op_name}: yield {yield_ratio:.2f} below min_yield {min_yield:.2f}"
            )
        if max_yield is not None and yield_ratio > max_yield:
            call.warnings.append(
                f"{op_name}: yield {yield_ratio:.2f} above max_yield {max_yield:.2f}"
            )
    if not _tracking.is_reporting():
        for warning in call.warnings:
            logger.warning(warning)
    return call

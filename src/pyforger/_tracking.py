"""Shared run-time state for ``@op`` calls, ``@step`` reports, and filter scores.

This module imports nothing from the rest of pyforger, so ``op.py``,
``steps.py``, ``acquire/model.py``, and the filters can all use it without
circular imports.

Context variables do not propagate into ``ThreadPoolExecutor`` workers (for
example, ``alter.dynamic_step`` runs ``dynamic_step_source`` in threads). Ops
called from such workers see no active step and are not itemized in its report,
unless the work is submitted as ``pool.submit(contextvars.copy_context().run,
fn, arg)``. A step whose ops all ran in workers gets a warning instead of the
"called no Forger operations" error, and a step called from a worker thread
still joins the pipeline when exactly one pipeline is running.

Token and cost totals are deltas of one process-wide counter, so the LLM usage
of worker threads lands in the enclosing op. When several ops or steps run at
the same time in different threads, each one's totals include the others'.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

_CATEGORIES = frozenset({"acquire", "alter", "filter", "inspect"})

# Op names that mark a step's own cache wrapper (see ``pyforger.steps``).
STEP_OP_PREFIX = "step."
DYNAMIC_STEP_OP = "alter.dynamic_step"


def category_of(op_name: str) -> str:
    """Return the op category from its name prefix, e.g. ``filter.by_length``."""
    head = op_name.split(".", 1)[0]
    return head if head in _CATEGORIES else "other"


class ScoreRecord(BaseModel):
    """Per-row scores a filter computed, and how it used them to keep rows.

    ``keep`` is the rule: ``ge`` keeps ``value >= threshold``, ``le`` keeps
    ``value <= threshold``, ``lt`` keeps ``value < threshold``, ``between`` keeps
    ``lower <= value <= upper``, and ``true`` means the value is ``True`` when kept.
    ``None`` values could not be scored (e.g. an unparseable judge response).
    """

    label: str
    values: list[float | int | bool | None]
    keep: Literal["ge", "le", "lt", "between", "true"]
    threshold: float | None = None
    lower: float | None = None
    upper: float | None = None
    # True when each score compares a row with the rows kept so far (e.g.
    # by_rouge_l): another threshold would change the scores themselves, so the
    # report cannot say what it would keep.
    pooled: bool = False


class OpCall(BaseModel):
    """One top-level op call inside a step or pipeline."""

    op_name: str
    category: str
    n_in: int | None
    n_out: int | None
    yield_ratio: float | None
    min_yield: float | None = None
    max_yield: float | None = None
    cache_hit: bool
    duration_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0
    scores: ScoreRecord | None = None
    warnings: list[str] = Field(default_factory=list)


class StepRecord(BaseModel):
    """Everything a ``@step`` call reports: its ops, totals, warnings, and samples.

    ``kind`` is ``"step"`` for a ``@step`` function, ``"dynamic_step"`` for
    ``alter.dynamic_step`` called directly in a pipeline, and ``"bare"`` for any
    other op called in a pipeline outside a step.
    """

    name: str
    doc: str = ""
    kind: Literal["step", "dynamic_step", "bare"] = "step"
    n_in: int | None = None
    n_out: int | None = None
    duration_s: float = 0.0
    cache_hit: bool = False
    ops: list[OpCall] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample: list[Any] = Field(default_factory=list)
    sample_shown: int = 3
    input_type: str | None = None
    output_type: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0


class PipelineRecord(BaseModel):
    """Everything a ``@pipeline`` call reports: its steps, totals, and warnings."""

    name: str
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0


_ACTIVE_STEP: ContextVar[StepRecord | None] = ContextVar(
    "forger_active_step", default=None
)
_ACTIVE_PIPELINE: ContextVar[PipelineRecord | None] = ContextVar(
    "forger_active_pipeline", default=None
)
_OP_DEPTH: ContextVar[int] = ContextVar("forger_op_depth", default=0)
_ACTIVE_SCORES: ContextVar[list[ScoreRecord] | None] = ContextVar(
    "forger_active_scores", default=None
)
# Set by the active step; ``op.py`` calls it on a step's fresh result before
# caching it, so a result that breaks the step's contract (or, under
# strict=True, has warnings) is never cached. It returns the step metadata that
# ``op.py`` stores next to the cached result (see ``step_meta``).
_STEP_RESULT_CHECK: ContextVar[Callable[[Any], str | None] | None] = ContextVar(
    "forger_step_result_check", default=None
)
# Set by the active pipeline; called with each ``alter.dynamic_step`` record
# so it is stored and printed like a step.
_ON_DYNAMIC_STEP: ContextVar[Callable[[StepRecord], None] | None] = ContextVar(
    "forger_on_dynamic_step", default=None
)


def active_step() -> StepRecord | None:
    """Return the step currently running in this context, if any."""
    return _ACTIVE_STEP.get()


def active_pipeline() -> PipelineRecord | None:
    """Return the pipeline currently running in this context, if any."""
    return _ACTIVE_PIPELINE.get()


def is_reporting() -> bool:
    """Return True when a step or pipeline will report this context's op calls."""
    return _ACTIVE_STEP.get() is not None or _ACTIVE_PIPELINE.get() is not None


@dataclass(frozen=True)
class ScopeToken:
    """Handle returned by ``enter_step`` / ``enter_pipeline`` to undo them."""

    owner_token: Token[Any]
    depth_token: Token[int]
    hook_token: Token[Any] | None = None


def enter_step(rec: StepRecord, check: Callable[[Any], str | None]) -> ScopeToken:
    """Make ``rec`` the active step; ops it calls are top level (depth 0) again.

    ``check`` validates the step's result; see ``check_step_result``.
    """
    return ScopeToken(
        _ACTIVE_STEP.set(rec), _OP_DEPTH.set(0), _STEP_RESULT_CHECK.set(check)
    )


def exit_step(token: ScopeToken) -> None:
    """Undo ``enter_step``."""
    if token.hook_token is not None:
        _STEP_RESULT_CHECK.reset(token.hook_token)
    _OP_DEPTH.reset(token.depth_token)
    _ACTIVE_STEP.reset(token.owner_token)


def check_step_result(result: Any) -> str | None:
    """Run the active step's result check; return the metadata to cache with it.

    The check raises on a contract violation, so the result is not cached.
    """
    check = _STEP_RESULT_CHECK.get()
    return check(result) if check is not None else None


def step_meta(rec: StepRecord) -> str:
    """Serialize the ops a step ran, to store next to its cached result.

    On a cache hit ``restore_step_meta`` puts them back (marked cached, with no
    time, usage, or scores), so the cached step reports the same ops and yield
    warnings, and ``strict=True`` still sees them.
    """
    ops = [
        call.model_copy(
            update={
                "cache_hit": True,
                "duration_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "token_units": 0.0,
                "scores": None,
            }
        ).model_dump(mode="json")
        for call in rec.ops
    ]
    return json.dumps({"version": 1, "ops": ops})


def restore_step_meta(text: str) -> None:
    """Give the active step the ops stored by ``step_meta`` (on a cache hit)."""
    step = _ACTIVE_STEP.get()
    if step is None:
        return
    try:
        step.ops = [OpCall.model_validate(d) for d in json.loads(text)["ops"]]
    except (ValueError, KeyError, TypeError):
        step.ops = []  # unreadable metadata: report the step as fully cached


_RUNNING_PIPELINES: list[PipelineRecord] = []
_RUNNING_LOCK = threading.Lock()


def enter_pipeline(
    rec: PipelineRecord, on_dynamic_step: Callable[[StepRecord], None] | None = None
) -> ScopeToken:
    """Make ``rec`` the active pipeline; ops it calls directly are top level.

    ``on_dynamic_step`` is called with the record of each ``alter.dynamic_step``
    call made directly in the pipeline.
    """
    with _RUNNING_LOCK:
        _RUNNING_PIPELINES.append(rec)
    return ScopeToken(
        _ACTIVE_PIPELINE.set(rec),
        _OP_DEPTH.set(0),
        _ON_DYNAMIC_STEP.set(on_dynamic_step),
    )


def exit_pipeline(token: ScopeToken) -> None:
    """Undo ``enter_pipeline``."""
    rec = _ACTIVE_PIPELINE.get()
    with _RUNNING_LOCK:
        for i, running in enumerate(_RUNNING_PIPELINES):
            if running is rec:
                del _RUNNING_PIPELINES[i]
                break
    if token.hook_token is not None:
        _ON_DYNAMIC_STEP.reset(token.hook_token)
    _OP_DEPTH.reset(token.depth_token)
    _ACTIVE_PIPELINE.reset(token.owner_token)


def pipeline_for_step() -> PipelineRecord | None:
    """Return the pipeline a finished step belongs to, if any.

    That is the pipeline active in this context or, for a step called from a
    worker thread (where context variables are unset), the only pipeline
    running in the process.
    """
    rec = _ACTIVE_PIPELINE.get()
    if rec is not None:
        return rec
    with _RUNNING_LOCK:
        return _RUNNING_PIPELINES[0] if len(_RUNNING_PIPELINES) == 1 else None


_OPS_STARTED = [0]
_OPS_LOCK = threading.Lock()


def ops_started() -> int:
    """Return how many ops (not steps) this process has started, in any thread."""
    with _OPS_LOCK:
        return _OPS_STARTED[0]


@dataclass(frozen=True)
class OpToken:
    """Opaque handle returned by ``enter_op`` and passed back to ``exit_op``."""

    is_step: bool
    depth: int
    depth_token: Token[int] | None = None
    scores_token: Token[list[ScoreRecord] | None] | None = None


def enter_op(op_name: str) -> OpToken:
    """Mark the start of an op call; nested calls fold into their parent."""
    if op_name.startswith(STEP_OP_PREFIX):
        # A step's own cache wrapper: the step itself does the reporting.
        return OpToken(is_step=True, depth=-1)
    with _OPS_LOCK:
        _OPS_STARTED[0] += 1
    depth = _OP_DEPTH.get()
    return OpToken(
        is_step=False,
        depth=depth,
        depth_token=_OP_DEPTH.set(depth + 1),
        scores_token=_ACTIVE_SCORES.set([]),
    )


def current_scores() -> list[ScoreRecord]:
    """Return the score records the innermost active op has recorded so far."""
    return list(_ACTIVE_SCORES.get() or [])


def exit_op(token: OpToken, call: OpCall | None) -> None:
    """Restore state from ``enter_op`` and hand a top-level call to its owner.

    ``call`` is None when the op raised. Top-level calls (depth 0) go to the
    active step, or to the active pipeline as a bare op when no step is active.
    """
    if token.is_step:
        step = _ACTIVE_STEP.get()
        if step is not None and call is not None:
            step.cache_hit = call.cache_hit
            step.warnings.extend(call.warnings)
        return
    if token.scores_token is not None:
        _ACTIVE_SCORES.reset(token.scores_token)
    if token.depth_token is not None:
        _OP_DEPTH.reset(token.depth_token)
    if call is None or token.depth != 0:
        return
    step = _ACTIVE_STEP.get()
    if step is not None:
        step.ops.append(call)
        return
    pipeline = _ACTIVE_PIPELINE.get()
    if pipeline is not None:
        rec = step_from_bare_op(call)
        pipeline.steps.append(rec)
        on_dynamic_step = _ON_DYNAMIC_STEP.get()
        if rec.kind == "dynamic_step" and on_dynamic_step is not None:
            on_dynamic_step(rec)


def step_from_bare_op(call: OpCall) -> StepRecord:
    """Wrap an op called directly in a pipeline as a one-op step record."""
    if call.op_name == DYNAMIC_STEP_OP:
        name, kind, warnings = "dynamic_step", "dynamic_step", []
    else:
        name, kind = f"(bare) {call.op_name}", "bare"
        warnings = ["called outside a @step"]
    return StepRecord(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        n_in=call.n_in,
        n_out=call.n_out,
        duration_s=call.duration_s,
        cache_hit=call.cache_hit,
        ops=[call],
        warnings=warnings + list(call.warnings),
        prompt_tokens=call.prompt_tokens,
        completion_tokens=call.completion_tokens,
        cost_usd=call.cost_usd,
        token_units=call.token_units,
    )


def record_scores(
    label: str,
    values: list[Any],
    *,
    keep: Literal["ge", "le", "lt", "between", "true"],
    threshold: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    pooled: bool = False,
) -> None:
    """Record one per-row score list for the active op; a no-op outside ops.

    Pass ``pooled=True`` when each score depends on the rows kept so far.
    """
    scores = _ACTIVE_SCORES.get()
    if scores is None:
        return
    scores.append(
        ScoreRecord(
            label=label,
            values=list(values),
            keep=keep,
            threshold=threshold,
            lower=lower,
            upper=upper,
            pooled=pooled,
        )
    )


def keep_where(items: list[Any], keep: list[bool]) -> list[Any]:
    """Record ``keep`` as a filter's per-row scores and return the kept items."""
    record_scores("kept", keep, keep="true")
    return [item for item, kept in zip(items, keep) if kept]


_USAGE_LOCK = threading.Lock()
_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost_usd": 0.0,
    "token_units": 0.0,
}


def _as_number(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def add_usage(
    prompt_tokens: Any,
    completion_tokens: Any,
    cost_usd: Any,
    token_units: Any = 0.0,
) -> None:
    """Add one LLM call's usage to the process-wide counter (NaN counts as 0)."""
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += int(_as_number(prompt_tokens))
        _USAGE["completion_tokens"] += int(_as_number(completion_tokens))
        _USAGE["cost_usd"] += _as_number(cost_usd)
        _USAGE["token_units"] += _as_number(token_units)


def usage_snapshot() -> tuple[int, int, float, float]:
    """Return ``(prompt_tokens, completion_tokens, cost_usd, token_units)``."""
    with _USAGE_LOCK:
        return (
            int(_USAGE["prompt_tokens"]),
            int(_USAGE["completion_tokens"]),
            float(_USAGE["cost_usd"]),
            float(_USAGE["token_units"]),
        )


def usage_since(
    start: tuple[int, int, float, float],
) -> tuple[int, int, float, float]:
    """Return the usage added since ``start`` (a ``usage_snapshot()``)."""
    now = usage_snapshot()
    return (
        now[0] - start[0],
        now[1] - start[1],
        max(0.0, now[2] - start[2]),
        max(0.0, now[3] - start[3]),
    )


_REGISTRY_LOCK = threading.Lock()
_STEP_HISTORY: dict[str, StepRecord] = {}
_LAST_PIPELINE: list[PipelineRecord] = []


def store_step(rec: StepRecord) -> None:
    """Keep ``rec`` as the latest run of its step; the newest step goes last."""
    with _REGISTRY_LOCK:
        _STEP_HISTORY.pop(rec.name, None)
        _STEP_HISTORY[rec.name] = rec


def store_pipeline(rec: PipelineRecord) -> None:
    """Keep ``rec`` as the latest pipeline run."""
    with _REGISTRY_LOCK:
        _LAST_PIPELINE[:] = [rec]


def step_records() -> dict[str, StepRecord]:
    """Return the latest record of each step, oldest first."""
    with _REGISTRY_LOCK:
        return dict(_STEP_HISTORY)


def last_pipeline() -> PipelineRecord | None:
    """Return the most recent pipeline record, if any pipeline has run."""
    with _REGISTRY_LOCK:
        return _LAST_PIPELINE[0] if _LAST_PIPELINE else None


def clear_step_records() -> None:
    """Forget all stored step and pipeline records."""
    with _REGISTRY_LOCK:
        _STEP_HISTORY.clear()
        _LAST_PIPELINE.clear()

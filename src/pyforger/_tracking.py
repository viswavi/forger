"""Shared run-time state for ``@op`` calls, ``@step`` reports, and filter scores.

This module imports nothing from the rest of pyforger, so ``op.py``,
``steps.py``, ``acquire/model.py``, and the filters can all use it without
circular imports.

Context variables do not propagate into ``ThreadPoolExecutor`` workers (for
example, ``alter.dynamic_step`` runs ``dynamic_step_source`` in threads). Ops
called from such workers see no active step and are not attributed to one; the
LLM usage they cause still lands in the process-wide counter, so the enclosing
op's token and cost deltas include it.
"""

from __future__ import annotations

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


class PipelineRecord(BaseModel):
    """Everything a ``@pipeline`` call reports: its steps, totals, and warnings."""

    name: str
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


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
# caching it, so a result that breaks the step's contract is never cached.
_STEP_RESULT_CHECK: ContextVar[Callable[[Any], None] | None] = ContextVar(
    "forger_step_result_check", default=None
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
    check_token: Token[Callable[[Any], None] | None] | None = None


def enter_step(rec: StepRecord, check: Callable[[Any], None]) -> ScopeToken:
    """Make ``rec`` the active step; ops it calls are top level (depth 0) again.

    ``check`` validates the step's result; see ``check_step_result``.
    """
    return ScopeToken(
        _ACTIVE_STEP.set(rec), _OP_DEPTH.set(0), _STEP_RESULT_CHECK.set(check)
    )


def exit_step(token: ScopeToken) -> None:
    """Undo ``enter_step``."""
    if token.check_token is not None:
        _STEP_RESULT_CHECK.reset(token.check_token)
    _OP_DEPTH.reset(token.depth_token)
    _ACTIVE_STEP.reset(token.owner_token)


def check_step_result(result: Any) -> None:
    """Run the active step's result check (it raises on a contract violation)."""
    check = _STEP_RESULT_CHECK.get()
    if check is not None:
        check(result)


def enter_pipeline(rec: PipelineRecord) -> ScopeToken:
    """Make ``rec`` the active pipeline; ops it calls directly are top level."""
    return ScopeToken(_ACTIVE_PIPELINE.set(rec), _OP_DEPTH.set(0))


def exit_pipeline(token: ScopeToken) -> None:
    """Undo ``enter_pipeline``."""
    _OP_DEPTH.reset(token.depth_token)
    _ACTIVE_PIPELINE.reset(token.owner_token)


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
        pipeline.steps.append(step_from_bare_op(call))


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
    )


def record_scores(
    label: str,
    values: list[Any],
    *,
    keep: Literal["ge", "le", "lt", "between", "true"],
    threshold: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    """Record one per-row score list for the active op; a no-op outside ops."""
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
        )
    )


def keep_where(items: list[Any], keep: list[bool]) -> list[Any]:
    """Record ``keep`` as a filter's per-row scores and return the kept items."""
    record_scores("kept", keep, keep="true")
    return [item for item, kept in zip(items, keep) if kept]


_USAGE_LOCK = threading.Lock()
_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def _as_number(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def add_usage(prompt_tokens: Any, completion_tokens: Any, cost_usd: Any) -> None:
    """Add one LLM call's usage to the process-wide counter (NaN counts as 0)."""
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += int(_as_number(prompt_tokens))
        _USAGE["completion_tokens"] += int(_as_number(completion_tokens))
        _USAGE["cost_usd"] += _as_number(cost_usd)


def usage_snapshot() -> tuple[int, int, float]:
    """Return the process-wide ``(prompt_tokens, completion_tokens, cost_usd)``."""
    with _USAGE_LOCK:
        return (
            int(_USAGE["prompt_tokens"]),
            int(_USAGE["completion_tokens"]),
            float(_USAGE["cost_usd"]),
        )


def usage_since(start: tuple[int, int, float]) -> tuple[int, int, float]:
    """Return the usage added since ``start`` (a ``usage_snapshot()``)."""
    now = usage_snapshot()
    return now[0] - start[0], now[1] - start[1], max(0.0, now[2] - start[2])


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

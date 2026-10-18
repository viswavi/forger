"""Shared run-time state for ``@op`` calls, ``@step`` reports, and filter scores.

This module imports nothing from the rest of pyforger but ``_spend`` and
``_rowflow`` (which import nothing), so ``op.py``, ``steps.py``,
``acquire/model.py``, and the filters can all use it without circular imports.

Context variables do not propagate into ``ThreadPoolExecutor`` workers. Ops
called from such workers see no active step and are not itemized in its report,
unless the work is submitted as ``pool.submit(contextvars.copy_context().run,
fn, arg)``. A step whose ops all ran in workers gets a warning instead of the
"called no Forger operations" error, and a step called from a worker thread
still joins the pipeline when exactly one pipeline is running.

Usage is counted in accumulators: entering a pipeline, step, or op pushes its
own ``Usage`` onto a context variable, and ``add_usage`` adds to every
accumulator open in the calling context. Workers started with
``copy_context().run`` share their parent's accumulators, so two ops running at
the same time each count only their own LLM calls. A step called from a
worker thread that did not copy the context joins the only running pipeline's
accumulator. Other usage from such threads reaches no step or pipeline total;
while exactly one pipeline runs it is shown as that pipeline's "unattributed"
usage, and it counts only toward the process total.
"""

from __future__ import annotations

import contextvars
import json
import math
import os
import threading
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from pyforger import _spend
from pyforger._rowflow import PipelineFlow, PythonGap, StepFlow

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
    # Set when the op raised, e.g. "ValueError: bad JSON"; its usage still counts.
    error: str | None = None
    # Positions in the step's ``ops`` of the earlier calls whose output rows this
    # call's input rows are (directly, or through other ops), by row identity.
    # The report sums chunks of one op into a row, but not a call fed by them.
    after: list[int] = Field(default_factory=list)
    # True when none of the input rows were the step's or an earlier call's
    # rows (rebuilt in Python, e.g. ``{**r, ...}``), False when some were, None
    # when unknown (no rows, or rows that have no identity).
    new_rows: bool | None = None


class FieldSummary(BaseModel):
    """What a report shows for one long value in a sample row.

    The stored sample keeps only the first 500 characters of a string, so the
    summaries are made from the whole value when the step returns. ``compact``
    is None when the value fits the step report as is, ``full`` when it fits
    ``inspect_step``. ``source`` is the clipped text an LLM summary reads, and
    ``llm`` that summary ("" when it failed, None before it is asked for).
    """

    shape: str  # e.g. "312 lines, 9.8k chars, python" or "[5 items]"
    compact: str | None = None
    full: str | None = None
    source: str = ""
    llm: str | None = None


class FieldInfo(BaseModel):
    """One field of a step's output rows, for the report's ``fields:`` line.

    ``types`` are the value type names seen, in order, ``"None"`` last (e.g.
    ``["int", "None"]``); ``optional`` is True when some rows lack the field.
    """

    name: str
    types: list[str]
    optional: bool = False


class RowStream(BaseModel):
    """One named list of rows: a step's input argument or one of its outputs.

    A step called as ``assemble(solvable, refusals)`` has two inputs, each with
    only ``name`` and ``n_rows`` set. A step that returns
    ``{"train": kept, "refusals": refused}`` has two output streams, each with
    its own fields, sample rows, and row type (``list[Train]`` when its type
    hint names one).
    """

    name: str
    n_rows: int
    fields: list[FieldInfo] | None = None
    fields_added: list[str] = Field(default_factory=list)
    fields_removed: list[str] = Field(default_factory=list)
    sample: list[Any] = Field(default_factory=list)
    sample_summaries: list[dict[int, FieldSummary]] = Field(default_factory=list)
    row_type: str | None = None


class UsageTotals(BaseModel):
    """Tokens and cost of one run, e.g. the run that cached a step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0
    duration_s: float = 0.0

    def is_zero(self) -> bool:
        """Return True when nothing was counted."""
        return not (
            self.prompt_tokens
            or self.completion_tokens
            or self.cost_usd
            or self.token_units
        )


class StepRecord(BaseModel):
    """Everything a ``@step`` call reports: its ops, totals, warnings, and samples.

    ``kind`` is ``"step"`` for a ``@step`` function, ``"dynamic_step"`` for
    ``alter.dynamic_step`` called directly in a pipeline, and ``"bare"`` for any
    other op called in a pipeline outside a step. ``sample_summaries[i]`` maps
    the position of a long field in ``sample[i]`` to its summary. On a cache
    hit the totals are this run's (zero) and ``original`` holds the time,
    tokens, and cost of the earlier run that cached the step.

    ``fields`` describes the output rows (None when they are not dicts);
    ``fields_added`` / ``fields_removed`` compare them with the input rows when
    those are dicts too. ``show`` names fields whose sample values print in full.
    ``python_gaps`` are rows that changed in plain Python between ops (see
    ``pyforger._rowflow``); a cached step keeps those of the run that cached it.

    ``n_in`` sums every rows argument; ``inputs`` names them when there are
    several (``solvable 111 + refusals 42``). A step that returned a dict of
    named row lists has ``streams`` (one per list, with its own fields and
    sample rows), ``n_out`` their sum, and no top-level ``sample``/``fields``.
    """

    name: str
    doc: str = ""
    kind: Literal["step", "dynamic_step", "bare"] = "step"
    n_in: int | None = None
    n_out: int | None = None
    inputs: list[RowStream] = Field(default_factory=list)
    streams: list[RowStream] | None = None
    duration_s: float = 0.0
    cache_hit: bool = False
    ops: list[OpCall] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample: list[Any] = Field(default_factory=list)
    sample_summaries: list[dict[int, FieldSummary]] = Field(default_factory=list)
    sample_shown: int = 1
    show: list[str] = Field(default_factory=list)
    fields: list[FieldInfo] | None = None
    fields_added: list[str] = Field(default_factory=list)
    fields_removed: list[str] = Field(default_factory=list)
    input_type: str | None = None
    output_type: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0
    original: UsageTotals | None = None
    python_gaps: list[PythonGap] = Field(default_factory=list)


class PipelineRecord(BaseModel):
    """Everything a ``@pipeline`` call reports: its steps, totals, and warnings.

    The totals are what this run's pipeline and steps spent. ``unattributed``
    is LLM usage from threads that did not copy the context, made while this
    was the only pipeline running; it is not in the totals, since such a
    thread may not belong to the pipeline.
    ``no_filter_steps`` and ``noop_filters`` (``(op, step, rows)``) are notes,
    not warnings: a filter that removed nothing may be a check that passed.
    ``python_gaps`` are rows dropped in plain Python between two steps, found
    by row identity (see ``pyforger._rowflow``).

    ``returned`` says what the pipeline returned: ``"rows"`` (a list of
    ``n_out`` rows), ``"streams"`` (a dict of row lists, in ``out_streams``,
    ``n_out`` their sum), ``"other"`` (anything else, e.g. a tuple or None:
    no final row count), or None for a record made before this was kept (the
    last step's output counts).
    """

    name: str
    returned: Literal["rows", "streams", "other"] | None = None
    n_out: int | None = None
    out_streams: list[RowStream] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    no_filter_steps: list[str] = Field(default_factory=list)
    noop_filters: list[tuple[str, str, int]] = Field(default_factory=list)
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    token_units: float = 0.0
    unattributed: UsageTotals = Field(default_factory=UsageTotals)
    max_cost_usd: float | None = None
    python_gaps: list[PythonGap] = Field(default_factory=list)

    @property
    def cached_cost_usd(self) -> float:
        """What the cached steps cost in the runs that cached them."""
        return sum(
            s.original.cost_usd for s in self.steps if s.cache_hit and s.original
        )


_USAGE_LOCK = threading.Lock()


class Usage:
    """Running usage totals of one pipeline, step, or op (see ``add_usage``).

    ``kind`` is ``"pipeline"``, ``"step"``, ``"op"``, or ``""`` for other
    totals; ``name`` is e.g. ``"filter.by_llm_score"``.
    """

    __slots__ = (
        "kind",
        "name",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "token_units",
    )

    def __init__(self, kind: str = "", name: str = "") -> None:
        """Start at zero."""
        self.kind = kind
        self.name = name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.token_units = 0.0

    def _add(self, prompt: int, completion: int, cost: float, units: float) -> None:
        # Callers hold _USAGE_LOCK.
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost
        self.token_units += units

    def add(self, totals: tuple[int, int, float, float]) -> None:
        """Add ``(prompt_tokens, completion_tokens, cost_usd, token_units)``."""
        with _USAGE_LOCK:
            self._add(*totals)

    def totals(self) -> tuple[int, int, float, float]:
        """Return ``(prompt_tokens, completion_tokens, cost_usd, token_units)``."""
        with _USAGE_LOCK:
            return (
                self.prompt_tokens,
                self.completion_tokens,
                self.cost_usd,
                self.token_units,
            )


# The accumulators open in this context, outermost first: e.g. (pipeline,
# step, op, nested op). Copied contexts share the same Usage objects.
_ACCUMULATORS: ContextVar[tuple[Usage, ...]] = ContextVar(
    "forger_usage_accumulators", default=()
)


def _open_usage(
    kind: str, name: str, joined: tuple[Usage, ...] = ()
) -> tuple[Usage, Token[tuple[Usage, ...]]]:
    """Push a new accumulator, after ``joined`` (accumulators to add to as well)."""
    usage = Usage(kind, name)
    return usage, _ACCUMULATORS.set(_ACCUMULATORS.get() + joined + (usage,))


def _pipeline_usage() -> Usage | None:
    """The accumulator of the pipeline this context counts toward, if any."""
    return next((u for u in _ACCUMULATORS.get() if u.kind == "pipeline"), None)


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
# The rows the active step's top-level ops took and returned (see _rowflow).
_ACTIVE_FLOW: ContextVar[StepFlow | None] = ContextVar(
    "forger_active_flow", default=None
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
    hook_token: Token[Any] | None
    usage: Usage
    usage_token: Token[tuple[Usage, ...]]
    flow_token: Token[StepFlow | None] | None = None


def enter_step(
    rec: StepRecord,
    check: Callable[[Any], str | None],
    flow: StepFlow | None = None,
) -> ScopeToken:
    """Make ``rec`` the active step; ops it calls are top level (depth 0) again.

    ``check`` validates the step's result; see ``check_step_result``. The
    step's usage is counted in ``token.usage``. A step called from a worker
    thread that did not copy the context joins the only running pipeline, so
    its usage counts toward that pipeline (and its cap) as it arrives. Its
    top-level ops' rows go to ``flow`` (see ``pyforger._rowflow``).
    """
    joined: tuple[Usage, ...] = ()
    if _ACTIVE_PIPELINE.get() is None and _pipeline_usage() is None:
        with _RUNNING_LOCK:
            if len(_RUNNING_PIPELINES) == 1:
                joined = (_RUNNING_PIPELINES[0].usage,)
    usage, usage_token = _open_usage("step", rec.name, joined)
    return ScopeToken(
        _ACTIVE_STEP.set(rec),
        _OP_DEPTH.set(0),
        _STEP_RESULT_CHECK.set(check),
        usage,
        usage_token,
        _ACTIVE_FLOW.set(flow if flow is not None else StepFlow(None, [])),
    )


def exit_step(token: ScopeToken) -> None:
    """Undo ``enter_step``."""
    if token.flow_token is not None:
        _ACTIVE_FLOW.reset(token.flow_token)
    _ACCUMULATORS.reset(token.usage_token)
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


# Score lists longer than this are not stored with a cached result (one
# number per row; the cached rows themselves are far larger).
MAX_STORED_SCORES = 100_000


def storable_scores(scores: ScoreRecord | None) -> ScoreRecord | None:
    """``scores`` when it is small enough to store with a cached result."""
    if scores is None or len(scores.values) > MAX_STORED_SCORES:
        return None
    return scores


def step_meta(rec: StepRecord, original: UsageTotals) -> str:
    """Serialize the ops a step ran, to store next to its cached result.

    On a cache hit ``restore_step_meta`` puts them back (marked cached, with no
    time or usage, and with their scores), so the cached step reports the same
    ops, yield warnings, and score histograms, and ``strict=True`` still sees
    them. ``original`` (the step's time, tokens, and cost in this run) becomes
    the cached step's ``original``.
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
                "scores": storable_scores(call.scores),
            }
        ).model_dump(mode="json")
        for call in rec.ops
    ]
    return json.dumps(
        {
            "version": 1,
            "ops": ops,
            "original": original.model_dump(mode="json"),
            "python_gaps": [g.model_dump(mode="json") for g in rec.python_gaps],
        }
    )


def restore_step_meta(text: str) -> None:
    """Give the active step the ops, usage, and (python) rows from ``step_meta``.

    Called on a cache hit. Metadata written before usage was stored has no
    ``original``, so the step shows no earlier cost.
    """
    step = _ACTIVE_STEP.get()
    if step is None:
        return
    try:
        meta = json.loads(text)
        step.ops = [OpCall.model_validate(d) for d in meta["ops"]]
    except (ValueError, KeyError, TypeError):
        step.ops = []  # unreadable metadata: report the step as fully cached
        return
    try:
        if meta.get("original") is not None:
            step.original = UsageTotals.model_validate(meta["original"])
    except ValueError:
        step.original = None
    try:
        step.python_gaps = [
            PythonGap.model_validate(g) for g in meta.get("python_gaps", [])
        ]
    except (ValueError, TypeError):
        step.python_gaps = []


@dataclass
class _Running:
    """A pipeline running in this process, with its usage so far."""

    rec: PipelineRecord
    usage: Usage
    unattributed: Usage
    flow: PipelineFlow = field(default_factory=PipelineFlow)


_RUNNING_PIPELINES: list[_Running] = []
_RUNNING_LOCK = threading.Lock()


def enter_pipeline(
    rec: PipelineRecord, on_dynamic_step: Callable[[StepRecord], None] | None = None
) -> ScopeToken:
    """Make ``rec`` the active pipeline; ops it calls directly are top level.

    ``on_dynamic_step`` is called with the record of each ``alter.dynamic_step``
    call made directly in the pipeline. ``rec.max_cost_usd`` caps its spend
    (see ``check_budget``).
    """
    usage, usage_token = _open_usage("pipeline", rec.name)
    with _RUNNING_LOCK:
        _RUNNING_PIPELINES.append(_Running(rec, usage, Usage("unattributed")))
    return ScopeToken(
        _ACTIVE_PIPELINE.set(rec),
        _OP_DEPTH.set(0),
        _ON_DYNAMIC_STEP.set(on_dynamic_step),
        usage,
        usage_token,
    )


def exit_pipeline(token: ScopeToken) -> None:
    """Undo ``enter_pipeline`` and set the record's usage totals.

    The totals are the pipeline's own usage; unattributed usage is kept apart.
    """
    rec = _ACTIVE_PIPELINE.get()
    unattributed = Usage()
    with _RUNNING_LOCK:
        for i, running in enumerate(_RUNNING_PIPELINES):
            if running.rec is rec:
                unattributed = running.unattributed
                del _RUNNING_PIPELINES[i]
                break
    _ACCUMULATORS.reset(token.usage_token)
    if token.hook_token is not None:
        _ON_DYNAMIC_STEP.reset(token.hook_token)
    _OP_DEPTH.reset(token.depth_token)
    _ACTIVE_PIPELINE.reset(token.owner_token)
    if rec is None:
        return
    extra = unattributed.totals()
    rec.unattributed = UsageTotals(
        prompt_tokens=extra[0],
        completion_tokens=extra[1],
        cost_usd=extra[2],
        token_units=extra[3],
    )
    (
        rec.prompt_tokens,
        rec.completion_tokens,
        rec.cost_usd,
        rec.token_units,
    ) = token.usage.totals()


def _running_for_context() -> _Running | None:
    """The running pipeline this context counts toward, if any.

    That is the pipeline whose accumulator is open here: the active pipeline,
    or the one a worker-thread step joined. A thread that did not copy the
    context and runs no step counts toward none.
    """
    usage = _pipeline_usage()
    if usage is None:
        return None
    with _RUNNING_LOCK:
        return next((r for r in _RUNNING_PIPELINES if r.usage is usage), None)


def pipeline_for_step() -> PipelineRecord | None:
    """Return the pipeline a finished step belongs to, if any.

    That is the pipeline active in this context or, for a step called from a
    worker thread (where context variables are unset), the only pipeline
    running in the process.
    """
    rec = _ACTIVE_PIPELINE.get()
    if rec is not None:
        return rec
    running = _running_for_context()
    if running is not None:
        return running.rec
    with _RUNNING_LOCK:
        return _RUNNING_PIPELINES[0].rec if len(_RUNNING_PIPELINES) == 1 else None


def _flow_of(rec: PipelineRecord) -> PipelineFlow | None:
    with _RUNNING_LOCK:
        running = next((r for r in _RUNNING_PIPELINES if r.rec is rec), None)
    return running.flow if running is not None else None


def step_starting(
    rec: PipelineRecord, rows_in: list[list[Any]], arg_lists: list[list[Any]]
) -> Any:
    """A step of ``rec`` starts with these rows; pass the result to ``join_pipeline``.

    ``rows_in`` are the step's rows arguments, each compared by identity with
    the previous step's output (each stream of it, when it returned a dict).
    ``arg_lists`` are all its list arguments.
    """
    flow = _flow_of(rec)
    return flow.start(rows_in, arg_lists) if flow is not None else None


def join_pipeline(
    rec: PipelineRecord, step: StepRecord, started: Any, rows_out: Any
) -> None:
    """Append a finished step (or bare op) to ``rec`` and hold its output rows.

    ``started`` is what ``step_starting`` returned for it.
    """
    flow = _flow_of(rec)
    with _RUNNING_LOCK:
        rec.steps.append(step)
        position = len(rec.steps) - 1
    if flow is not None:
        flow.finish(started, position, step.name, rows_out)


def close_pipeline_flow(rec: PipelineRecord, result: Any) -> None:
    """Set ``rec.python_gaps`` once the pipeline returned ``result``."""
    flow = _flow_of(rec)
    if flow is not None:
        rec.python_gaps = flow.close(result)


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
    usage: Usage | None = None
    usage_token: Token[tuple[Usage, ...]] | None = None

    def usage_totals(self) -> tuple[int, int, float, float]:
        """The op's own usage so far (zero for a step's cache wrapper)."""
        return self.usage.totals() if self.usage is not None else (0, 0, 0.0, 0.0)


def enter_op(op_name: str) -> OpToken:
    """Mark the start of an op call; nested calls fold into their parent."""
    if op_name.startswith(STEP_OP_PREFIX):
        # A step's own cache wrapper: the step itself does the reporting.
        return OpToken(is_step=True, depth=-1)
    if _COUNT_USAGE.get():  # report summaries must not look like a step's ops
        with _OPS_LOCK:
            _OPS_STARTED[0] += 1
    depth = _OP_DEPTH.get()
    usage, usage_token = _open_usage("op", op_name)
    return OpToken(
        is_step=False,
        depth=depth,
        depth_token=_OP_DEPTH.set(depth + 1),
        scores_token=_ACTIVE_SCORES.set([]),
        usage=usage,
        usage_token=usage_token,
    )


def current_scores() -> list[ScoreRecord]:
    """Return the score records the innermost active op has recorded so far."""
    return list(_ACTIVE_SCORES.get() or [])


def scores_meta(scores: list[ScoreRecord]) -> str | None:
    """Serialize an op's score records, to store next to its cached result.

    None when there is nothing to store (no scores, or too many values).
    """
    kept = [s for s in scores if storable_scores(s) is not None]
    if not kept:
        return None
    return json.dumps(
        {"version": 1, "scores": [s.model_dump(mode="json") for s in kept]}
    )


def restore_scores(text: str) -> None:
    """Record the scores from ``scores_meta`` for the active op (a cache hit)."""
    scores = _ACTIVE_SCORES.get()
    if scores is None:
        return
    try:
        records = [ScoreRecord.model_validate(d) for d in json.loads(text)["scores"]]
    except (ValueError, KeyError, TypeError):
        return  # unreadable: the report says the histogram is missing
    scores.extend(records)


def exit_op(
    token: OpToken, call: OpCall | None, rows: tuple[Any, Any] = (None, None)
) -> None:
    """Restore state from ``enter_op`` and hand a top-level call to its owner.

    Top-level calls (depth 0) go to the active step, or to the active
    pipeline as a bare op when no step is active. A call that raised (``error``
    set) goes only to a step, which may have caught the error; ``call`` is
    None when there is nothing to report. ``rows`` is the op's
    ``(first argument, result)``: the step's ``StepFlow`` uses them to tell
    chunks from chained calls and to find (python) rows; they are not kept
    past the step.
    """
    if token.is_step:
        step = _ACTIVE_STEP.get()
        if step is not None and call is not None:
            step.cache_hit = call.cache_hit
            step.warnings.extend(call.warnings)
        return
    if token.usage_token is not None:
        _ACCUMULATORS.reset(token.usage_token)
    if token.scores_token is not None:
        _ACTIVE_SCORES.reset(token.scores_token)
    if token.depth_token is not None:
        _OP_DEPTH.reset(token.depth_token)
    if call is None or token.depth != 0:
        return
    step = _ACTIVE_STEP.get()
    if step is not None:
        flow = _ACTIVE_FLOW.get()
        if flow is not None:
            flow.add(step.ops, call, *rows)
        else:
            step.ops.append(call)
        return
    pipeline = _ACTIVE_PIPELINE.get()
    if pipeline is not None and call.error is None:
        rec = step_from_bare_op(call)
        rows_in, rows_out = rows
        stream = (
            rows_in if isinstance(rows_in, list) and call.n_in is not None else None
        )
        lists = [stream] if stream is not None else []
        started = step_starting(pipeline, lists, lists)
        join_pipeline(pipeline, rec, started, rows_out)
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


# All LLM usage in this process, report summaries included; the
# FORGER_MAX_COST_USD cap is checked against it.
_PROCESS_USAGE = Usage()
# False inside ``run_uncounted``: LLM calls Forger makes for its own reports
# (summaries of sample values) stay out of every op, step, and pipeline total.
_COUNT_USAGE: ContextVar[bool] = ContextVar("forger_count_usage", default=True)


def run_uncounted(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)`` outside every step, pipeline, and op; skip its usage.

    It runs in a fresh context: its ops join no step or pipeline report, add
    nothing to ``ops_started``, and its LLM usage is added to no op, step, or
    pipeline (only to the process total and the spend ledger). Call it again
    from each worker thread, since threads do not inherit the context.
    """

    def run() -> Any:
        _COUNT_USAGE.set(False)
        return fn(*args)

    return contextvars.Context().run(run)


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
    *,
    model_id: str | None = None,
) -> None:
    """Record one LLM result's usage (NaN counts as 0).

    It is added to every accumulator open in this context (the pipeline, the
    step, and each enclosing op), to the process total, and to the spend
    ledger. Usage from a thread with no step or pipeline in its context (one
    that did not copy the context) is unattributed: when exactly one pipeline
    is running it shows on that pipeline's unattributed line, but it is not in
    the pipeline's totals and does not count toward its ``max_cost_usd``
    (the thread may not be the pipeline's). It always counts toward the
    process total and ``FORGER_MAX_COST_USD``.
    """
    totals = (
        int(_as_number(prompt_tokens)),
        int(_as_number(completion_tokens)),
        _as_number(cost_usd),
        _as_number(token_units),
    )
    accumulators = _ACCUMULATORS.get()
    counted = _COUNT_USAGE.get()
    step = _ACTIVE_STEP.get()
    pipeline_usage = _pipeline_usage()
    unattributed: list[Usage] = []
    if counted and step is None and pipeline_usage is None:
        with _RUNNING_LOCK:
            if len(_RUNNING_PIPELINES) == 1:
                unattributed = [_RUNNING_PIPELINES[0].unattributed]
    with _USAGE_LOCK:
        _PROCESS_USAGE._add(*totals)
        if counted:
            for usage in (*accumulators, *unattributed):
                usage._add(*totals)
    op = next((u.name for u in accumulators if u.kind == "op"), None)
    _spend.append(
        pipeline=pipeline_usage.name if pipeline_usage is not None else None,
        step=step.name if step is not None else None,
        op=op,
        model_id=model_id,
        prompt_tokens=totals[0],
        completion_tokens=totals[1],
        cost_usd=totals[2],
    )


def usage_snapshot() -> tuple[int, int, float, float]:
    """Return the process total, report summaries included.

    That is ``(prompt_tokens, completion_tokens, cost_usd, token_units)``.
    """
    return _PROCESS_USAGE.totals()


class BudgetExceededError(RuntimeError):
    """Spend reached ``@pipeline(max_cost_usd=...)`` or ``FORGER_MAX_COST_USD``."""


def _env_cap() -> float | None:
    raw = os.environ.get("FORGER_MAX_COST_USD", "").strip()
    if not raw:
        return None
    try:
        cap = float(raw)
    except ValueError:
        cap = math.nan
    if not cap >= 0:  # also rejects nan
        raise ValueError(
            f"FORGER_MAX_COST_USD must be a number of dollars >= 0, got {raw!r}"
        )
    return cap


def check_budget() -> None:
    """Raise ``BudgetExceededError`` when a spending cap is reached.

    Models call this before each request batch, and ``dynamic_step`` before
    and during each agent conversation. The caps are ``max_cost_usd`` of the
    pipeline this context counts toward (its spend this run, unattributed
    usage excluded) and ``FORGER_MAX_COST_USD`` (all spend in this process).
    Requests already sent finish and are recorded, so spend can pass a cap by
    at most one batch (per thread sending batches).
    """
    cap = _env_cap()
    if cap is not None:
        spent = _PROCESS_USAGE.totals()[2]
        if spent >= cap:
            raise BudgetExceededError(
                f"spent ${spent:.4f} in this process, reaching "
                f"FORGER_MAX_COST_USD=${cap:g}; no further LLM requests are sent"
            )
    running = _running_for_context()
    if running is None or running.rec.max_cost_usd is None:
        return
    spent = running.usage.totals()[2]
    if spent >= running.rec.max_cost_usd:
        raise BudgetExceededError(
            f"pipeline {running.rec.name} spent ${spent:.4f}, reaching its "
            f"max_cost_usd=${running.rec.max_cost_usd:g}; no further LLM requests "
            "are sent"
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

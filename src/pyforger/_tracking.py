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

import contextlib
import contextvars
import json
import math
import os
import threading
from collections.abc import Callable, Iterable, Iterator
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
    by row identity (see ``pyforger._rowflow``). ``notes`` are other
    informational lines (see ``close_retrieval_reach``).

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
    # Informational lines, e.g. how few retrieved documents reached a prompt.
    notes: list[str] = Field(default_factory=list)

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


def step_meta(rec: StepRecord, original: UsageTotals) -> str:
    """Serialize the ops a step ran, to store next to its cached result.

    On a cache hit ``restore_step_meta`` puts them back (marked cached, with no
    time, usage, or scores), so the cached step reports the same ops and yield
    warnings, and ``strict=True`` still sees them. ``original`` (the step's
    time, tokens, and cost in this run) becomes the cached step's ``original``.
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
    reach: _Reach = field(default_factory=lambda: _Reach())


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


# --------------------------------------------------------------------------- #
# Retrieval reach: did retrieved text get into a prompt or the output?
# --------------------------------------------------------------------------- #

_SHINGLE_CHARS = 60  # one fingerprint is 60 characters of a document's text
_SHINGLES_PER_DOC = 3
_MIN_DOC_CHARS = 200  # shorter documents (titles, snippets) are not fingerprinted
_SHINGLE_SPAN = 1500  # fingerprints come from a document's first 1,500 characters
_MAX_DOCS = 2000  # documents fingerprinted per pipeline run
_MAX_SCAN_CHARS = 200_000  # characters of one prompt or returned value checked
_MAX_SCAN_VALUES = 100_000  # string values of the returned rows checked
_FEW_USED = 0.10  # under this share of documents used, the summary notes it


def _squash(text: str) -> str:
    """Collapse whitespace runs, so a prompt's ``"a  b"`` matches ``"a b"``."""
    return " ".join(text[:_MAX_SCAN_CHARS].split())


def _main_text(row: Any) -> str | None:
    """A document's main text: its ``text`` field, else its longest string."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None
    text = row.get("text")
    if isinstance(text, str):
        return text
    strings = [v for v in row.values() if isinstance(v, str)]
    return max(strings, key=len) if strings else None


def fingerprints(text: str) -> list[str]:
    """Up to three 60-character pieces of ``text``, each starting at a word.

    They are spread over the first 1,500 characters, whitespace collapsed, so
    a prompt that quotes the document's opening (``doc[:1000]``) holds one.
    Texts shorter than 200 characters get none.
    """
    flat = _squash(text[: 2 * _SHINGLE_SPAN])
    if len(flat) < _MIN_DOC_CHARS:
        return []
    last = min(len(flat), _SHINGLE_SPAN) - _SHINGLE_CHARS
    out: list[str] = []
    for k in range(_SHINGLES_PER_DOC):
        space = flat.find(" ", last * k // _SHINGLES_PER_DOC, last)
        if space < 0:
            break
        piece = flat[space + 1 : space + 1 + _SHINGLE_CHARS]
        if piece not in out:
            out.append(piece)
    return out


def _found_in(text: str, pending: set[str]) -> set[str]:
    """The fingerprints in ``pending`` that occur in ``text`` (after a space)."""
    flat = _squash(text)
    found: set[str] = set()
    at = flat.find(" ") + 1
    while at:
        piece = flat[at : at + _SHINGLE_CHARS]
        if piece in pending:
            found.add(piece)
        at = flat.find(" ", at) + 1
    return found


def _prompt_text(prompt: Any) -> str:
    """The text of a string prompt or of a chat-message list's contents."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for message in prompt if isinstance(prompt, (list, tuple)) else ():
        content = message.get("content") if isinstance(message, dict) else message
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):  # [{"type": "text", "text": "..."}]
            parts.extend(
                p["text"]
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
    return "\n".join(parts)


class ReachLog:
    """What one op or step call added to its pipeline's retrieval check.

    ``op.py`` caches it next to the call's result, and a cache hit replays it:
    the documents a cached retrieval returned, and the fingerprints a cached
    op's prompts held. ``blind`` is True when the call sent prompts while
    every registered fingerprint had already been found, so they were not
    checked; replaying it makes the check inconclusive.
    """

    __slots__ = ("returned", "docs", "found", "prompts", "blind")

    def __init__(self) -> None:
        """Start empty."""
        self.returned: dict[str, int] = {}  # op label -> documents returned
        self.docs: list[tuple[str, list[str]]] = []  # (op label, fingerprints)
        self.found: set[str] = set()
        self.prompts = 0
        self.blind = False

    def dumps(self) -> str:
        """Serialize, to store next to the call's cached result."""
        return json.dumps(
            {
                "version": 1,
                "returned": self.returned,
                "docs": self.docs,
                "found": sorted(self.found),
                "prompts": self.prompts,
                "blind": self.blind,
            }
        )


class _Reach:
    """The retrieved documents of one pipeline run, and which reached a prompt."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.returned: dict[str, int] = {}  # op label -> documents returned
        self.docs: dict[tuple[str, ...], int] = {}  # fingerprints -> document
        self.pending: dict[str, int] = {}  # fingerprint -> document not yet seen
        self.used: set[int] = set()
        self.seen: set[str] = set()  # also fingerprints replayed before their doc
        self.prompts = 0
        self.blind = False

    def add(
        self,
        logs: Iterable[ReachLog],
        *,
        returned: dict[str, int] | None = None,
        docs: Iterable[tuple[str, list[str]]] = (),
        found: Iterable[str] = (),
        prompts: int = 0,
        blind: bool = False,
    ) -> None:
        """Add documents, found fingerprints, and prompts; ``logs`` get them too."""
        returned = returned or {}
        found = set(found)
        with self.lock:
            kept: list[tuple[str, list[str]]] = []
            for label, n in returned.items():
                self.returned[label] = self.returned.get(label, 0) + n
            for label, fps in docs:
                key = tuple(fps)
                if not fps or key in self.docs or len(self.docs) >= _MAX_DOCS:
                    continue
                index = self.docs[key] = len(self.docs)
                kept.append((label, fps))
                if self.seen.intersection(fps):
                    self.used.add(index)
                else:
                    self.pending.update((fp, index) for fp in fps)
            self.seen |= found
            for fp in found:
                used = self.pending.get(fp)
                if used is not None:  # the document is used: stop looking for it
                    self.used.add(used)
                    for other in [k for k, v in self.pending.items() if v == used]:
                        del self.pending[other]
            self.prompts += prompts
            self.blind = self.blind or blind
            for log in logs:
                for label, n in returned.items():
                    log.returned[label] = log.returned.get(label, 0) + n
                log.docs.extend(kept)
                log.found |= found
                log.prompts += prompts
                log.blind = log.blind or blind


_REACH_LOGS: ContextVar[tuple[ReachLog, ...]] = ContextVar(
    "forger_reach_logs", default=()
)


def _active_reach() -> _Reach | None:
    running = _running_for_context()
    return running.reach if running is not None else None


def reach_active() -> bool:
    """Return True when this context's calls count toward a pipeline's check."""
    return _active_reach() is not None


@contextlib.contextmanager
def reach_scope() -> Iterator[ReachLog | None]:
    """Collect what the calls inside add to the retrieval check (None outside)."""
    if _active_reach() is None:
        yield None
        return
    log = ReachLog()
    token = _REACH_LOGS.set(_REACH_LOGS.get() + (log,))
    try:
        yield log
    finally:
        _REACH_LOGS.reset(token)


def replay_reach(text: str) -> None:
    """Add a cached call's ``ReachLog.dumps()`` to the running pipeline's check."""
    reach = _active_reach()
    if reach is None:
        return
    try:
        data = json.loads(text)
        returned = {str(k): int(v) for k, v in data.get("returned", {}).items()}
        docs = [(str(label), [str(fp) for fp in fps]) for label, fps in data["docs"]]
        found = [str(fp) for fp in data.get("found", [])]
        prompts = int(data.get("prompts", 0))
        blind = bool(data.get("blind", False))
    except (ValueError, KeyError, TypeError, AttributeError):
        return  # unreadable: the call counts as adding nothing
    reach.add(
        _REACH_LOGS.get(),
        returned=returned,
        docs=docs,
        found=found,
        prompts=prompts,
        blind=blind,
    )


def mark_reach_blind() -> None:
    """Say a cached call's prompts are unknown, so the check stays silent.

    e.g. a generation op cached before the check existed: it may well have sent
    the retrieved text, and a "none of their text appeared" warning would be
    false.
    """
    reach = _active_reach()
    if reach is not None:
        reach.add(_REACH_LOGS.get(), blind=True)


def register_documents(label: str, rows: list[Any]) -> None:
    """Fingerprint the documents a retrieval op returned, for the pipeline's check.

    ``label`` names the op, e.g. ``acquire.retrieve_from_web``. When the
    pipeline returns, ``close_retrieval_reach`` warns if none of their text
    reached a model prompt (see ``note_prompt_texts``) or the returned rows.
    Does nothing outside a pipeline. At most 2,000 documents per pipeline run
    are fingerprinted; the others are only counted.
    """
    reach = _active_reach()
    if reach is None or not rows:
        return
    room = _MAX_DOCS - len(reach.docs)
    docs: list[tuple[str, list[str]]] = []
    for row in rows:
        if len(docs) >= room:
            break
        text = _main_text(row)
        fps = fingerprints(text) if text else []
        if fps:
            docs.append((label, fps))
    reach.add(_REACH_LOGS.get(), returned={label: len(rows)}, docs=docs)


def note_prompt_texts(prompts: Iterable[Any]) -> None:
    """Check prompts about to be sent for retrieved text (see ``register_documents``).

    Backend-agnostic: each model class calls it once per batch, before it
    sends the prompts or looks them up in a cache, as one line:
    ``_tracking.note_prompt_texts(prompts)``. Prompts are strings or
    chat-message lists. Outside a pipeline it does nothing; when no retrieved
    document is left unseen, it only counts the prompts.
    """
    reach = _active_reach()
    if reach is None:
        return
    prompts = list(prompts)
    with reach.lock:
        pending = set(reach.pending)
        blind = bool(prompts) and not pending and bool(reach.docs)
    found: set[str] = set()
    for prompt in prompts:
        if not pending:
            break
        hits = _found_in(_prompt_text(prompt), pending)
        found |= hits
        pending -= hits  # a found fingerprint is not looked for again
    reach.add(_REACH_LOGS.get(), found=found, prompts=len(prompts), blind=blind)


def _strings(value: Any, budget: list[int]) -> Iterator[str]:
    """String values in ``value`` (also in dicts and lists), ``budget[0]`` at most."""
    if budget[0] <= 0:
        return
    if isinstance(value, str):
        budget[0] -= 1
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v, budget)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v, budget)


def close_retrieval_reach(rec: PipelineRecord, result: Any) -> None:
    """Warn when no retrieved document reached a model prompt or ``result``.

    Called when the pipeline returns ``result``; its string values count as
    use too (retrieved text kept as data). Adds a warning to ``rec`` when
    retrieval ops returned fingerprinted documents and none was seen, and a
    note when under 10% were. Adds nothing when a cached call could not look
    (``ReachLog.blind``).
    """
    with _RUNNING_LOCK:
        running = next((r for r in _RUNNING_PIPELINES if r.rec is rec), None)
    if running is None or not running.reach.docs:
        return
    reach = running.reach
    with reach.lock:
        pending = set(reach.pending)
    found: set[str] = set()
    for text in _strings(result, [_MAX_SCAN_VALUES]):
        if not pending:
            break
        hits = _found_in(text, pending)
        found |= hits
        pending -= hits
    reach.add((), found=found)
    labels = list(reach.returned)
    who = " and ".join(
        [", ".join(labels[:-1]), labels[-1]] if len(labels) > 1 else labels
    )
    if not reach.used:
        if reach.blind:
            return
        sent = (
            "the 1 prompt"
            if reach.prompts == 1
            else f"any of the {reach.prompts:,} prompts"
        )
        rec.warnings.append(
            f"{who}: none of the {sum(reach.returned.values()):,} retrieved documents "
            f"were used in {sent} or the output; put their text in your prompts"
        )
    elif len(reach.used) < _FEW_USED * len(reach.docs):
        rec.notes.append(
            f"{who}: only {len(reach.used):,} of {len(reach.docs):,} retrieved "
            f"documents ({len(reach.used) / len(reach.docs):.0%}) were used in a "
            "prompt or the output"
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

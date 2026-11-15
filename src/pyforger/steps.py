"""``@step`` and ``@pipeline``: named, checked, and reported groups of Forger ops.

A step is a function from rows to rows built from Forger ops::

    @step
    def write_solutions(problems: list[Problem], model) -> list[Solved]:
        rows = alter.generate_from_context(problems, PROMPT, model)
        return filter.by_llm_score(rows, model=judge, ..., min_yield=0.3)

    @pipeline
    def main(model):
        return write_solutions(find_problems(model), model)

A step may take several lists of rows (``assemble(solvable, refusals)``; the
report counts ``solvable 111 + refusals 42``) and may return a dict of named
row lists (``{"train": kept, "refusals": refused}``), each reported as its own
stream. A later step takes one stream: ``format_train(out["train"])``.

``my_step.submit(rows, ...)`` starts a step in a background thread and returns
a ``Future``, so independent steps run at the same time::

    impossible = make_impossible.submit(verified[:n], writer, judge)
    hard = student_difficulty(verified[n:], student)  # runs meanwhile
    rows = assemble(impossible.result(), hard)

Each step is cached like an ``@op`` and prints a report when it returns: the
output rows' fields, phrases and names that many of them repeat (see
``pyforger._phrases``), every op it ran with rows in/out, yield, time, tokens, and
cost (repeated calls of one op share a row), a histogram of each filter's
scores, warnings, and a sample row. Rows that plain Python dropped or added
between ops, or between steps, show as dim ``(python)`` rows (see
``pyforger._rowflow``). A cached step prints one line. A pipeline
prints a summary table of its steps, with notes on steps without a filter and
filters that removed nothing.

Warnings are data problems (rows dropped, cut off, or unscored), and each says
what happened to the rows. Suggestions are about how the code is written (a
step with no Forger ops, no inspect op); they print dimmed. The pipeline
summary ends with every step's data problems, so the last lines of a log show
them. ``@step(prod=True)``, ``@pipeline(prod=True)``, or ``FORGER_STEP_REPORT=0``
silence the reports; ``strict=True`` turns warnings and suggestions into errors.

A few can be suppressed with a reason: ``@step(suppress_truncation_warning=
"<why these rows are still fine>")`` turns that step's truncation warnings into
one dim ``suppressed:`` line with their size, and the pipeline summary lists
every suppression (see ``_tracking.STEP_SUPPRESS_FLAGS``).

Long sample values are summarized by shape (``[5 items] a.py, b.py, …``).
``set_report_model(model)`` or ``FORGER_REPORT_MODEL`` adds a one-sentence LLM
summary of each; see ``pyforger._summaries``.
"""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import os
import queue
import re
import sys
import threading
import time
import typing
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from pyforger import _phrases, _summaries, _tracking
from pyforger._render import (
    print_pipeline,
    print_step,
    render_pipeline,
    render_scores,
    render_step,
    sample_summaries,
)
from pyforger._rowflow import StepFlow
from pyforger._summaries import set_report_model
from pyforger._tracking import (
    BudgetExceededError,
    FieldInfo,
    PipelineRecord,
    RowStream,
    StepRecord,
    UsageTotals,
)
from pyforger.op import UNSEEDED_HASH, _cache_dir, op, spread_indices
from pyforger.utils import _helper_key, calls_unseeded_hash, get_formatted_logger

F = TypeVar("F", bound=Callable[..., Any])

_logger = get_formatted_logger(__name__)

_DIVERSITY_FILTERS = frozenset(
    {
        "filter.by_embedding_similarity",
        "filter.by_exact_match",
        "filter.by_rouge_l",
        "filter.by_n_gram_overlap",
        "filter.by_topic_frequency",
    }
)
# Every other op of category "filter" counts as a quality filter, including a
# user op declared with @op(category="filter"). by_length and by_predicate
# check structure ("has ≥3 papers"), and any lambda would pass as a judge.
_NON_QUALITY_FILTERS = _DIVERSITY_FILTERS | {"filter.by_length", "filter.by_predicate"}

_SAMPLE_ROWS_KEPT = 10  # rows stored per record; inspect_step shows all of them
_SAMPLE_FIELD_CHARS = 500  # stored per string field; inspect_step shows all of it
_COMPACT_FIELD_CHARS = 80  # per field in the step report's sample lines
_WIDTH = 120  # report lines never exceed this many characters
_RULE_WIDTH = 100  # header rules are padded to this width
_NAME_CHARS = 40  # step and op names in report tables are cut to this length
_NOT_DICTS = "rows are not dicts; schema checks and samples skipped"
_SCHEMA_ROWS = 5000  # the fields line reads at most this many rows, evenly spaced


class StepContractError(RuntimeError):
    """A ``@step`` broke its contract (bad rows, nesting, or strict mode)."""


def _saved_step_path(name: str) -> Path:
    """Where the latest record of step ``name`` is saved, under ``FORGER_CACHE``."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return _cache_dir() / "step_reports" / f"{safe}-{digest}.json"


def _save_step(rec: StepRecord) -> None:
    """Save ``rec`` as its step's latest run, for ``inspect_step`` in another process.

    e.g. a pipeline run under nohup, inspected later from a shell. Values that
    are not JSON (sample rows holding objects) are saved as their repr. Never
    raises: a full disk must not fail the step.
    """
    path = _saved_step_path(rec.name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec.model_dump(), default=repr))
        os.replace(tmp, path)
    except Exception as exc:
        _logger.debug("could not save the report of step %s: %s", rec.name, exc)


def _save_run(rec: PipelineRecord) -> None:
    """Save a finished pipeline run, for ``inspect.runs``.

    Each run gets its own file, ``pipeline_runs/<name>-<time>-<pid>.json``, so
    two variants of a pipeline never overwrite each other's record. Sets
    ``rec.run_id`` to the file's stem. Never raises.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec.name)[:80]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec.run_id = f"{safe}-{stamp}-{os.getpid()}-{time.perf_counter_ns() % 10_000:04d}"
    path = _cache_dir() / "pipeline_runs" / f"{rec.run_id}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(rec.model_dump(), default=repr))
        os.replace(tmp, path)
    except Exception as exc:
        _logger.debug("could not save the run of pipeline %s: %s", rec.name, exc)


def saved_runs() -> list[PipelineRecord]:
    """Every saved pipeline run under ``FORGER_CACHE``, oldest first."""
    folder = _cache_dir() / "pipeline_runs"
    runs: list[PipelineRecord] = []
    for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime):
        try:
            if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
                continue
            runs.append(PipelineRecord.model_validate_json(path.read_text()))
        except (OSError, ValueError) as exc:
            _logger.debug("skipping saved run %s: %s", path, exc)
    return runs


def saved_step_records() -> dict[str, StepRecord]:
    """The saved latest record of every step under ``FORGER_CACHE``, oldest first.

    Files another user wrote, or that do not parse, are skipped.
    """
    folder = _cache_dir() / "step_reports"
    paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
    records: dict[str, StepRecord] = {}
    for path in paths:
        try:
            if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
                continue
            rec = StepRecord.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            _logger.debug("skipping saved step report %s: %s", path, exc)
            continue
        records.pop(rec.name, None)
        records[rec.name] = rec
    return records


class UntestedFullRunError(RuntimeError):
    """A full run of pipeline code that never finished a limited run."""


def _full_run_ok() -> bool:
    return os.environ.get("FORGER_FULL_RUN", "").strip().lower() in ("1", "true", "yes")


def _code_key(fn: Any) -> str:
    """A hash of a pipeline's code, as ``@op`` keys functions.

    It covers the pipeline's body, the steps and helpers it calls, and the
    plain-data globals they read (prompts, constants).
    """
    try:
        key: Any = _helper_key(fn)
    except Exception:  # noqa: BLE001 - fall back to the name
        key = getattr(fn, "__qualname__", repr(fn))
    return hashlib.sha256(repr(key).encode()).hexdigest()[:20]


def _limited_marker(name: str, code: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80]
    return _cache_dir() / "limited_runs" / f"{safe}-{code}.json"


def _check_tested(name: str, fn: Any) -> None:
    """Refuse a full run of code with no finished limited run (see ``pipeline``)."""
    if _full_run_ok() or _limited_marker(name, _code_key(fn)).exists():
        return
    raise UntestedFullRunError(
        f"pipeline {name}: this code has not finished a limited run, so a full run "
        "would find its bugs at full size and full cost. Run it first with "
        "FORGER_LIMIT=3 (a smoke run) or FORGER_LIMIT=50 (a pilot); once a limited "
        "run of the same code finishes, the full run starts. Any edit to the "
        "pipeline, its steps, their helpers, or the prompts and constants they read "
        "is new code; sizes passed as arguments are not."
    )


def _mark_tested(name: str, fn: Any, limit: int) -> None:
    """Record that a limited run of this code finished. Never raises."""
    path = _limited_marker(name, _code_key(fn))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"limit": limit, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        )
    except OSError as exc:
        _logger.debug("could not record the limited run of %s: %s", name, exc)


class PipelineContractError(RuntimeError):
    """A ``@pipeline`` broke its contract (no steps, nesting, or strict mode)."""


def _reports_enabled(prod: bool) -> bool:
    flag = os.environ.get("FORGER_STEP_REPORT", "").strip().lower()
    return not prod and flag not in {"0", "false", "no"}


# --------------------------------------------------------------------------- #
# Step.submit: steps in background threads
# --------------------------------------------------------------------------- #

# At most this many submitted steps run at once in the process; later ones
# wait, in submission order, for a free thread.
SUBMIT_WORKERS = 8


class _StepFuture(Future):  # type: ignore[type-arg]
    """The Future ``.submit`` returns; it remembers whether its outcome was read."""

    def __init__(self, step_name: str) -> None:
        """Start pending."""
        super().__init__()
        self.step_name = step_name
        self.read = False

    def result(self, timeout: float | None = None) -> Any:
        """Wait for the step; return its rows or raise its exception."""
        self.read = True
        return super().result(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Wait for the step; return the exception it raised, if any."""
        self.read = True
        return super().exception(timeout)


class _StepPool:
    """``SUBMIT_WORKERS`` daemon threads that run submitted steps in order.

    Not a ``ThreadPoolExecutor``: its threads are joined at interpreter exit,
    so Ctrl-C during a slow judge step would wait for the step to finish.
    Daemon threads end with the process.
    """

    def __init__(self, workers: int) -> None:
        """Start no threads until the first submission."""
        self._workers = workers
        self._queue: queue.SimpleQueue[
            tuple[Future, Callable[[], Any]]
        ] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def submit(self, future: Future, fn: Callable[[], Any]) -> None:
        """Run ``fn()`` in a worker thread and set ``future`` to its outcome."""
        self._queue.put((future, fn))
        with self._lock:
            if len(self._threads) < self._workers:
                thread = threading.Thread(
                    target=self._work,
                    name=f"forger-step-{len(self._threads)}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def _work(self) -> None:
        while True:
            future, fn = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue  # cancelled while queued
            try:
                value = fn()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(value)
            del future, fn  # let go of the step's rows


_POOL = _StepPool(SUBMIT_WORKERS)
# The steps the running pipeline submitted, in submission order.
_SUBMITTED: contextvars.ContextVar[list[_StepFuture] | None] = contextvars.ContextVar(
    "forger_submitted_steps", default=None
)


def _await_submitted(submitted: list[_StepFuture]) -> None:
    """Wait for every submitted step; raise the first error nobody read.

    A step whose ``.result()`` the pipeline never called still finishes before
    the pipeline does, so its report and usage are in the summary, and its
    exception is not lost.
    """
    wait_futures(submitted)
    for future in submitted:
        if future.read or future.cancelled():
            continue
        exc = Future.exception(future)
        if exc is not None:
            typing.cast(Any, exc).add_note(  # mypy's target predates add_note
                f"raised by step {future.step_name}, run with .submit() and never "
                "awaited"
            )
            raise exc


# --------------------------------------------------------------------------- #
# Row schemas from type hints
# --------------------------------------------------------------------------- #


def _row_model(hint: Any) -> type[BaseModel] | None:
    """Return ``X`` when ``hint`` is ``list[X]`` with ``X`` a pydantic model."""
    if typing.get_origin(hint) is not list:
        return None
    args = typing.get_args(hint)
    if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        return args[0]
    return None


def _is_rows_hint(hint: Any) -> bool:
    """``list[dict]``, ``list[dict[str, Any]]``, or ``list[X]`` with ``X`` a model."""
    if _row_model(hint) is not None:
        return True
    args = typing.get_args(hint) if typing.get_origin(hint) is list else ()
    return len(args) == 1 and (args[0] is dict or typing.get_origin(args[0]) is dict)


def _type_label(model: type[BaseModel] | None) -> str:
    return f"list[{model.__name__}]" if model is not None else "any"


@dataclass
class _OutHint:
    """What a step's return annotation says about its rows.

    ``list[X]``: one list of ``X`` rows (``streams`` is None).
    ``dict[str, list[X]]``: any streams, each of ``X`` rows (``every``).
    A ``TypedDict`` of ``list[X]`` values: those streams (``streams``, value
    None for a stream without a row model), ``required`` among them.
    """

    model: type[BaseModel] | None = None
    every: type[BaseModel] | None = None
    streams: dict[str, type[BaseModel] | None] | None = None
    required: frozenset[str] = frozenset()
    name: str = ""

    @property
    def wants_dict(self) -> bool:
        return self.every is not None or self.streams is not None

    def stream_model(self, stream: str) -> type[BaseModel] | None:
        if self.streams is not None:
            return self.streams.get(stream)
        return self.every

    def label(self) -> str:
        if self.model is not None:
            return _type_label(self.model)
        if self.every is not None:
            return f"dict[str, {_type_label(self.every)}]"
        return " · ".join(
            f"{k} {_type_label(m)}" for k, m in (self.streams or {}).items()
        )


def _out_hint(hint: Any, localns: dict[str, Any] | None) -> _OutHint | None:
    """Read a return annotation: ``list[X]``, ``dict[str, list[X]]``, or a TypedDict."""
    model = _row_model(hint)
    if model is not None:
        return _OutHint(model=model)
    if typing.get_origin(hint) is dict:
        args = typing.get_args(hint)
        every = _row_model(args[1]) if len(args) == 2 else None
        return _OutHint(every=every) if every is not None else None
    if typing.is_typeddict(hint):
        fields = typing.get_type_hints(hint, localns=localns)
        return _OutHint(
            streams={k: _row_model(v) for k, v in fields.items()},
            required=frozenset(getattr(hint, "__required_keys__", fields)),
            name=hint.__name__,
        )
    return None


@dataclass
class _Hints:
    """Row types from a step's annotations (see ``_step_hints``)."""

    params: dict[str, Any] = field(default_factory=dict)  # name -> resolved hint
    out: _OutHint | None = None
    warning: str | None = None

    def model(self, param: str) -> type[BaseModel] | None:
        return _row_model(self.params.get(param))


def _step_hints(
    fn: Callable[..., Any], localns: dict[str, Any] | None = None
) -> _Hints:
    """Row types of ``fn``'s parameters and return value, from its annotations.

    ``localns`` holds the names of the scope that defined ``fn``, so string
    annotations naming a class defined in a function body still resolve. A
    hint that cannot be resolved gives a warning instead.
    """
    params = list(inspect.signature(fn).parameters)
    try:
        hints = typing.get_type_hints(fn, localns=localns)
        out = _out_hint(hints.get("return"), localns)
    except Exception as exc:
        annotations = getattr(fn, "__annotations__", {}) or {}
        # Warn only when a hint could name rows: the return value, the first
        # parameter, or any list/dict hint (not ``model: "Model"``).
        if not (
            {"return", *params[:1]} & set(annotations)
            or any("list" in str(a) or "dict" in str(a) for a in annotations.values())
        ):
            return _Hints()
        return _Hints(
            warning="type hints could not be resolved "
            f"({type(exc).__name__}: {exc}); schema checks skipped"
        )
    return _Hints({k: v for k, v in hints.items() if k in params}, out)


def _pydantic_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(row)"
        parts.append(f"{loc}: {err.get('msg', '')}")
    more = len(exc.errors()) - 3
    if more > 0:
        parts.append(f"and {more} more")
    return "; ".join(parts)


def _validate_rows(
    rows: list[Any], model: type[BaseModel], step_name: str, which: str
) -> None:
    """Raise StepContractError naming the first row that does not match ``model``."""
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise StepContractError(
                f"step {step_name}: {which} row {i} is a {type(row).__name__}, "
                f"not a dict matching {model.__name__}"
            )
        try:
            model.model_validate(row)
        except ValidationError as exc:
            raise StepContractError(
                f"step {step_name}: {which} row {i} does not match "
                f"{model.__name__}: {_pydantic_message(exc)}"
            ) from None


def _spread(rows: list[Any], limit: int = _SCHEMA_ROWS) -> list[Any]:
    """All rows, or ``limit`` of them evenly spaced for a huge list."""
    if len(rows) <= limit:
        return rows
    return [rows[i * len(rows) // limit] for i in range(limit)]


def _keys(rows: list[dict]) -> list[Any]:
    """Every key of ``rows``, in order of first appearance."""
    return list(dict.fromkeys(k for row in _spread(rows) for k in row))


def _type_name(value: Any) -> str:
    return "None" if value is None else type(value).__name__


def _field_infos(rows: list[dict]) -> list[FieldInfo]:
    """Field names of ``rows`` with the value types seen, e.g. ``score int|None``."""
    scanned = _spread(rows)
    types: dict[Any, dict[str, None]] = {}
    present: dict[Any, int] = {}
    for row in scanned:
        for key, value in row.items():
            types.setdefault(key, {})[_type_name(value)] = None
            present[key] = present.get(key, 0) + 1
    return [
        FieldInfo(
            name=str(key),
            # None goes last: "int|None", not "None|int".
            types=sorted(seen, key=lambda t: t == "None"),
            optional=present[key] < len(scanned),
        )
        for key, seen in types.items()
    ]


def _truncate(value: Any, limit: int) -> Any:
    """Shorten strings (also inside lists and dicts) to ``limit`` characters."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, limit) for v in list(value)[:20]]
    return value


# --------------------------------------------------------------------------- #
# @step
# --------------------------------------------------------------------------- #


def step(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    cache: bool = True,
    prod: bool = False,
    strict: bool = False,
    sample: int = 1,
    show: list[str] | tuple[str, ...] | str = (),
    suppress_truncation_warning: str | None = None,
    suppress_no_ops_warning: str | None = None,
    **unknown: Any,
) -> Any:
    """Decorate a function from rows to rows built from Forger ops.

    The step is cached like an ``@op`` (under the op name ``step.<name>``) and
    prints a report when it returns. Its input is every argument that holds
    rows (see ``_row_arguments``): with several, the report shows each count,
    e.g. ``solvable 111 + refusals 42 → 153 rows``, and the yield uses the sum.
    It returns a list of rows or a dict of named row lists (streams), e.g.
    ``{"train": kept, "refusals": refused}``; each stream gets its own count,
    fields line, and sample rows. In a limited run (``@pipeline(limit=3)`` or
    ``FORGER_LIMIT=3``) each rows argument is cut to 3 rows spread over it
    (``pyforger.op.spread_indices``) before the call.

    If a parameter or the return value is annotated ``list[X]`` with ``X`` a
    pydantic model, every row must be a dict that validates against ``X``
    (rows stay dicts). For streams, annotate ``dict[str, list[X]]`` (every
    stream holds ``X`` rows) or a ``TypedDict`` whose values are ``list[X]``
    (those streams, each with its own row model).

    ``my_step.submit(*args, **kwargs)`` runs the step in a background thread
    (at most ``SUBMIT_WORKERS`` at once) and returns a ``Future``; see
    ``pipeline`` for what happens to submitted steps when the pipeline ends.

    Errors (``StepContractError``): the result is neither a list nor a dict of
    lists, rows do not match the annotated row model (or the streams those a
    ``TypedDict`` names), or the step was called inside another step.
    Warnings (data problems): an op missed its ``min_yield`` / ``max_yield``
    (a suggestion in a limited run on < 20 rows), or its completions stopped at
    max_tokens.
    Suggestions: the step called no Forger ops, the rows are not dicts,
    ``show`` names a field no output row has, or the type hints could not be
    resolved.

    Args:
        _fn: The function, when used as ``@step`` without parentheses.
        name: Step name in reports and the cache; defaults to the function name.
        cache: Cache the step's result on disk, like ``@op(cache=...)``.
        prod: Do not print the report.
        strict: Raise ``StepContractError`` when the step has any warning or
            suggestion; such a result is not cached.
        sample: Number of output rows shown in the report, per stream
            (``inspect_step`` shows up to 10).
        show: Fields of the sample rows to print in full, each as an indented
            block under its row, e.g. ``show=["statement"]``. Other fields are
            cut or summarized.
        suppress_truncation_warning: Why this step's completions cut off at
            max_tokens do not hurt its rows. Its truncation warnings then print
            as one dim ``suppressed:`` line with their size and the reason, and
            leave the data-problem count and ``strict``. If no truncation
            occurs in any run of the step, that is a warning (remove the flag).
        suppress_no_ops_warning: Why this step has no Forger ops; the
            "called no Forger operations" suggestion is suppressed the same way.
        **unknown: Any other keyword raises TypeError with the closest flag.
            A reason under 3 words, or a placeholder, raises ValueError.
    """
    shown_fields = [show] if isinstance(show, str) else list(show)
    suppress = _tracking.check_suppress(
        "step",
        {
            "suppress_truncation_warning": suppress_truncation_warning,
            "suppress_no_ops_warning": suppress_no_ops_warning,
        },
        unknown,
    )
    # The scope that defines the step (where ``step`` is called, both for
    # ``@step`` and ``@step(...)``), for resolving string annotations.
    frame = sys._getframe(1)
    localns = dict(frame.f_locals) if frame.f_locals is not frame.f_globals else None
    del frame

    def decorate(fn: F) -> F:
        step_name = name or fn.__name__
        cached = op(fn, name=f"{_tracking.STEP_OP_PREFIX}{step_name}", cache=cache)
        signature = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        unseeded_hash = calls_unseeded_hash(fn)
        hints: list[_Hints] = []  # resolved on the first call, when names exist

        def check_not_nested() -> None:
            outer = _tracking.active_step()
            if outer is not None:
                raise StepContractError(
                    f"step {step_name} was called inside step {outer.name}; steps "
                    "cannot nest, so call both from the @pipeline"
                )

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            check_not_nested()
            if not hints or hints[0].warning is not None:
                hints[:] = [_step_hints(fn, localns)]
            hint = hints[0]
            row_args = _row_arguments(signature, hint, args, kwargs)
            # The pipeline flow compares the uncut lists with the previous
            # step's output, so a smoke cut is not shown as (python) rows.
            full_rows = [rows for _, _, rows in row_args]
            full_lists = [a for a in (*args, *kwargs.values()) if isinstance(a, list)]
            smoke = _tracking.smoke_rows()
            if smoke is not None:
                args, kwargs, row_args = _smoke_cut(row_args, args, kwargs, smoke)
            several = len(row_args) > 1
            models = [(label, hint.model(param)) for label, param, _ in row_args]
            if several:
                input_type = (
                    " + ".join(f"{label} {_type_label(m)}" for label, m in models)
                    if any(m for _, m in models)
                    else None
                )
            else:
                first = models[0][1] if models else None
                params = list(signature.parameters)
                first = first or (hint.model(params[0]) if params else None)
                input_type = _type_label(first) if first else None
            rec = StepRecord(
                name=step_name,
                doc=doc,
                sample_shown=sample,
                show=shown_fields,
                input_type=input_type,
                output_type=hint.out.label() if hint.out else None,
                suppress=suppress,
            )
            if hint.warning:
                _tracking.add_problem(rec, "type_hints", hint.warning, suggestion=True)
            if unseeded_hash:
                _tracking.add_problem(
                    rec, "unseeded_hash", f"this step {UNSEEDED_HASH}", suggestion=True
                )
            if row_args:
                rec.n_in = sum(len(rows) for _, _, rows in row_args)
            if smoke is not None:
                rec.smoke_rows = smoke
                if row_args:
                    rec.smoke_n_in = sum(len(rows) for rows in full_rows)
            if several:
                rec.inputs = [
                    RowStream(name=label, n_rows=len(rows))
                    for label, _, rows in row_args
                ]
            for (label, _, rows), (_, model) in zip(row_args, models):
                if model is not None:
                    which = f"input {label}" if several else "input"
                    _validate_rows(rows, model, step_name, which)
            filled = [rows for _, _, rows in row_args if rows]
            input_keys: list[Any] | None = None
            if filled and all(isinstance(r, dict) for rows in filled for r in rows):
                input_keys = _keys([r for rows in filled for r in rows])

            # (python) rows inside the step compare the rows of one stream; with
            # several list arguments the step's input is unclear, so it has none.
            arg_lists = [a for a in (*args, *kwargs.values()) if isinstance(a, list)]
            one_list = (
                len(row_args) == 1
                and len(arg_lists) == 1
                and arg_lists[0] is row_args[0][2]
            )
            stream = row_args[0][2] if one_list else None
            flow = StepFlow(stream, arg_lists)
            pipeline_rec = _tracking.pipeline_for_step()
            started = (
                _tracking.step_starting(pipeline_rec, full_rows, full_lists)
                if pipeline_rec is not None
                else None
            )

            checked: list[bool] = []
            strict_failed: list[bool] = []
            ops_before = _tracking.ops_started()

            def check(result: Any) -> str:
                # Called by @op on a fresh result, before it is cached: raising
                # here keeps a result that breaks the contract out of the cache.
                checked.append(True)
                if isinstance(result, list):
                    rec.python_gaps = flow.gaps(result)
                elif _is_streams(result):  # every stream's rows, as one return
                    rec.python_gaps = flow.gaps([r for v in result.values() for r in v])
                _check_step_result(
                    rec,
                    result,
                    hint.out,
                    _tracking.ops_started() - ops_before,
                    input_keys,
                )
                _tracking.apply_suppress(rec, suppress, step_name)
                if strict and (rec.warnings or rec.suggestions):
                    strict_failed.append(True)
                    raise _strict_error(rec)
                return _tracking.step_meta(rec, _usage_totals(scope.usage, start))

            start = time.perf_counter()

            def finish(result: Any = None) -> None:
                rec.duration_s = time.perf_counter() - start
                usage = scope.usage.totals()
                (
                    rec.prompt_tokens,
                    rec.completion_tokens,
                    rec.cost_usd,
                    rec.token_units,
                ) = usage
                if _reports_enabled(prod):  # only printed reports pay for it
                    _add_repeats(rec, result, filled if input_keys is not None else [])
                _tracking.store_step(rec)
                if pipeline_rec is not None:
                    _tracking.join_pipeline(pipeline_rec, rec, started, result)
                if _reports_enabled(prod):
                    if not rec.cache_hit:  # a cached step prints one line
                        _summaries.add_llm_summaries(rec, rows=rec.sample_shown)
                    print_step(rec)
                _save_step(rec)

            scope = _tracking.enter_step(rec, check, flow)
            try:
                result = cached(*args, **kwargs)
            except StepContractError:
                if strict_failed:  # report the step, then raise
                    finish()
                raise
            finally:
                _tracking.exit_step(scope)
                flow.close()  # let go of the rows the step's ops saw

            if not checked:  # a cache hit
                _check_step_result(rec, result, hint.out, 0, input_keys)
            if smoke is not None and rec.n_in and result == []:
                _tracking.add_problem(
                    rec,
                    "empty_limited",
                    f"step {step_name} returned 0 of {rec.n_in} rows in a limited "
                    f"run (limit={smoke}), so the steps after it check nothing; fix "
                    "this step or raise the limit",
                )
            # Again: the step's own min_yield warning came after ``check``.
            _tracking.apply_suppress(rec, suppress, step_name)
            if pipeline_rec is None and smoke is None:  # else the pipeline checks
                for kind in _tracking.stale_suppressions(suppress, rec.suppressed):
                    _tracking.add_stale(rec, kind, step_name)
            finish(result)
            if strict and (rec.warnings or rec.suggestions):
                raise _strict_error(rec)
            return result

        def submit(*args: Any, **kwargs: Any) -> Future:
            """Start the step in a background thread; return a ``Future`` of its rows.

            The step runs in a copy of the caller's context, so it joins the
            same pipeline, counts toward its spending caps, and gets its own
            report. ``.result()`` returns the step's rows or raises its error.
            """
            check_not_nested()
            future = _StepFuture(step_name)
            pipeline_rec = _tracking.active_pipeline()
            seq = (
                _tracking.reserve_step_seq(pipeline_rec)
                if pipeline_rec is not None
                else None
            )
            submitted = _SUBMITTED.get()
            if submitted is not None:
                submitted.append(future)
            context = contextvars.copy_context()
            _POOL.submit(
                future,
                lambda: context.run(
                    _tracking.run_reserved, seq, lambda: wrapper(*args, **kwargs)
                ),
            )
            return future

        wrapper.submit = submit  # type: ignore[attr-defined]
        return typing.cast(F, wrapper)

    if _fn is not None:
        return decorate(_fn)
    return decorate


def _add_repeats(rec: StepRecord, result: Any, inputs: list[list[Any]]) -> None:
    """Set the ``repeated:`` items of the step's output rows (of each stream).

    ``inputs`` are the step's rows arguments when they are dicts, else empty.
    """
    before = [row for rows in inputs for row in rows]
    if isinstance(result, list):
        rec.repeated = _phrases.step_repeats(result, before)
    elif _is_streams(result) and rec.streams:
        for stream in rec.streams:
            stream.repeated = _phrases.step_repeats(result.get(stream.name, []), before)


def _usage_totals(usage: _tracking.Usage, start: float) -> UsageTotals:
    """A step's usage and time so far, stored with its cached result."""
    prompt, completion, cost, units = usage.totals()
    return UsageTotals(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost,
        token_units=units,
        duration_s=time.perf_counter() - start,
    )


def _strict_error(rec: StepRecord) -> StepContractError:
    return StepContractError(
        f"step {rec.name} has warnings (strict=True):\n"
        + "\n".join(f"  - {w}" for w in [*rec.warnings, *rec.suggestions])
    )


def _row_arguments(
    signature: inspect.Signature, hints: _Hints, args: tuple, kwargs: dict
) -> list[tuple[str, str, list[Any]]]:
    """The step's arguments that hold rows, as ``(label, parameter, rows)``.

    The first argument counts when it is a list (also a list of seed topic
    names, as a step's input always has). Any other argument counts when it
    is a non-empty list of dicts, or a list annotated ``list[dict]`` or
    ``list[X]``. Models, numbers, strings, and other lists (topic names
    after the first argument, a list of judges) do not count. ``*rows``
    arguments are labelled ``rows[0]``, ``rows[1]``, ...
    """
    # Per-call controls of the step's @op wrapper, not the function's own.
    reserved = ("_no_cache", "_cache_only", "min_yield", "max_yield")
    own = {k: v for k, v in kwargs.items() if k not in reserved}
    try:
        bound = signature.bind(*args, **own)
    except TypeError:
        return []  # the call itself raises
    out: list[tuple[str, str, list[Any]]] = []
    for i, (param, value) in enumerate(bound.arguments.items()):
        kind = signature.parameters[param].kind
        if kind is inspect.Parameter.VAR_POSITIONAL:
            items = [(f"{param}[{j}]", v) for j, v in enumerate(value)]
        elif kind is inspect.Parameter.VAR_KEYWORD:
            items = [(str(k), v) for k, v in value.items()]
        else:
            items = [(param, value)]
        hinted = _is_rows_hint(hints.params.get(param))
        for j, (label, v) in enumerate(items):
            if not isinstance(v, list):
                continue
            dicts = bool(v) and all(isinstance(row, dict) for row in v)
            if (i == 0 and j == 0) or hinted or dicts:
                out.append((label, param, v))
    return out


def _smoke_cut(
    row_args: list[tuple[str, str, list[Any]]],
    args: tuple,
    kwargs: dict,
    limit: int,
) -> tuple[tuple, dict, list[tuple[str, str, list[Any]]]]:
    """Cut each rows argument to ``limit`` rows spread over it, for a smoke run.

    Rows arguments of the same length are cut at the same positions (those of
    the first one), so lists that go together stay aligned, e.g. rows and
    their labels. Returns the step's new ``args``, ``kwargs``, and rows
    arguments. The rows arguments are the very objects in ``args`` /
    ``kwargs`` (also each ``*rows`` item), so they are swapped by identity.
    """
    keep: dict[int, list[int]] = {}  # length -> positions
    cut = {}
    for _, _, rows in row_args:
        if len(rows) > limit:
            positions = keep.setdefault(len(rows), spread_indices(rows, limit))
            cut[id(rows)] = [rows[i] for i in positions]
    if not cut:
        return args, kwargs, row_args
    return (
        tuple(cut.get(id(a), a) for a in args),
        {k: cut.get(id(v), v) for k, v in kwargs.items()},
        [(label, param, cut.get(id(rows), rows)) for label, param, rows in row_args],
    )


def _is_streams(result: Any) -> bool:
    """True for a dict of named row lists: ``{"train": [...], "refusals": [...]}``."""
    return isinstance(result, dict) and all(
        isinstance(k, str) and isinstance(v, list) for k, v in result.items()
    )


def _not_streams(name: str, result: dict) -> StepContractError:
    key, value = next(
        (k, v)
        for k, v in result.items()
        if not (isinstance(k, str) and isinstance(v, list))
    )
    what = (
        f"key {key!r} is not a string"
        if not isinstance(key, str)
        else f"value {key!r} is {type(value).__name__}, not a list"
    )
    return StepContractError(
        f"step {name} returned dict whose {what}; a step returns rows or a dict "
        "of named row lists"
    )


def _describe(
    rows: list[Any], show: list[str], input_keys: list[Any] | None, into: Any
) -> list[Any] | None:
    """Fill ``into``'s sample and fields from dict ``rows``; return their keys.

    ``into`` is the step record (one list of rows) or one ``RowStream``.
    Returns None, filling nothing, when the rows are not all dicts.
    """
    if not all(isinstance(row, dict) for row in rows):
        return None
    kept = rows[:_SAMPLE_ROWS_KEPT]
    shown = set(show)
    # Fields named in show= are kept whole; they print in full.
    into.sample = [
        {
            k: v if k in shown else _truncate(v, _SAMPLE_FIELD_CHARS)
            for k, v in row.items()
        }
        for row in kept
    ]
    # Summaries of long values need the whole value, not the cut sample.
    into.sample_summaries = [sample_summaries(row) for row in kept]
    into.fields = _field_infos(rows)
    keys = _keys(rows)
    if input_keys is not None and rows:
        into.fields_added = [str(k) for k in keys if k not in input_keys]
        into.fields_removed = [str(k) for k in input_keys if k not in keys]
    return keys


def _check_streams(rec: StepRecord, result: dict, out: _OutHint | None) -> None:
    """Check the stream names against the return hint, when it names streams."""
    if out is not None and out.model is not None:
        raise StepContractError(
            f"step {rec.name} returned dict but its return annotation is "
            f"{out.label()}; annotate dict[str, list[X]] or a TypedDict of streams"
        )
    if out is None or out.streams is None:
        return
    missing = [k for k in out.streams if k in out.required and k not in result]
    extra = [k for k in result if k not in out.streams]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"no stream {', '.join(map(repr, missing))}")
        if extra:
            parts.append(f"stream {', '.join(map(repr, extra))} not in {out.name}")
        raise StepContractError(
            f"step {rec.name} returned {' and '.join(parts)} (streams of "
            f"{out.name}: {', '.join(out.streams)})"
        )


def _check_step_result(
    rec: StepRecord,
    result: Any,
    out: _OutHint | None,
    ops_started: int,
    input_keys: list[Any] | None = None,
) -> None:
    """Validate a step's result; fill in its row counts, fields, samples, and warnings.

    The result is a list of rows or a dict of named row lists (streams).
    ``ops_started`` is how many ops the process started while the step ran; ops
    from worker threads count there but not in ``rec.ops``. ``input_keys`` are
    the input rows' fields when those rows are dicts.
    """
    output_keys: list[Any] = []
    if isinstance(result, list):
        if out is not None and out.wants_dict:
            raise StepContractError(
                f"step {rec.name} returned a list but its return annotation is "
                f"{out.name or out.label()}; return a dict of named row lists"
            )
        rec.n_out = len(result)
        if out is not None and out.model is not None:
            _validate_rows(result, out.model, rec.name, "output")
        keys = _describe(result, rec.show, input_keys, rec)
        if keys is None:
            _tracking.add_problem(rec, "not_dicts", _NOT_DICTS, suggestion=True)
        output_keys = keys or []
    elif isinstance(result, dict):
        if not _is_streams(result):
            raise _not_streams(rec.name, result)
        _check_streams(rec, result, out)
        rec.n_out = sum(len(rows) for rows in result.values())
        rec.streams = []
        not_dicts = []
        for name, rows in result.items():
            model = out.stream_model(name) if out is not None else None
            if model is not None:
                _validate_rows(rows, model, rec.name, f"output stream {name!r}")
            stream = RowStream(
                name=name,
                n_rows=len(rows),
                row_type=_type_label(model) if model is not None else None,
            )
            keys = _describe(rows, rec.show, input_keys, stream)
            if keys is None:
                not_dicts.append(name)
            output_keys += [k for k in keys or [] if k not in output_keys]
            rec.streams.append(stream)
        if not_dicts:
            _tracking.add_problem(
                rec,
                "not_dicts",
                f"rows of {', '.join(not_dicts)} are not dicts; schema checks and "
                "samples skipped",
                suggestion=True,
            )
    else:
        raise StepContractError(
            f"step {rec.name} returned {type(result).__name__}; a step returns rows "
            "or a dict of named row lists"
        )
    missing = [f for f in rec.show if output_keys and f not in output_keys]
    if missing:
        _tracking.add_problem(
            rec,
            "show_missing",
            f"show= names fields no output row has: {', '.join(missing)}",
            suggestion=True,
        )

    if not rec.cache_hit and not rec.ops:
        if ops_started > 0:
            _tracking.add_problem(
                rec,
                "worker_threads",
                "its ops ran in worker threads and are not itemized; submit with "
                "pool.submit(contextvars.copy_context().run, fn, arg) to list them",
                suggestion=True,
            )
            return
        # A suggestion, not an error: an error pushed agents to wrap pure-Python
        # steps in no-op filters (by_predicate(rows, lambda r: True)).
        changed = (
            f"rows went {rec.n_in} → {rec.n_out} in plain Python, and "
            if rec.n_in is not None and rec.n_out is not None and rec.n_in != rec.n_out
            else ""
        )
        _tracking.add_problem(
            rec,
            "no_forger_ops",
            f"called no Forger operations: {changed}this report cannot show what "
            "the step kept or changed; write its checks as Forger ops where one fits "
            "(e.g. filter.by_predicate with the real check, alter.run_programs, "
            "filter.by_tests), and do not add a no-op filter only to quiet this",
            suggestion=True,
        )
        return
    for call in rec.ops:
        _tracking.copy_problems(call, rec)


# --------------------------------------------------------------------------- #
# @pipeline
# --------------------------------------------------------------------------- #


def pipeline(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    prod: bool = False,
    strict: bool = False,
    max_cost_usd: float | None = None,
    max_token_units: float | None = None,
    limit: int | None = None,
    smoke: int | None = None,
    suppress_no_quality_filter_warning: str | None = None,
    suppress_no_diversity_filter_warning: str | None = None,
    suppress_no_inspect_warning: str | None = None,
    suppress_retrieval_warning: str | None = None,
    suppress_low_yield_warning: str | None = None,
    **unknown: Any,
) -> Any:
    """Decorate a function that calls ``@step`` functions; print a summary at the end.

    ``alter.dynamic_step`` called directly in the pipeline counts as a step; any
    other op called outside a step is listed as a bare op with a warning. The
    pipeline may return anything; the summary's final row count is that of a
    returned list, or of a dict of row lists (with each stream), and is left out
    for anything else (a tuple, None).

    Errors (``PipelineContractError``): no step ran, or the pipeline was called
    inside another pipeline. Warnings are data problems: those its steps
    reported (e.g. completions cut off at max_tokens), retrieval ops that
    returned documents whose text reached no model prompt and not the returned
    rows, and under 10% of the first step's rows reaching the output (10-30% is
    a note); the summary lists them last, under "data problems". Suggestions are
    about how the pipeline is written: only one step ran; no quality filter,
    no diversity filter, or no inspect op ran anywhere; ops ran outside a
    step. Notes, which never raise: steps that ran no filter, filters that
    removed nothing (a passed check, or one to tighten), and under 10% of
    retrieved documents used.

    Steps run with ``my_step.submit(...)`` are listed in the order they were
    submitted. When the function returns, the pipeline waits for submitted
    steps still running, then raises the error of any whose ``.result()`` was
    never called. When it raises (or on Ctrl-C), submitted steps still queued
    are cancelled and running ones are not waited for.

    Args:
        _fn: The function, when used as ``@pipeline`` without parentheses.
        name: Pipeline name in the report; defaults to the function name.
        prod: Do not print the summary.
        strict: Raise ``PipelineContractError`` when the pipeline has any
            warning or suggestion (notes do not count).
        max_cost_usd: Stop when this run has spent this many dollars: the next
            model request raises ``BudgetExceededError``. Requests already in
            flight finish, so spend can pass the cap by what they cost.
        max_token_units: The same cap in token units (``pyforger.token_units``),
            for models priced in units rather than dollars. Calls to models
            without fixed weights add no units.
        limit: Run on at most this many rows per step: every step's rows
            arguments are cut to their first ``limit`` rows, and
            ``acquire.sample_from_LLM`` / ``acquire.retrieve*`` cap their counts
            to it. ``limit=3`` is a smoke test of every op; ``limit=50`` a
            pilot whose yields mean something. Defaults to ``FORGER_LIMIT``
            (unset or 0: off); ``limit=0`` turns it off whatever the variable
            says. Limited results are cached apart from a full run's.
        smoke: The old name of ``limit`` (and ``FORGER_SMOKE_ROWS`` of
            ``FORGER_LIMIT``); still accepted.
        suppress_no_quality_filter_warning: Why the rows need no quality
            filter, e.g. every row is checked against a reference answer. The
            suggestion then prints as one dim ``suppressed:`` line.
        suppress_no_diversity_filter_warning: Why the rows need no diversity
            filter.
        suppress_no_inspect_warning: Why no inspect op is needed.
        suppress_low_yield_warning: Why keeping under 10% of the rows is intended.
        suppress_retrieval_warning: Why retrieved text that reached no prompt
            is fine (a data problem: its size stays in the last block).
        **unknown: Any other keyword raises TypeError with the closest flag (a
            step's flag names the decorator to put it on). A step's problems
            are suppressed on that step, where the reason is about its rows.
            A reason under 3 words, or a placeholder, raises ValueError. A flag
            whose problem never fired is a warning (retrieval) or a
            suggestion (the rest), and so is a step's (not in a limited run).
    """
    suppress = _tracking.check_suppress(
        "pipeline",
        {
            "suppress_no_quality_filter_warning": suppress_no_quality_filter_warning,
            "suppress_no_diversity_filter_warning": (
                suppress_no_diversity_filter_warning
            ),
            "suppress_no_inspect_warning": suppress_no_inspect_warning,
            "suppress_low_yield_warning": suppress_low_yield_warning,
            "suppress_retrieval_warning": suppress_retrieval_warning,
        },
        unknown,
    )
    for label, cap in (
        ("max_cost_usd", max_cost_usd),
        ("max_token_units", max_token_units),
    ):
        if cap is not None and not cap >= 0:
            raise ValueError(f"{label} must be a number >= 0, got {cap}")
    if limit is not None and smoke is not None:
        raise ValueError("pass limit= or its old name smoke=, not both")
    if smoke is not None:
        _check_smoke(smoke, "smoke")
        limit = smoke
    if limit is not None:
        _check_smoke(limit, "limit")

    def decorate(fn: F) -> F:
        pipeline_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            outer = _tracking.active_pipeline()
            if outer is not None:
                raise PipelineContractError(
                    f"pipeline {pipeline_name} was called inside pipeline "
                    f"{outer.name}; pipelines cannot nest"
                )
            rec = PipelineRecord(
                name=pipeline_name,
                max_cost_usd=max_cost_usd,
                max_token_units=max_token_units,
                smoke_rows=(limit or None) if limit is not None else _env_smoke(),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            if rec.smoke_rows is None:
                _check_tested(pipeline_name, fn)
            start = time.perf_counter()

            def on_dynamic_step(step_rec: StepRecord) -> None:
                _tracking.store_step(step_rec)
                if _reports_enabled(prod):
                    print_step(step_rec)
                _save_step(step_rec)

            scope = _tracking.enter_pipeline(rec, on_dynamic_step)
            submitted: list[_StepFuture] = []
            submitted_token = _SUBMITTED.set(submitted)
            try:
                result = fn(*args, **kwargs)
                _await_submitted(submitted)
                _tracking.close_pipeline_flow(rec, result)
                _tracking.close_retrieval_reach(rec, result)
                _count_returned(rec, result)
            except BaseException:
                for future in submitted:
                    future.cancel()  # steps still queued never start
                raise
            finally:
                _SUBMITTED.reset(submitted_token)
                _tracking.exit_pipeline(scope)
            rec.duration_s = time.perf_counter() - start
            if rec.smoke_rows is not None:
                _mark_tested(pipeline_name, fn, rec.smoke_rows)
            _check_pipeline(rec)
            _suppress_pipeline(rec, suppress)
            _tracking.store_pipeline(rec)
            _save_run(rec)
            if _reports_enabled(prod):
                print_pipeline(rec)
            if strict and (rec.warnings or rec.suggestions):
                raise PipelineContractError(
                    f"pipeline {pipeline_name} has warnings (strict=True):\n"
                    + "\n".join(f"  - {w}" for w in [*rec.warnings, *rec.suggestions])
                )
            return result

        return typing.cast(F, wrapper)

    if _fn is not None:
        return decorate(_fn)
    return decorate


def _check_smoke(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a row count >= 0, got {value!r}")


def _env_smoke() -> int | None:
    """``FORGER_LIMIT`` (or its old name ``FORGER_SMOKE_ROWS``) as a row limit.

    None when unset, empty, or 0.
    """
    name = "FORGER_LIMIT" if os.environ.get("FORGER_LIMIT") else "FORGER_SMOKE_ROWS"
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value: Any = int(raw)
    except ValueError:
        value = raw
    _check_smoke(value, name)
    return value or None


def _count_returned(rec: PipelineRecord, result: Any) -> None:
    """The pipeline's final row count: a list, or a dict of row lists summed.

    Anything else (a tuple, None, a path) has no row count; the summary leaves
    it out.
    """
    if isinstance(result, list):
        rec.returned, rec.n_out = "rows", len(result)
    elif isinstance(result, dict) and result and _is_streams(result):
        rec.returned = "streams"
        rec.out_streams = [RowStream(name=k, n_rows=len(v)) for k, v in result.items()]
        rec.n_out = sum(s.n_rows for s in rec.out_streams)
    else:
        rec.returned = "other"


def _check_pipeline(rec: PipelineRecord) -> None:
    """Raise when no step ran; otherwise fill in the pipeline-level warnings."""
    real = [s for s in rec.steps if s.kind != "bare"]
    if not real:
        raise PipelineContractError(
            f"pipeline {rec.name} called no steps; wrap groups of Forger ops in "
            "@step functions and call those from the pipeline"
        )
    if len(real) == 1:
        _tracking.add_problem(
            rec,
            "one_step",
            "only one step ran; split the work into separate @step functions "
            "(e.g. generate, then filter) so each gets a report",
            suggestion=True,
        )
    # A step cached by an older Forger version has no record of its ops; skip
    # the checks for which ops ran rather than warn about ops that may have run.
    ops_known = all(s.ops or not s.cache_hit for s in rec.steps)
    # Categories come from the op registry, not names: a user's
    # @op(name="filter.keep_all") is "other" and satisfies nothing here.
    filters = {c.op_name for s in rec.steps for c in s.ops if c.category == "filter"}
    if ops_known:
        if not filters - _NON_QUALITY_FILTERS:
            _tracking.add_problem(
                rec,
                "no_quality_filter",
                "no quality filter ran",
                suggestion=True,
            )
        if not filters & _DIVERSITY_FILTERS:
            _tracking.add_problem(
                rec,
                "no_diversity_filter",
                "no diversity filter ran",
                suggestion=True,
            )
        if not any(c.category == "inspect" for s in rec.steps for c in s.ops):
            _tracking.add_problem(
                rec,
                "no_inspect",
                "no inspect op ran",
                suggestion=True,
            )
    _filter_notes(rec)
    _yield_check(rec, real)
    bare = [s.ops[0].op_name for s in rec.steps if s.kind == "bare"]
    if bare:
        listed = ", ".join(dict.fromkeys(bare))
        _tracking.add_problem(
            rec,
            "outside_step",
            f"{len(bare)} op call{'s' if len(bare) != 1 else ''} outside a @step: "
            f"{listed}",
            suggestion=True,
        )


def _suppress_pipeline(rec: PipelineRecord, suppress: dict[str, str]) -> None:
    """Apply the pipeline's suppress flags; report flags whose problem never fired.

    Also its steps' flags: ``suppress_truncation_warning`` on a step none of
    whose runs truncated. A limited run checks no flag (3 rows rarely show a
    problem), nor a step cached by an older Forger (its ops are unknown).
    """
    _tracking.apply_suppress(rec, suppress, f"pipeline {rec.name}")
    if rec.smoke_rows is not None:
        return
    for kind in _tracking.stale_suppressions(suppress, rec.suppressed):
        _tracking.add_stale(rec, kind, f"pipeline {rec.name}")
    runs: dict[str, list[StepRecord]] = {}
    for s in rec.steps:
        if s.suppress:
            runs.setdefault(s.name, []).append(s)
    for name, recs in runs.items():
        if any(s.cache_hit and not s.ops for s in recs):
            continue
        hidden = [x for s in recs for x in s.suppressed]
        for kind in _tracking.stale_suppressions(recs[-1].suppress, hidden):
            _tracking.add_stale(rec, kind, name)


# The pipeline's yield (rows returned / rows out of its first step): a warning
# below the first, a note below the second.
_LOW_YIELD = 0.1
_NOTE_YIELD = 0.3


def _yield_check(rec: PipelineRecord, real: list[StepRecord]) -> None:
    """Warn when few of the first step's rows reach the output, naming the worst step.

    In a limited run only ``limit`` rows go on from the first step, so the
    yield is out of those, not the first step's full output.

    e.g. "3 of 536 rows (0.6%) from step briefs reached the output; validate
    dropped the most (533 -> 3)". A limited run also says what 100 rows give.
    """
    first = next((s for s in real if s.n_out), None)
    if first is None or first.n_out is None or rec.n_out is None or len(real) < 2:
        return
    # A limited run passes at most `limit` rows from the first step to the next.
    base = first.n_out
    if rec.smoke_rows is not None:
        base = min(base, rec.smoke_rows)
    kept = rec.n_out / base
    if kept >= _NOTE_YIELD:
        return
    drops = [
        (s.n_in - s.n_out, s)
        for s in real
        if s.n_in is not None and s.n_out is not None and s.n_in > s.n_out
    ]
    message = (
        f"{rec.n_out} of {base} rows ({kept:.1%}) from step {first.name} "
        "reached the output"
    )
    if drops:
        _, worst = max(drops, key=lambda d: d[0])
        message += f"; {worst.name} dropped the most ({worst.n_in} -> {worst.n_out})"
    if rec.smoke_rows is not None:
        message += (
            f"; at this yield every 100 rows from {first.name} give about "
            f"{100 * kept:.0f}"
        )
    if kept < _LOW_YIELD:
        _tracking.add_problem(rec, "low_yield", message, size=f"{kept:.1%} kept")
    else:
        rec.notes.append(message)


def _filter_notes(rec: PipelineRecord) -> None:
    """Note the steps that ran no filter and the filters that removed nothing.

    These are notes, not warnings: a generate-only step followed by a filter
    step is fine, and a filter that removed nothing may be a check that passed.
    Steps whose ops are unknown (cached by an older Forger, or run in worker
    threads without a copied context) are skipped. A step that ran several
    times counts once; its filter calls are summed per op.
    """
    filtered: dict[str, bool] = {}
    totals: dict[tuple[str, str], list[int]] = {}
    for s in rec.steps:
        if not s.ops:
            continue
        if s.kind == "step":
            has_filter = any(c.category == "filter" for c in s.ops)
            filtered[s.name] = filtered.get(s.name, False) or has_filter
        for call in s.ops:
            if call.category != "filter" or call.n_in is None or call.n_out is None:
                continue
            seen = totals.setdefault((call.op_name, s.name), [0, 0])
            seen[0] += call.n_in
            seen[1] += call.n_out
    rec.no_filter_steps = [name for name, has in filtered.items() if not has]
    rec.noop_filters = [
        (op_name, step_name, n_in)
        for (op_name, step_name), (n_in, n_out) in totals.items()
        if n_in and n_out == n_in
    ]


__all__ = [
    "BudgetExceededError",
    "PipelineContractError",
    "StepContractError",
    "pipeline",
    "print_pipeline",
    "print_step",
    "render_pipeline",
    "render_scores",
    "render_step",
    "set_report_model",
    "step",
]

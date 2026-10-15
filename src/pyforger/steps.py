"""``@step`` and ``@pipeline``: named, checked, and reported groups of Forger ops.

A step is a function from rows to rows built from Forger ops::

    @step
    def write_solutions(problems: list[Problem], model) -> list[Solved]:
        rows = alter.generate_from_context(problems, PROMPT, model)
        return filter.by_llm_score(rows, model=judge, ..., min_yield=0.3)

    @pipeline
    def main(model):
        return write_solutions(find_problems(model), model)

Each step is cached like an ``@op`` and prints a report when it returns: every
op it ran with rows in/out, yield, time, tokens, and cost, a histogram of each
filter's scores, warnings, and a few sample rows. A pipeline prints a summary
table of its steps. ``@step(prod=True)``, ``@pipeline(prod=True)``, or
``FORGER_STEP_REPORT=0`` silence the reports; ``strict=True`` turns warnings
into errors.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import textwrap
import time
import typing
from collections import Counter
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from pyforger import _tracking
from pyforger._tracking import OpCall, PipelineRecord, ScoreRecord, StepRecord
from pyforger.op import op

F = TypeVar("F", bound=Callable[..., Any])

_DIVERSITY_FILTERS = frozenset(
    {"filter.by_exact_match", "filter.by_rouge_l", "filter.by_n_gram_overlap"}
)
# Every other filter.* op counts as a quality filter.
_NON_QUALITY_FILTERS = _DIVERSITY_FILTERS | {"filter.by_length"}

_SAMPLE_ROWS_KEPT = 10  # rows stored per record; inspect_step shows all of them
_SAMPLE_FIELD_CHARS = 500  # stored per string field; inspect_step shows all of it
_COMPACT_FIELD_CHARS = 80  # per field in the step report's sample lines
_WIDTH = 120  # report lines never exceed this many characters
_RULE_WIDTH = 100  # header rules are padded to this width
_NAME_CHARS = 40  # step and op names in report tables are cut to this length
_NOT_DICTS = "rows are not dicts; schema checks and samples skipped"


class StepContractError(RuntimeError):
    """A ``@step`` broke its contract (bad rows, no ops, nesting, or strict mode)."""


class PipelineContractError(RuntimeError):
    """A ``@pipeline`` broke its contract (no steps, nesting, or strict mode)."""


def _reports_enabled(prod: bool) -> bool:
    flag = os.environ.get("FORGER_STEP_REPORT", "").strip().lower()
    return not prod and flag not in {"0", "false", "no"}


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


def _row_models(
    fn: Callable[..., Any], localns: dict[str, Any] | None = None
) -> tuple[type[BaseModel] | None, type[BaseModel] | None, str | None]:
    """Row models for ``fn``'s first parameter and return value, if any.

    ``localns`` holds the names of the scope that defined ``fn``, so string
    annotations naming a class defined in a function body still resolve. The
    third item is a warning when the annotations could not be resolved.
    """
    params = list(inspect.signature(fn).parameters)
    try:
        hints = typing.get_type_hints(fn, localns=localns)
    except Exception as exc:
        annotations = getattr(fn, "__annotations__", {}) or {}
        if not ({"return", *params[:1]} & set(annotations)):
            return None, None, None
        return (
            None,
            None,
            "type hints could not be resolved "
            f"({type(exc).__name__}: {exc}); schema checks skipped",
        )
    first = hints.get(params[0]) if params else None
    return _row_model(first), _row_model(hints.get("return")), None


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
    sample: int = 3,
) -> Any:
    """Decorate a function from rows to rows built from Forger ops.

    The step is cached like an ``@op`` (under the op name ``step.<name>``) and
    prints a report when it returns. If the first parameter or the return value
    is annotated ``list[X]`` with ``X`` a pydantic model, every row must be a dict
    that validates against ``X`` (rows stay dicts).

    Errors (``StepContractError``): the result is not a list, the step called no
    Forger ops, rows do not match the annotated row model, or the step was called
    inside another step. Warnings: the last op is not a filter, an op missed its
    ``min_yield`` / ``max_yield``, the rows are not dicts, or the type hints
    could not be resolved.

    Args:
        _fn: The function, when used as ``@step`` without parentheses.
        name: Step name in reports and the cache; defaults to the function name.
        cache: Cache the step's result on disk, like ``@op(cache=...)``.
        prod: Do not print the report.
        strict: Raise ``StepContractError`` when the step has any warning; such
            a result is not cached.
        sample: Number of output rows shown in the report.
    """
    # The scope that defines the step (where ``step`` is called, both for
    # ``@step`` and ``@step(...)``), for resolving string annotations.
    frame = sys._getframe(1)
    localns = dict(frame.f_locals) if frame.f_locals is not frame.f_globals else None
    del frame

    def decorate(fn: F) -> F:
        step_name = name or fn.__name__
        cached = op(fn, name=f"{_tracking.STEP_OP_PREFIX}{step_name}", cache=cache)
        params = list(inspect.signature(fn).parameters)
        first_param = params[0] if params else None
        doc = inspect.getdoc(fn) or ""
        hints: list[Any] = []  # resolved on the first call, when names exist

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            outer = _tracking.active_step()
            if outer is not None:
                raise StepContractError(
                    f"step {step_name} was called inside step {outer.name}; steps "
                    "cannot nest, so call both from the @pipeline"
                )
            if not hints or hints[0][2] is not None:
                hints[:] = [_row_models(fn, localns)]
            in_model, out_model, hint_warning = hints[0]
            rec = StepRecord(
                name=step_name,
                doc=doc,
                sample_shown=sample,
                input_type=f"list[{in_model.__name__}]" if in_model else None,
                output_type=f"list[{out_model.__name__}]" if out_model else None,
            )
            if hint_warning:
                rec.warnings.append(hint_warning)
            rows_in = args[0] if args else kwargs.get(first_param or "")
            if isinstance(rows_in, list):
                rec.n_in = len(rows_in)
                if in_model is not None:
                    _validate_rows(rows_in, in_model, step_name, "input")

            checked: list[bool] = []
            strict_failed: list[bool] = []
            ops_before = _tracking.ops_started()

            def check(result: Any) -> str:
                # Called by @op on a fresh result, before it is cached: raising
                # here keeps a result that breaks the contract out of the cache.
                checked.append(True)
                _check_step_result(
                    rec, result, out_model, _tracking.ops_started() - ops_before
                )
                if strict and rec.warnings:
                    strict_failed.append(True)
                    raise _strict_error(rec)
                return _tracking.step_meta(rec)

            usage_start = _tracking.usage_snapshot()
            start = time.perf_counter()

            def finish() -> None:
                rec.duration_s = time.perf_counter() - start
                usage = _tracking.usage_since(usage_start)
                rec.prompt_tokens, rec.completion_tokens, rec.cost_usd = usage
                _tracking.store_step(rec)
                pipeline_rec = _tracking.pipeline_for_step()
                if pipeline_rec is not None:
                    pipeline_rec.steps.append(rec)
                if _reports_enabled(prod):
                    print("\n" + render_step(rec), flush=True)

            scope = _tracking.enter_step(rec, check)
            try:
                result = cached(*args, **kwargs)
            except StepContractError:
                if strict_failed:  # report the step, then raise
                    finish()
                raise
            finally:
                _tracking.exit_step(scope)

            if not checked:  # a cache hit
                _check_step_result(rec, result, out_model, 0)
            finish()
            if strict and rec.warnings:
                raise _strict_error(rec)
            return result

        return typing.cast(F, wrapper)

    if _fn is not None:
        return decorate(_fn)
    return decorate


def _strict_error(rec: StepRecord) -> StepContractError:
    return StepContractError(
        f"step {rec.name} has warnings (strict=True):\n"
        + "\n".join(f"  - {w}" for w in rec.warnings)
    )


def _check_step_result(
    rec: StepRecord,
    result: Any,
    out_model: type[BaseModel] | None,
    ops_started: int,
) -> None:
    """Validate a step's result and fill in its row count, sample, and warnings.

    ``ops_started`` is how many ops the process started while the step ran; ops
    from worker threads count there but not in ``rec.ops``.
    """
    if not isinstance(result, list):
        raise StepContractError(
            f"step {rec.name} returned {type(result).__name__}; a step must return "
            "a list of rows"
        )
    rec.n_out = len(result)
    if out_model is not None:
        _validate_rows(result, out_model, rec.name, "output")
    if all(isinstance(row, dict) for row in result):
        rec.sample = [
            _truncate(row, _SAMPLE_FIELD_CHARS) for row in result[:_SAMPLE_ROWS_KEPT]
        ]
    else:
        rec.warnings.append(_NOT_DICTS)

    if not rec.cache_hit and not rec.ops:
        if ops_started > 0:
            rec.warnings.append(
                "its ops ran in worker threads and are not itemized; submit with "
                "pool.submit(contextvars.copy_context().run, fn, arg) to list them"
            )
            return
        raise StepContractError(
            f"step {rec.name} called no Forger operations; build steps from "
            "pyforger.acquire/alter/filter/inspect ops and keep pure-Python helpers "
            "as plain functions"
        )
    for call in rec.ops:
        rec.warnings.extend(call.warnings)
    if rec.ops and rec.ops[-1].category != "filter":
        rec.warnings.append("ends without a filter")


# --------------------------------------------------------------------------- #
# @pipeline
# --------------------------------------------------------------------------- #


def pipeline(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    prod: bool = False,
    strict: bool = False,
) -> Any:
    """Decorate a function that calls ``@step`` functions; print a summary at the end.

    ``alter.dynamic_step`` called directly in the pipeline counts as a step; any
    other op called outside a step is listed as a bare op with a warning.

    Errors (``PipelineContractError``): no step ran, or the pipeline was called
    inside another pipeline. Warnings: only one step ran; no quality filter, no
    diversity filter, or no inspect op ran anywhere; ops ran outside a step.

    Args:
        _fn: The function, when used as ``@pipeline`` without parentheses.
        name: Pipeline name in the report; defaults to the function name.
        prod: Do not print the summary.
        strict: Raise ``PipelineContractError`` when the pipeline has any warning.
    """

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
            rec = PipelineRecord(name=pipeline_name)
            usage_start = _tracking.usage_snapshot()
            start = time.perf_counter()

            def on_dynamic_step(step_rec: StepRecord) -> None:
                _tracking.store_step(step_rec)
                if _reports_enabled(prod):
                    print("\n" + render_step(step_rec), flush=True)

            scope = _tracking.enter_pipeline(rec, on_dynamic_step)
            try:
                result = fn(*args, **kwargs)
            finally:
                _tracking.exit_pipeline(scope)
            rec.duration_s = time.perf_counter() - start
            usage = _tracking.usage_since(usage_start)
            rec.prompt_tokens, rec.completion_tokens, rec.cost_usd = usage

            _check_pipeline(rec)
            _tracking.store_pipeline(rec)
            if _reports_enabled(prod):
                print("\n" + render_pipeline(rec), flush=True)
            if strict and rec.warnings:
                raise PipelineContractError(
                    f"pipeline {pipeline_name} has warnings (strict=True):\n"
                    + "\n".join(f"  - {w}" for w in rec.warnings)
                )
            return result

        return typing.cast(F, wrapper)

    if _fn is not None:
        return decorate(_fn)
    return decorate


def _check_pipeline(rec: PipelineRecord) -> None:
    """Raise when no step ran; otherwise fill in the pipeline-level warnings."""
    real = [s for s in rec.steps if s.kind != "bare"]
    if not real:
        raise PipelineContractError(
            f"pipeline {rec.name} called no steps; wrap groups of Forger ops in "
            "@step functions and call those from the pipeline"
        )
    if len(real) == 1:
        rec.warnings.append(
            "only one step ran; split the work into separate @step functions "
            "(e.g. generate, then filter) so each gets a report"
        )
    # A step cached by an older Forger version has no record of its ops; skip
    # the checks for which ops ran rather than warn about ops that may have run.
    ops_known = all(s.ops or not s.cache_hit for s in rec.steps)
    names = {call.op_name for s in rec.steps for call in s.ops}
    if ops_known:
        if not any(
            n.startswith("filter.") and n not in _NON_QUALITY_FILTERS for n in names
        ):
            rec.warnings.append("no quality filter ran (e.g. filter.by_llm_score)")
        if not names & _DIVERSITY_FILTERS:
            rec.warnings.append(
                "no diversity filter ran (filter.by_exact_match, by_rouge_l, or "
                "by_n_gram_overlap)"
            )
        if not any(n.startswith("inspect.") for n in names):
            rec.warnings.append("no inspect op ran")
    bare = [s.ops[0].op_name for s in rec.steps if s.kind == "bare"]
    if bare:
        listed = ", ".join(dict.fromkeys(bare))
        rec.warnings.append(
            f"{len(bare)} op call{'s' if len(bare) != 1 else ''} outside a @step: "
            f"{listed}"
        )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_int(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def _fmt_yield(y: float | None) -> str:
    return "-" if y is None or not math.isfinite(y) else f"{y:.2f}"


def _fmt_secs(s: float) -> str:
    return f"{s:,.1f}s" if math.isfinite(s) else "-"


def _fmt_cost(c: float) -> str:
    if not c or not math.isfinite(c):
        return "-"
    return f"${c:,.2f}" if c >= 1 else f"${c:.4f}"


def _fmt_tokens(prompt: int, completion: int) -> str:
    if not prompt and not completion:
        return "-"
    return f"{prompt:,}/{completion:,}"


def _fmt_num(x: float) -> str:
    """Compact number: ``3``, ``0.7``, ``0.25``, ``1,204``."""
    if float(x).is_integer():
        return f"{int(x):,}"
    return f"{x:.2f}".rstrip("0") if abs(x) < 10 else f"{x:,.1f}"


def _rule(char: str, text: str) -> str:
    line = f"{char * 2} {text} "
    return line + char * max(2, _RULE_WIDTH - len(line))


def _header(char: str, title: str, parts: list[str]) -> list[str]:
    """Rule line with ``title: parts``; the parts move to a second line if long."""
    line = _rule(char, f"{title}: " + " · ".join(parts))
    if len(line) <= _WIDTH:
        return [line]
    return [_clip(_rule(char, title)), _clip("  " + " · ".join(parts))]


def _clip(line: str, width: int = _WIDTH) -> str:
    return line if len(line) <= width else line[: width - 1] + "…"


def _table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], left: Sequence[int]
) -> list[str]:
    """Align ``rows`` under ``headers``; columns in ``left`` are left-aligned.

    The last column holds free-form notes and is never padded.
    """
    ncols = len(headers)
    rows = [[_clip(r[0], _NAME_CHARS), *r[1:]] for r in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(ncols - 1)
    ]

    def fmt(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells[:-1]):
            out.append(cell.ljust(widths[i]) if i in left else cell.rjust(widths[i]))
        out.append(cells[-1])
        return "  ".join(out).rstrip()

    return [fmt(headers)] + [fmt(r) for r in rows]


def _op_notes(call: OpCall) -> str:
    notes = []
    y = call.yield_ratio
    if call.min_yield is not None:
        status = "-" if y is None else ("ok" if y >= call.min_yield else "LOW")
        if call.n_in == 0 and call.min_yield > 0:
            status = "LOW"
        notes.append(f"min {call.min_yield:.2f} {status}")
    if call.max_yield is not None:
        status = "-" if y is None else ("ok" if y <= call.max_yield else "HIGH")
        notes.append(f"max {call.max_yield:.2f} {status}")
    if call.cache_hit:
        notes.append("(cached)")
    return "  ".join(notes)


def _show_yield(call: OpCall) -> bool:
    return (
        call.category == "filter"
        or call.min_yield is not None
        or call.max_yield is not None
        or (call.n_in is not None and call.n_out != call.n_in)
    )


_KEEP_SYMBOL = {"ge": ">=", "le": "<=", "lt": "<"}


def _kept(values: list[float], keep: str, t: float) -> int:
    if keep == "ge":
        return sum(1 for v in values if v >= t)
    if keep == "le":
        return sum(1 for v in values if v <= t)
    return sum(1 for v in values if v < t)


def _wrap_items(prefix: str, items: list[str], indent: int) -> list[str]:
    """Join ``items`` after ``prefix`` two spaces apart, wrapping before ``_WIDTH``."""
    lines, line = [], prefix
    for item in items:
        if len(line) + 2 + len(item) > _WIDTH - indent and line != prefix:
            lines.append(line)
            line = " " * len(prefix)
        line += "  " + item
    lines.append(line)
    return lines


def _nice_width(span: float, integer: bool) -> float:
    """Smallest round bin width (1, 2, 2.5, or 5 times a power of ten) for 10 bins."""
    raw = span / 10 if span > 0 else 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        width = step * power
        if width >= raw * (1 - 1e-9) and (not integer or width.is_integer()):
            return width
    return 10.0 * power


def _fmt_edge(x: float) -> str:
    """Compact bin edge: ``40``, ``2.5``, ``2,500``, ``12.5k``."""
    if abs(x) >= 10_000:
        return f"{x / 1000:,.1f}".rstrip("0").rstrip(".") + "k"
    return _fmt_num(round(x, 6))


def render_scores(scores: ScoreRecord, indent: int = 4) -> list[str]:
    """Render a filter's per-row scores as histogram lines, ``indent`` spaces in.

    Ints with up to 12 distinct values get one column each. Floats in [0, 1]
    get ten bins labelled by their lower edge. Other values get round-width
    bins labelled by their lower edge, with the width noted.
    """
    pad = " " * indent
    total = len(scores.values)
    good: list[Any] = [
        v
        for v in scores.values
        if v is not None and not (isinstance(v, float) and not math.isfinite(v))
    ]
    unparsed = total - len(good)
    note = f"   unparsed {unparsed:,}" if unparsed else ""
    if not total:
        return [f"{pad}no rows scored"]
    if scores.keep == "true" or (good and all(isinstance(v, bool) for v in good)):
        kept = sum(1 for v in good if v)
        return [f"{pad}kept {kept:,}  dropped {len(good) - kept:,}{note}"]
    if not good:
        return [f"{pad}{scores.label}: all {unparsed:,} rows unparsed"]

    nums = [float(v) for v in good]
    distinct = sorted(set(good))
    is_int = all(isinstance(v, int) for v in good)
    lo, hi = min(nums), max(nums)
    labels: list[str]
    counts: list[int]
    thresholds: list[float]
    if is_int and len(distinct) <= 12:
        tally = Counter(good)
        labels = [_fmt_num(v) for v in distinct]
        counts = [tally[v] for v in distinct]
        thresholds = [float(v) for v in distinct]
    elif not is_int and lo >= 0.0 and hi <= 1.0:
        counts = [0] * 10
        for v in nums:
            # The epsilon keeps edge values in their own bin (0.3 / 0.1 < 3).
            counts[min(int(v * 10 + 1e-9), 9)] += 1
        labels = [f"{i / 10:.1f}" for i in range(10)]
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        width = _nice_width(hi - lo, is_int)
        first = math.floor(lo / width + 1e-9) * width
        nbins = int((hi - first) / width + 1e-9) + 1
        counts = [0] * nbins
        for v in nums:
            counts[min(int((v - first) / width + 1e-9), nbins - 1)] += 1
        edges = [first + i * width for i in range(nbins)]
        labels = [_fmt_edge(e) for e in edges]
        thresholds = [e for e in edges if lo < e <= hi]
        note = f"   (bins of {_fmt_edge(width)}){note}"
    if scores.threshold is not None and scores.threshold not in thresholds:
        thresholds = sorted(thresholds + [scores.threshold])

    head = max(len(scores.label), len("keep at"))
    widths = [max(len(a), len(f"{b:,}")) for a, b in zip(labels, counts)]
    label_line = scores.label.ljust(head) + "".join(
        "  " + a.rjust(w) for a, w in zip(labels, widths)
    )
    count_line = "rows".ljust(head) + "".join(
        "  " + f"{b:,}".rjust(w) for b, w in zip(counts, widths)
    )
    lines = [pad + label_line + note, pad + count_line]

    if scores.keep == "between":
        lower = scores.lower if scores.lower is not None else -math.inf
        upper = scores.upper if scores.upper is not None else math.inf
        kept = sum(1 for v in nums if lower <= v <= upper)
        bounds = f"[{_fmt_num(lower)}, {_fmt_num(upper)}]".replace("inf", "∞")
        lines.append(
            f"{pad}{'keep'.ljust(head)}  {bounds}: {kept:,} of {total:,} "
            f"({kept / total:.2f})"
        )
        return lines

    symbol = _KEEP_SYMBOL[scores.keep]
    items = []
    for t in thresholds:
        is_current = scores.threshold is not None and t == scores.threshold
        if scores.pooled and not is_current:
            # Each score compares a row with the rows kept at this threshold;
            # another threshold keeps other rows, so these scores cannot say.
            continue
        frac = _kept(nums, scores.keep, t) / total
        if frac in (0.0, 1.0) and not is_current:
            continue
        item = f"{symbol}{_fmt_num(t)} {frac:.2f}"
        items.append(f"[{item}]" if is_current else item)
    if scores.pooled and items:
        items.append("(scored against the kept rows; rerun to try other thresholds)")
    if items:
        lines.extend(pad + ln for ln in _wrap_items("keep at".ljust(head), items, 4))
    return lines


def _fmt_value(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return f'"{text}"'
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sample_lines(rec: StepRecord, full: bool) -> list[str]:
    if not rec.sample:
        return []
    shown = rec.sample if full else rec.sample[: max(0, rec.sample_shown)]
    if not shown:
        return []
    lines = [f"  sample ({len(shown)} of {_fmt_int(rec.n_out)}):"]
    for i, row in enumerate(shown):
        if not full:
            fields = " ".join(
                f"{k}={_fmt_value(v, _COMPACT_FIELD_CHARS)}" for k, v in row.items()
            )
            lines.append(_clip(f"    [{i}] {fields}"))
            continue
        lines.append(f"    [{i}]")
        for k, v in row.items():
            text = f"{k}: {_fmt_value(v, _SAMPLE_FIELD_CHARS + 2)}"
            lines.extend(
                textwrap.wrap(
                    text,
                    width=_WIDTH,
                    initial_indent="      ",
                    subsequent_indent="        ",
                )
                or ["      " + text]
            )
    return lines


def render_step(rec: StepRecord, *, full: bool = False) -> str:
    """Render a step record as the report ``@step`` prints.

    ``full=True`` (used by ``inspect.inspect_step``) shows the whole docstring,
    up to 10 sample rows, and up to 500 characters per field.
    """
    parts = [f"{_fmt_int(rec.n_in)} → {_fmt_int(rec.n_out)} rows"]
    parts.append(_fmt_secs(rec.duration_s))
    if rec.cache_hit:
        parts.append("cached")
    if rec.prompt_tokens or rec.completion_tokens:
        parts.append(f"{rec.prompt_tokens:,} in / {rec.completion_tokens:,} out tokens")
    if rec.cost_usd:
        parts.append(_fmt_cost(rec.cost_usd))
    lines = _header("─", f"step {rec.name}", parts)

    doc_lines = rec.doc.strip().splitlines()
    if doc_lines:
        if full:
            lines.extend(_clip(f"  {ln}".rstrip()) for ln in doc_lines)
        else:
            lines.append(_clip(f"  {doc_lines[0]}"))
    if rec.input_type or rec.output_type:
        types = []
        if rec.input_type:
            types.append(f"in: {rec.input_type} ok")
        if rec.output_type:
            types.append(f"out: {rec.output_type} ok")
        lines.append("  " + "   ".join(types))

    if rec.ops:
        headers = ["op", "in", "out", "yield", "time", "tokens in/out", "cost", ""]
        rows = [
            [
                call.op_name,
                _fmt_int(call.n_in),
                _fmt_int(call.n_out),
                _fmt_yield(call.yield_ratio) if _show_yield(call) else "-",
                _fmt_secs(call.duration_s),
                _fmt_tokens(call.prompt_tokens, call.completion_tokens),
                _fmt_cost(call.cost_usd),
                _op_notes(call),
            ]
            for call in rec.ops
        ]
        table = _table(headers, rows, left=[0])
        lines.append(_clip("  " + table[0]))
        for call, row_line in zip(rec.ops, table[1:]):
            lines.append(_clip("  " + row_line))
            if call.scores is not None and not call.cache_hit:
                lines.extend(_clip(ln) for ln in render_scores(call.scores))
    elif rec.cache_hit:
        lines.append(
            "  (cached) result loaded from the step cache; its ops did not run"
        )

    if rec.warnings:
        lines.append("  warnings:")
        lines.extend(_clip(f"    - {w}") for w in rec.warnings)
    lines.extend(_sample_lines(rec, full))
    return "\n".join(lines)


def render_pipeline(rec: PipelineRecord) -> str:
    """Render a pipeline record as the summary ``@pipeline`` prints."""
    n_steps = sum(1 for s in rec.steps if s.kind != "bare")
    first = rec.steps[0].n_in if rec.steps else None
    last = rec.steps[-1].n_out if rec.steps else None
    parts = [
        f"{n_steps} step{'s' if n_steps != 1 else ''}",
        f"{_fmt_int(first)} → {_fmt_int(last)} rows",
        _fmt_secs(rec.duration_s),
    ]
    if rec.prompt_tokens or rec.completion_tokens:
        parts.append(f"{rec.prompt_tokens:,} in / {rec.completion_tokens:,} out tokens")
    if rec.cost_usd:
        parts.append(_fmt_cost(rec.cost_usd))
    lines = _header("═", f"pipeline {rec.name}", parts)

    headers = ["step", "in", "out", "yield", "time", "cost", ""]
    rows = []
    for s in rec.steps:
        y = s.n_out / s.n_in if s.n_in and s.n_out is not None else None
        rows.append(
            [
                s.name,
                _fmt_int(s.n_in),
                _fmt_int(s.n_out),
                _fmt_yield(y),
                _fmt_secs(s.duration_s),
                _fmt_cost(s.cost_usd),
                "cached" if s.cache_hit else "",
            ]
        )
    table = _table(headers, rows, left=[0])
    lines.append(_clip("  " + table[0]))
    for s, row_line in zip(rec.steps, table[1:]):
        lines.append(_clip("  " + row_line))
        lines.extend(_clip(f"      ! {w}") for w in dict.fromkeys(s.warnings))
    if rec.warnings:
        lines.append("  pipeline warnings:")
        lines.extend(_clip(f"    - {w}") for w in rec.warnings)
    return "\n".join(lines)

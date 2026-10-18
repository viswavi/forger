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
import os
import sys
import time
import typing
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from pyforger import _tracking
from pyforger._render import (
    print_pipeline,
    print_step,
    render_pipeline,
    render_scores,
    render_step,
)
from pyforger._tracking import PipelineRecord, StepRecord
from pyforger.op import op

F = TypeVar("F", bound=Callable[..., Any])

_DIVERSITY_FILTERS = frozenset(
    {"filter.by_exact_match", "filter.by_rouge_l", "filter.by_n_gram_overlap"}
)
# Every other op of category "filter" counts as a quality filter, including a
# user op declared with @op(category="filter").
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
                (
                    rec.prompt_tokens,
                    rec.completion_tokens,
                    rec.cost_usd,
                    rec.token_units,
                ) = usage
                _tracking.store_step(rec)
                pipeline_rec = _tracking.pipeline_for_step()
                if pipeline_rec is not None:
                    pipeline_rec.steps.append(rec)
                if _reports_enabled(prod):
                    print_step(rec)

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
                    print_step(step_rec)

            scope = _tracking.enter_pipeline(rec, on_dynamic_step)
            try:
                result = fn(*args, **kwargs)
            finally:
                _tracking.exit_pipeline(scope)
            rec.duration_s = time.perf_counter() - start
            usage = _tracking.usage_since(usage_start)
            (
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.cost_usd,
                rec.token_units,
            ) = usage

            _check_pipeline(rec)
            _tracking.store_pipeline(rec)
            if _reports_enabled(prod):
                print_pipeline(rec)
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
    # Categories come from the op registry, not names: a user's
    # @op(name="filter.keep_all") is "other" and satisfies nothing here.
    filters = {c.op_name for s in rec.steps for c in s.ops if c.category == "filter"}
    if ops_known:
        if not filters - _NON_QUALITY_FILTERS:
            rec.warnings.append("no quality filter ran (e.g. filter.by_llm_score)")
        if not filters & _DIVERSITY_FILTERS:
            rec.warnings.append(
                "no diversity filter ran (filter.by_exact_match, by_rouge_l, or "
                "by_n_gram_overlap)"
            )
        if not any(c.category == "inspect" for s in rec.steps for c in s.ops):
            rec.warnings.append("no inspect op ran")
    bare = [s.ops[0].op_name for s in rec.steps if s.kind == "bare"]
    if bare:
        listed = ", ".join(dict.fromkeys(bare))
        rec.warnings.append(
            f"{len(bare)} op call{'s' if len(bare) != 1 else ''} outside a @step: "
            f"{listed}"
        )


__all__ = [
    "PipelineContractError",
    "StepContractError",
    "pipeline",
    "print_pipeline",
    "print_step",
    "render_pipeline",
    "render_scores",
    "render_step",
    "step",
]

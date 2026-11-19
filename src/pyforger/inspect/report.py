"""Step reports: re-print the full report of a ``@step`` run, or load its output."""

from __future__ import annotations

from typing import Any, Literal, overload

from pyforger import _summaries, _tracking
from pyforger._tracking import StepRecord
from pyforger.op import _UNCACHED_STEP_SUFFIX, _cache_dir, _read_result
from pyforger.steps import _reports_enabled, print_step, saved_step_records
from pyforger.utils import get_formatted_logger

logger = get_formatted_logger(__name__)

# Warnings that say why a step's result was kept out of the cache.
_NOT_CACHED_KINDS = ("provider_failure", "not_cached")


@overload
def inspect_step(
    name: str | None = None, *, rows: Literal[False] = False
) -> StepRecord:
    ...


@overload
def inspect_step(name: str | None = None, *, rows: Literal[True]) -> Any:
    ...


def inspect_step(name: str | None = None, *, rows: bool = False) -> Any:
    """Print the full report of the latest run of step ``name`` and return it.

    With no name, use the most recent step. Steps that ran in this process come
    first; otherwise the report is read from the one every step saves under
    ``FORGER_CACHE``, so a pipeline run in another process (e.g. under nohup)
    can be inspected from a shell. The full report shows the whole docstring, up
    to 10 sample rows, and up to 500 characters per field; a longer value is
    summarized by its shape, and with a report model
    (``pyforger.steps.set_report_model``) also in one sentence.

    With ``rows=True``, print nothing and return that run's whole output (what
    the step returned: its rows, or its dict of row lists) from the step's
    cache entry, e.g. ``inspect_step("write_targets", rows=True)`` after the
    pipeline died in a later step. A limited run (``FORGER_LIMIT``) has its own
    entry, so its few rows come back if it ran last. A run kept out of the
    cache (e.g. 10 of 357 requests timed out) still saved its output for this,
    and loading it logs a warning that the rows may be partial; a rerun of the
    step runs it again, reusing the completions that arrived.

    Returns:
        The step's ``StepRecord``, as printed; with ``rows=True``, its output.

    Raises:
        KeyError: No step with that name has run; the message lists known steps.
        FileNotFoundError: ``rows=True`` and that run's output is not in the
            cache (the step has ``cache=False``, the cache was cleared, or an
            older Forger saved the report).
    """
    rec = _find_record(name)
    if rows:
        return _load_rows(rec)
    if _reports_enabled(prod=False):
        _summaries.add_llm_summaries(rec, rows=None)
    print_step(rec, full=True)
    return rec


def _find_record(name: str | None) -> StepRecord:
    """The latest record of step ``name`` (the most recent step when None)."""
    records = _tracking.step_records()
    if name is None and records:
        return list(records.values())[-1]
    if name is not None and name in records:
        return records[name]
    saved = saved_step_records()
    if name is None and saved:
        return list(saved.values())[-1]
    if name is not None and name in saved:
        return saved[name]
    known = list(records) + [n for n in saved if n not in records]
    where = f"in this process or under {_cache_dir() / 'step_reports'}"
    if not known:
        raise KeyError(f"no @step has run {where}")
    raise KeyError(f"no step named {name!r} {where}; known steps: {', '.join(known)}")


def _load_rows(rec: StepRecord) -> Any:
    """Read the output of ``rec``'s run from its cache entry.

    A run kept out of the cache saved its output beside the entry instead
    (``<key>.uncached.pkl``); that is loaded with a warning saying why.
    """
    if rec.cache_file is None:
        raise FileNotFoundError(
            f"the latest run of step {rec.name!r} was not cached (cache=False, "
            "or a report saved by an older Forger); run the step again to cache "
            "its output"
        )
    root = _cache_dir()
    path = root / rec.cache_file
    try:
        result = _read_result(path, root)
    except FileNotFoundError as exc:  # also a blob that is gone
        raise FileNotFoundError(
            f"the cached output of step {rec.name!r} is gone ({path}); was "
            "FORGER_CACHE cleared? Run the step again"
        ) from exc
    if rec.cache_file.endswith(_UNCACHED_STEP_SUFFIX):
        logger.warning(_not_cached_note(rec))
    return result


def _not_cached_note(rec: StepRecord) -> str:
    """Why ``rec``'s output was not cached, e.g. after provider errors.

    e.g. "step 'solutions' was not cached, so its rows may be partial (e.g.
    None for failed requests): alter.generate_from_context: 10 of 357 prompts
    failed ...; a rerun runs the step again".
    """
    reasons = [w for w in rec.warnings if rec.kinds.get(w) in _NOT_CACHED_KINDS]
    why = "; ".join(reasons) or "an op kept its result out of the cache"
    return (
        f"step {rec.name!r} was not cached, so its rows may be partial (e.g. None "
        f"for failed requests): {why}; a rerun runs the step again, and the "
        "request cache reuses the completions that arrived"
    )

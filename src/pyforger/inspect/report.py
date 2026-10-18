"""Step reports: re-print the full report of a ``@step`` run."""

from __future__ import annotations

from pyforger import _summaries, _tracking
from pyforger._tracking import StepRecord
from pyforger.op import _cache_dir
from pyforger.steps import _reports_enabled, print_step, saved_step_records


def inspect_step(name: str | None = None) -> StepRecord:
    """Print the full report of the latest run of step ``name`` and return it.

    With no name, use the most recent step. Steps that ran in this process come
    first; otherwise the report is read from the one every step saves under
    ``FORGER_CACHE``, so a pipeline run in another process (e.g. under nohup)
    can be inspected from a shell. The full report shows the whole docstring, up
    to 10 sample rows, and up to 500 characters per field; a longer value is
    summarized by its shape, and with a report model
    (``pyforger.steps.set_report_model``) also in one sentence.

    Returns:
        The step's ``StepRecord``, as printed.

    Raises:
        KeyError: No step with that name has run; the message lists known steps.
    """
    records = _tracking.step_records()
    if name is None and records:
        rec = list(records.values())[-1]
    elif name is not None and name in records:
        rec = records[name]
    else:
        saved = saved_step_records()
        if name is None and saved:
            rec = list(saved.values())[-1]
        elif name is not None and name in saved:
            rec = saved[name]
        else:
            known = list(records) + [n for n in saved if n not in records]
            where = f"in this process or under {_cache_dir() / 'step_reports'}"
            if not known:
                raise KeyError(f"no @step has run {where}")
            raise KeyError(
                f"no step named {name!r} {where}; known steps: {', '.join(known)}"
            )
    if _reports_enabled(prod=False):
        _summaries.add_llm_summaries(rec, rows=None)
    print_step(rec, full=True)
    return rec

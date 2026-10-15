"""Step reports: re-print the full report of a ``@step`` run."""

from __future__ import annotations

from pyforger import _summaries, _tracking
from pyforger._tracking import StepRecord
from pyforger.steps import _reports_enabled, print_step


def inspect_step(name: str | None = None) -> StepRecord:
    """Print the full report of the latest run of step ``name`` and return it.

    With no name, use the most recent step that ran in this process. The full
    report shows the whole docstring, up to 10 sample rows, and up to 500
    characters per field; a longer value is summarized by its shape, and with
    a report model (``pyforger.steps.set_report_model``) also in one sentence.

    Raises:
        KeyError: No step with that name has run; the message lists known steps.
    """
    records = _tracking.step_records()
    if not records:
        raise KeyError("no @step has run in this process yet")
    if name is None:
        rec = list(records.values())[-1]
    elif name in records:
        rec = records[name]
    else:
        raise KeyError(f"no step named {name!r}; known steps: {', '.join(records)}")
    if _reports_enabled(prod=False):
        _summaries.add_llm_summaries(rec, rows=len(rec.sample))
    print_step(rec, full=True)
    return rec

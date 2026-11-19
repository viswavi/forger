"""Saved pipeline runs: ``runs`` lists them.

Every finished ``@pipeline`` run is saved under ``$FORGER_CACHE/pipeline_runs``
(see ``pyforger.steps._save_run``), one file per run, so two variants of a
pipeline never overwrite each other's record.
"""

from __future__ import annotations

from collections import defaultdict

from pyforger import _stats
from pyforger._tracking import PipelineRecord, StepRecord
from pyforger.op import _cache_dir
from pyforger.steps import _DIVERSITY_FILTERS, saved_runs

_RUN_YIELDS = 4  # step yields shown per run
# Ops whose score for a row depends on the other rows (an occurrence count,
# ROUGE-L against the rows kept so far), so its mean says little.
_CROSS_ROW_OPS = _DIVERSITY_FILTERS | {
    "filter.by_comparative_difficulty",
    "filter.by_group_agreement",
    "inspect.topic_based_clustering",
}


def runs(name: str | None = None, last: int = 10) -> list[PipelineRecord]:
    """Print the latest saved pipeline runs, oldest first, and return them.

    At most ``last`` runs, of pipeline ``name`` if given.

    Per run, oldest first: its id, start time, rows out, this run's cost, and
    its warnings, e.g. ``main-20261022-130510-4242  13:05:10  3 steps → 61 rows
    $0.41  2 warnings``; then the yields of steps that dropped rows (with 95%
    intervals) and the mean of the last filter score, e.g. ``judge 0.40
    (0.28–0.54) · mean filter.by_llm_score score 3.2 ± 0.2``.
    """
    found = [r for r in saved_runs() if name is None or r.name == name][-last:]
    if not found:
        print(f"no saved pipeline runs under {_cache_dir() / 'pipeline_runs'}")
        return []
    for rec in found:
        when = (rec.started_at or "?")[-8:]
        rows = "?" if rec.n_out is None else f"{rec.n_out:,}"
        smoke = f"  LIMIT {rec.smoke_rows}" if rec.smoke_rows else ""
        warn = f"  {len(rec.warnings)} warnings" if rec.warnings else ""
        print(
            f"  {rec.run_id}  {when}  {len(rec.steps)} steps → {rows} rows  "
            f"${rec.cost_usd:.2f}{warn}{smoke}"
        )
        headline = _headline(rec)
        if headline:
            print(f"      {headline}")
    return found


def _headline(rec: PipelineRecord) -> str:
    """Step yields and the last score's mean of one run."""
    parts = []
    dropped = [
        s for s in rec.steps if s.n_in and s.n_out is not None and s.n_out < s.n_in
    ]
    for s in dropped[:_RUN_YIELDS]:
        kept, seen = s.n_out or 0, s.n_in or 0
        lo, hi = _stats.wilson(kept, seen)
        parts.append(f"{s.name} {kept / seen:.2f} ({_stats.fmt_range(lo, hi)})")
    if len(dropped) > _RUN_YIELDS:
        parts.append(f"+{len(dropped) - _RUN_YIELDS} more")
    scores: tuple[str, list[float]] | None = None
    for step in rec.steps:
        for label, values in _score_values(step).items():
            scores = (label, values)  # the last one wins
    if scores is not None and len(set(scores[1])) > 1:
        parts.append(f"mean {scores[0]} {_mean_text(scores[1])}")
    return " · ".join(parts)


def _score_values(step: StepRecord) -> dict[str, list[float]]:
    """Numeric per-row scores per ``"<op> <label>"``, over chunks and calls.

    Scores that compare rows with each other (an occurrence count, ROUGE-L
    against the rows kept so far) are left out: their mean says little.
    """
    values: dict[str, list[float]] = defaultdict(list)
    for call in step.ops:
        if call.scores is None or call.scores.pooled or call.op_name in _CROSS_ROW_OPS:
            continue
        for value in call.scores.values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"{call.op_name} {call.scores.label}"].append(float(value))
    return {k: v for k, v in values.items() if v}


def _mean_text(values: list[float] | None) -> str:
    if not values:
        return "-"
    spread = _stats.mean_interval(values)
    mean = sum(values) / len(values)
    return f"{mean:.2f}" if spread is None else f"{mean:.2f} ± {spread[1]:.2f}"

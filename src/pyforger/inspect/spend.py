"""Spend ledger totals: what each run, pipeline, and step spent on LLM calls."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger import _spend
from pyforger._render import (
    WIDTH,
    _console,
    _emit,
    _fit_table,
    _plain,
    _rule,
    fmt_cost,
    fmt_tokens,
)


class SpendTotal(BaseModel):
    """What one step (or a pipeline outside its steps) spent in one run.

    ``pipeline`` and ``step`` are None for LLM calls made outside them.
    ``first`` and ``last`` are the ISO timestamps of its first and last call.
    """

    run_id: str
    pipeline: str | None = None
    step: str | None = None
    first: str
    last: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class SpendReport(BaseModel):
    """The ledger's totals, one row per run, pipeline, and step, oldest first."""

    path: str
    rows: list[SpendTotal] = Field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def spend(
    path: str | Path | None = None, *, since: datetime | str | None = None
) -> SpendReport:
    """Print and return the spend ledger's totals by run, pipeline, and step.

    The ledger (see ``pyforger._spend``) holds one line per LLM result, written
    as it is recorded, so it also covers runs that were killed or raised.

    Args:
        path: The ledger file; defaults to ``$FORGER_SPEND_LEDGER``, else
            ``$FORGER_CACHE/spend.jsonl``.
        since: Count only calls at or after this time (a datetime or an ISO
            string such as ``"2026-10-16T09:00"``; without a zone it is local
            time).

    Raises:
        ValueError: No path was given and ``FORGER_SPEND_LEDGER=0``.
    """
    ledger = Path(path).expanduser() if path is not None else _spend.ledger_path()
    if ledger is None:
        raise ValueError("the spend ledger is off (FORGER_SPEND_LEDGER=0); pass path=")
    start = _spend.parse_time(since) if since is not None else None
    records = _spend.read(ledger, start) if ledger.exists() else []

    report = SpendReport(path=str(ledger))
    groups: dict[tuple[str, str | None, str | None], SpendTotal] = {}
    for r in records:
        key = (str(r.get("run_id", "")), r.get("pipeline"), r.get("step"))
        total = groups.get(key)
        if total is None:
            total = groups[key] = SpendTotal(
                run_id=key[0],
                pipeline=key[1],
                step=key[2],
                first=r["timestamp"],
                last=r["timestamp"],
            )
        prompt = int(r.get("prompt_tokens") or 0)
        completion = int(r.get("completion_tokens") or 0)
        cost = float(r.get("cost_usd") or 0.0)
        total.last = r["timestamp"]
        total.calls += 1
        total.prompt_tokens += prompt
        total.completion_tokens += completion
        total.cost_usd += cost
        report.calls += 1
        report.prompt_tokens += prompt
        report.completion_tokens += completion
        report.cost_usd += cost
    report.rows = list(groups.values())
    console = _console()
    _emit(_spend_renderable(report, since, width=console.width), console)
    return report


def render_spend(report: SpendReport) -> str:
    """Render a ``spend()`` report as plain text."""
    return _plain(_spend_renderable(report, None, width=WIDTH))


def _started(timestamp: str) -> str:
    """``2026-10-16 11:02`` in local time."""
    try:
        return _spend.parse_time(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return timestamp


def _spend_renderable(
    report: SpendReport, since: datetime | str | None, *, width: int
) -> RenderableType:
    stats = Text()
    stats.append(f"{report.calls:,} LLM calls", style="bold")
    if report.prompt_tokens or report.completion_tokens:
        stats.append(
            f" · {report.prompt_tokens:,} in / {report.completion_tokens:,} out tokens"
        )
    stats.append(f" · {fmt_cost(report.cost_usd)}")
    if since is not None:
        stats.append(f" · since {since}", style="dim")
    body: list[RenderableType] = [stats, Text(report.path, style="dim")]
    if report.rows:
        rows = [
            [
                Text(" · ".join(x or "(none)" for x in (t.pipeline, t.step))),
                Text(t.run_id),
                Text(_started(t.first)),
                Text(f"{t.calls:,}"),
                Text(fmt_tokens(t.prompt_tokens)),
                Text(fmt_tokens(t.completion_tokens)),
                Text(fmt_cost(t.cost_usd)),
            ]
            for t in report.rows
        ]
        headers = [
            "pipeline · step",
            "run",
            "started",
            "calls",
            "tokens in",
            "tokens out",
            "cost",
        ]
        body += [
            Text(""),
            _fit_table(headers, rows, width - 2, left={"run", "started"}),
        ]
    return Group(_rule("spend", "ledger"), Padding(Group(*body), (0, 0, 0, 2)))

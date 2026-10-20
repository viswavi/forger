"""Render ``@step`` and ``@pipeline`` reports with ``rich``.

Each ``render_*`` function returns plain text (no color) at 120 columns, which
tests and log readers can use. Each ``print_*`` function prints the same report
to stdout: in color and at the terminal's width (up to 120) on a terminal, and
as plain 120-column text when stdout is a file.

A report has no side borders, so log files are not padded with box characters.
All text that comes from user data (docstrings, op names, sample rows,
warnings) goes through ``Text`` so rich never parses it as markup.
"""

from __future__ import annotations

import io
import json
import math
import sys
import threading
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from pyforger import _summaries
from pyforger._tracking import (
    FieldSummary,
    OpCall,
    PipelineRecord,
    ScoreRecord,
    StepRecord,
)

WIDTH = 120  # reports never exceed this many columns
COMPACT_FIELD_CHARS = 80  # per sample field in the step report
FULL_FIELD_CHARS = 500  # per sample field in inspect_step
_FULL_FIELD_LINES = 30  # lines kept per multi-line sample field in inspect_step
_SAMPLE_ITEMS = 20  # items of a list kept in a step's stored sample
_BAR_CHARS = 28  # length of the longest histogram bar
_MAX_TABLE_FIELDS = 4  # samples with more fields are listed as field / value
_MAX_DISTINCT_BINS = 12  # up to this many distinct values get one bin each
_TOKENS_MIN_WIDTH = 110  # narrower terminals drop the per-op token columns
_EIGHTHS = " ▏▎▍▌▋▊▉"
_KEEP_SYMBOL = {"ge": "≥", "le": "≤", "lt": "<"}
_LINE_BREAK = _summaries.LINE_BREAK  # where a compact sample value had a newline
_MIN_NAME = 12  # narrowest the op / step name column gets before others drop
_PRINT_LOCK = threading.Lock()  # keeps reports from threads apart


# --------------------------------------------------------------------------- #
# Number formatting
# --------------------------------------------------------------------------- #


def fmt_int(n: int | None) -> str:
    """Format a row count with thousands separators ("-" when unknown)."""
    return "-" if n is None else f"{n:,}"


def fmt_yield(y: float | None) -> str:
    """Format a yield ratio with two decimals ("-" when unknown)."""
    return "-" if y is None or not math.isfinite(y) else f"{y:.2f}"


def fmt_secs(s: float) -> str:
    """Format seconds as ``12.3s``, or ``4m 05s`` from a minute up."""
    if not math.isfinite(s):
        return "-"
    if s < 60:
        return f"{s:.1f}s"
    minutes, seconds = divmod(int(round(s)), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_cost(c: float) -> str:
    """Format dollars: ``$12.34`` from a dollar up, else two significant digits."""
    if not c or not math.isfinite(c):
        return "-"
    if c >= 1:
        return f"${c:,.2f}"
    if c < 0.0001:
        return "<$0.0001"
    return "$" + f"{c:.2g}"


def fmt_units(units: float) -> str:
    """Format token units compactly: 950, 12.3K, 4.81M ("-" when zero or unknown)."""
    if not units or not math.isfinite(units):
        return "-"
    return _compact(units)


def _compact(n: float) -> str:
    """Exact below 10,000 (``9,840``), else ``38.9K`` or ``7.65M``."""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 10_000:
        return f"{n / 1e3:.1f}K"
    return f"{n:,.0f}"


def fmt_tokens(n: int) -> str:
    """Format one token count compactly ("-" when zero)."""
    return _compact(n) if n else "-"


def _strip_zeros(text: str) -> str:
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_num(x: float) -> str:
    """Compact number: ``3``, ``0.7``, ``0.25``, ``1,204``, ``12.5k``, ``7.65M``."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    a = abs(x)
    if a >= 1e12:
        return f"{x:.3g}"
    if a >= 1e9:
        return _strip_zeros(f"{x / 1e9:,.2f}") + "B"
    if a >= 1e6:
        return _strip_zeros(f"{x / 1e6:,.2f}") + "M"
    if a >= 10_000:
        return _strip_zeros(f"{x / 1e3:,.1f}") + "k"
    if float(x).is_integer():
        return f"{int(x):,}"
    if a < 0.01:
        return f"{x:.2g}"
    return _strip_zeros(f"{x:.2f}") if a < 10 else _strip_zeros(f"{x:,.1f}")


def _fmt_cut(x: float) -> str:
    """A threshold, precise enough to tell apart: ``0.999``, ``2/3``, ``0.65``."""
    label = _fmt_value_label(x)
    if "/" in label or not math.isfinite(x) or float(x).is_integer():
        return label
    if 0.01 <= abs(x) < 10_000:
        return _strip_zeros(f"{x:.4g}") if "e" not in f"{x:.4g}" else f"{x:.4g}"
    return fmt_num(x)


def _unique_labels(values: list[float], fmt: Any) -> list[str]:
    """Format values with ``fmt``; add digits until no two labels collide."""
    labels = [fmt(v) for v in values]
    digits = 3
    while len(set(labels)) < len(set(values)) and digits <= 15:
        labels = [f"{v:.{digits}g}" for v in values]
        digits += 1
    return labels


def _fmt_value_label(v: float) -> str:
    """Label one distinct score: ints as is, k/n fractions as ``2/3``."""
    if float(v).is_integer():
        return fmt_num(v)
    frac = Fraction(v).limit_denominator(12)
    if abs(float(frac) - v) < 1e-9 and frac.denominator in (3, 6, 7, 9, 11, 12):
        return f"{frac.numerator}/{frac.denominator}"
    return fmt_num(v)


def _clean(text: Any, *, single_line: bool = True) -> str:
    """User text made safe for a terminal or log: no control characters.

    Tabs become spaces; newlines become spaces when ``single_line``; any other
    C0/C1 control character (ANSI escapes, bells, carriage returns) becomes �.
    Unicode formatting characters (emoji joiners, direction marks) are kept.
    """
    out = []
    for ch in str(text):
        if ch == "\n":
            out.append(" " if single_line else ch)
        elif ch == "\t":
            out.append(" ")
        elif unicodedata.category(ch) == "Cc":
            out.append("�")
        elif unicodedata.category(ch) in ("Zl", "Zp"):
            out.append(" " if single_line else "\n")
        else:
            out.append(ch)
    return "".join(out)


def _usage_summary(prompt: int, completion: int, units: float, cost: float) -> str:
    parts = []
    if prompt or completion:
        parts.append(f"{prompt:,} in / {completion:,} out tokens")
    if units and math.isfinite(units):
        parts.append(f"{fmt_units(units)} units")
    if cost and math.isfinite(cost):
        parts.append(fmt_cost(cost))
    return " · ".join(parts)


def _originally(rec: StepRecord) -> str:
    """`` (originally 37.9s · 1,204 in / 310 out tokens · $0.50)``, cached steps."""
    if rec.original is None:
        return ""
    o = rec.original
    parts = [fmt_secs(o.duration_s)]
    usage = _usage_summary(
        o.prompt_tokens, o.completion_tokens, o.token_units, o.cost_usd
    )
    if usage:
        parts.append(usage)
    return f" (originally {' · '.join(parts)})"


def _rows_text(n_in: int | None, n_out: int | None, *, generated: str) -> str:
    """``143 → 61 rows (yield 0.43)``, or ``276 rows generated`` without input."""
    if n_in is None:
        return f"{fmt_int(n_out)} rows {generated}"
    text = f"{fmt_int(n_in)} → {fmt_int(n_out)} rows"
    if n_in and n_out is not None:
        text += f" (yield {n_out / n_in:.2f})"
    return text


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #


@dataclass
class _Bin:
    label: str
    count: int
    cut: str = ""  # e.g. "≥ 4": a threshold placed at this bin
    kept: int | None = None  # rows that threshold would keep
    share: float | None = None  # kept / all scored rows
    status: str = ""  # between filters: "yes", "no", or "partly"
    current: bool = False
    empty_run: int = 0  # > 0: this row stands for that many empty bins
    dim: bool = False  # overflow and empty-run rows


@dataclass
class _Histogram:
    label: str
    title: str
    bins: list[_Bin] = field(default_factory=list)
    unparsed: int = 0
    mode: str = "cut"  # "cut", "between", "pooled", or "none"
    note: str = ""


def _finite(values: Iterable[Any]) -> list[Any]:
    return [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and not math.isfinite(v))
    ]


def _kept_count(values: list[float], keep: str, t: float) -> int:
    if keep == "ge":
        return sum(1 for v in values if v >= t)
    if keep == "le":
        return sum(1 for v in values if v <= t)
    return sum(1 for v in values if v < t)


def _nice_width(span: float, integer: bool) -> float:
    """Smallest round bin width (1, 2, 2.5, or 5 times a power of ten) for 10 bins."""
    raw = span / 10 if span > 0 else 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        width = step * power
        if width >= raw * (1 - 1e-9) and (not integer or width.is_integer()):
            return width
    return 10.0 * power


def _percentile(sorted_nums: list[float], q: float) -> float:
    i = min(len(sorted_nums) - 1, max(0, int(round(q * (len(sorted_nums) - 1)))))
    return sorted_nums[i]


@dataclass
class _RawBin:
    label: str
    count: int
    lo: float  # smallest value in the bin
    hi: float  # largest value in the bin (distinct bins: == lo)
    upper: float  # exclusive upper edge for ranges, == hi for distinct values
    overflow: bool = False
    distinct: bool = False  # one bin per value


def _raw_bins(good: list[Any], scores: ScoreRecord) -> list[_RawBin]:
    """Bin the finite scores: one bin per distinct value, or round-width ranges."""
    nums = sorted(float(v) for v in good)
    distinct = sorted(set(nums))
    if len(distinct) <= _MAX_DISTINCT_BINS:
        tally = Counter(nums)
        labels = _unique_labels(distinct, _fmt_value_label)
        return [
            _RawBin(label, tally[v], v, v, v, distinct=True)
            for label, v in zip(labels, distinct)
        ]

    is_int = all(isinstance(v, int) for v in good)
    lo, hi = nums[0], nums[-1]
    if not is_int and lo >= 0.0 and hi <= 1.0:
        width, first, last = 0.1, 0.0, 1.0
    else:
        # Bin the middle 98% so one outlier does not stretch every bin; the
        # tails get their own overflow rows. Keep the thresholds in range.
        p_lo, p_hi = _percentile(nums, 0.01), _percentile(nums, 0.99)
        for t in (scores.threshold, scores.lower, scores.upper):
            if t is not None and math.isfinite(t) and lo <= t <= hi:
                p_lo, p_hi = min(p_lo, t), max(p_hi, t)
        width = _nice_width(p_hi - p_lo, is_int)
        first = math.floor(p_lo / width + 1e-9) * width
        last = (math.floor(p_hi / width + 1e-9) + 1) * width
    nbins = max(1, int(round((last - first) / width)))
    counts = [0] * nbins
    below = above = 0
    for v in nums:
        if v < first - 1e-9:
            below += 1
        elif v >= last - 1e-9 and not (last == 1.0 and width == 0.1):
            above += 1
        else:
            # The epsilon keeps edge values in their own bin (0.3 / 0.1 < 3).
            counts[min(int((v - first) / width + 1e-9), nbins - 1)] += 1
    edges = [first + i * width for i in range(nbins + 1)]
    names = _unique_labels(edges, fmt_num)
    out: list[_RawBin] = []
    if below:
        out.append(_RawBin(f"< {names[0]}", below, lo, first, first, True))
    for i, count in enumerate(counts):
        a, b = edges[i], edges[i + 1]
        if is_int and width == 1:
            label = names[i]
        elif is_int:
            label = f"{names[i]}–{fmt_num(b - 1)}"
        else:
            label = f"{names[i]}–{names[i + 1]}"
        out.append(_RawBin(label, count, a, b - 1 if is_int else b, b))
    if above:
        out.append(_RawBin(f"≥ {names[-1]}", above, last, hi, math.inf, True))
    return out


def _cut_for(raw: _RawBin, keep: str) -> float:
    """The threshold that keeps this bin and everything on the kept side of it."""
    if keep == "ge":
        return raw.lo
    if keep == "le" and (raw.distinct or raw.hi != raw.upper):
        return raw.hi  # one value, or an int range a..b-1
    if keep == "le":
        # Float range [a, b): "<= t" keeps it all only just below b.
        return math.nextafter(raw.upper, -math.inf)
    # keep == "lt": below the bin's exclusive upper edge; one value: just above it.
    return math.nextafter(raw.hi, math.inf) if raw.distinct else raw.upper


def _cut_label(keep: str, t: float, raw: _RawBin) -> str:
    if keep == "lt" and raw.distinct:
        return f"≤ {raw.label}"  # same rows as "< the next value"
    if keep == "le" and not raw.distinct and raw.hi == raw.upper:
        return f"< {_fmt_cut(raw.upper)}"  # float range: everything below b
    return f"{_KEEP_SYMBOL[keep]} {_fmt_cut(t)}"


def _current_index(raws: list[_RawBin], keep: str, current: float) -> int | None:
    """The bin at the edge of the kept rows for the current threshold.

    For ``ge`` that is the lowest bin with a kept value; for ``le`` and
    ``lt`` the highest. Range bins are half open, ``[lo, upper)``.
    """
    if math.isnan(current):
        return None
    if keep == "ge":
        for i, r in enumerate(raws):
            if (r.lo >= current) if r.distinct else (r.upper > current):
                return i
        return None
    for i in range(len(raws) - 1, -1, -1):
        r = raws[i]
        float_range = not r.distinct and r.hi == r.upper
        if keep == "lt" or (keep == "le" and float_range):
            if r.lo < current:
                return i
        elif r.lo <= current:
            return i
    return None


def _merge_empty_runs(bins: list[_Bin]) -> list[_Bin]:
    """Fold two or more consecutive empty, unmarked bins into one row."""
    out: list[_Bin] = []
    run: list[_Bin] = []

    def flush() -> None:
        if len(run) >= 2:
            label = f"{run[0].label.split('–')[0]}–{run[-1].label.split('–')[-1]}"
            out.append(_Bin(label, 0, empty_run=len(run), dim=True))
        else:
            out.extend(run)
        run.clear()

    for b in bins:
        if b.count == 0 and not b.current:
            run.append(b)
            continue
        flush()
        out.append(b)
    flush()
    return out


def _histogram(scores: ScoreRecord) -> _Histogram:
    """Bin a filter's per-row scores and work out what each threshold keeps.

    Up to 12 distinct values get one bin each (``2/3`` for judge fractions).
    Otherwise floats in [0, 1] get ten bins of 0.1 and other values about ten
    round-width bins over their middle 98%, with overflow rows for the tails.
    Each bin row says which rows a threshold placed there would keep, and the
    bin at the edge of the current threshold is marked.
    """
    total = len(scores.values)
    good = _finite(scores.values)
    label = _clean(scores.label)
    hist = _Histogram(label=label, title=label)
    hist.unparsed = total - len(good)
    if label.startswith("approval >="):
        hist.label = f"judges ≥{label.removeprefix('approval >=')}"
        hist.title = f"share of judges scoring ≥{label.removeprefix('approval >=')}"
    if not total:
        hist.mode, hist.title = "none", "no rows scored"
        return hist
    if scores.keep == "true" or (good and all(isinstance(v, bool) for v in good)):
        kept = sum(1 for v in good if v)
        hist.mode = "none"
        hist.title = (
            f"kept {kept:,} of {total:,} ({kept / total:.2f}) · "
            f"dropped {total - kept:,}"
        )
        return hist
    if not good:
        hist.mode, hist.title = "none", f"{label}: all {total:,} rows unparsed"
        return hist

    nums = [float(v) for v in good]
    raws = _raw_bins(good, scores)

    if scores.keep == "between":
        lower = scores.lower if scores.lower is not None else -math.inf
        upper = scores.upper if scores.upper is not None else math.inf
        kept = sum(1 for v in nums if lower <= v <= upper)
        hist.mode = "between"
        hist.title += (
            f" · keeps {_fmt_cut(lower)} to {_fmt_cut(upper)}: "
            f"{kept:,} of {total:,} ({kept / total:.2f})"
        )
        for r in raws:
            if lower <= r.lo and r.hi <= upper:
                status = "yes"
            elif r.hi < lower or r.lo > upper:
                status = "no"
            else:
                status = "partly"
            hist.bins.append(_Bin(r.label, r.count, status=status, dim=r.overflow))
        hist.bins = _merge_empty_runs(hist.bins)
        return hist

    symbol = _KEEP_SYMBOL[scores.keep]
    current = scores.threshold
    marked = _current_index(raws, scores.keep, current) if current is not None else None
    if current is not None:
        kept_now = _kept_count(nums, scores.keep, current)
        hist.title += (
            f" · keeps {symbol} {_fmt_cut(current)}: "
            f"{kept_now:,} of {total:,} ({kept_now / total:.2f})"
        )
    hist.mode = "pooled" if scores.pooled else "cut"
    if scores.pooled:
        hist.note = (
            "No what-if columns: each score depends on the rows kept before it. "
            "Rerun to compare thresholds."
        )
    for i, r in enumerate(raws):
        is_current = i == marked
        if is_current and current is not None:
            # The marked row shows the real threshold, which may sit mid-bin.
            t, cut = current, f"{symbol} {_fmt_cut(current)}"
        else:
            t = _cut_for(r, scores.keep)
            cut = _cut_label(scores.keep, t, r)
        n_kept = _kept_count(nums, scores.keep, t)
        hist.bins.append(
            _Bin(
                r.label,
                r.count,
                cut="" if scores.pooled else cut,
                kept=None if scores.pooled else n_kept,
                share=None if scores.pooled else n_kept / total,
                current=is_current,
                dim=r.overflow,
            )
        )
    hist.bins = _merge_empty_runs(hist.bins)
    return hist


def _bar(count: int, largest: int, size: int = _BAR_CHARS) -> str:
    if not count or not largest:
        return ""
    eighths = max(1, round(count / largest * size * 8))
    full, rest = divmod(eighths, 8)
    return "█" * full + (_EIGHTHS[rest] if rest else "")


def _histogram_renderable(
    hist: _Histogram, op_name: str, width: int = WIDTH
) -> RenderableType:
    heading = Text()
    heading.append(_clean(op_name), style="bold")
    heading.append(f" · {hist.title}")
    if hist.mode == "none" or not hist.bins:
        return heading

    def style_of(b: _Bin) -> str:
        return "bold" if b.current else ("dim" if b.dim else "")

    def label_of(b: _Bin) -> str:
        return f"{b.label}  ({b.empty_run} empty bins)" if b.empty_run else b.label

    head = hist.label if len(hist.label) <= 20 else hist.label[:19] + "…"
    # (header, right-aligned, cell per bin); "bar" is filled in below.
    columns: list[tuple[str | Text, bool, list[Text]]] = [
        ("", False, [Text("▶" if b.current else "", style="green") for b in hist.bins]),
        (Text(head), True, [Text(label_of(b), style=style_of(b)) for b in hist.bins]),
        ("rows", True, [Text(f"{b.count:,}", style=style_of(b)) for b in hist.bins]),
        ("bar", False, []),
    ]
    if hist.mode == "cut":
        blank = [b.empty_run > 0 for b in hist.bins]
        columns += [
            (
                "if cut at",
                False,
                [
                    Text("" if e else b.cut, style=style_of(b))
                    for b, e in zip(hist.bins, blank)
                ],
            ),
            (
                "keeps",
                True,
                [
                    Text("" if e else fmt_int(b.kept), style=style_of(b))
                    for b, e in zip(hist.bins, blank)
                ],
            ),
            (
                "share",
                True,
                [
                    Text("" if e else fmt_yield(b.share), style=style_of(b))
                    for b, e in zip(hist.bins, blank)
                ],
            ),
        ]
    elif hist.mode == "between":
        columns.append(
            (
                "kept",
                False,
                [
                    Text("" if b.empty_run else b.status, style=style_of(b))
                    for b in hist.bins
                ],
            )
        )
    columns.append(
        (
            "pointer",
            False,
            [
                Text("◀ current threshold" if b.current else "", style="green")
                for b in hist.bins
            ],
        )
    )
    if hist.unparsed:
        for header, _, cells in columns:
            name = header.plain if isinstance(header, Text) else header
            if name == head:
                cells.append(Text("unparsed", style="yellow"))
            elif name == "rows":
                cells.append(Text(f"{hist.unparsed:,}", style="yellow"))
            elif name == "if cut at":
                cells.append(Text("never kept", style="dim"))
            elif name == "kept":
                cells.append(Text("no", style="dim"))
            elif name != "bar":
                cells.append(Text(""))

    def width_of(column: tuple[str | Text, bool, list[Text]]) -> int:
        header, _, cells = column
        name = header.plain if isinstance(header, Text) else header
        if name in ("bar", "pointer"):
            name = ""
        return max([len(name), *(c.cell_len for c in cells)])

    def fixed() -> int:
        cols = [c for c in columns if c[0] != "bar"]
        # Two spaces between columns, and the table is indented by 2.
        return 2 + sum(width_of(c) for c in cols) + 2 * (len(cols))

    # Narrow consoles lose, in order: the pointer text, then the bar, then
    # the kept-row counts (the share stays).
    for drop in ("pointer", "bar", "keeps"):
        if width - fixed() >= (6 if any(c[0] == "bar" for c in columns) else 0):
            break
        columns = [c for c in columns if c[0] != drop]
    bar = max(0, min(_BAR_CHARS, width - fixed()))
    if bar < 6:
        columns = [c for c in columns if c[0] != "bar"]

    largest = max(b.count for b in hist.bins)
    table = Table(
        box=None, show_header=True, pad_edge=False, padding=(0, 1), header_style="dim"
    )
    rows: list[list[Text]] = [[] for _ in range(len(hist.bins) + bool(hist.unparsed))]
    for header, right, cells in columns:
        if header == "bar":
            table.add_column("", no_wrap=True, width=bar)
            bars = [Text(_bar(b.count, largest, bar), style="cyan") for b in hist.bins]
            cells = bars + ([Text("")] if hist.unparsed else [])
        else:
            shown = "" if header == "pointer" else header
            table.add_column(
                shown,
                justify="right" if right else "left",
                no_wrap=True,
                width=width_of((header, right, cells)),
            )
        for row, cell in zip(rows, cells):
            row.append(cell)
    for row in rows:
        table.add_row(*row)
    parts: list[RenderableType] = [heading, Padding(table, (0, 0, 0, 2))]
    if hist.note:
        parts.append(Padding(Text(hist.note, style="dim"), (0, 0, 0, 2)))
    return Group(*parts)


# --------------------------------------------------------------------------- #
# Step report
# --------------------------------------------------------------------------- #


def _show_yield(call: OpCall) -> bool:
    return (
        call.category == "filter"
        or call.min_yield is not None
        or call.max_yield is not None
        or (call.n_in is not None and call.n_out != call.n_in)
    )


def _check_text(call: OpCall) -> Text:
    """The op table's yield-check cell, verdict first: ``LOW · min 0.60``."""
    text = Text()
    if call.error is not None:
        text.append("RAISED", style="bold red")
        return text
    y = call.yield_ratio
    if y is not None and not math.isfinite(y):
        y = None
    checks: list[tuple[str, str]] = []
    if call.min_yield is not None and math.isfinite(call.min_yield):
        status = "-" if y is None else ("ok" if y >= call.min_yield else "LOW")
        if call.n_in == 0 and call.min_yield > 0:
            status = "LOW"
        checks.append((status, f"min {call.min_yield:.2f}"))
    if call.max_yield is not None and math.isfinite(call.max_yield):
        status = "-" if y is None else ("ok" if y <= call.max_yield else "HIGH")
        checks.append((status, f"max {call.max_yield:.2f}"))
    for status, bound in checks:
        if text:
            text.append("  ")
        text.append(status, style="green" if status == "ok" else "bold red")
        text.append(f" · {bound}")
    return text


def _first_column_room(headers: list[str], rows: list[list[Text]], width: int) -> int:
    """Columns left for the first column after every other column is full width."""
    others = sum(
        max([len(h), *(r[i].cell_len for r in rows)])
        for i, h in enumerate(headers)
        if i
    )
    # SIMPLE_HEAD tables: one space of padding each side plus a separator.
    return width - others - 3 * (len(headers) - 1)


def _fit_table(
    headers: list[str], rows: list[list[Text]], width: int, *, left: set[str]
) -> Table:
    """A table whose first column alone is cut (with …) to fit ``width``.

    rich would otherwise crop whole columns off the right edge.
    """
    room = _first_column_room(headers, rows, width)
    # Still too narrow: the last column (yield check, flags) wraps instead.
    squeeze = max(0, 8 - room)
    room = max(8, room)
    table = Table(
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold"
    )
    last = len(headers) - 1
    for i, name in enumerate(headers):
        cells = [r[i] for r in rows]
        widest = max([len(name), *(c.cell_len for c in cells)])
        if i == 0:
            table.add_column(name, no_wrap=True, width=min(widest, room))
            for c in cells:
                c.truncate(min(widest, room), overflow="ellipsis")
        elif i == last and squeeze:
            table.add_column(name, overflow="fold", width=max(4, widest - squeeze))
        else:
            table.add_column(
                name,
                justify="left" if name in left or i == last else "right",
                no_wrap=True,
                width=widest,
            )
    for cells in rows:
        table.add_row(*cells)
    return table


def _lines_view(headers: list[str], rows: list[list[Text]]) -> RenderableType:
    """Too narrow for a table: one wrapped line per row, ``name  in → out ...``."""
    lines: list[RenderableType] = []
    for cells in rows:
        line = Text()
        line.append_text(cells[0])
        line.append("  ")
        parts: list[Text] = []
        values = dict(zip(headers, cells))
        if "in" in values and "out" in values:
            parts.append(Text(f"{values['in'].plain} → {values['out'].plain}"))
        for name, cell in zip(headers[1:], cells[1:]):
            if name in ("in", "out") or not cell.plain or cell.plain == "-":
                continue
            label = Text(f"{name} ") if name in ("yield", "units") else Text("")
            parts.append(label + cell)
        line.append_text(Text(" · ").join(parts))
        lines.append(line)
    return Group(*lines)


def _drop_columns(
    headers: list[str], rows: list[list[Text]], width: int, optional: list[str]
) -> tuple[list[str], list[list[Text]]]:
    """Drop ``optional`` columns, in order, until the first column has room."""
    for name in optional:
        if _first_column_room(headers, rows, width) >= _MIN_NAME:
            break
        drop = {i for i, h in enumerate(headers) if h == name}
        headers = [h for i, h in enumerate(headers) if i not in drop]
        rows = [[c for i, c in enumerate(r) if i not in drop] for r in rows]
    return headers, rows


def _op_table(ops: list[OpCall], width: int) -> Table:
    show_tokens = width >= _TOKENS_MIN_WIDTH and any(
        c.prompt_tokens or c.completion_tokens for c in ops
    )
    show_units = any(c.token_units and math.isfinite(c.token_units) for c in ops)
    show_cost = any(c.cost_usd and math.isfinite(c.cost_usd) for c in ops)
    checks = [_check_text(call) for call in ops]
    show_check = any(checks)
    show_cached = any(c.cache_hit for c in ops)

    rows: list[list[Text]] = []
    for call, check in zip(ops, checks):
        cells = [
            Text(_clean(call.op_name)),
            Text(fmt_int(call.n_in)),
            Text(fmt_int(call.n_out)),
            Text(fmt_yield(call.yield_ratio) if _show_yield(call) else "-"),
            Text("-" if call.cache_hit else fmt_secs(call.duration_s)),
        ]
        if show_tokens:
            cells.append(Text(fmt_tokens(call.prompt_tokens)))
            cells.append(Text(fmt_tokens(call.completion_tokens)))
        if show_units:
            cells.append(Text(fmt_units(call.token_units)))
        if show_cost:
            cells.append(Text(fmt_cost(call.cost_usd)))
        if show_cached:
            cells.append(Text("cached" if call.cache_hit else "", style="dim"))
        if show_check:
            cells.append(check)
        rows.append(cells)

    headers = ["op", "in", "out", "yield", "time"]
    if show_tokens:
        headers += ["tokens in", "tokens out"]
    if show_units:
        headers.append("units")
    if show_cost:
        headers.append("cost")
    if show_cached:
        headers.append("")
    if show_check:
        headers.append("yield check")

    # Too narrow: drop columns whose totals the stats line already shows.
    headers, rows = _drop_columns(
        headers, rows, width, ["tokens in", "tokens out", "", "units", "cost"]
    )
    if _first_column_room(headers, rows, width) < _MIN_NAME:
        return _lines_view(headers, rows)
    return _fit_table(headers, rows, width, left={"", "yield check"})


def _value_text(value: Any, *, full: bool) -> tuple[str, bool]:
    """The value as report text, uncut, and whether it has too many lines."""
    if isinstance(value, str):
        lines = [line.rstrip() for line in value.strip("\n").splitlines()]
        if full:
            return "\n".join(lines), len(lines) > _FULL_FIELD_LINES
        return _LINE_BREAK.join(" ".join(line.split()) for line in lines if line), False
    try:
        return json.dumps(value, ensure_ascii=False, default=str), False
    except (TypeError, ValueError):
        return repr(value), False


def _fits(value: Any, *, full: bool) -> bool:
    text, too_tall = _value_text(value, full=full)
    return not too_tall and len(text) <= (
        FULL_FIELD_CHARS if full else COMPACT_FIELD_CHARS
    )


def sample_summaries(row: dict) -> dict[int, FieldSummary]:
    """Summaries of the long values in one sample row, keyed by field position.

    Made from the whole row when the step returns (the stored sample is cut to
    500 characters per string). A value that cannot be summarized is skipped.
    """
    out: dict[int, FieldSummary] = {}
    for i, value in enumerate(row.values()):
        try:
            compact_fits = _fits(value, full=False)
            if compact_fits:
                continue
            # The stored sample keeps only the first 20 items of a list.
            cut_list = isinstance(value, (list, tuple)) and len(value) > _SAMPLE_ITEMS
            full_fits = _fits(value, full=True) and not cut_list
            compact = _summaries.compact_summary(value, COMPACT_FIELD_CHARS)
            full = (
                None if full_fits else _summaries.full_summary(value, FULL_FIELD_CHARS)
            )
            if compact is None and full is None:
                continue
            out[i] = FieldSummary(
                shape=_summaries.shape_of(value),
                compact=compact,
                full=full,
                source=_summaries.summary_source(value),
            )
        except Exception:  # an odd value keeps the plain, cut view
            continue
    return out


def _cut_value(text: str, limit: int, *, full: bool) -> str:
    # Control characters (ANSI escapes, tabs) would move the cursor in a log.
    text = _clean(text, single_line=not full)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_value(
    value: Any, limit: int, *, full: bool, summary: FieldSummary | None = None
) -> str:
    """One sample value as report text: as is, summarized, or cut to ``limit``.

    ``summary`` comes from the step run; without one, a value too long for
    ``limit`` is summarized from the (possibly already cut) sample value.
    """
    if summary is None and not _fits(value, full=full):
        summary = sample_summaries({"": value}).get(0)
    shown = None if summary is None else (summary.full if full else summary.compact)
    llm = summary.llm if summary is not None else None
    if full:
        if shown is None:
            text, too_tall = _value_text(value, full=True)
            if too_tall:
                text = "\n".join(text.split("\n")[:_FULL_FIELD_LINES] + ["…"])
            shown = _cut_value(text, limit, full=True)
        shown = _cut_value(shown, 2 * limit, full=True)
        return f"{shown}\nLLM: {_clean(llm)}" if llm else shown
    # The value's own text comes first; the LLM sentence is a second line, so
    # it adds to what the row says instead of standing in for it.
    if llm and shown is not None:
        shown = _cut_value(shown, 2 * limit, full=False)
        return f"{shown}\nLLM: {_cut_value(llm, 2 * limit, full=False)}"
    if shown is not None:
        return _cut_value(shown, 2 * limit, full=False)
    return _cut_value(_value_text(value, full=False)[0], limit, full=False)


def _samples(rec: StepRecord, full: bool, width: int = WIDTH) -> RenderableType | None:
    if not rec.sample:
        return None
    shown = rec.sample if full else rec.sample[: max(0, rec.sample_shown)]
    if not shown:
        return None
    limit = FULL_FIELD_CHARS if full else COMPACT_FIELD_CHARS
    heading = Text(f"sample rows ({len(shown)} of {fmt_int(rec.n_out)})", style="bold")
    keys = list(dict.fromkeys(k for row in shown for k in row))
    fields = list(dict.fromkeys(_clean(k) for k in keys))
    summaries = list(rec.sample_summaries)

    def value_text(i: int, j: int, value: Any) -> Text:
        summary = summaries[i].get(j) if i < len(summaries) else None
        return Text(_fmt_value(value, limit, full=full, summary=summary))

    def cell(i: int, row: dict, name: str) -> Text:
        for j, (k, v) in enumerate(row.items()):
            if _clean(k) == name:
                return value_text(i, j, v)
        return Text("")

    # The table view needs one column per key; colliding names (1 and "1")
    # fall back to the block view, which lists every key.
    if not full and len(fields) <= _MAX_TABLE_FIELDS and len(fields) == len(keys):
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
        table.add_column("#", justify="right", style="dim", no_wrap=True)
        for name in fields:
            table.add_column(Text(name), overflow="fold", ratio=1)
        for i, row in enumerate(shown):
            table.add_row(Text(str(i)), *(cell(i, row, name) for name in fields))
        return Group(heading, table)
    # Many fields, or the full view: one block per row, one line per field.
    blocks: list[RenderableType] = [heading]
    for i, row in enumerate(shown):
        grid = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1))
        key_width = max(4, min(30, width // 3))
        grid.add_column(style="dim", no_wrap=True, max_width=key_width)
        grid.add_column(overflow="fold")
        for j, (k, v) in enumerate(row.items()):
            key = Text(_clean(k))
            key.truncate(key_width, overflow="ellipsis")
            grid.add_row(key, value_text(i, j, v))
        blocks.append(Text(f"[{i}]", style="dim"))
        blocks.append(Padding(grid, (0, 0, 0, 2)))
    return Group(*blocks)


def _warning_lines(warnings: Iterable[str], prefix: str = "") -> list[Text]:
    out: list[Text] = []
    for w in dict.fromkeys(warnings):
        line = Text()
        line.append("⚠ ", style="yellow")
        if prefix:
            line.append(f"{_clean(prefix)}: ", style="bold")
        line.append(_clean(w))
        out.append(line)
    return out


def _rule(kind: str, name: str) -> Rule:
    title = Text()
    title.append(f"{kind} ", style="dim")
    title.append(_clean(name), style="bold")
    return Rule(title, align="left", style="blue" if kind == "step" else "magenta")


def step_renderable(
    rec: StepRecord, *, full: bool = False, width: int = WIDTH
) -> RenderableType:
    """Build the rich renderable for a step report (see ``render_step``)."""
    body: list[RenderableType] = []
    doc_lines = [_clean(line) for line in rec.doc.strip().splitlines()]
    if doc_lines:
        doc = "\n".join(doc_lines) if full else doc_lines[0]
        body.append(Text(doc, style="italic"))
        if full and len(doc_lines) > 1:
            body.append(Text(""))

    one_plain_op = (
        len(rec.ops) == 1 and not _check_text(rec.ops[0]) and not rec.ops[0].cache_hit
    )
    stats = Text()
    stats.append(
        _rows_text(rec.n_in, rec.n_out, generated="generated"),
        style="bold red" if rec.warnings else "bold",
    )
    if one_plain_op:
        stats.append(f" · {_clean(rec.ops[0].op_name)}")
    stats.append(f" · {fmt_secs(rec.duration_s)}")
    usage = _usage_summary(
        rec.prompt_tokens, rec.completion_tokens, rec.token_units, rec.cost_usd
    )
    if usage:
        stats.append(f" · {usage}")
    if rec.cache_hit:
        stats.append(" · cached" + _originally(rec), style="dim")
    body.append(stats)
    if rec.input_type or rec.output_type:
        arrow = f"{rec.input_type or 'any'} → {rec.output_type or 'any'}"
        body.append(Text(f"rows validated: {arrow}", style="dim"))
    body.extend(_warning_lines(rec.warnings))

    if rec.ops and not one_plain_op:
        body.append(Text(""))
        body.append(_op_table(rec.ops, width - 2))
    elif rec.cache_hit and not rec.ops:
        body.append(
            Text("result loaded from the step cache; its ops did not run", style="dim")
        )
    for call in rec.ops:
        if call.scores is not None and not call.cache_hit:
            body.append(Text(""))
            body.append(
                _histogram_renderable(_histogram(call.scores), call.op_name, width - 2)
            )
        elif call.cache_hit and call.category == "filter":
            body.append(Text(""))
            body.append(
                Text(
                    f"{_clean(call.op_name)}: no score histogram for a cached result; "
                    "rerun with _no_cache=True to see it",
                    style="dim",
                )
            )

    samples = _samples(rec, full, width - 2)
    if samples is not None:
        body.append(Text(""))
        body.append(samples)
    return Group(_rule("step", rec.name), Padding(Group(*body), (0, 0, 0, 2)))


# --------------------------------------------------------------------------- #
# Pipeline summary
# --------------------------------------------------------------------------- #


def pipeline_renderable(rec: PipelineRecord, *, width: int = WIDTH) -> RenderableType:
    """Build the rich renderable for a pipeline summary (see ``render_pipeline``)."""
    n_steps = sum(1 for s in rec.steps if s.kind != "bare")
    first = rec.steps[0].n_in if rec.steps else None
    last = rec.steps[-1].n_out if rec.steps else None
    any_warnings = rec.warnings or any(s.warnings for s in rec.steps)
    stats = Text()
    stats.append(f"{n_steps} step{'s' if n_steps != 1 else ''} · ")
    stats.append(
        _rows_text(first, last, generated="out"),
        style="bold red" if any_warnings else "bold",
    )
    stats.append(f" · {fmt_secs(rec.duration_s)}")
    usage = _usage_summary(
        rec.prompt_tokens, rec.completion_tokens, rec.token_units, rec.cost_usd
    )
    cached_cost = rec.cached_cost_usd
    if cached_cost and math.isfinite(cached_cost):
        stats.append(f" · {usage or 'nothing'} spent this run")
        stats.append(f" · cached (originally {fmt_cost(cached_cost)})", style="dim")
    elif usage:
        stats.append(f" · {usage}")

    def original_cost(s: StepRecord) -> float:
        return s.original.cost_usd if s.cache_hit and s.original else 0.0

    show_units = any(s.token_units and math.isfinite(s.token_units) for s in rec.steps)
    show_cost = any(
        (s.cost_usd and math.isfinite(s.cost_usd)) or original_cost(s)
        for s in rec.steps
    )
    rows: list[list[Text]] = []
    for s in rec.steps:
        y = s.n_out / s.n_in if s.n_in and s.n_out is not None else None
        n_warn = len(dict.fromkeys(s.warnings))
        flags = Text()
        if n_warn:
            flags.append(f"⚠ {n_warn}", style="yellow")
        if s.cache_hit:
            flags.append(("  " if flags else "") + "cached", style="dim")
        cells = [
            Text(_clean(s.name), style="italic" if s.kind != "step" else ""),
            Text(fmt_int(s.n_in)),
            Text(fmt_int(s.n_out)),
            Text(fmt_yield(y)),
            Text("-" if s.cache_hit else fmt_secs(s.duration_s)),
        ]
        if show_units:
            cells.append(Text(fmt_units(s.token_units)))
        if show_cost and original_cost(s):
            cells.append(Text(f"({fmt_cost(original_cost(s))})", style="dim"))
        elif show_cost:
            cells.append(Text(fmt_cost(s.cost_usd)))
        cells.append(flags)
        rows.append(cells)
    headers = ["step", "in", "out", "yield", "time"]
    if show_units:
        headers.append("units")
    if show_cost:
        headers.append("cost")
    headers.append("")
    headers, rows = _drop_columns(headers, rows, width - 2, ["units", "cost"])
    table: RenderableType
    if _first_column_room(headers, rows, width - 2) < _MIN_NAME:
        table = _lines_view(headers, rows)
    else:
        table = _fit_table(headers, rows, width - 2, left={""})

    body: list[RenderableType] = [stats, Text(""), table]
    extra = rec.unattributed
    if not extra.is_zero():
        usage = _usage_summary(
            extra.prompt_tokens,
            extra.completion_tokens,
            extra.token_units,
            extra.cost_usd,
        )
        body.append(Text(""))
        body.append(
            Text(
                f"unattributed: {usage or 'no cost'}, from threads started "
                "without contextvars.copy_context().run",
                style="yellow",
            )
        )
    step_warnings = [
        line for s in rec.steps for line in _warning_lines(s.warnings, prefix=s.name)
    ]
    if step_warnings:
        body.append(Text(""))
        body.append(Text("step warnings", style="bold"))
        body.extend(step_warnings)
    if rec.warnings:
        body.append(Text(""))
        body.append(Text("pipeline warnings", style="bold"))
        body.extend(_warning_lines(rec.warnings))
    return Group(_rule("pipeline", rec.name), Padding(Group(*body), (0, 0, 0, 2)))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _plain(renderable: RenderableType) -> str:
    buf = io.StringIO()
    console = Console(
        file=buf,
        width=WIDTH,
        color_system=None,
        force_terminal=False,
        highlight=False,
        emoji=False,
        markup=False,
    )
    console.print(renderable)
    return "\n".join(line.rstrip() for line in buf.getvalue().splitlines())


def _console() -> Console:
    console = Console(highlight=False, emoji=False, markup=False)
    # Log files get the full width; narrow terminals get their own width.
    console.width = min(console.width, WIDTH) if console.is_terminal else WIDTH
    return console


def render_step(rec: StepRecord, *, full: bool = False) -> str:
    """Render a step report as plain text.

    ``full=True`` (used by ``inspect.inspect_step``) shows the whole docstring,
    every stored sample row, and up to 500 characters per field with line
    breaks kept. Longer values are summarized (see ``pyforger._summaries``).
    """
    return _plain(step_renderable(rec, full=full))


def render_pipeline(rec: PipelineRecord) -> str:
    """Render a pipeline summary as plain text."""
    return _plain(pipeline_renderable(rec))


def render_scores(scores: ScoreRecord, op_name: str = "scores") -> str:
    """Render one filter's score histogram as plain text."""
    return _plain(_histogram_renderable(_histogram(scores), op_name, WIDTH))


def _emit(renderable: RenderableType, console: Console) -> None:
    with _PRINT_LOCK:
        if console.is_terminal:
            console.print(Group(Text(""), renderable))
        else:
            # Logs: plain text without the trailing padding rich adds to each line.
            sys.stdout.write("\n" + _plain(renderable) + "\n")
            sys.stdout.flush()


def print_step(rec: StepRecord, *, full: bool = False) -> None:
    """Print a step report to stdout (in color on a terminal)."""
    console = _console()
    _emit(step_renderable(rec, full=full, width=console.width), console)


def print_pipeline(rec: PipelineRecord) -> None:
    """Print a pipeline summary to stdout (in color on a terminal)."""
    console = _console()
    _emit(pipeline_renderable(rec, width=console.width), console)

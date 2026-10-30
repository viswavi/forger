"""How rows' feature distributions compare with a reference set's."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import Any, Union

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger._render import WIDTH, _clean, _console, _emit, _fit_table, _plain, _rule
from pyforger.op import op

_OP = "compare_to_reference"
_MAX_LABELS = 20  # more distinct short strings than this are text, not labels
_LABEL_CHARS = 40  # longer strings are text
_SHOWN_LABELS = 8
# A numeric feature is flagged when this share of rows falls outside the
# reference's 5th-95th percentiles (10% when the distributions are equal), a
# label feature when the total variation distance reaches _TVD_FLAG.
_OUTSIDE_FLAG = 0.25
_TVD_FLAG = 0.2

Spec = Union[str, Callable[[dict], Any]]
Feature = Union[Spec, tuple[Spec, Spec]]


class Summary(BaseModel):
    """One set's numbers: count, mean, 10th/50th/90th percentiles, share over max."""

    n: int
    mean: float
    p10: float
    p50: float
    p90: float
    over_max: float | None = None


class FeatureComparison(BaseModel):
    """One feature in both sets.

    ``kind`` is ``"length"`` (text, measured in ``unit``), ``"number"``, or
    ``"label"``. Numeric kinds fill ``rows``, ``reference``, and
    ``outside_reference`` (the share of rows outside the reference's 5th-95th
    percentiles); labels fill ``shares`` (value -> [rows share, reference
    share]) and ``tvd`` (total variation distance, 0 to 1).
    """

    name: str
    kind: str
    unit: str | None = None
    rows: Summary | None = None
    reference: Summary | None = None
    outside_reference: float | None = None
    shares: dict[str, list[float]] = Field(default_factory=dict)
    tvd: float | None = None
    missing: int = 0
    reference_missing: int = 0
    flagged: bool = False


class ReferenceReport(BaseModel):
    """What ``compare_to_reference`` measured; ``model_dump()`` gives a dict."""

    n_rows: int
    n_reference: int
    max_length: int | None = None
    features: list[FeatureComparison] = Field(default_factory=list)


def compare_to_reference(
    rows: list[dict],
    reference: list[dict],
    *,
    features: dict[str, Feature] | list[str] | None = None,
    tokenizer: str | Any | None = None,
    max_length: int | None = None,
    **op_controls: Any,
) -> ReferenceReport:
    """Print each feature of ``rows`` beside the same feature of ``reference``.

    Text is compared by length, numbers by quantiles, labels by share; ``!``
    marks a large gap. Cheap: no model calls.

    Args:
        rows: E.g. your training rows.
        reference: E.g. the validation rows.
        features: Name -> field, ``row -> value`` function, or a ``(rows_spec,
            reference_spec)`` pair; default the fields both sets share.
        tokenizer: Count text in tokens (a Hugging Face id or anything with
            ``encode``) instead of words.
        max_length: Also report each set's share of texts longer than this.
        op_controls: Per-call ``@op`` controls (``_no_cache``, ``_cache_only``).

    Returns:
        A ``ReferenceReport``, also printed. A numeric feature is flagged when
        at least 25% of rows fall outside the reference's 5th-95th
        percentiles; a label when its shares differ by at least 0.2 in total
        (total variation distance).
    """
    if not rows or not reference:
        raise ValueError(f"{_OP} needs rows and reference rows")
    specs = _feature_specs(rows, reference, features)
    names = list(specs)
    raw = [
        [[_value(row, spec) for row in rows] for spec, _ in specs.values()],
        [[_value(row, ref) for row in reference] for _, ref in specs.values()],
    ]
    values, ref_values = _measure_all(raw, tokenizer)
    report = _compare(
        names,
        values,
        ref_values,
        unit="words" if tokenizer is None else "tokens",
        max_length=max_length,
        **op_controls,
    )
    console = _console()
    _emit(_renderable(report, width=console.width), console)
    return report


def _feature_specs(
    rows: list[dict], reference: list[dict], features: Any
) -> dict[str, tuple[Spec, Spec]]:
    if features is None:
        features = [k for k in rows[0] if k in reference[0]]
        if not features:
            raise ValueError(
                f"{_OP}: the sets share no field; pass features= with "
                "(rows_spec, reference_spec) pairs"
            )
    if isinstance(features, list):
        features = {name: name for name in features}
    return {
        name: (spec[0], spec[1]) if isinstance(spec, tuple) else (spec, spec)
        for name, spec in features.items()
    }


def _value(row: dict, spec: Spec) -> Any:
    return spec(row) if callable(spec) else row.get(spec)


def _measure_all(raw: list[list[list[Any]]], tokenizer: Any) -> list[list[list[Any]]]:
    """Numbers stay numbers, long text becomes ``("length", n)``, the rest labels.

    All texts are counted in one batch, which a fast tokenizer parallelizes.
    """
    flat = [_text(v) for side in raw for feature in side for v in feature]
    long = list(
        dict.fromkeys(t for t in flat if isinstance(t, str) and len(t) > _LABEL_CHARS)
    )
    counts = dict(zip(long, _count(long, tokenizer)))
    out, it = [], iter(flat)
    for side in raw:
        out.append(
            [
                [
                    ("length", counts[t]) if t in counts else t
                    for t in (next(it) for _ in f)
                ]
                for f in side
            ]
        )
    return out


def _text(value: Any) -> Any:
    """Chat messages as their joined contents; other values unchanged."""
    if isinstance(value, list) and all(isinstance(m, dict) for m in value):
        return "\n".join(str(m.get("content", "")) for m in value)
    return value


def _count(texts: list[str], tokenizer: Any) -> list[int]:
    if tokenizer is None:
        return [len(t.split()) for t in texts]
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if callable(tokenizer):  # a Hugging Face tokenizer: one batched call
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(i) for i in ids]
    return [len(tokenizer.encode(t)) for t in texts]


@op(name="inspect.compare_to_reference")
def _compare(
    names: list[str],
    values: list[list[Any]],
    ref_values: list[list[Any]],
    *,
    unit: str,
    max_length: int | None,
) -> ReferenceReport:
    report = ReferenceReport(
        n_rows=len(values[0]) if values else 0,
        n_reference=len(ref_values[0]) if ref_values else 0,
        max_length=max_length,
    )
    for name, vs, refs in zip(names, values, ref_values):
        report.features.append(_feature(name, vs, refs, unit, max_length))
    return report


def _feature(
    name: str, vs: list[Any], refs: list[Any], unit: str, max_length: int | None
) -> FeatureComparison:
    present = [v for v in vs if v is not None]
    ref_present = [v for v in refs if v is not None]
    out = FeatureComparison(
        name=name,
        kind="label",
        missing=len(vs) - len(present),
        reference_missing=len(refs) - len(ref_present),
    )
    both = present + ref_present
    labels = {str(v) for v in both if not _numeric(v)}
    texts = any(isinstance(v, tuple) for v in both)
    if texts or (both and all(_numeric(v) for v in both)) or len(labels) > _MAX_LABELS:
        nums = [_number(v) for v in present]
        ref_nums = [_number(v) for v in ref_present]
        texts = texts or bool(labels)
        out.kind, out.unit = ("length", unit) if texts else ("number", None)
        cap = max_length if texts else None
        out.rows = _summary(nums, cap) if nums else None
        out.reference = _summary(ref_nums, cap) if ref_nums else None
        if nums and ref_nums:
            lo, hi = _quantile(ref_nums, 0.05), _quantile(ref_nums, 0.95)
            out.outside_reference = sum(not lo <= x <= hi for x in nums) / len(nums)
            out.flagged = out.outside_reference >= _OUTSIDE_FLAG
        return out
    counts, ref_counts = Counter(map(str, present)), Counter(map(str, ref_present))
    n, m = max(1, len(present)), max(1, len(ref_present))
    keys = sorted(
        set(counts) | set(ref_counts), key=lambda k: -ref_counts[k] - counts[k]
    )
    out.shares = {k: [counts[k] / n, ref_counts[k] / m] for k in keys}
    out.tvd = 0.5 * sum(abs(a - b) for a, b in out.shares.values())
    out.flagged = out.tvd >= _TVD_FLAG
    return out


def _numeric(v: Any) -> bool:
    return isinstance(v, tuple) or (
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    )


def _number(v: Any) -> float:
    if isinstance(v, tuple):
        return float(v[1])
    if isinstance(v, str):  # a short text among long ones
        return float(len(v.split()))
    return float(v)


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[round(q * (len(s) - 1))]


def _summary(xs: list[float], max_length: int | None) -> Summary:
    return Summary(
        n=len(xs),
        mean=sum(xs) / len(xs),
        p10=_quantile(xs, 0.1),
        p50=_quantile(xs, 0.5),
        p90=_quantile(xs, 0.9),
        over_max=None
        if max_length is None
        else sum(x > max_length for x in xs) / len(xs),
    )


def render_reference_report(report: ReferenceReport) -> str:
    """Render a ``compare_to_reference`` report as plain text."""
    return _plain(_renderable(report, width=WIDTH))


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _quantiles(s: Summary | None) -> str:
    return "-" if s is None else " / ".join(_num(x) for x in (s.p10, s.p50, s.p90))


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 or x == int(x) else f"{x:.2f}"


def _renderable(report: ReferenceReport, *, width: int) -> RenderableType:
    head = Text(f"{report.n_rows:,} rows vs {report.n_reference:,} reference rows")
    body: list[RenderableType] = [head]
    numeric = [f for f in report.features if f.rows or f.reference]
    if numeric:
        headers = ["feature", "rows p10 / p50 / p90", "reference p10 / p50 / p90"]
        headers += ["outside ref 5-95%"]
        if report.max_length is not None:
            headers.append(f"over {report.max_length:,}")
        lines = []
        for f in numeric:
            name = f.name + (f" ({f.unit})" if f.unit else "")
            cells = [
                Text(("! " if f.flagged else "") + _clean(name)),
                Text(_quantiles(f.rows)),
                Text(_quantiles(f.reference)),
                Text(_pct(f.outside_reference)),
            ]
            if report.max_length is not None:
                over = [s.over_max if s else None for s in (f.rows, f.reference)]
                cells.append(Text(f"{_pct(over[0])} vs {_pct(over[1])}"))
            lines.append(cells + [Text("")])
        table = _fit_table(headers + [""], lines, width - 2, left=set())
        body += [Text(""), table]
    for f in report.features:
        if f.kind == "label" and f.tvd is not None:
            shown = list(f.shares.items())[:_SHOWN_LABELS]
            text = ", ".join(f"{_clean(k)} {a:.0%} vs {b:.0%}" for k, (a, b) in shown)
            more = len(f.shares) - len(shown)
            line = Text(
                ("! " if f.flagged else "") + f"{_clean(f.name)}: ", style="bold"
            )
            line.append(text + (f", +{more} more" if more > 0 else ""))
            line.append(f"  (distance {f.tvd:.2f})", style="dim")
            body += [Text(""), line]
    for f in report.features:
        if f.missing or f.reference_missing:
            body.append(
                Text(
                    f"{_clean(f.name)}: missing in {f.missing:,} rows and "
                    f"{f.reference_missing:,} reference rows",
                    style="dim",
                )
            )
    return Group(_rule("inspect", _OP), Padding(Group(*body), (0, 0, 0, 2)))

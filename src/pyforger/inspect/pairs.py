"""How often a rubric scores the better of two responses higher."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.text import Text

from pyforger._render import _console, _emit
from pyforger.filter.rubric import Grade, _grade_column
from pyforger.inspect.scores import (
    PASS,
    _check_max_grades,
    _criterion_text,
    _plan_grades,
    _sample_rows,
)
from pyforger.op import tracked
from pyforger.utils import get_field

Outcome = Literal["win", "tie", "loss"]


class PairResult(BaseModel):
    """One ordered pair: ``better`` should score above ``worse`` on ``row``.

    ``better_index`` and ``worse_index`` place each text in its field (0 for a
    field holding one text). Scores hold one grade per repeat, None when
    ungradable; ``margin`` is mean(better) - mean(worse), None when either
    side has an ungradable repeat.
    """

    row: int
    better_index: int
    worse_index: int
    better: str
    worse: str
    better_scores: list[float | None]
    worse_scores: list[float | None]
    margin: float | None = None
    outcome: Outcome | None = None


class CriterionPairs(BaseModel):
    """One criterion scored alone on every pair of its row.

    A requirement (weight > 0) passes a text whose mean score is at least 0.5,
    a mistake (weight < 0) one whose mean is below it. ``right`` counts pairs
    where it passes the better text and fails the worse, ``wrong`` the
    reverse, ``same`` both passing or both failing, ``ungradable`` the rest.
    """

    row: int
    index: int
    criterion: str
    weight: float
    right: int = 0
    wrong: int = 0
    same: int = 0
    ungradable: int = 0
    errors: list[str] = Field(default_factory=list)


class RubricPairReport(BaseModel):
    """What ``rubric_pair_accuracy`` measured; ``model_dump()`` gives JSON data.

    ``accuracy`` is (wins + ties / 2) / compared, over pairs graded on both
    sides; None when no pair was. ``wins``, ``ties`` and ``losses`` are kept
    apart because rubrics with yes/no criteria tie often.
    """

    n_rows: int
    epsilon: float = 0.05
    rows: list[int]
    repeats: int
    pairs: list[PairResult] = Field(default_factory=list)
    compared: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0
    ungradable: int = 0
    missing: int = 0
    accuracy: float | None = None
    mean_margin: float | None = None
    identical_response_gaps: list[float] = Field(default_factory=list)
    criteria: list[CriterionPairs] = Field(default_factory=list)
    rows_without_rubric: list[int] = Field(default_factory=list)
    zero_weight_criteria: int = 0
    errors: list[str] = Field(default_factory=list)

    def losing_pairs(self) -> list[PairResult]:
        """Pairs whose worse text scored higher, largest reversal first."""
        lost = [p for p in self.pairs if p.outcome == "loss"]
        return sorted(lost, key=lambda p: (p.margin or 0, p.row))

    def wrong_way_criteria(self) -> list[CriterionPairs]:
        """Criteria that passed a worse text over its better one, most often first.

        Use (entry.row, entry.index) to inspect or select the criterion. A
        wrong-way criterion may prefer a detail the worse text happens to
        state, the judge may misread it, or the pair may be mislabeled.
        """
        found = [c for c in self.criteria if c.wrong]
        return sorted(
            found, key=lambda c: (-(c.wrong - c.right), -abs(c.weight), c.row, c.index)
        )


def _texts(value: Any, name: str) -> list[str | None]:
    """A field's texts: one string, or a list of strings or None."""
    if value is None or isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(
        v is None or isinstance(v, str) for v in value
    ):
        return list(value)
    raise TypeError(f"{name} must hold text or a list of texts")


def _usable(text: str | None) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _mean(values: list[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return mean(float(v) for v in values if v is not None)


def _passes(score: float, weight: float) -> bool:
    return score >= PASS if weight > 0 else score < PASS


@tracked("inspect.rubric_pair_accuracy", smoke=("n",))
def rubric_pair_accuracy(
    rows: list[dict],
    grade: Grade,
    *,
    better_field: str = "better",
    worse_field: str = "worse",
    n: int | None = 50,
    repeats: int = 1,
    epsilon: float = 0.05,
    per_criterion: bool = True,
    rubric_field: str = "rubric",
    seed: int = 0,
    workers: int = 8,
    max_grades: int | None = None,
) -> RubricPairReport:
    """Report how often ``grade`` scores the better response of a pair higher.

    Each row holds the responses it ranks: ``better_field`` and
    ``worse_field`` each hold one text or a list of texts, and every
    (better, worse) combination on the row is a pair. You decide what makes
    one better, e.g. a reference against a copy with a key fact removed or a
    mistake added, or a strong model's reply against a weak model's. For
    each pair, the report records the margin mean(better) - mean(worse) and
    a win when it is above ``epsilon`` (default 0.05), a loss below
    -epsilon, else a tie, then ``accuracy`` = (wins + ties / 2) /
    compared and ``mean_margin``. ``report.losing_pairs()`` lists every pair
    whose worse text won, with its row.

    ``repeats`` grades each text that many times with fresh grades (Forger
    caches are bypassed when repeats > 1) and records the range of each
    text's grades in ``identical_response_gaps``: a pair whose margin is
    within that range may be judge noise, not the rubric's ranking.

    ``per_criterion=True`` also scores each nonzero-weight criterion of
    ``rubric_field`` alone (``grade`` gets the row with that field replaced
    by ``[[criterion, 1]]``, so it must read the row's rubric; pass False for
    a grade that does not). A criterion is right on a pair when it passes the
    better text and fails the worse, wrong for the reverse (a mistake, weight
    < 0, passes a text that does not make it). ``report.wrong_way_criteria()``
    lists the criteria wrong on any pair, with row and index. Nothing is
    removed or changed.

    Grade calls: ``repeats`` per distinct text, plus as many per criterion
    with ``per_criterion``. ``n`` samples rows with ``seed`` (default 50;
    None for all). The planned number of grade calls is printed before any
    call, and ``max_grades`` raises instead when the plan is above it.

    Returns:
        A serializable report with every pair, its scores and outcome, and
        per-criterion counts.
    """
    for name, value in (("repeats", repeats), ("workers", workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ValueError("n must be a positive integer or None")
    _check_max_grades(max_grades)
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not (math.isfinite(epsilon) and epsilon >= 0)
    ):
        raise ValueError("epsilon must be finite and nonnegative")
    if better_field == worse_field:
        raise ValueError("better_field and worse_field must differ")
    picked = _sample_rows(len(rows), n, seed)
    sample = [rows[i] for i in picked]
    report = RubricPairReport(
        n_rows=len(rows), rows=picked, repeats=repeats, epsilon=epsilon
    )
    # Per row: distinct texts to grade and the (better, worse) pairs among them.
    texts: list[list[str]] = []
    pairs: list[list[tuple[int, int, int, int]]] = []  # (bi, wi, b slot, w slot)
    for row in sample:
        better = _texts(get_field(row, better_field, "rubric_pair_accuracy"), "better")
        worse = _texts(get_field(row, worse_field, "rubric_pair_accuracy"), "worse")
        own: list[str] = []
        slot: dict[str, int] = {}

        def place(text: str) -> int:
            if text not in slot:
                slot[text] = len(own)
                own.append(text)
            return slot[text]

        row_pairs = []
        for bi, b in enumerate(better):
            for wi, w in enumerate(worse):
                if not (_usable(b) and _usable(w)):
                    report.missing += 1
                    continue
                row_pairs.append((bi, wi, place(str(b)), place(str(w))))
        texts.append(own)
        pairs.append(row_pairs)
    criterion_jobs: list[tuple[int, int, Any, float]] = []
    if per_criterion:
        for i, row in enumerate(sample):
            if not pairs[i]:
                continue
            rubric = row.get(rubric_field) if isinstance(row, dict) else None
            if not isinstance(rubric, (list, tuple)):
                report.rows_without_rubric.append(picked[i])
                continue
            for ci, entry in enumerate(rubric):
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(
                        f"row {picked[i]} criterion {ci}: expected [criterion, weight]"
                    )
                criterion, weight = entry
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, (int, float))
                    or not math.isfinite(weight)
                ):
                    raise ValueError(
                        f"row {picked[i]} criterion {ci}: weight must be finite"
                    )
                if weight == 0:
                    report.zero_weight_criteria += 1
                    continue
                criterion_jobs.append((i, ci, criterion, float(weight)))
    planned = (
        sum(len(own) for own in texts)
        + sum(len(texts[i]) for i, _, _, _ in criterion_jobs)
    ) * repeats
    _plan_grades(
        "inspect.rubric_pair_accuracy",
        planned,
        f"{len(sample)} rows, {sum(len(own) for own in texts)} distinct texts and "
        f"{len(criterion_jobs)} criteria scored alone, x {repeats} repeats",
        max_grades,
    )
    fresh = repeats > 1
    expanded = [[t for t in own for _ in range(repeats)] for own in texts]
    scores, errors = _grade_column(
        sample, expanded, grade=grade, workers=workers, _no_cache=fresh
    )

    def grouped(values: Sequence[float | None]) -> list[list[float | None]]:
        return [list(values[j : j + repeats]) for j in range(0, len(values), repeats)]

    margins = []
    for i, row_pairs in enumerate(pairs):
        groups = grouped(scores[i])
        if repeats > 1:
            for group in groups:
                if group and None not in group:
                    values = [float(v) for v in group if v is not None]
                    report.identical_response_gaps.append(max(values) - min(values))
        for bi, wi, bs, ws in row_pairs:
            result = PairResult(
                row=picked[i],
                better_index=bi,
                worse_index=wi,
                better=texts[i][bs],
                worse=texts[i][ws],
                better_scores=groups[bs],
                worse_scores=groups[ws],
            )
            hi, lo = _mean(groups[bs]), _mean(groups[ws])
            if hi is None or lo is None:
                report.ungradable += 1
            else:
                result.margin = hi - lo
                result.outcome = (
                    "win"
                    if hi - lo > epsilon
                    else "loss"
                    if lo - hi > epsilon
                    else "tie"
                )
                margins.append(result.margin)
            report.pairs.append(result)
        report.errors.extend(e for e in errors[i] if e is not None)
    report.wins = sum(p.outcome == "win" for p in report.pairs)
    report.ties = sum(p.outcome == "tie" for p in report.pairs)
    report.losses = sum(p.outcome == "loss" for p in report.pairs)
    report.compared = len(margins)
    if margins:
        report.accuracy = (report.wins + report.ties / 2) / report.compared
        report.mean_margin = mean(margins)
    if criterion_jobs:
        isolated = [
            {**sample[i], rubric_field: [[criterion, 1]]}
            for i, _, criterion, _ in criterion_jobs
        ]
        c_scores, c_errors = _grade_column(
            isolated,
            [expanded[i] for i, _, _, _ in criterion_jobs],
            grade=grade,
            workers=workers,
            _no_cache=fresh,
        )
        for (i, ci, criterion, weight), values, messages in zip(
            criterion_jobs, c_scores, c_errors
        ):
            entry = CriterionPairs(
                row=picked[i],
                index=ci,
                criterion=_criterion_text(criterion),
                weight=weight,
                errors=[m for m in messages if m is not None],
            )
            means = [_mean(g) for g in grouped(values)]
            for _, _, bs, ws in pairs[i]:
                hi, lo = means[bs], means[ws]
                if hi is None or lo is None:
                    entry.ungradable += 1
                    continue
                good, bad = _passes(hi, weight), _passes(lo, weight)
                if good and not bad:
                    entry.right += 1
                elif bad and not good:
                    entry.wrong += 1
                else:
                    entry.same += 1
            report.criteria.append(entry)
    _render(report)
    return report


def _render(report: RubricPairReport) -> None:
    lines = [
        f"Rubric pair accuracy: {len(report.rows)} of {report.n_rows} rows; "
        f"{report.compared} pairs compared, {report.ungradable} ungradable, "
        f"{report.missing} missing a text."
    ]
    if report.accuracy is not None and report.mean_margin is not None:
        lines.append(
            f"Better text scored higher by more than {report.epsilon:g} in "
            f"{report.wins}, within it in {report.ties}, "
            f"lower in {report.losses} (accuracy {report.accuracy:.1%}, ties "
            f"count half; mean margin {report.mean_margin:+.3f})."
        )
    gaps = report.identical_response_gaps
    if gaps:
        lines.append(
            f"Identical-text repeat variation: mean range {mean(gaps):.3g}, "
            f"max {max(gaps):.3g} over {len(gaps)} texts."
        )
    if report.criteria:
        wrong = report.wrong_way_criteria()
        lines.append(
            f"Criteria scored alone: {len(report.criteria)}; {len(wrong)} passed a "
            "worse text over its better one on some pair."
        )
        for c in wrong[:3]:
            lines.append(
                f"  row {c.row}, index {c.index}, weight {c.weight:g}: right "
                f"{c.right}, wrong {c.wrong}: {c.criterion[:200]}"
            )
    for p in report.losing_pairs()[:3]:
        lines.append(
            f"  losing pair row {p.row} ({p.better_index} vs {p.worse_index}): "
            f"margin {p.margin:+.3f}"
        )
    _emit(Text("\n".join(lines)), _console())

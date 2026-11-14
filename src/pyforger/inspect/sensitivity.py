"""Measure reward changes under candidate meaning-preserving rewrites."""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field
from rich.text import Text

from pyforger._render import _console, _emit
from pyforger.alter.paraphrase import paraphrase
from pyforger.filter.rubric import Grade, _grade_column
from pyforger.inspect.scores import _check_max_grades, _plan_grades, _sample_rows
from pyforger.op import tracked
from pyforger.utils import get_field


class SensitivityPair(BaseModel):
    """Original and candidate scores; row indexes the caller's input rows.

    equivalent is an optional independent check, not the paraphraser's claim.
    False excludes the pair from the comparison; None leaves it unverified.
    """

    row: int
    original: str
    candidate: str
    original_scores: list[float | None]
    candidate_scores: list[float | None]
    equivalent: bool | None = None
    gap: float | None = None
    changed: bool | None = None


class ChangeRate(BaseModel):
    """A changed/compared count; fraction is None when nothing was comparable."""

    changed: int = 0
    compared: int = 0
    fraction: float | None = None


class CriterionSensitivity(BaseModel):
    """A criterion occurrence, identified by original input row and rubric index.

    Pairs retain full texts and raw repeat scores. The original weight is
    metadata: isolated scoring uses weight +1, so negative weights cannot
    clip every isolated score to zero or invert the measured differences.
    """

    row: int
    index: int
    criterion: Any
    weight: float
    pairs: list[SensitivityPair] = Field(default_factory=list)
    pair_changes: ChangeRate = Field(default_factory=ChangeRate)
    verified_pair_changes: ChangeRate = Field(default_factory=ChangeRate)
    identical_response_gaps: list[float] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RubricSensitivityReport(BaseModel):
    """Candidate sensitivity is a diagnostic, not proof of equal utility.

    rubric_pair_changes counts original/paraphrase pairs; rubric_changes
    counts each sampled input row once if any eligible pair changed;
    criterion_changes counts each (row, rubric index) once if any pair changed.
    Denominators include only units with at least one complete, non-rejected
    comparison. verified_* rates require an accepted equivalence review.
    The original tie-rate fields are retained for compatibility.

    candidate_tie_rate includes unverified pairs; verified_tie_rate includes
    only pairs accepted by equivalent. Both use absolute differences between
    repeated-score means at epsilon. identical_response_gaps measures the
    range across fresh grades of each identical text, a separate noise check.
    Missing grades are never filled with zero. Only complete repeat groups
    enter comparisons. External grader caches must also be disabled by the
    caller; the inspection bypasses Forger caches when grading.
    """

    rubric_pair_changes: ChangeRate = Field(default_factory=ChangeRate)
    rubric_changes: ChangeRate = Field(default_factory=ChangeRate)
    criterion_changes: ChangeRate = Field(default_factory=ChangeRate)
    verified_rubric_pair_changes: ChangeRate = Field(default_factory=ChangeRate)
    verified_rubric_changes: ChangeRate = Field(default_factory=ChangeRate)
    verified_criterion_changes: ChangeRate = Field(default_factory=ChangeRate)
    criteria: list[CriterionSensitivity] = Field(default_factory=list)
    per_criterion: bool = True
    criterion_epsilon: float = 0.05
    rows_without_rubric: list[int] = Field(default_factory=list)
    zero_weight_criteria: int = 0

    n_rows: int
    rows: list[int]
    epsilon: float
    repeats: int
    pairs: list[SensitivityPair] = Field(default_factory=list)
    unchanged: int = 0
    duplicates: int = 0
    missing: int = 0
    ungradable_pairs: int = 0
    rejected: int = 0
    compared: int = 0
    changed: int = 0
    verified: int = 0
    candidate_tie_rate: float | None = None
    verified_tie_rate: float | None = None
    mean_abs_gap: float | None = None
    max_abs_gap: float | None = None
    identical_response_gaps: list[float] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def sensitive_criteria(
        self, *, verified_only: bool = False
    ) -> list[CriterionSensitivity]:
        """Return criteria with at least one eligible changed pair, largest gap first.

        Use (entry.row, entry.index) to inspect or select original criteria.
        This does not mutate data or automatically remove criteria; a changed
        score is a diagnostic candidate, not proof that a criterion is wrong.
        """
        found = [
            c
            for c in self.criteria
            if (c.verified_pair_changes if verified_only else c.pair_changes).changed
        ]
        return sorted(
            found,
            key=lambda c: max(
                (abs(p.gap or 0) for p in _eligible(c.pairs, verified_only)), default=0
            ),
            reverse=True,
        )


def _eligible(
    pairs: list[SensitivityPair], verified_only: bool = False
) -> list[SensitivityPair]:
    """Keep complete comparisons whose equivalence review did not reject them."""
    return [
        p
        for p in pairs
        if p.gap is not None
        and (p.equivalent is True if verified_only else p.equivalent is not False)
    ]


def _rate(flags: list[bool]) -> ChangeRate:
    """Count unique units supplied by the caller, never infer missing stability."""
    return ChangeRate(
        changed=sum(flags),
        compared=len(flags),
        fraction=sum(flags) / len(flags) if flags else None,
    )


def _pair_and_row_rates(
    pairs: list[SensitivityPair], verified_only: bool = False
) -> tuple[ChangeRate, ChangeRate]:
    """Return pair and unique-row rates from the same eligible comparisons."""
    eligible = _eligible(pairs, verified_only)
    by_row: dict[int, bool] = {}
    for pair in eligible:
        by_row[pair.row] = by_row.get(pair.row, False) or pair.changed is True
    return _rate([p.changed is True for p in eligible]), _rate(list(by_row.values()))


def _complete(values: list[float | None]) -> list[float] | None:
    """Return a complete group, or None if any fresh grading was unavailable."""
    if not values or any(v is None for v in values):
        return None
    return [float(v) for v in values if v is not None]


@tracked("inspect.rubric_sensitivity", smoke=("n",))
def rubric_sensitivity(
    rows: list[dict],
    grade: Grade,
    *,
    model: Any = None,
    variants_field: str | None = None,
    response_field: str = "response",
    prompt_field: str | None = "prompt",
    n: int | None = 50,
    k: int = 2,
    repeats: int = 2,
    epsilon: float = 0.05,
    per_criterion: bool = True,
    rubric_field: str = "rubric",
    criterion_epsilon: float | None = None,
    equivalent: Callable[[str, str, dict], bool | None] | None = None,
    seed: int = 0,
    workers: int = 8,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_grades: int | None = None,
) -> RubricSensitivityReport:
    """Inspect reward stability across candidate paraphrases.

    Supply model to generate paraphrases, or variants_field holding existing
    lists of strings/None. Report the share whose mean reward changes by more
    than epsilon, plus the noise across repeated fresh grades of identical
    text. Three rates have distinct denominators: rubric/paraphrase pairs,
    input rows with any changed pair, and criterion occurrences with any
    changed pair. This probes specificity, not discrimination: a constant reward
    passes this check. Use rubric_scores controls and teacher checks too.

    A paraphrase may change quality. equivalent(original, candidate, row)
    can independently accept/reject it; None is unverified. Rejected pairs
    are retained for review but excluded from summary metrics. Without this
    check, results describe candidate sensitivity, not verified specificity.

    Grading bypasses Forger caches, including nested @op calls, so repetitions
    are fresh. A callable with its own external cache must bypass that too.
    Paraphrase generation is cached separately. At most n * (k + 1) * repeats
    grade calls plus n * k generations; each grade may call many rubric judges.
    With variants_field, the number of supplied variants replaces k. The
    planned number of grade calls is printed before any model call, and
    max_grades raises instead when the plan is above it.

    per_criterion=True additionally grades each nonzero-weight criterion
    from rubric_field separately: grade receives the original row with that
    field replaced by [[criterion, 1.0]]. The grade function MUST honor this
    substituted rubric; set per_criterion=False for an opaque reward that
    ignores it. Negative weights are preserved as metadata, not used in
    isolated scoring. criterion_epsilon defaults to epsilon. Missing rubrics
    and zero weights are counted separately. This adds (k + 1) * repeats
    grade calls per nonzero criterion, with the same cache bypass. Full-rubric
    and isolated grades are independent draws, so noise can make them differ.

    Samples keep original row indices; no rows are removed or changed.
    report.sensitive_criteria() returns ranked criterion entries with row,
    index, exact criterion, weight, paired texts, raw scores and repeat noise.
    Use verified_only=True to select only independently accepted comparisons.

    Returns:
        A serializable report with paired texts, scores, omissions, errors,
        and separate candidate/verified tie rates at the chosen tolerance.
    """
    if (model is None) == (variants_field is None):
        raise ValueError("supply exactly one of model or variants_field")
    for name, value in (("repeats", repeats), ("workers", workers), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ValueError("n must be a positive integer or None")
    _check_max_grades(max_grades)
    criterion_epsilon = epsilon if criterion_epsilon is None else criterion_epsilon
    if not math.isfinite(criterion_epsilon) or criterion_epsilon < 0:
        raise ValueError("criterion_epsilon must be finite and nonnegative")
    picked = _sample_rows(len(rows), n, seed)
    sample = [rows[i] for i in picked]
    report = RubricSensitivityReport(
        n_rows=len(rows),
        rows=picked,
        epsilon=epsilon,
        repeats=repeats,
        per_criterion=per_criterion,
        criterion_epsilon=criterion_epsilon,
    )
    # Validate the criterion schema before any model calls. A missing rubric
    # is allowed for generic grade functions, but never counted as stable.
    criterion_jobs = []
    if per_criterion:
        for i, row in enumerate(sample):
            rubric = row.get(rubric_field)
            if rubric is None:
                report.rows_without_rubric.append(picked[i])
                continue
            if not isinstance(rubric, (list, tuple)):
                raise TypeError(f"row {picked[i]}: rubric must be a sequence of pairs")
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
    # The original plus each paraphrase (k, or the variants supplied).
    texts_per_row = []
    for row in sample:
        if model is not None:
            texts_per_row.append(1 + k)
        else:
            supplied = get_field(row, str(variants_field), "rubric_sensitivity")
            texts_per_row.append(
                1 + (len(supplied) if isinstance(supplied, list) else 0)
            )
    per_criterion_texts = sum(texts_per_row[i] for i, _, _, _ in criterion_jobs)
    _plan_grades(
        "inspect.rubric_sensitivity",
        (sum(texts_per_row) + per_criterion_texts) * repeats,
        f"{len(sample)} rows, {sum(texts_per_row)} texts and "
        f"{len(criterion_jobs)} criteria scored alone, x {repeats} repeats",
        max_grades,
    )
    if model is not None:
        # Avoid writing even a temporary output field into the caller's rows.
        inputs = []
        for row in sample:
            item = {"response": get_field(row, response_field, "rubric_sensitivity")}
            if prompt_field is not None:
                item["prompt"] = get_field(row, prompt_field, "rubric_sensitivity")
            inputs.append(item)
        generated = paraphrase(
            inputs,
            model=model,
            prompt_field="prompt" if prompt_field else None,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        variants = [row["paraphrases"] for row in generated]
    else:
        assert variants_field is not None  # exactly one source was required above
        variants = [
            get_field(row, variants_field, "rubric_sensitivity") for row in sample
        ]
    texts: list[list[str | None]] = []
    candidates: list[list[str]] = []
    for row, choices in zip(sample, variants):
        original = get_field(row, response_field, "rubric_sensitivity")
        if not isinstance(original, str):
            raise TypeError("response_field must contain text")
        if not isinstance(choices, list):
            raise TypeError("variants_field must contain lists")
        seen = {original}
        unique = []
        for choice in choices:
            if choice is None or (isinstance(choice, str) and not choice.strip()):
                report.missing += 1
                continue
            if not isinstance(choice, str):
                raise TypeError("variants must be text or None")
            if choice == original:
                report.unchanged += 1
            elif choice in seen:
                report.duplicates += 1
            else:
                unique.append(choice)
                seen.add(choice)
        candidates.append(unique)
        texts.append([text for text in [original, *unique] for _ in range(repeats)])
    # _no_cache propagates into nested ops and worker contexts. Reusing cached
    # grades of identical strings would manufacture a zero-noise baseline.
    scores, errors = _grade_column(
        sample, texts, grade=grade, workers=workers, _no_cache=True
    )
    for i, (row, unique, values) in enumerate(zip(sample, candidates, scores)):
        groups = [values[j : j + repeats] for j in range(0, len(values), repeats)]
        for group in groups:
            complete = _complete(group)
            if complete is not None and repeats > 1:
                report.identical_response_gaps.append(max(complete) - min(complete))
        original = str(texts[i][0])
        for candidate, group in zip(unique, groups[1:]):
            accepted = equivalent(original, candidate, row) if equivalent else None
            if accepted is not None and not isinstance(accepted, bool):
                raise TypeError("equivalent must return bool or None")
            pair = SensitivityPair(
                row=picked[i],
                original=original,
                candidate=candidate,
                original_scores=groups[0],
                candidate_scores=group,
                equivalent=accepted,
            )
            a, b = _complete(groups[0]), _complete(group)
            if a is not None and b is not None:
                pair.gap = mean(b) - mean(a)
                pair.changed = abs(pair.gap) > epsilon
            report.pairs.append(pair)
        report.errors.extend(error for error in errors[i] if error is not None)
    report.rejected = sum(p.equivalent is False for p in report.pairs)
    report.ungradable_pairs = sum(p.gap is None for p in report.pairs)
    eligible = [
        p for p in report.pairs if p.equivalent is not False and p.gap is not None
    ]
    verified = [p for p in eligible if p.equivalent is True]
    report.compared = len(eligible)
    report.changed = sum(p.changed is True for p in eligible)
    report.verified = len(verified)
    if eligible:
        report.candidate_tie_rate = 1 - report.changed / len(eligible)
        gaps = [abs(p.gap) for p in eligible if p.gap is not None]
        report.mean_abs_gap, report.max_abs_gap = mean(gaps), max(gaps)
    if verified:
        report.verified_tie_rate = 1 - sum(p.changed is True for p in verified) / len(
            verified
        )
    report.rubric_pair_changes, report.rubric_changes = _pair_and_row_rates(
        report.pairs
    )
    (
        report.verified_rubric_pair_changes,
        report.verified_rubric_changes,
    ) = _pair_and_row_rates(report.pairs, True)
    if criterion_jobs:
        isolated = [
            {**sample[i], rubric_field: [[criterion, 1.0]]}
            for i, _, criterion, _ in criterion_jobs
        ]
        criterion_scores, criterion_errors = _grade_column(
            isolated,
            [texts[i] for i, _, _, _ in criterion_jobs],
            grade=grade,
            workers=workers,
            _no_cache=True,
        )
        # Reuse each response pair's equivalence verdict; never ask a separate
        # equivalence judge for each rubric criterion.
        reviews = {(p.row, p.candidate): p.equivalent for p in report.pairs}
        for job, values, messages in zip(
            criterion_jobs, criterion_scores, criterion_errors
        ):
            i, ci, criterion, weight = job
            entry = CriterionSensitivity(
                row=picked[i], index=ci, criterion=criterion, weight=weight
            )
            groups = [values[j : j + repeats] for j in range(0, len(values), repeats)]
            for group in groups:
                complete = _complete(group)
                if complete is not None and repeats > 1:
                    entry.identical_response_gaps.append(max(complete) - min(complete))
            for candidate, group in zip(candidates[i], groups[1:]):
                pair = SensitivityPair(
                    row=picked[i],
                    original=str(texts[i][0]),
                    candidate=candidate,
                    original_scores=groups[0],
                    candidate_scores=group,
                    equivalent=reviews[(picked[i], candidate)],
                )
                a, b = _complete(groups[0]), _complete(group)
                if a is not None and b is not None:
                    pair.gap = mean(b) - mean(a)
                    pair.changed = abs(pair.gap) > criterion_epsilon
                entry.pairs.append(pair)
            entry.errors = [m for m in messages if m is not None]
            entry.pair_changes = _pair_and_row_rates(entry.pairs)[0]
            entry.verified_pair_changes = _pair_and_row_rates(entry.pairs, True)[0]
            report.criteria.append(entry)
        report.criterion_changes = _rate(
            [
                c.pair_changes.changed > 0
                for c in report.criteria
                if c.pair_changes.compared
            ]
        )
        report.verified_criterion_changes = _rate(
            [
                c.verified_pair_changes.changed > 0
                for c in report.criteria
                if c.verified_pair_changes.compared
            ]
        )
    noise = report.identical_response_gaps
    noise_text = (
        f"mean range {mean(noise):.3g}, max {max(noise):.3g}, "
        f"{sum(gap > epsilon for gap in noise)}/{len(noise)} exceed tolerance"
        if noise
        else "not measured (need two complete grades per text)"
    )
    _emit(
        Text(
            f"Rubric sensitivity: {len(picked)} of {len(rows)} rows sampled; "
            f"{report.changed}/{report.compared} candidate pairs changed by more "
            f"than {epsilon:g}; {report.verified} pairs independently accepted.\n"
            f"Identical-text repeat variation: {noise_text}.\n"
            f"unchanged rewrites: {report.unchanged}; duplicates: {report.duplicates}; "
            f"missing: {report.missing}; rejected: {report.rejected}; "
            f"ungradable pairs: {report.ungradable_pairs}.\n"
            "Review paired texts: semantic equivalence is not guaranteed. "
            "No comparisons means no evidence of stability."
        ),
        _console(),
    )
    for label, rate in (
        ("Rubric–paraphrase pairs", report.rubric_pair_changes),
        ("Rubrics with any changed paraphrase", report.rubric_changes),
        ("Criteria with any changed paraphrase", report.criterion_changes),
    ):
        percentage = (
            f"{100 * rate.fraction:.1f}%"
            if rate.fraction is not None
            else "not measured"
        )
        _emit(
            Text(f"{label}: {rate.changed}/{rate.compared} ({percentage})"), _console()
        )
    _emit(
        Text(
            f"Criterion coverage: {len(report.criteria)} nonzero-weight criteria; "
            f"{len(report.rows_without_rubric)} rows without rubric; "
            f"{report.zero_weight_criteria} zero-weight criteria skipped."
        ),
        _console(),
    )
    for criterion in report.sensitive_criteria()[:3]:
        _emit(
            Text(
                f"Sensitive criterion: row {criterion.row}, index {criterion.index}, "
                f"weight {criterion.weight:g}: {str(criterion.criterion)[:240]}"
            ),
            _console(),
        )
    largest = sorted(eligible, key=lambda p: abs(p.gap or 0), reverse=True)[:3]
    for pair in largest:
        # Keep full text in the returned report; terminal examples stay short.
        _emit(
            Text(
                f"row {pair.row}: gap {pair.gap:+.3f}; "
                f"equivalence {'accepted' if pair.equivalent else 'unverified'}\n"
                f"  original: {pair.original[:240]!r}\n"
                f"  rewrite:  {pair.candidate[:240]!r}"
            ),
            _console(),
        )
    return report

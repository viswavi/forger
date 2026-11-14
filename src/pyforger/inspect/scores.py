"""How a grade function scores responses, beside responses it should not reward."""

from __future__ import annotations

import math
import random
import string
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.text import Text

from pyforger._render import WIDTH, _clean, _console, _emit, _fit_table, _plain, _rule
from pyforger.filter.model_difficulty import _field_prompt, _samples
from pyforger.filter.rubric import Grade, _grade_column
from pyforger.op import tracked
from pyforger.utils import get_formatted_logger

Probe = Callable[[dict], "str | None"]

BINS = ["0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 0.75]", "(0.75, 1)", "1"]
CONTROLS = ("empty", "random", "other_row")
# A criterion scored alone passes at this score or above; a mistake (negative
# weight) passes below it, e.g. "recommends aspirin" scored 0.0 is avoided.
PASS = 0.5
TEACHER_FLAG = (
    0.2  # flag when the teacher fails this share of reference-passed criteria
)
EASY_FLAG = 0.9  # flag a sampled column whose mean is at least this
_EPS = 1e-9  # 0.9999999999 from a weighted mean is 1
_RANDOM_CHARS = 200
_ALPHABET = string.ascii_letters + string.digits + string.punctuation + " \n"


class Example(BaseModel):
    """One graded response: its row (index into the rows given) and score."""

    row: int
    score: float
    response: str


class Paired(BaseModel):
    """One column against the reference column, row by row.

    ``rows`` counts rows both graded (a sampled column's row score is its
    samples' mean). ``below``, ``tied`` and ``above`` count rows where the
    column scored lower than, the same as, or higher than ``reference``;
    ``mean_gap`` is the mean of reference minus column. A rubric that
    separates strong from weak responses puts a weak column mostly below.
    """

    reference: str
    rows: int = 0
    below: int = 0
    tied: int = 0
    above: int = 0
    mean_gap: float | None = None


class ColumnScores(BaseModel):
    """The scores of one column of responses.

    ``kind`` is ``"field"`` (a row field, e.g. the SFT target), ``"sampled"``
    (the model's ``k`` samples per prompt), ``"probe"``, or ``"control"``.
    ``scores[i]`` holds the scores of the responses to the report's
    ``rows[i]`` (one, or ``k`` for sampled), None when ungradable.
    ``counts`` is aligned with ``BINS``; shares are of the gradable
    responses. Scores outside [0, 1] are counted in ``out_of_range``, not in
    a bin, and are in ``mean``.
    """

    name: str
    kind: str
    scores: list[list[float | None]] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=lambda: [0] * len(BINS))
    responses: int = 0
    gradable: int = 0
    ungradable: int = 0
    errors: int = 0
    error: str | None = None
    out_of_range: int = 0
    mean: float | None = None
    share_at_0: float | None = None
    share_at_1: float | None = None
    lowest: list[Example] = Field(default_factory=list)
    highest: list[Example] = Field(default_factory=list)
    # Sampled columns: prompts with 2+ gradable samples, and how many of them
    # gave every sample the same score (no RL gradient), all 0, or all 1.
    prompts_compared: int = 0
    constant_prompts: int = 0
    constant_at_0: int = 0
    constant_at_1: int = 0

    # Field, probe, and sampled columns other than the reference: how each
    # row's score compares with the reference's on the same row.
    vs_reference: Paired | None = None

    def histogram(self) -> dict[str, int]:
        """``{bin label: count}``, e.g. ``{"0": 3, ..., "1": 47}``."""
        return dict(zip(BINS, self.counts))


class CriterionCheck(BaseModel):
    """One criterion scored alone: on the reference and on each teacher sample.

    ``row`` indexes the rows given; ``index`` is the criterion's place in the
    row's rubric. ``teacher`` holds one score per sample, None when ungradable.
    """

    row: int
    index: int
    criterion: str
    weight: float
    reference: float
    teacher: list[float | None]


class TeacherCriteria(BaseModel):
    """Which rubric criteria the reference passes and fresh teacher replies fail.

    Each criterion with a nonzero weight is scored alone (the row's rubric
    replaced by ``[[criterion, 1]]``) on the reference and on every teacher
    sample. A positive-weight criterion passes at ``PASS`` (0.5) or above; a
    negative-weight one names a mistake and passes below it, e.g. "Recommends
    aspirin for a child" [-75] passes when the reply scores 0 on it. Of the
    ``reference_passes`` criteria, ``fails_all`` fail every gradable teacher
    sample and ``fails_some`` fail some but not all. ``skipped`` counts
    criteria with a weight of 0, which change no score. ``failed`` lists every
    criterion that fails every sample, heaviest (largest absolute weight)
    first, with its row and index, e.g. to drop or rewrite them;
    ``examples`` is its first five, as printed.
    """

    reference: str
    rows: int = 0
    criteria: int = 0
    skipped: int = 0
    reference_passes: int = 0
    fails_all: int = 0
    fails_some: int = 0
    share_fails_all: float | None = None
    examples: list[CriterionCheck] = Field(default_factory=list)
    failed: list[CriterionCheck] = Field(default_factory=list)


class RubricScoreReport(BaseModel):
    """What ``rubric_scores`` measured; ``model_dump()`` gives plain JSON data.

    Only a sample is scored: ``rows`` are the indices of the ``len(rows)``
    sampled rows of the ``n_rows`` given, in order, and a column's
    ``scores[i]`` belongs to ``rows[i]``. A row not in ``rows`` has no score,
    e.g. 50 of 300 rows with the default ``n=50``; ``filter.by_rubric_score``
    scores every row. ``flags`` has one line per control or probe that earns
    at least half the reference column's mean (the first field column, else
    the sampled one), one when the teacher fails ``TEACHER_FLAG`` of the
    criteria the reference passes, and one when the sampled column's mean is
    ``EASY_FLAG`` or more.
    """

    grade: str
    n_rows: int
    rows: list[int]
    seed: int
    bins: list[str] = Field(default_factory=lambda: list(BINS))
    model_id: str | None = None
    k: int = 0
    teacher_id: str | None = None
    teacher_k: int = 0
    columns: list[ColumnScores] = Field(default_factory=list)
    teacher_criteria: TeacherCriteria | None = None
    flags: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnScores:
        """Look up a column by its name, e.g. ``"empty"``."""
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"no column {name!r}; columns: {[c.name for c in self.columns]}")


def _bin(score: float) -> int | None:
    """Index into BINS, or None outside [0, 1]."""
    if abs(score) <= _EPS:
        return 0
    if abs(score - 1) <= _EPS:
        return len(BINS) - 1
    if score < 0 or score > 1:
        return None
    return min(4, max(1, math.ceil(score * 4)))


def _sample_rows(n_rows: int, n: int | None, seed: int) -> list[int]:
    """``n`` row indices drawn with ``seed``, in order; all rows when n is None."""
    if n is None or n >= n_rows:
        return list(range(n_rows))
    return sorted(random.Random(seed).sample(range(n_rows), n))


def _derangement(n: int, seed: int) -> list[int]:
    """``other[i] != i`` for every i: one cycle through a shuffled order."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    other = [0] * n
    for a, b in zip(order, order[1:] + order[:1]):
        other[a] = b
    return other


def _random_text(seed: int, i: int) -> str:
    rng = random.Random(f"{seed}:{i}")
    return "".join(rng.choice(_ALPHABET) for _ in range(_RANDOM_CHARS))


def _summarize(
    col: ColumnScores,
    responses: list[list[str | None]],
    errors: list[list[str | None]],
    rows: list[int],
    examples: int,
) -> None:
    flat = [
        (rows[i], s, responses[i][j])
        for i, scores in enumerate(col.scores)
        for j, s in enumerate(scores)
    ]
    col.responses = len(flat)
    graded = [(r, s, text) for r, s, text in flat if s is not None]
    col.gradable = len(graded)
    col.ungradable = col.responses - col.gradable
    messages = [e for row in errors for e in row if e is not None]
    col.errors = len(messages)
    col.error = messages[0] if messages else None
    for _, s, _ in graded:
        b = _bin(s)
        if b is None:
            col.out_of_range += 1
        else:
            col.counts[b] += 1
    if graded:
        col.mean = sum(s for _, s, _ in graded) / len(graded)
        col.share_at_0 = col.counts[0] / len(graded)
        col.share_at_1 = col.counts[-1] / len(graded)
        ranked = sorted(graded, key=lambda g: (g[1], g[0]))
        pick = lambda g: Example(  # noqa: E731
            row=g[0], score=g[1], response=(g[2] or "")[:500]
        )
        col.lowest = [pick(g) for g in ranked[: max(0, examples)]]
        if ranked[-1][1] > ranked[0][1]:
            col.highest = [pick(g) for g in ranked[::-1][: max(0, examples)]]
    if col.kind == "sampled":
        for scores in col.scores:
            good = [s for s in scores if s is not None]
            if len(good) < 2:
                continue
            col.prompts_compared += 1
            if max(good) - min(good) <= _EPS:
                col.constant_prompts += 1
                col.constant_at_0 += _bin(good[0]) == 0
                col.constant_at_1 += _bin(good[0]) == len(BINS) - 1


@tracked("inspect.rubric_scores")
def rubric_scores(
    rows: list[dict],
    grade: Grade,
    *,
    n: int | None = 50,
    response_fields: list[str] | str | None = "response",
    reference_field: str | None = None,
    probes: dict[str, Probe] | None = None,
    model: Any = None,
    k: int = 4,
    teacher: Any = None,
    teacher_k: int = 2,
    rubric_field: str = "rubric",
    prompt_field: str = "prompt",
    temperature: float | None = None,
    max_tokens: int | None = None,
    control_columns: bool = True,
    seed: int = 0,
    workers: int = 8,
    examples: int = 1,
) -> RubricScoreReport:
    """Print and return how ``grade`` scores a sample of ``n`` rows, binned, by column.

    Before RL, look at the reward's distribution: on what the policy will
    sample, on the targets, and on responses it should not reward. ``grade(
    response, row)`` returns a score in [0, 1] (a bool counts as 0 or 1), or
    None when the response cannot be graded; a code rubric, an LLM judge, or
    exact match all fit. Callables are part of the cache key (an object with
    ``__call__`` by its ``cache_key()`` if it has one), so the grades are
    cached; a grading that raised is counted with its message and not cached.

    It scores a sample: ``n`` rows drawn with ``seed`` (default 50), not every
    row. The report names how many, e.g. "50 of 300 rows sampled", and
    ``report.rows`` lists them; other rows have no score, so do not keep or
    drop rows by the report. ``n=None`` scores every row, and
    ``filter.by_rubric_score`` keeps rows by their score. Columns, one per
    kind:

    - each of ``response_fields``, e.g. a reference solution and a weak
      baseline's response;
    - each of ``probes``: ``probe(row)`` builds a response that should not
      score well, e.g. a program printing a valid but trivial answer, or the
      example output hard-coded. Probes earning half the reference's mean are
      flagged, like the controls;
    - with ``model``, ``k`` samples per prompt: what RL sees. The report counts
      prompts whose samples all got the same score, which give GRPO-style RL
      no gradient. A mean of 0.9 or more is flagged: little is left to learn;
    - with ``teacher``, ``teacher_k`` replies per prompt from a model trusted
      to answer well, which sees only the prompt: does the reward pay good
      replies, or only the reference's own? When rows carry a rubric in
      ``rubric_field`` (``[[criterion, weight], ...]``), each criterion with a
      nonzero weight is also scored alone on the reference and on every
      teacher reply (``grade`` gets the row with ``[[criterion, 1]]`` in that
      field), and the report counts the criteria the reference passes and
      every teacher reply fails, e.g. "teacher ×2 fails 31 of the 120
      criteria the reference passes on every reply", listing the heaviest.
      A negative-weight criterion names a mistake: it passes when the reply
      scores below 0.5 on it. Read them: such a criterion may demand a
      detail, number or choice that a good reply can leave out, a mistake
      may be docked on a reply that names it only to warn against it, or
      the teacher may be wrong. 20% or more is
      flagged. The stronger the teacher, the stronger the check; one that did
      not write the rubric or the reference does not share their habits.
      This calls ``grade`` ``1 + teacher_k`` times per criterion;
    - with ``control_columns``, responses no task should reward: ``empty``
      (``""``), ``random`` (200 random characters), and ``other_row`` (the
      first column's response to another sampled row, never its own). A
      rubric that gives 1.0 to any response without a code block puts
      ``empty`` at 1; it is flagged.

    The first field column (else the sampled one) is the reference. Every
    other field, probe, and sampled column is also compared with it row by
    row: on how many rows it scores below, tied with, or above the reference,
    and the mean gap, e.g. "baseline: below reference on 31, tied 15, above 4
    of 50 rows, mean gap 0.21". Many ties between a reference and a weak
    baseline mean the rubric cannot tell them apart.

    Bins are ``0``, ``(0, 0.25]``, ``(0.25, 0.5]``, ``(0.5, 0.75]``,
    ``(0.75, 1)``, ``1``. Rows are not modified.

    Args:
        rows: The rows (dicts); each is passed to ``grade`` as is.
        grade: ``(response, row) -> float | bool | None``.
        n: Rows to sample (default 50); None grades every row. A warning is
            logged when it leaves rows out.
        response_fields: Fields holding responses to grade as they are; the
            first is the reference unless ``reference_field`` names one.
        reference_field: The field holding the reference, e.g.
            ``"ground_truth"``: an answer already stored in each row, which
            every other column is compared with and on which the teacher
            check scores each criterion. It is graded even when not in
            ``response_fields``. The teacher, by contrast, is a model that
            writes fresh replies from the prompt.
        probes: ``{name: probe(row) -> response or None}``; None is
            ungradable. Probes run on every call; their responses' grades
            are cached like the others.
        model: Model whose samples to grade, e.g. the SFT checkpoint; needs
            ``generate_batch``. Completions are graded whole, thinking
            included.
        k: Samples per prompt from ``model``.
        teacher: Model whose replies make the ``teacher`` column; needs
            ``generate_batch``. It sees only the prompt.
        teacher_k: Replies per prompt from ``teacher`` (default 2). With
            two or more, a criterion every reply fails is told apart from
            one reply's chance miss.
        rubric_field: Field holding the row's rubric as ``[[criterion,
            weight], ...]``, scored per criterion with ``teacher`` and a
            response field; rows without one are left out of that check.
        prompt_field: Field holding the prompt sent to ``model`` and
            ``teacher`` (text or chat messages).
        temperature: Sampling temperature (the model's default if None);
            must be > 0 when ``k > 1``.
        max_tokens: Max tokens per sample (the model's default if None).
        control_columns: Add the ``empty``, ``random``, and ``other_row``
            columns.
        seed: Seeds the row sample, the random text, and ``other_row``.
        workers: ``grade`` calls run at a time.
        examples: Lowest- and highest-scoring responses kept per column.
    """
    fields = (
        [response_fields]
        if isinstance(response_fields, str)
        else list(response_fields or [])
    )
    if reference_field is not None:
        if not isinstance(reference_field, str) or not reference_field:
            raise TypeError("reference_field must be a field name")
        fields = [reference_field, *(f for f in fields if f != reference_field)]
    probes = dict(probes or {})
    clash = set(fields) & {*CONTROLS, "sampled", "teacher"}
    if clash:
        raise ValueError(f"response_fields {sorted(clash)} clash with column names")
    clash = set(probes) & {*fields, *CONTROLS, "sampled", "teacher"}
    if clash:
        raise ValueError(f"probes {sorted(clash)} clash with column names")
    if model is not None and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if teacher is not None and teacher_k < 1:
        raise ValueError(f"teacher_k must be at least 1, got {teacher_k}")
    several = (model is not None and k > 1) or (teacher is not None and teacher_k > 1)
    if several and temperature is not None and temperature <= 0:
        raise ValueError(
            "temperature must be > 0 when k or teacher_k > 1: samples would match"
        )
    if not fields and model is None and teacher is None and not probes:
        raise ValueError(
            "pass response_fields, probes, a model, or a teacher: there is nothing "
            "to grade"
        )
    picked = _sample_rows(len(rows), n, seed)
    if len(picked) < len(rows):
        get_formatted_logger("RubricScores").warning(
            "inspect.rubric_scores scores a sample of %d of %d rows (n=%d); the "
            "others get no score. Pass n=None to score every row, or use "
            "filter.by_rubric_score to keep rows by score.",
            len(picked),
            len(rows),
            n,
        )
    sample = [rows[i] for i in picked]
    report = RubricScoreReport(
        grade=_name(grade), n_rows=len(rows), rows=picked, seed=seed
    )
    columns: list[tuple[str, str, list[list[str | None]]]] = []
    for name in fields:
        texts = [r.get(name) if isinstance(r, dict) else None for r in sample]
        columns.append(
            (name, "field", [[None if t is None else str(t)] for t in texts])
        )
    prompts = (
        [
            _field_prompt(r, prompt_field, "inspect.rubric_scores", "prompt_field")
            for r in sample
        ]
        if sample and (model is not None or teacher is not None)
        else []
    )
    if model is not None and sample:
        report.model_id = _model_name(model)
        report.k = k
        completions, _ = _samples(
            prompts,
            model=model,
            n=k,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=None,
        )
        columns.append(("sampled", "sampled", completions))
    teacher_replies: list[list[str | None]] = []
    if teacher is not None and sample:
        report.teacher_id = _model_name(teacher)
        report.teacher_k = teacher_k
        teacher_replies, _ = _samples(
            prompts,
            model=teacher,
            n=teacher_k,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=None,
        )
        columns.append(("teacher", "teacher", teacher_replies))
    for name, probe in probes.items():
        built = [probe(r) for r in sample]
        columns.append(
            (name, "probe", [[None if t is None else str(t)] for t in built])
        )
    if control_columns and sample:
        columns.append(("empty", "control", [[""] for _ in sample]))
        columns.append(("random", "control", [[_random_text(seed, i)] for i in picked]))
        if len(sample) >= 2:
            first = [r[0] for r in columns[0][2]]
            other = _derangement(len(sample), seed)
            columns.append(("other_row", "control", [[first[j]] for j in other]))
    for name, kind, responses in columns:
        scores, errors = _grade_column(sample, responses, grade=grade, workers=workers)
        col = ColumnScores(name=name, kind=kind, scores=scores)
        _summarize(col, responses, errors, picked, examples)
        report.columns.append(col)
    _pair_with_reference(report)
    if teacher_replies and fields:
        report.teacher_criteria = _teacher_criteria(
            sample,
            picked,
            [r[0] for r in columns[0][2]],
            teacher_replies,
            reference=fields[0],
            rubric_field=rubric_field,
            grade=grade,
            workers=workers,
        )
    report.flags = _flags(report)
    console = _console()
    _emit(_renderable(report, width=console.width), console)
    return report


_CRITERION_EXAMPLES = 5


def _model_name(model: Any) -> str:
    return str(getattr(model, "model_id", getattr(model, "_model_id", "")) or model)[
        :120
    ]


def _criterion_text(criterion: Any) -> str:
    """A criterion's words: the string, or a dict's ``prompt`` or ``code``."""
    if isinstance(criterion, dict):
        text = criterion.get("prompt") or criterion.get("code") or str(criterion)
    else:
        text = str(criterion)
    return text[:300]


def _teacher_criteria(
    sample: list[dict],
    picked: list[int],
    references: list[str | None],
    replies: list[list[str | None]],
    *,
    reference: str,
    rubric_field: str,
    grade: Grade,
    workers: int,
) -> TeacherCriteria | None:
    """Score each nonzero-weight criterion alone on the reference and every reply.

    None when no sampled row has both a reference and a rubric list.
    """
    out = TeacherCriteria(reference=reference)
    jobs: list[
        tuple[int, int, Any, float]
    ] = []  # (sample i, criterion index, criterion, weight)
    for i, row in enumerate(sample):
        rubric = row.get(rubric_field) if isinstance(row, dict) else None
        if references[i] is None or not isinstance(rubric, list | tuple):
            continue
        out.rows += 1
        for ci, pair in enumerate(rubric):
            try:
                criterion, weight = pair
                weight = float(weight)
            except (TypeError, ValueError):
                continue
            if weight == 0:
                out.skipped += 1
                continue
            jobs.append((i, ci, criterion, weight))
    if not out.rows:
        return None
    if not jobs:
        return out
    rows = [{**sample[i], rubric_field: [[c, 1]]} for i, _, c, _ in jobs]
    responses = [[references[i], *replies[i]] for i, _, _, _ in jobs]
    scores, _ = _grade_column(rows, responses, grade=grade, workers=workers)
    failed: list[CriterionCheck] = []
    for (i, ci, criterion, weight), row_scores in zip(jobs, scores):
        ref, own = row_scores[0], row_scores[1:]
        if ref is None:
            continue
        out.criteria += 1
        good = [s for s in own if s is not None]
        if not _passes(ref, weight) or not good:
            continue
        out.reference_passes += 1
        misses = sum(not _passes(s, weight) for s in good)
        if misses == len(good):
            out.fails_all += 1
            failed.append(
                CriterionCheck(
                    row=picked[i],
                    index=ci,
                    criterion=_criterion_text(criterion),
                    weight=weight,
                    reference=ref,
                    teacher=own,
                )
            )
        elif misses:
            out.fails_some += 1
    if out.reference_passes:
        out.share_fails_all = out.fails_all / out.reference_passes
    failed.sort(key=lambda c: (-abs(c.weight), c.row, c.index))
    out.failed = failed
    out.examples = failed[:_CRITERION_EXAMPLES]
    return out


def _passes(score: float, weight: float) -> bool:
    """A requirement passes when met, a mistake (weight < 0) when not made."""
    return score >= PASS if weight > 0 else score < PASS


def _name(grade: Grade) -> str:
    """``rubric_score()`` for an object with a repr, ``check`` for a function."""
    if getattr(grade, "__code__", None) is not None:
        return str(getattr(grade, "__qualname__", "grade"))
    return repr(grade)[:120]


def _reference(report: RubricScoreReport) -> ColumnScores | None:
    """The first field column, else the sampled one."""
    for kind in ("field", "sampled"):
        for c in report.columns:
            if c.kind == kind:
                return c
    return None


def _row_score(scores: list[float | None]) -> float | None:
    good = [s for s in scores if s is not None]
    return sum(good) / len(good) if good else None


def _pair_with_reference(report: RubricScoreReport) -> None:
    reference = _reference(report)
    if reference is None:
        return
    for c in report.columns:
        if c is reference or c.kind == "control":
            continue
        paired = Paired(reference=reference.name)
        gaps = []
        for ref_scores, scores in zip(reference.scores, c.scores):
            ref, own = _row_score(ref_scores), _row_score(scores)
            if ref is None or own is None:
                continue
            paired.rows += 1
            gap = ref - own
            gaps.append(gap)
            if abs(gap) <= _EPS:
                paired.tied += 1
            elif gap > 0:
                paired.below += 1
            else:
                paired.above += 1
        if gaps:
            paired.mean_gap = sum(gaps) / len(gaps)
        c.vs_reference = paired


def _flags(report: RubricScoreReport) -> list[str]:
    reference = _reference(report)
    if reference is None or not reference.mean:
        return []
    flags = []
    for c in report.columns:
        unwanted = c.kind in ("control", "probe")
        if unwanted and c.mean is not None and c.mean >= reference.mean / 2:
            flags.append(
                f"{c.name} scores {c.mean:.2f} on average, at least half of "
                f"{reference.name}'s {reference.mean:.2f}: RL can earn it too"
            )
        if c.kind == "sampled" and c.mean is not None and c.mean >= EASY_FLAG:
            flags.append(
                f"sampled scores {c.mean:.2f} on average: little is left for RL "
                "to raise"
            )
    t = report.teacher_criteria
    if (
        t is not None
        and t.share_fails_all is not None
        and t.share_fails_all >= TEACHER_FLAG
    ):
        flags.append(
            f"teacher fails {t.fails_all} of the {t.reference_passes} criteria "
            f"{t.reference} passes ({t.share_fails_all:.0%}) on every reply: "
            "read them below"
        )
    return flags


def render_rubric_score_report(report: RubricScoreReport) -> str:
    """Render a ``rubric_scores`` report as plain text."""
    return _plain(_renderable(report, width=WIDTH))


def _share(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _label(c: ColumnScores, report: RubricScoreReport) -> str:
    if c.kind == "sampled":
        return f"sampled ×{report.k}"
    if c.kind == "teacher":
        return f"teacher ×{report.teacher_k}"
    return c.name


def _renderable(report: RubricScoreReport, *, width: int) -> RenderableType:
    head = Text()
    head.append(f"{len(report.rows):,} of {report.n_rows:,} rows", style="bold")
    if len(report.rows) < report.n_rows:
        # e.g. "50 of 300 rows sampled (seed 0; the other 250 have no score)"
        head.append(" sampled", style="bold")
        head.append(
            f" (seed {report.seed}; the other "
            f"{report.n_rows - len(report.rows):,} have no score)"
        )
    else:
        head.append(f" (seed {report.seed})")
    head.append(f" · grade {_clean(report.grade)}")
    if report.model_id:
        head.append(f" · sampled: {_clean(report.model_id)}")
    if report.teacher_id:
        head.append(f" · teacher: {_clean(report.teacher_id)}")
    body: list[RenderableType] = [head]
    cols = report.columns
    if not cols or not report.rows:
        body.append(Text("nothing graded", style="dim"))
        return Group(
            _rule("inspect", "rubric_scores"), Padding(Group(*body), (0, 0, 0, 2))
        )
    lines = [
        [Text(b)] + [Text(f"{c.counts[i]:,}") for c in cols] for i, b in enumerate(BINS)
    ]
    if any(c.out_of_range for c in cols):
        lines.append(
            [Text("outside [0, 1]")] + [Text(f"{c.out_of_range:,}") for c in cols]
        )
    dim = [
        ("ungradable", lambda c: f"{c.ungradable:,}"),
        ("mean", lambda c: "-" if c.mean is None else f"{c.mean:.2f}"),
        ("at 0", lambda c: _share(c.share_at_0)),
        ("at 1", lambda c: _share(c.share_at_1)),
    ]
    for name, cell in dim:
        lines.append(
            [Text(name, style="dim")] + [Text(cell(c), style="dim") for c in cols]
        )
    # An empty last column: _fit_table left-aligns the last one.
    headers = ["score"] + [_clean(_label(c, report)) for c in cols] + [""]
    lines = [cells + [Text("")] for cells in lines]
    body += [Text(""), _fit_table(headers, lines, width - 2, left=set())]
    notes: list[Text] = []
    for c in cols:
        p = c.vs_reference
        if p is not None and p.rows:
            gap = "" if p.mean_gap is None else f", mean gap {p.mean_gap:+.2f}"
            notes.append(
                Text(
                    f"{_label(c, report)}: below {_clean(p.reference)} on "
                    f"{p.below}, tied {p.tied}, above {p.above} of {p.rows} "
                    f"rows{gap}"
                )
            )
    for c in cols:
        if c.kind == "sampled" and c.prompts_compared:
            notes.append(
                Text(
                    f"sampled: {c.constant_prompts}/{c.prompts_compared} prompts gave "
                    f"all samples the same score ({c.constant_at_0} all 0, "
                    f"{c.constant_at_1} all 1): no RL gradient there"
                )
            )
        if c.errors:
            notes.append(
                Text(
                    f"{c.name}: grade raised on {c.errors} responses, "
                    f"e.g. {_clean(c.error)}"
                )
            )
    t = report.teacher_criteria
    if t is not None:
        # e.g. "per criterion: teacher ×2 fails 31 of the 120 criteria reference
        # passes on every reply, 18 on some (152 criteria on 40 rows; 3 skipped)"
        skipped = f"; {t.skipped} with weight 0 skipped" if t.skipped else ""
        notes.append(
            Text(
                f"per criterion: teacher ×{report.teacher_k} fails {t.fails_all} of "
                f"the {t.reference_passes} criteria {_clean(t.reference)} passes on "
                f"every reply, {t.fails_some} on some ({t.criteria} criteria on "
                f"{t.rows} rows{skipped})"
            )
        )
    for flag in report.flags:
        notes.append(Text(f"! {flag}", style="bold red"))
    if notes:
        body.append(Text(""))
        body += notes
    if t is not None and t.examples:
        body.append(Text(""))
        for e in t.examples:
            scores = ", ".join("-" if s is None else f"{s:.2f}" for s in e.teacher)
            item = Text(
                f"teacher fails: row {e.row} criterion {e.index} [{e.weight:g}] "
                f"({_clean(t.reference)} {e.reference:.2f}; teacher {scores})  ",
                style="dim",
            )
            item.append(_clean(e.criterion))
            item.truncate(width - 2, overflow="ellipsis")
            body.append(item)
    # Empty and random responses need no example; a flagged other_row does.
    flagged = {f.split(" ", 1)[0] for f in report.flags}
    shown = [
        c
        for c in cols
        if c.kind not in ("control", "probe") or c.name in flagged - {"empty", "random"}
    ]
    for c in shown:
        for tag, picks in (("lowest", c.lowest), ("highest", c.highest)):
            for e in picks:
                item = Text(
                    f"{_label(c, report)} {tag}: row {e.row} {e.score:.2f}  ",
                    style="dim",
                )
                item.append(_clean(e.response) or '""')
                item.truncate(width - 2, overflow="ellipsis")
                body.append(item)
    return Group(_rule("inspect", "rubric_scores"), Padding(Group(*body), (0, 0, 0, 2)))

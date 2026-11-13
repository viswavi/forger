"""Keep rows by a grade function's score on each row's response.

``by_rubric_score`` grades every row, unlike ``inspect.rubric_scores``, which
grades a sample of ``n`` rows to show the score distribution. Both grade with
the same cached op, so their grades of the same rows are shared.
"""

from __future__ import annotations

import contextvars
import logging
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pyforger import _tracking
from pyforger.op import op, skip_cache_write
from pyforger.utils import get_field, get_formatted_logger

Grade = Callable[[str, dict], "float | bool | None"]


def _logger() -> logging.Logger:
    # Looked up per call: a module-level logger read inside an @op body would
    # trigger @op's "reads globals" warning.
    return get_formatted_logger("RubricScore")


def _call(grade: Grade, response: str, row: dict) -> tuple[float | None, str | None]:
    """``(score, error)``: a raised error is ungradable and keeps its message."""
    try:
        value = grade(response, row)
    except Exception as exc:  # a crash in one response's grading
        return None, f"{type(exc).__name__}: {exc}"[:300]
    if value is None:
        return None, None
    if not isinstance(value, (bool, int, float)):
        raise TypeError(
            f"grade must return a number, a bool, or None; got {value!r} from {grade!r}"
        )
    if not math.isfinite(value):
        return None, f"grade returned {value}"
    return float(value), None


# Named for inspect.rubric_scores, which came first: the name is part of the
# cache key, so grades cached before this op existed are still found.
@op(name="inspect.rubric_scores.grades", category="inspect")
def _grade_column(
    rows: list[dict],
    responses: list[list[str | None]],
    *,
    grade: Grade,
    workers: int,
) -> tuple[list[list[float | None]], list[list[str | None]]]:
    """Grade ``responses[i][j]`` against ``rows[i]``; a None response is ungradable.

    Returns the scores and, aligned, each error message. A grading that raised
    keeps the result out of the cache, so a rerun tries it again.
    """
    scores: list[list[float | None]] = [[None] * len(r) for r in responses]
    errors: list[list[str | None]] = [[None] * len(r) for r in responses]
    jobs = [
        (i, j)
        for i, row in enumerate(responses)
        for j, response in enumerate(row)
        if response is not None
    ]

    def run(i: int, j: int) -> tuple[float | None, str | None]:
        return _call(grade, responses[i][j], rows[i])  # type: ignore[arg-type]

    def run_in(
        context: contextvars.Context, i: int, j: int
    ) -> tuple[float | None, str | None]:
        return context.run(run, i, j)

    if workers <= 1 or len(jobs) <= 1:
        results = [run(i, j) for i, j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            # Each call runs in a copy of the caller's context, so ops it calls
            # are cached and reported as without threads.
            futures = [
                pool.submit(run_in, contextvars.copy_context(), i, j) for i, j in jobs
            ]
            results = [f.result() for f in futures]
    for (i, j), (score, error) in zip(jobs, results):
        scores[i][j], errors[i][j] = score, error
    if any(e is not None for row in errors for e in row):
        skip_cache_write()
    return scores, errors


@op(name="filter.by_rubric_score")
def by_rubric_score(
    items: list[dict],
    grade: Grade,
    *,
    response_field: str = "response",
    min_score: float | None = None,
    max_score: float | None = None,
    score_field: str | None = None,
    keep_ungradable: bool = False,
    workers: int = 8,
) -> list[dict]:
    """Grade every row's response with ``grade``; keep rows scoring within the bounds.

    ``grade(response, row)`` is the same function ``inspect.rubric_scores``
    takes: a score (a bool counts as 0 or 1), or None when the response cannot
    be graded. A row is kept when ``min_score <= score <= max_score``; a bound
    left as None is open, so with neither bound every gradable row is kept and
    ``score_field`` gives each its score, e.g. ``{**row, "rubric_score": 0.75}``.

    A row whose response is None, or that ``grade`` returns None for, is
    ungradable, and so is one where ``grade`` raised: a warning gives the
    count and the first message, and the result is not cached, so a rerun
    grades those rows again. Ungradable rows are dropped unless
    ``keep_ungradable``, which keeps them with ``score_field`` None.

    Example::

        kept = filter.by_rubric_score(
            rows, grade, response_field="solution", min_score=0.5,
            score_field="rubric_score",
        )

    Args:
        items: Rows; each is passed to ``grade`` as is.
        grade: ``(response, row) -> float | bool | None``; part of the cache
            key (an object with ``__call__`` by its ``cache_key()``).
        response_field: Field holding the response to grade; a value that is
            not text is graded as ``str(value)``.
        min_score: Lowest score kept (inclusive); None for no lower bound.
        max_score: Highest score kept (inclusive); None for no upper bound.
        score_field: When set, kept rows carry their score under this key.
        keep_ungradable: Keep rows with no score instead of dropping them.
        workers: ``grade`` calls run at a time; ``grade`` must be safe to call
            from several threads when this is above 1.

    Returns:
        The kept rows, in input order; unchanged unless ``score_field`` is set.
    """
    for name, bound in (("min_score", min_score), ("max_score", max_score)):
        if bound is not None and not math.isfinite(bound):
            raise ValueError(f"{name} must be finite, got {bound}")
    if min_score is not None and max_score is not None and min_score > max_score:
        raise ValueError(
            f"min_score ({min_score}) must not exceed max_score ({max_score})"
        )
    if workers < 1:
        raise ValueError("workers must be at least 1")
    texts = [
        get_field(item, response_field, "filter.by_rubric_score") for item in items
    ]
    responses = [None if t is None else str(t) for t in texts]
    graded, errors = _grade_column(
        items, [[r] for r in responses], grade=grade, workers=workers
    )
    scores = [g[0] for g in graded]
    messages = [e[0] for e in errors]
    raised = [m for m in messages if m is not None]
    if raised:
        warning = (
            f"filter.by_rubric_score: grade raised on {len(raised)} of "
            f"{len(items)} rows, e.g. {raised[0]}; "
            + ("kept with no score" if keep_ungradable else "dropped")
            + " (not cached, so a rerun grades them again)"
        )
        _logger().warning(warning)
        _tracking.add_op_warning(warning, "grade_error")
    ungradable = scores.count(None)
    note = None
    if ungradable:
        # e.g. "5 of 40 rows ungradable (3 no response, 2 grade raised): dropped"
        empty = sum(r is None for r in responses)
        causes = [
            f"{count} {cause}"
            for count, cause in (
                (empty, "no response"),
                (len(raised), "grade raised"),
                (ungradable - empty - len(raised), "grade returned None"),
            )
            if count
        ]
        note = (
            f"{ungradable} of {len(items)} rows ungradable ({', '.join(causes)}): "
            + ("kept" if keep_ungradable else "dropped")
        )
    _tracking.record_scores(
        "rubric score",
        scores,
        keep="between",
        lower=min_score,
        upper=max_score,
        notes=messages,
        previews=responses,
        note=note,
    )
    kept = []
    for item, score in zip(items, scores):
        if score is None:
            if not keep_ungradable:
                continue
        elif (min_score is not None and score < min_score) or (
            max_score is not None and score > max_score
        ):
            continue
        kept.append(item if score_field is None else {**item, score_field: score})
    return kept

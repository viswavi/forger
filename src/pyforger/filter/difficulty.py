"""Difficulty filters.

``by_judged_difficulty`` asks an LLM judge to rate each task from 1 (trivial) to 5
(very hard) against three editable rubric anchors, and keeps the tasks whose
rating falls in ``[min_difficulty, max_difficulty]``.
"""
from __future__ import annotations

from pyforger import _tracking
from pyforger.acquire._model_utils import _RequestFailures
from pyforger.acquire.model import Message, Model, _cache_kwargs
from pyforger.filter.judge import _parse_score_response
from pyforger.op import op
from pyforger.utils import get_field, get_formatted_logger

logger = get_formatted_logger("Difficulty")

MAX_DIFFICULTY = 5

# Rubric anchors for levels 1, 3, and 5; levels 2 and 4 sit between them.
DEFAULT_MINIMUM_DIFFICULTY_CRITERION = (
    "This is a straightforward task that any person should be able to readily solve."
)
DEFAULT_MEDIUM_DIFFICULTY_CRITERION = (
    "This is a challenging task that roughly 25% of skilled workers or "
    "professionals in the relevant area would require at least a minute to work "
    "through."
)
DEFAULT_MAXIMUM_DIFFICULTY_CRITERION = (
    "This is a very challenging task that roughly 90% of skilled workers or "
    "professionals in the relevant area would require at least 10 minutes to work "
    "through, and at least 25% would make a mistake."
)

DIFFICULTY_SYSTEM_PROMPT = (
    "Given a task description, judge the difficulty of this task on a 5 point "
    "scale. Rate how hard it is to solve the task correctly, not how hard the "
    "description is to read. Do not solve the task.\n\n"
    "1: {minimum}\n"
    "2: Between levels 1 and 3.\n"
    "3: {medium}\n"
    "4: Between levels 3 and 5.\n"
    "5: {maximum}\n\n"
    'Respond with JSON: {{"explanation": "<one or two sentences>", '
    '"score": <integer from 1 to 5>}}'
)


@op(name="filter.by_judged_difficulty")
def by_judged_difficulty(
    items: list[dict],
    *,
    model: Model,
    task_field: str,
    min_difficulty: float = 3,
    max_difficulty: float = 5,
    num_judges: int = 1,
    temperature: float = 0.0,
    minimum_difficulty_criterion: str = DEFAULT_MINIMUM_DIFFICULTY_CRITERION,
    medium_difficulty_criterion: str = DEFAULT_MEDIUM_DIFFICULTY_CRITERION,
    maximum_difficulty_criterion: str = DEFAULT_MAXIMUM_DIFFICULTY_CRITERION,
) -> list[dict]:
    """Keep items whose judged difficulty is in [min_difficulty, max_difficulty].

    Each judge rates ``task_field`` from 1 to 5; with ``num_judges > 1`` the
    row's difficulty is the mean of the parsed ratings (e.g. 3, 4, 4 -> 3.67).
    A row with no parseable rating is dropped, as is one whose judge request
    failed (a 504 or a timeout after the model's retries; a warning counts
    them). Ratings are cached separately from the bounds, so re-running with
    other bounds makes no new calls.
    """
    if num_judges < 1:
        raise ValueError("num_judges must be at least 1")
    if not (1 <= min_difficulty <= max_difficulty <= MAX_DIFFICULTY):
        raise ValueError(
            f"need 1 <= min_difficulty ({min_difficulty}) <= max_difficulty "
            f"({max_difficulty}) <= {MAX_DIFFICULTY}"
        )
    if num_judges > 1 and temperature == 0.0:
        raise ValueError(
            "temperature must be > 0.0 when num_judges > 1; "
            "deterministic sampling defeats multi-judge consensus"
        )

    system = DIFFICULTY_SYSTEM_PROMPT.format(
        minimum=minimum_difficulty_criterion,
        medium=medium_difficulty_criterion,
        maximum=maximum_difficulty_criterion,
    )
    tasks = [_task_text(item, task_field) for item in items]
    # Building prompts here puts their exact text in the cache key of the ratings.
    prompts = [
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task:\n{task}"},
        ]
        for task in tasks
    ]
    all_scores = _difficulty_scores(
        prompts, model=model, num_judges=num_judges, temperature=temperature
    )
    kept: list[dict] = []
    difficulties: list[float | int | None] = []
    parse_failures = 0
    for item, scores in zip(items, all_scores):
        parsed = [score for score in scores if score is not None]
        parse_failures += len(scores) - len(parsed)
        if not parsed:
            difficulties.append(None)
            continue
        difficulty = parsed[0] if num_judges == 1 else sum(parsed) / len(parsed)
        difficulties.append(difficulty)
        if min_difficulty <= difficulty <= max_difficulty:
            kept.append(item)
    if parse_failures:
        logger.warning(
            "by_judged_difficulty: %d judge responses unparseable", parse_failures
        )
    _tracking.record_scores(
        "difficulty",
        difficulties,
        keep="between",
        lower=min_difficulty,
        upper=max_difficulty,
    )
    return kept


def _task_text(item: dict, task_field: str) -> str:
    """Return the task under ``task_field`` as text for the judge.

    Chat messages are written out one block each, the role line and then the
    content, e.g. an RL row's ``[{"role": "user", "content": "Add 2 and 2."}]``
    becomes "user:" and "Add 2 and 2." on two lines, not the repr of the list.
    A number is sent as its text.
    """
    value = get_field(item, task_field, "by_judged_difficulty")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):  # bool too: e.g. 7 → "7"
        return str(value)
    if (
        isinstance(value, list)
        and value
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in value)
    ):
        return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in value)
    raise ValueError(
        f"task_field {task_field!r} must hold text or a list of chat messages "
        f'({{"role": ..., "content": ...}} dicts), got {type(value).__name__}'
    )


@op(name="filter.by_judged_difficulty.scores", samples_arg="num_judges")
def _difficulty_scores(
    prompts: list[list[Message]],
    *,
    model: Model,
    num_judges: int,
    temperature: float,
) -> list[list[int | float | None]]:
    """Return each prompt's ``num_judges`` parsed ratings (None when unparseable).

    A prompt whose request still fails after the model's retries with a
    transient error (a 504, a timeout), or that hits a row error (the content
    filter, an over-long prompt), gets ``None`` ratings, one warning names the
    count and the last error, and the result is not cached; any other error
    raises, as does a row error on every prompt.
    """
    failures = _RequestFailures()
    ratings: list[list[int | float | None]] = []
    for prompt in prompts:
        # Each prompt is built from one row, so it keys that row in the request
        # cache: a new row costs only its own judges.
        try:
            raws = model.generate(
                prompt,
                n=num_judges,
                temperature=temperature,
                **_cache_kwargs(model.generate, cache_row=prompt),
            )
        except Exception as exc:
            failures.failed(exc)  # raises unless transient or a row error
            ratings.append([None] * num_judges)
            continue
        ratings.append([_parse_score_response(raw, MAX_DIFFICULTY)[0] for raw in raws])
    failures.raise_if_systemic(len(prompts))
    failures.report(
        f"filter.by_judged_difficulty: {len(failures)} of {len(prompts)} judge "
        "requests failed after retries; those rows are unrated"
    )
    return ratings

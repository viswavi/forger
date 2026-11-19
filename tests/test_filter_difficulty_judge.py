"""Tests for the LLM-judged difficulty filter."""

import json

import pytest

from pyforger import _tracking
from pyforger.filter import by_judged_difficulty
from pyforger.filter.difficulty import (
    DEFAULT_MAXIMUM_DIFFICULTY_CRITERION,
    DEFAULT_MEDIUM_DIFFICULTY_CRITERION,
    DEFAULT_MINIMUM_DIFFICULTY_CRITERION,
)
from pyforger.steps import pipeline, step


class _ScriptedJudge:
    """Fake judge that rates a task by looking up its text in ``ratings``.

    ``ratings["Add 2 and 2."] == [1, 2]`` makes judge i answer ``ratings[...][i]``;
    a string entry is returned as-is (e.g. ``"not json"``).
    """

    def __init__(self, ratings):
        """Store the per-task ratings and start counting calls."""
        self.ratings = ratings
        self.prompts = []
        self.call_count = 0
        self.last_n = None
        self.last_temperature = None

    def cache_key(self):
        """Keep the call counters out of the @op cache key."""
        return ("_ScriptedJudge", json.dumps(self.ratings, sort_keys=True))

    def generate(self, prompt, *, n=1, temperature=None, max_tokens=None):
        """Return ``n`` JSON ratings for the task in the user message."""
        self.call_count += 1
        self.prompts.append(prompt)
        self.last_n = n
        self.last_temperature = temperature
        task = prompt[-1]["content"].removeprefix("Task:\n")
        answers = self.ratings[task]
        return [
            a if isinstance(a, str) else json.dumps({"explanation": "x", "score": a})
            for a in answers[:n]
        ]


TASKS = {
    "Add 2 and 2.": [1, 1, 2],
    "Prove there are infinitely many primes.": [3, 3, 4],
    "Find all integer solutions of x^3 + y^3 = z^3.": [5, 5, 4],
}


def _rows():
    return [{"task": task} for task in TASKS]


def test_prompt_contains_criteria_and_task():
    """The judge sees all three default anchors and the task text."""
    judge = _ScriptedJudge(TASKS)
    by_judged_difficulty(_rows()[:1], model=judge, task_field="task")
    system, user = judge.prompts[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert "judge the difficulty of this task on a 5 point scale" in system["content"]
    for criterion in (
        DEFAULT_MINIMUM_DIFFICULTY_CRITERION,
        DEFAULT_MEDIUM_DIFFICULTY_CRITERION,
        DEFAULT_MAXIMUM_DIFFICULTY_CRITERION,
    ):
        assert criterion in system["content"]
    assert "Add 2 and 2." in user["content"]


def test_custom_criteria_replace_the_defaults():
    """User-supplied anchors appear in the prompt instead of the defaults."""
    judge = _ScriptedJudge(TASKS)
    by_judged_difficulty(
        _rows()[:1],
        model=judge,
        task_field="task",
        minimum_difficulty_criterion="EASY ANCHOR",
        medium_difficulty_criterion="MEDIUM ANCHOR",
        maximum_difficulty_criterion="HARD ANCHOR",
    )
    system = judge.prompts[0][0]["content"]
    assert "1: EASY ANCHOR" in system
    assert "3: MEDIUM ANCHOR" in system
    assert "5: HARD ANCHOR" in system
    assert DEFAULT_MEDIUM_DIFFICULTY_CRITERION not in system


def test_default_bounds_keep_medium_and_hard_tasks():
    """The default range [3, 5] drops the easy task."""
    kept = by_judged_difficulty(_rows(), model=_ScriptedJudge(TASKS), task_field="task")
    assert [r["task"] for r in kept] == list(TASKS)[1:]


@pytest.mark.parametrize(
    "lower, upper, expected",
    [(1, 1, [0]), (3, 3, [1]), (1, 3, [0, 1]), (5, 5, [2]), (2, 2, [])],
)
def test_bounds_are_inclusive(lower, upper, expected):
    """A rating equal to either bound is kept."""
    rows = _rows()
    kept = by_judged_difficulty(
        rows,
        model=_ScriptedJudge(TASKS),
        task_field="task",
        min_difficulty=lower,
        max_difficulty=upper,
    )
    assert kept == [rows[i] for i in expected]


def test_several_judges_are_averaged(monkeypatch):
    """With num_judges > 1 the row's difficulty is the mean rating."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    judge = _ScriptedJudge(TASKS)

    @step
    def judged(rows):
        return by_judged_difficulty(
            rows,
            model=judge,
            task_field="task",
            num_judges=3,
            temperature=0.7,
            min_difficulty=3.5,
        )

    kept = judged(_rows())
    assert (judge.last_n, judge.last_temperature) == (3, 0.7)
    # Means are 4/3, 10/3, and 14/3; only the last reaches 3.5.
    assert [r["task"] for r in kept] == list(TASKS)[2:]
    scores = _tracking.step_records()["judged"].ops[0].scores
    assert scores is not None
    assert scores.values == pytest.approx([4 / 3, 10 / 3, 14 / 3])


def test_unparsed_ratings_drop_the_row_and_warn(caplog):
    """A row with no parseable rating is dropped and the failures are logged."""
    judge = _ScriptedJudge(
        {"Add 2 and 2.": ["not json"], "Hard task.": [json.dumps({"score": 9})]}
    )
    rows = [{"task": "Add 2 and 2."}, {"task": "Hard task."}]
    kept = by_judged_difficulty(rows, model=judge, task_field="task", min_difficulty=1)
    assert kept == []
    assert "2 judge responses unparseable" in caplog.text


def test_partly_parsed_judges_average_the_parsed_ratings():
    """Unparseable judges are left out of the mean rather than counted as 0."""
    judge = _ScriptedJudge({"Add 2 and 2.": ["oops", 4, 5]})
    kept = by_judged_difficulty(
        [{"task": "Add 2 and 2."}],
        model=judge,
        task_field="task",
        num_judges=3,
        temperature=0.7,
        min_difficulty=4.5,
    )
    assert len(kept) == 1


def test_changing_bounds_makes_no_new_judge_calls():
    """Ratings are cached apart from the bounds."""
    judge = _ScriptedJudge(TASKS)
    wide = by_judged_difficulty(
        _rows(), model=judge, task_field="task", min_difficulty=1
    )
    assert judge.call_count == len(TASKS)
    narrow = by_judged_difficulty(
        _rows(), model=judge, task_field="task", min_difficulty=5
    )
    assert judge.call_count == len(TASKS)
    assert len(narrow) < len(wide) == len(TASKS)


def test_changing_criteria_asks_the_judges_again():
    """A new rubric anchor changes the prompt, so the ratings are not reused."""
    judge = _ScriptedJudge(TASKS)
    by_judged_difficulty(_rows(), model=judge, task_field="task")
    by_judged_difficulty(
        _rows(), model=judge, task_field="task", medium_difficulty_criterion="Other."
    )
    assert judge.call_count == 2 * len(TASKS)


def test_scores_recorded_as_between(monkeypatch):
    """Inside a step the ratings are recorded with keep='between' and the bounds."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    judge = _ScriptedJudge({**TASKS, "Bad.": ["not json"]})

    @step
    def hard_only(rows):
        return by_judged_difficulty(
            rows, model=judge, task_field="task", min_difficulty=2, max_difficulty=4
        )

    hard_only(_rows() + [{"task": "Bad."}])
    scores = _tracking.step_records()["hard_only"].ops[0].scores
    assert scores is not None
    assert (scores.label, scores.keep, scores.lower, scores.upper) == (
        "difficulty",
        "between",
        2,
        4,
    )
    assert scores.values == [1, 3, 5, None]


def test_counts_as_a_quality_filter(monkeypatch):
    """A pipeline that runs it gets no 'no quality filter' warning."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    judge = _ScriptedJudge(TASKS)

    @step
    def hard(rows):
        return by_judged_difficulty(rows, model=judge, task_field="task")

    @pipeline
    def run(rows):
        return hard(rows)

    run(_rows())
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert not any("no quality filter" in w for w in rec.warnings)


def test_empty_input_makes_no_calls():
    """Empty input returns [] without asking the judge."""
    judge = _ScriptedJudge(TASKS)
    assert by_judged_difficulty([], model=judge, task_field="task") == []
    assert judge.call_count == 0


def test_missing_field_raises():
    """A row without ``task_field`` raises before any judge call."""
    judge = _ScriptedJudge(TASKS)
    with pytest.raises(KeyError, match="missing required field 'task'"):
        by_judged_difficulty([{"other": "x"}], model=judge, task_field="task")
    assert judge.call_count == 0


def test_chat_message_task_is_rendered_as_text():
    """An RL row's message list reaches the judge as readable text, not its repr."""
    rendered = "system:\nBe brief.\n\nuser:\nAdd 2 and 2."
    judge = _ScriptedJudge({rendered: [1], "7": [4]})
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Add 2 and 2."},
    ]
    kept = by_judged_difficulty(
        [{"prompt": messages}, {"prompt": 7}], model=judge, task_field="prompt"
    )
    assert [user["content"] for _, user in judge.prompts] == [
        f"Task:\n{rendered}",
        "Task:\n7",
    ]
    assert kept == [{"prompt": 7}]


@pytest.mark.parametrize("value", [{"role": "user"}, [], ["Add 2 and 2."], None])
def test_other_task_types_raise(value):
    """A dict, an empty list, a list of non-messages, or None raises before any call."""
    judge = _ScriptedJudge(TASKS)
    with pytest.raises(ValueError, match="list of chat messages"):
        by_judged_difficulty([{"task": value}], model=judge, task_field="task")
    assert judge.call_count == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(min_difficulty=4, max_difficulty=3), "min_difficulty"),
        (dict(min_difficulty=0), "min_difficulty"),
        (dict(max_difficulty=6), "max_difficulty"),
        (dict(num_judges=0), "num_judges"),
        (dict(num_judges=2, temperature=0.0), "temperature"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    """Bad bounds, judge counts, and zero temperature with several judges raise."""
    with pytest.raises(ValueError, match=match):
        by_judged_difficulty(
            _rows(), model=_ScriptedJudge(TASKS), task_field="task", **kwargs
        )

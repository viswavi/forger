"""Tests for inspect.rubric_scores: binned reward scores beside control responses."""

from __future__ import annotations

import json

import pytest

from pyforger.inspect import RubricScoreReport, rubric_scores
from pyforger.inspect.scores import BINS, _bin, _derangement, render_rubric_score_report


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Silence the step report."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")


def _rows(n: int = 20) -> list[dict]:
    """Coding rows: every fifth target is wrong."""
    return [
        {
            "prompt": [{"role": "user", "content": f"task {i}"}],
            "answer": str(i),
            "response": "<think>t</think>```python\n"
            + ("wrong" if i % 5 == 0 else str(i))
            + "\n```",
        }
        for i in range(n)
    ]


def abstain_bug(response: str, row: dict) -> float:
    """The real bug: 1.0 for any response without a python block."""
    if "```python" not in response:
        return 1.0
    code = response.split("```python")[-1].split("```")[0]
    return float(code.strip() == row["answer"])


def strict(response: str, row: dict) -> float:
    """Full reward only for the row's own correct target."""
    return 1.0 if response == row["response"] and "wrong" not in response else 0.0


class Counting:
    """A grade function that counts its calls; keyed by cache_key."""

    def __init__(self, fn, key="counting"):
        """Wrap ``fn``; ``key`` is the cache key."""
        self.fn, self.key, self.calls = fn, key, 0

    def cache_key(self):
        """The key Forger's op cache uses."""
        return ("Counting", self.key)

    def __call__(self, response, row):
        """Count the call and grade."""
        self.calls += 1
        return self.fn(response, row)


class Student:
    """Odd prompts: one right, one wrong, one abstention. Even: right every time."""

    model_id = "toy-student"

    def cache_key(self):
        """The key Forger's op cache uses."""
        return ("Student",)

    def generate_batch(self, prompts, n=1, **kwargs):
        """Up to three completions per prompt."""
        self.prompts = prompts
        out = []
        for p in prompts:
            i = int(p[-1]["content"].split()[-1])
            right = f"```python\n{i}\n```"
            out.append(
                (
                    [right, "```python\nwrong\n```", "I cannot"]
                    if i % 2
                    else [right] * 3
                )[:n]
            )
        return out


def test_empty_control_catches_a_rubric_that_rewards_abstention():
    """The targets look fine; the empty response at 1 shows the bug."""
    report = rubric_scores(_rows(), abstain_bug, n=None)
    assert isinstance(report, RubricScoreReport)
    assert [c.name for c in report.columns] == [
        "response",
        "empty",
        "random",
        "other_row",
    ]
    target, empty = report.column("response"), report.column("empty")
    assert target.histogram() == dict(zip(BINS, [4, 0, 0, 0, 0, 16]))
    assert empty.counts[-1] == 20 and empty.mean == 1.0 and empty.share_at_1 == 1.0
    assert report.column("random").mean == 1.0
    assert report.column("other_row").counts[0] == 20
    assert [f.split()[0] for f in report.flags] == ["empty", "random"]
    data = json.loads(report.model_dump_json())
    assert data["bins"] == BINS
    assert data["columns"][1]["counts"] == [0, 0, 0, 0, 0, 20]


def test_a_sound_rubric_scores_controls_zero():
    """No control earns reward, so nothing is flagged."""
    report = rubric_scores(_rows(), strict, n=None)
    assert report.column("response").mean == pytest.approx(0.8)
    for name in ("empty", "random", "other_row"):
        assert report.column(name).counts[0] == 20
    assert report.flags == []


def test_other_row_is_never_its_own_response():
    """other_row pairs each row with another row's response."""
    for n in range(2, 30):
        assert all(j != i for i, j in enumerate(_derangement(n, seed=n)))
    own = Counting(lambda r, row: float(r == row["response"]))
    assert rubric_scores(_rows(), own, n=None).column("other_row").mean == 0.0


def test_samples_n_rows_with_the_seed():
    """The same seed draws the same rows."""
    rows = _rows(100)
    a = rubric_scores(rows, strict, n=10, seed=3)
    assert len(a.rows) == 10 and a.rows == sorted(a.rows)
    assert a.rows == rubric_scores(rows, strict, n=10, seed=3).rows
    assert a.rows != rubric_scores(rows, strict, n=10, seed=4).rows
    assert len(a.column("response").scores) == 10


def test_sampled_column_counts_prompts_without_gradient():
    """Prompts whose k samples all score the same are counted."""
    student = Student()
    report = rubric_scores(
        _rows(10), abstain_bug, n=None, model=student, k=3, control_columns=False
    )
    sampled = report.column("sampled")
    assert report.model_id == "toy-student" and report.k == 3
    assert student.prompts[0] == [{"role": "user", "content": "task 0"}]
    assert sampled.responses == 30
    # Even prompts: three right answers. Odd: right, wrong, and a rewarded abstention.
    assert sampled.prompts_compared == 10
    assert (sampled.constant_prompts, sampled.constant_at_1) == (5, 5)
    assert sampled.counts[0] == 5 and sampled.counts[-1] == 25
    assert "5/10 prompts gave all samples the same score" in (
        render_rubric_score_report(report)
    )


def test_grades_are_cached_by_the_grade_function():
    """A second call grades nothing; another grade function grades again."""
    rows = _rows()
    grade = Counting(strict)
    rubric_scores(rows, grade, n=None)
    first = grade.calls
    assert first == 80  # 20 rows x (response, empty, random, other_row)
    rubric_scores(rows, grade, n=None)
    assert grade.calls == first
    other = Counting(strict, key="another")
    rubric_scores(rows, other, n=None)
    assert other.calls == first


def test_none_and_errors_are_ungradable_and_errors_are_not_cached():
    """None and raised gradings are ungradable; raised ones run again."""

    def flaky(response, row):
        """Raise on empty responses; None on wrong ones."""
        if response == "":
            raise RuntimeError("sandbox did not start")
        return None if "wrong" in response else 1.0

    grade = Counting(flaky)
    report = rubric_scores(_rows(), grade, n=None)
    target, empty = report.column("response"), report.column("empty")
    assert (target.ungradable, target.errors, target.gradable) == (4, 0, 16)
    assert empty.errors == 20 and empty.error == "RuntimeError: sandbox did not start"
    assert empty.mean is None
    text = render_rubric_score_report(report)
    assert "grade raised on 20 responses" in text
    calls = grade.calls
    rubric_scores(_rows(), grade, n=None)
    assert grade.calls == calls + 20  # only the column that raised runs again


def test_bools_bins_and_out_of_range():
    """Bools count as 0 or 1; scores outside [0, 1] get no bin."""
    assert [_bin(x) for x in (0.0, 1e-12, 0.1, 0.25, 0.26, 0.75, 0.9, 1.0)] == [
        0,
        0,
        1,
        1,
        2,
        3,
        4,
        5,
    ]
    assert _bin(1 - 1e-12) == 5 and _bin(-0.5) is None and _bin(3.0) is None
    report = rubric_scores(_rows(), lambda r, row: "wrong" not in r, n=None)
    assert report.column("response").counts == [4, 0, 0, 0, 0, 16]
    scaled = rubric_scores(_rows(), lambda r, row: 7.0, n=None, control_columns=False)
    assert scaled.column("response").out_of_range == 20
    assert "outside [0, 1]" in render_rubric_score_report(scaled)
    with pytest.raises(TypeError, match="grade must return a number"):
        rubric_scores(_rows(), lambda r, row: "yes", n=None)


def test_rendered_report():
    """The text report has the bins, columns, flags, and examples."""
    text = render_rubric_score_report(rubric_scores(_rows(), abstain_bug, n=None))
    assert "inspect rubric_scores" in text
    assert "20 of 20 rows (seed 0) · grade abstain_bug" in text
    for label in BINS + ["ungradable", "mean", "at 1", "other_row"]:
        assert label in text
    assert "! empty scores 1.00 on average" in text
    assert "response lowest: row 0 0.00" in text


def test_needs_something_to_grade():
    """No response field and no model is an error, as is a clashing name."""
    with pytest.raises(ValueError, match="nothing to grade"):
        rubric_scores(_rows(), strict, response_fields=None)
    with pytest.raises(ValueError, match="clash"):
        rubric_scores(_rows(), strict, response_fields=["empty"])


def _graded_rows(n: int = 20) -> list[dict]:
    """A right answer, and a baseline right on even rows, trivial on odd ones."""
    return [
        {
            "prompt": f"task {i}",
            "answer": str(i + 1),
            "baseline": str(i + 1) if i % 2 == 0 else "0",
        }
        for i in range(n)
    ]


def partial(response: str, row: dict) -> float:
    """1 for the answer, 0.5 for the trivial "0", else 0."""
    return 1.0 if response == row["answer"] else 0.5 if response == "0" else 0.0


def test_paired_comparison_with_the_reference():
    """Other columns are compared with the first field row by row."""
    report = rubric_scores(
        _graded_rows(), partial, n=None, response_fields=["answer", "baseline"]
    )
    assert report.column("answer").vs_reference is None
    paired = report.column("baseline").vs_reference
    assert paired is not None
    assert (paired.reference, paired.rows) == ("answer", 20)
    assert (paired.below, paired.tied, paired.above) == (10, 10, 0)
    assert paired.mean_gap == pytest.approx(0.25)
    assert report.column("empty").vs_reference is None  # controls are not paired
    text = render_rubric_score_report(report)
    assert "baseline: below answer on 10, tied 10, above 0 of 20 rows" in text
    assert "mean gap +0.25" in text


def test_probes_are_graded_flagged_and_paired():
    """A probe earning half the reference's mean is flagged like a control."""
    report = rubric_scores(
        _graded_rows(),
        partial,
        n=None,
        response_fields="answer",
        probes={"trivial": lambda row: "0", "unbuilt": lambda row: None},
    )
    trivial = report.column("trivial")
    assert (trivial.kind, trivial.mean) == ("probe", 0.5)
    assert trivial.vs_reference is not None and trivial.vs_reference.below == 20
    assert report.column("unbuilt").ungradable == 20
    assert any(f.startswith("trivial scores 0.50") for f in report.flags)
    assert "trivial lowest: row 0 0.50" in render_rubric_score_report(report)
    only = rubric_scores(
        _graded_rows(),
        partial,
        n=None,
        response_fields=None,
        probes={"trivial": lambda row: "0"},
        control_columns=False,
    )
    assert [c.name for c in only.columns] == ["trivial"]
    with pytest.raises(ValueError, match="probes .* clash"):
        rubric_scores(_graded_rows(), partial, probes={"empty": lambda row: ""})

"""Tests for inspect.rubric_scores: binned reward scores beside control responses."""

from __future__ import annotations

import json
import logging

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


def test_a_sample_is_named_in_the_report_and_a_warning(caplog):
    """An n below the row count warns and says how many rows have no score."""
    with caplog.at_level(logging.WARNING):
        report = rubric_scores(_rows(100), strict, n=10)
    assert "scores a sample of 10 of 100 rows (n=10)" in caplog.text
    assert "filter.by_rubric_score" in caplog.text
    text = render_rubric_score_report(report)
    assert "10 of 100 rows sampled (seed 0; the other 90 have no score)" in text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rubric_scores(_rows(), strict, n=50)  # 20 rows: all scored
    assert "scores a sample" not in caplog.text


def test_missing_prompt_names_rubric_scores():
    """A missing prompt_field names the op the user called, not another."""
    with pytest.raises(
        KeyError, match="inspect.rubric_scores: item is missing required field 'task'"
    ):
        rubric_scores(_rows(), strict, model=Student(), prompt_field="task", n=None)
    rows = [{**r, "prompt": {"text": "x"}} for r in _rows()]
    with pytest.raises(
        ValueError, match="inspect.rubric_scores: prompt_field 'prompt' must hold"
    ) as err:
        rubric_scores(rows, strict, model=Student(), n=None)
    assert "prompt_template" not in str(err.value)  # rubric_scores has none


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


def keyword_rubric(response: str, row: dict) -> float:
    """Weighted mean over criteria "mentions WORD" (positive weights only)."""
    pairs = [(c, w) for c, w in row["rubric"] if w > 0]
    total = sum(w for _, w in pairs)
    met = sum(w for c, w in pairs if c.split(" ", 1)[1] in response)
    return met / total


def _rubric_rows(n: int = 6) -> list[dict]:
    """The reference names a dose a good reply may leave out."""
    return [
        {
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "reference": "Give aspirin 81 mg; watch for bleeding.",
            "rubric": [
                ["mentions aspirin", 50],
                ["mentions 81 mg", 100],
                ["mentions bleeding", 25],
                ["mentions warfarin", -10],
            ],
        }
        for i in range(n)
    ]


class Teacher:
    """Replies name aspirin and bleeding; odd prompts' first reply names the dose."""

    model_id = "toy-teacher"

    def cache_key(self):
        """The key Forger's op cache uses."""
        return ("Teacher",)

    def generate_batch(self, prompts, n=1, **kwargs):
        """``n`` replies per prompt; the prompts are recorded."""
        self.prompts = prompts
        out = []
        for p in prompts:
            i = int(p[-1]["content"].split()[-1])
            plain = "Aspirin is reasonable: aspirin, but watch bleeding."
            dosed = plain + " 81 mg daily."
            out.append([dosed if (i % 2 and j == 0) else plain for j in range(n)])
        return out


def test_teacher_finds_criteria_only_the_reference_meets():
    """The dose criterion fails every teacher reply on even rows, some on odd ones."""
    teacher = Teacher()
    report = rubric_scores(
        _rubric_rows(),
        keyword_rubric,
        n=None,
        response_fields="reference",
        teacher=teacher,
        control_columns=False,
    )
    assert report.teacher_id == "toy-teacher" and report.teacher_k == 2
    assert teacher.prompts[0] == [{"role": "user", "content": "question 0"}]
    assert [c.name for c in report.columns] == ["reference", "teacher"]
    t = report.teacher_criteria
    # "mentions warfarin" [-10] is a mistake both avoid: it passes everywhere.
    assert (t.rows, t.criteria, t.skipped) == (6, 24, 0)
    assert (t.reference_passes, t.fails_all, t.fails_some) == (24, 3, 3)
    assert t.share_fails_all == pytest.approx(3 / 24)
    assert [(e.row, e.index, e.criterion, e.weight) for e in t.examples] == [
        (0, 1, "mentions 81 mg", 100.0),
        (2, 1, "mentions 81 mg", 100.0),
        (4, 1, "mentions 81 mg", 100.0),
    ]
    assert t.examples[0].reference == 1.0 and t.examples[0].teacher == [0.0, 0.0]
    assert report.flags == []  # 12.5% is under the 20% flag
    text = render_rubric_score_report(report)
    assert "teacher: toy-teacher" in text and "teacher ×2" in text
    assert (
        "per criterion: teacher ×2 fails 3 of the 24 criteria reference passes" in text
    )
    assert "teacher fails: row 0 criterion 1 [100]" in text and "mentions 81 mg" in text


def test_teacher_flag_and_one_reply():
    """With one reply, even rows fail the dose criterion: 3 of 12, flagged."""
    rows = _rubric_rows()
    for r in rows:
        r["rubric"] = r["rubric"][:2]  # aspirin, dose: the dose misses are 3 of 12
    report = rubric_scores(
        rows,
        keyword_rubric,
        n=None,
        response_fields="reference",
        teacher=Teacher(),
        teacher_k=1,
        control_columns=False,
    )
    t = report.teacher_criteria
    assert (t.reference_passes, t.fails_all, t.fails_some) == (12, 3, 0)
    assert report.flags == [
        "teacher fails 3 of the 12 criteria reference passes (25%) on every reply: "
        "read them below"
    ]


def test_teacher_without_a_rubric_or_reference():
    """Without a rubric field the teacher is a column paired with the reference."""
    rows = [{k: v for k, v in r.items() if k != "rubric"} for r in _rubric_rows()]
    report = rubric_scores(
        rows,
        lambda resp, row: float("81 mg" in resp),
        n=None,
        response_fields="reference",
        teacher=Teacher(),
        control_columns=False,
    )
    assert report.teacher_criteria is None
    paired = report.column(
        "teacher"
    ).vs_reference  # odd rows: one reply of two names the dose
    assert (paired.below, paired.mean_gap) == (6, pytest.approx(0.75))
    alone = rubric_scores(
        _rubric_rows(),
        keyword_rubric,
        n=None,
        response_fields=None,
        teacher=Teacher(),
        control_columns=False,
    )
    assert alone.teacher_criteria is None and alone.column("teacher").mean is not None
    with pytest.raises(ValueError, match="teacher_k"):
        rubric_scores(_rubric_rows(), keyword_rubric, teacher=Teacher(), teacher_k=0)
    with pytest.raises(ValueError, match="clash"):
        rubric_scores(_rubric_rows(), keyword_rubric, response_fields="teacher")


def test_a_sampled_column_near_one_is_flagged():
    """A student that already scores 0.9 or more leaves RL little to raise."""

    class Copier(Teacher):
        def cache_key(self):
            return ("Copier",)

        def generate_batch(self, prompts, n=1, **kwargs):
            return [["aspirin 81 mg, bleeding"] * n for _ in prompts]

    report = rubric_scores(
        _rubric_rows(),
        keyword_rubric,
        n=None,
        response_fields="reference",
        model=Copier(),
        k=1,
        control_columns=False,
    )
    assert report.column("sampled").mean >= 0.9
    assert any(f.startswith("sampled scores") for f in report.flags)


def test_teacher_check_flips_mistake_criteria():
    """A mistake every teacher reply makes and the reference avoids is listed."""
    rows = [
        {
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "reference": "Give aspirin 81 mg; watch for bleeding.",
            "rubric": [
                ["mentions aspirin", 50],
                ["mentions reasonable", -75],  # every teacher reply says it
                ["mentions daily", -25],  # only odd rows' first reply says it
                ["mentions warfarin", 0],
            ],
        }
        for i in range(4)
    ]
    report = rubric_scores(
        rows,
        keyword_rubric,
        n=None,
        response_fields="reference",
        teacher=Teacher(),
        control_columns=False,
    )
    t = report.teacher_criteria
    assert (t.rows, t.criteria, t.skipped) == (4, 12, 4)
    assert (t.reference_passes, t.fails_all, t.fails_some) == (12, 4, 2)
    assert [(e.row, e.criterion, e.weight) for e in t.examples] == [
        (i, "mentions reasonable", -75.0) for i in range(4)
    ]
    assert t.examples[0].reference == 0.0 and t.examples[0].teacher == [1.0, 1.0]
    text = render_rubric_score_report(report)
    assert "4 with weight 0 skipped" in text
    assert "teacher fails: row 0 criterion 1 [-75]" in text


def test_teacher_check_lists_every_failing_criterion():
    """``failed`` holds all criteria every teacher reply fails; ``examples`` five."""
    report = rubric_scores(
        _rubric_rows(12),
        keyword_rubric,
        n=None,
        response_fields=None,
        reference_field="reference",
        teacher=Teacher(),
        teacher_k=1,
        control_columns=False,
    )
    t = report.teacher_criteria
    assert t.reference == "reference"
    assert [c.name for c in report.columns] == ["reference", "teacher"]
    assert [(e.row, e.index) for e in t.failed] == [(r, 1) for r in range(0, 12, 2)]
    assert t.examples == t.failed[:5] and len(t.failed) == 6


def test_reference_field_comes_first():
    """reference_field is the reference even when listed later in response_fields."""
    rows = [dict(r, weak="aspirin only") for r in _rubric_rows(4)]
    report = rubric_scores(
        rows,
        keyword_rubric,
        n=None,
        response_fields=["weak", "reference"],
        reference_field="reference",
        control_columns=False,
    )
    assert [c.name for c in report.columns] == ["reference", "weak"]
    with pytest.raises(TypeError, match="reference_field"):
        rubric_scores(rows, keyword_rubric, reference_field="")

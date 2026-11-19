"""rubric_pair_accuracy counts how often a rubric ranks the better text higher."""

from __future__ import annotations

import pytest

from pyforger.inspect import rubric_pair_accuracy


def keyword_rubric(response: str, row: dict) -> float:
    """sum(w * mentions) / sum(positive w), clipped; criteria are "mentions WORD"."""
    pairs = row["rubric"]
    total = sum(w for _, w in pairs if w > 0) or 1
    met = sum(w for c, w in pairs if c.split(" ", 1)[1] in response)
    return min(1.0, max(0.0, met / total))


def _rows():
    return [
        {
            "rubric": [
                ["mentions aspirin", 2],
                ["mentions dose", 1],
                ["mentions warfarin", -1],
            ],
            "better": "aspirin at a low dose",
            "worse": ["aspirin", "warfarin dose", "aspirin at a low dose"],
        },
        {
            "rubric": [["mentions rest", 1]],
            "better": ["rest", None],
            "worse": "rest and fluids",
        },
    ]


def test_wins_ties_losses_and_margin():
    """Row 0: two wins and a tie between identical texts; row 1: a tie."""
    report = rubric_pair_accuracy(_rows(), keyword_rubric, per_criterion=False)
    outcomes = [(p.row, p.better_index, p.worse_index, p.outcome) for p in report.pairs]
    assert outcomes == [
        (0, 0, 0, "win"),  # 1.0 vs 2/3
        (0, 0, 1, "win"),  # 1.0 vs (1 - 1) / 3 = 0
        (0, 0, 2, "tie"),  # identical texts
        (1, 0, 0, "tie"),  # both mention rest
    ]
    assert (report.wins, report.ties, report.losses, report.missing) == (2, 2, 0, 1)
    assert report.accuracy == pytest.approx(3 / 4)
    assert report.mean_margin == pytest.approx((1 / 3 + 1) / 4)
    assert report.losing_pairs() == []


def test_losing_pairs_and_wrong_way_criteria():
    """A criterion the worse text meets and the better one misses is wrong-way."""
    rows = [
        {
            "rubric": [
                ["mentions 81 mg", 1],
                ["mentions bleeding", 1],
                ["mentions warfarin", -1],
            ],
            "better": "watch bleeding",
            "worse": ["81 mg", "81 mg with warfarin, watch bleeding"],
        }
    ]
    report = rubric_pair_accuracy(rows, keyword_rubric)
    assert [p.outcome for p in report.pairs] == ["tie", "tie"]
    by_index = {c.index: (c.right, c.wrong, c.same) for c in report.criteria}
    assert by_index == {0: (0, 2, 0), 1: (1, 0, 1), 2: (1, 0, 1)}
    assert [(c.row, c.index) for c in report.wrong_way_criteria()] == [(0, 0)]
    rows[0]["better"] = "x"
    report = rubric_pair_accuracy(rows, keyword_rubric, per_criterion=False)
    assert [p.worse_index for p in report.losing_pairs()] == [0, 1]


def test_repeats_grade_fresh_and_record_noise():
    """With repeats > 1, each distinct text is regraded and its spread reported."""
    calls = []

    def noisy(response, row):
        calls.append(response)
        return 1.0 if len(calls) % 2 else 0.0

    rows = [{"better": "a", "worse": ["b", "a"]}]
    report = rubric_pair_accuracy(rows, noisy, repeats=2, per_criterion=False)
    assert sorted(calls) == ["a", "a", "b", "b"]  # each distinct text twice
    assert report.identical_response_gaps == [1.0, 1.0]
    assert all(len(p.better_scores) == 2 for p in report.pairs)
    assert report.rows_without_rubric == []
    report = rubric_pair_accuracy(rows, noisy, repeats=2)
    assert report.rows_without_rubric == [0]


def test_validation():
    """Bad arguments fail before any grading."""
    with pytest.raises(ValueError, match="repeats"):
        rubric_pair_accuracy([], keyword_rubric, repeats=0)
    with pytest.raises(ValueError, match="epsilon"):
        rubric_pair_accuracy([], keyword_rubric, epsilon=-1)
    with pytest.raises(ValueError, match="differ"):
        rubric_pair_accuracy([], keyword_rubric, better_field="x", worse_field="x")
    with pytest.raises(TypeError, match="better"):
        rubric_pair_accuracy([{"better": 3, "worse": "a"}], keyword_rubric)


def test_max_grades_raises_before_any_call():
    """The plan counts distinct texts and lone criteria; above max_grades, no call."""
    calls = []

    def grade(response, row):
        calls.append(response)
        return 1.0

    rows = [
        {
            "rubric": [["mentions a", 1], ["mentions b", 0]],
            "better": "a",
            "worse": ["b", "c"],
        }
    ]
    # 3 texts, plus 3 for the one nonzero criterion, x 2 repeats = 12.
    with pytest.raises(ValueError, match="up to 12 times"):
        rubric_pair_accuracy(rows, grade, repeats=2, max_grades=11)
    assert calls == []
    rubric_pair_accuracy(rows, grade, repeats=2, max_grades=12)
    assert len(calls) == 12
    with pytest.raises(ValueError, match="max_grades"):
        rubric_pair_accuracy(rows, grade, max_grades=0)


def test_epsilon_counts_small_margins_as_ties():
    """A margin of 0.02 is a tie at the default 0.05 and a win at epsilon=0."""
    rows = [{"better": "aa", "worse": "a"}]

    def grade(response, row):
        return 0.5 + 0.01 * len(response)

    assert rubric_pair_accuracy(rows, grade).pairs[0].outcome == "tie"
    report = rubric_pair_accuracy(rows, grade, epsilon=0)
    assert report.pairs[0].outcome == "win" and report.epsilon == 0

"""Sensitivity checks preserve evidence and separate rewrites from judge noise."""

from __future__ import annotations

import copy
import json

import pytest

from pyforger.alter import paraphrase
from pyforger.inspect import rubric_sensitivity
from pyforger.op import op


class Rewriter:
    """Offline model with observable prompts and configurable completions."""

    def __init__(self, choices=None):
        """Use two conservative rewrites unless choices are supplied."""
        self.choices = (
            choices if choices is not None else ["It is blue.", "Blue it is."]
        )
        self.calls = 0
        self.prompts = []

    def cache_key(self):
        """Keep test model state out of the cache key."""
        return ("rewriter", self.choices)

    def generate_batch(self, prompts, n=1, **kwargs):
        """Record the payload and return fixed text without any API calls."""
        self.calls += 1
        self.prompts = prompts
        return [self.choices[:n] for _ in prompts]


def _rows():
    """Give equivalent candidate texts different surface forms."""
    return [{"prompt": "What color?", "response": "Blue.", "variants": ["It is blue."]}]


def _surface_grade(response, row):
    """A deliberately oversensitive reward."""
    return float(response.startswith("Blue"))


def test_paraphrase_preserves_rows_and_hides_rubric():
    """The rewriter sees only the task and response, never gold or rubric."""
    rows = [
        {"prompt": "Color?", "response": "Blue.", "rubric": "secret", "gold": "hidden"}
    ]
    saved = copy.deepcopy(rows)
    model = Rewriter()
    out = paraphrase(rows, model=model)
    assert rows == saved and "paraphrases" not in rows[0]
    assert out[0]["paraphrases"] == ["It is blue.", "Blue it is."]
    assert json.loads(model.prompts[0][1]["content"]) == {
        "task": "Color?",
        "response": "Blue.",
    }
    assert paraphrase(rows, model=model) == out
    assert model.calls == 1


def test_paraphrase_rejects_overwrites_and_malformed_groups():
    """A bad generation shape cannot silently truncate the dataset."""
    with pytest.raises(ValueError, match="new"):
        paraphrase(_rows(), model=Rewriter(), output_field="response")
    with pytest.raises(ValueError, match="k completions"):
        paraphrase(_rows(), model=Rewriter(["one"]), k=2)
    with pytest.raises(ValueError, match="positive integer"):
        paraphrase(_rows(), model=Rewriter(), k=True)


def test_truncated_or_blank_paraphrases_are_missing():
    """A truncated generation is not a valid perturbation."""

    class Completions(list):
        finish_reasons = ["length", "stop"]

    class Truncated(Rewriter):
        def generate_batch(self, prompts, n=1, **kwargs):
            """Return one truncated and one empty completion."""
            return [Completions(["Unfinished", " "]) for _ in prompts]

    out = paraphrase(_rows(), model=Truncated())
    assert out[0]["paraphrases"] == [None, None]


def test_surface_changes_detected_with_original_row_indices():
    """A lexical reward changes on every equivalent pair in a seeded sample."""
    rows = _rows() * 9
    saved = copy.deepcopy(rows)
    report = rubric_sensitivity(rows, _surface_grade, variants_field="variants", n=3)
    assert report.compared == report.changed == 3
    assert report.candidate_tie_rate == 0
    assert report.verified_tie_rate is None
    assert report.max_abs_gap == report.mean_abs_gap == 1
    assert [p.row for p in report.pairs] == report.rows
    assert report.identical_response_gaps == [0] * 6
    assert rows == saved
    assert json.loads(report.model_dump_json())["compared"] == 3


def test_equivalence_gate_excludes_changed_meaning():
    """Rejected pairs remain visible but do not count as reward sensitivity."""
    rows = _rows()
    rows[0]["variants"] += ["Red."]
    report = rubric_sensitivity(
        rows,
        _surface_grade,
        variants_field="variants",
        equivalent=lambda a, b, row: "blue" in b.lower(),
    )
    assert report.rejected == 1 and report.verified == 1
    assert report.compared == 1 and len(report.pairs) == 2
    assert report.verified_tie_rate == 0


def test_duplicate_unchanged_and_missing_variants_are_not_successes():
    """Copies cannot inflate the apparent specificity of a reward."""
    rows = _rows()
    rows[0]["variants"] = ["Blue.", "It is blue.", "It is blue.", None, ""]
    report = rubric_sensitivity(rows, _surface_grade, variants_field="variants")
    assert (report.unchanged, report.duplicates, report.missing) == (1, 1, 2)
    assert report.compared == 1


def test_no_comparisons_is_not_perfect_specificity():
    """Empty and wholly unchanged input yields no evidence."""
    report = rubric_sensitivity([], _surface_grade, variants_field="variants")
    assert report.candidate_tie_rate is None
    rows = _rows()
    rows[0]["variants"] = ["Blue."]
    report = rubric_sensitivity(rows, _surface_grade, variants_field="variants")
    assert report.compared == 0 and report.candidate_tie_rate is None


def test_identical_text_noise_is_measured_with_nested_cache_bypassed():
    """Cached grader results must not manufacture perfect repeatability."""
    calls = []

    @op(name="test.sensitivity.noisy_grade")
    def noisy(response, row):
        """Alternate scores on repeated calls to identical text."""
        calls.append(response)
        return float(len(calls) % 2)

    for _ in range(2):
        report = rubric_sensitivity(
            _rows(), noisy, variants_field="variants", repeats=2, workers=1
        )
        assert report.identical_response_gaps == [1, 1]
        assert report.candidate_tie_rate == 1
    assert len(calls) == 8


def test_missing_grades_do_not_turn_into_zero():
    """Partially scored groups cannot enter the paired comparison."""

    def missing(response, row):
        """The rewritten response cannot be graded."""
        if response != row["response"]:
            raise ValueError("unavailable")
        return 1.0

    report = rubric_sensitivity(_rows(), missing, variants_field="variants")
    assert report.compared == 0 and report.candidate_tie_rate is None
    assert len(report.errors) == 2
    assert report.pairs[0].candidate_scores == [None, None]


def test_tolerance_is_explicit_and_does_not_threshold_each_grade():
    """A small fractional change is a tie only at the requested tolerance."""

    def fractional(response, row):
        """Use continuous scores as the RL judge currently does."""
        return 0.7 if response == row["response"] else 0.72

    a = rubric_sensitivity(_rows(), fractional, variants_field="variants", epsilon=0.03)
    b = rubric_sensitivity(_rows(), fractional, variants_field="variants", epsilon=0.01)
    assert a.changed == 0 and b.changed == 1


def test_inspection_can_generate_variants():
    """The convenience model argument runs the separate cached paraphrase op."""
    report = rubric_sensitivity(_rows(), _surface_grade, model=Rewriter())
    assert report.compared == 2 and report.changed == 1


@pytest.mark.parametrize(
    "kwargs", [{}, {"epsilon": -1}, {"epsilon": float("nan")}, {"repeats": 0}]
)
def test_invalid_arguments_fail_before_grading(kwargs):
    """Reject ambiguous inputs and invalid measurement parameters."""
    if kwargs:
        kwargs["variants_field"] = "variants"
    with pytest.raises(ValueError):
        rubric_sensitivity(_rows(), _surface_grade, **kwargs)


def _rubric_grade(response, row):
    """A signed reward with a deliberately lexical blue criterion."""
    predicates = {
        "blue": float(response.startswith("Blue")),
        "not_blue": float(not response.startswith("Blue")),
        "always": 1.0,
    }
    rubric = row["rubric"]
    return max(
        0.0,
        min(
            1.0,
            sum(w * predicates[c] for c, w in rubric)
            / sum(w for _, w in rubric if w > 0),
        ),
    )


def test_pair_rubric_and_criterion_rates_have_distinct_denominators():
    """Four pairs, two rubrics and four criterion occurrences are separate units."""
    rows = [
        {
            "response": "Blue yes",
            "variants": ["blue yes", "Blue YES"],
            "rubric": [["blue", 1], ["always", 1]],
        },
        {
            "response": "Red yes",
            "variants": ["red yes", "Red YES"],
            "rubric": [["always", 1], ["always", 1]],
        },
    ]
    report = rubric_sensitivity(rows, _rubric_grade, variants_field="variants")
    assert report.rubric_pair_changes.model_dump() == {
        "changed": 1,
        "compared": 4,
        "fraction": 0.25,
    }
    assert report.rubric_changes.model_dump() == {
        "changed": 1,
        "compared": 2,
        "fraction": 0.5,
    }
    assert report.criterion_changes.model_dump() == {
        "changed": 1,
        "compared": 4,
        "fraction": 0.25,
    }
    found = report.sensitive_criteria()
    assert [(c.row, c.index, c.criterion) for c in found] == [(0, 0, "blue")]
    assert found[0].pairs[0].original_scores == [1, 1]
    assert found[0].pairs[0].candidate_scores == [0, 0]
    assert report.sensitive_criteria(verified_only=True) == []
    assert report.verified_criterion_changes.fraction is None


def test_criterion_changes_cannot_hide_behind_aggregate_cancellation():
    """Two opposing criterion flips leave the full rubric unchanged."""
    rows = [
        {
            "response": "Blue",
            "variants": ["blue"],
            "rubric": [["blue", 1], ["not_blue", 1]],
        }
    ]
    report = rubric_sensitivity(rows, _rubric_grade, variants_field="variants")
    assert report.rubric_changes.fraction == 0
    assert report.criterion_changes.fraction == 1
    assert len(report.sensitive_criteria()) == 2


def test_negative_criterion_is_scored_with_positive_unit_weight():
    """Isolated negative weights must not cause every score to clip to zero."""
    rows = [
        {
            "response": "Blue",
            "variants": ["blue"],
            "rubric": [["always", 1], ["blue", -2], ["always", 0]],
        }
    ]
    original = copy.deepcopy(rows)
    report = rubric_sensitivity(
        rows,
        _rubric_grade,
        variants_field="variants",
        equivalent=lambda a, b, row: True,
    )
    assert report.zero_weight_criteria == 1
    assert report.criterion_changes.fraction == 0.5
    assert report.verified_criterion_changes.fraction == 0.5
    found = report.sensitive_criteria(verified_only=True)
    assert len(found) == 1 and found[0].weight == -2 and found[0].index == 1
    assert found[0].pairs[0].gap == -1
    assert rows == original


def test_rejected_and_ungradable_criteria_do_not_count_as_stable():
    """A missing criterion comparison is absent from the denominator."""
    rows = [
        {
            "response": "Blue",
            "variants": ["blue"],
            "rubric": [["blue", 1], ["always", 1]],
        }
    ]

    def incomplete(response, row):
        """Fail only the isolated always criterion on the paraphrase."""
        if row["rubric"] == [["always", 1]] and response == "blue":
            return None
        return _rubric_grade(response, row)

    report = rubric_sensitivity(rows, incomplete, variants_field="variants")
    assert report.criterion_changes.compared == 1
    assert report.criterion_changes.changed == 1
    rejected = rubric_sensitivity(
        rows,
        _rubric_grade,
        variants_field="variants",
        equivalent=lambda a, b, row: False,
    )
    assert rejected.criterion_changes.fraction is None
    assert rejected.rubric_changes.fraction is None
    assert rejected.sensitive_criteria() == []
    assert len(rejected.criteria[0].pairs) == 1


def test_criterion_inspection_can_be_disabled_and_has_separate_tolerance():
    """Opting out spends no calls on isolated criteria; the tolerance is explicit."""
    rows = [
        {
            "response": "Blue",
            "variants": ["blue"],
            "rubric": [["blue", 1], ["always", 1]],
        }
    ]
    report = rubric_sensitivity(
        rows, _rubric_grade, variants_field="variants", per_criterion=False
    )
    assert report.criteria == [] and report.criterion_changes.fraction is None
    report = rubric_sensitivity(
        rows,
        _rubric_grade,
        variants_field="variants",
        epsilon=0.6,
        criterion_epsilon=0.1,
    )
    assert report.rubric_changes.fraction == 0
    assert report.criterion_changes.fraction == 0.5


def test_malformed_rubric_fails_before_paraphrase_generation():
    """The more expensive diagnostic validates input before spending model calls."""
    model = Rewriter()
    rows = [{"response": "Blue", "rubric": [["blue", float("nan")]]}]
    with pytest.raises(ValueError, match="weight must be finite"):
        rubric_sensitivity(rows, _rubric_grade, model=model)
    assert model.calls == 0


def test_sampled_criteria_keep_input_row_indices_and_missing_rubric_coverage():
    """Returned indices remain useful for downstream selection on the source rows."""
    rows = [
        {"response": "Blue", "variants": ["blue"], "rubric": [["blue", 1]]}
        for _ in range(10)
    ]
    report = rubric_sensitivity(rows, _rubric_grade, variants_field="variants", n=3)
    assert [c.row for c in report.criteria] == report.rows
    missing = rubric_sensitivity(_rows(), _surface_grade, variants_field="variants")
    assert missing.rows_without_rubric == [0]
    assert missing.criterion_changes.compared == 0


def test_sensitivity_max_grades_raises_before_paraphrasing():
    """1 row x (original + 2 rewrites) x 2 repeats = 6 calls; above max_grades, none."""
    model = Rewriter()
    with pytest.raises(ValueError, match="up to 6 times"):
        rubric_sensitivity(
            _rows(), lambda r, row: 1.0, model=model, per_criterion=False, max_grades=5
        )
    assert model.calls == 0

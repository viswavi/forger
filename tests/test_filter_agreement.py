"""Tests for agreement-based filter operations."""

import pytest

from pyforger.filter import by_exact_agreement, by_group_agreement, by_rouge_l_agreement
from pyforger.op import trace


class TestByExactAgreement:
    """by_exact_agreement keeps items where two fields are identical."""

    def test_keeps_matching_items(self):
        """Items where field_a == field_b are kept."""
        data = [
            {"output_1": "42", "output_2": "42"},
            {"output_1": "yes", "output_2": "no"},
        ]
        result = by_exact_agreement(data, field_a="output_1", field_b="output_2")
        assert result == [data[0]]

    def test_removes_disagreeing_items(self):
        """Items where field_a != field_b are removed."""
        data = [
            {"output_1": "cat", "output_2": "dog"},
            {"output_1": "red", "output_2": "blue"},
        ]
        result = by_exact_agreement(data, field_a="output_1", field_b="output_2")
        assert result == []

    def test_all_agree(self):
        """When all items agree, none are removed."""
        data = [
            {"output_1": "yes", "output_2": "yes"},
            {"output_1": "no", "output_2": "no"},
        ]
        result = by_exact_agreement(data, field_a="output_1", field_b="output_2")
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        data = []
        result = by_exact_agreement(data, field_a="a", field_b="b")
        assert result == []

    def test_case_sensitive(self):
        """Comparison is case-sensitive."""
        data = [{"output_1": "Yes", "output_2": "yes"}]
        result = by_exact_agreement(data, field_a="output_1", field_b="output_2")
        assert result == []

    def test_whitespace_ignored(self):
        """Leading/trailing whitespace is stripped before comparison."""
        data = [
            {"output_1": "hello ", "output_2": "hello"},
            {"output_1": " hello", "output_2": "hello"},
            {"output_1": "hello", "output_2": " hello"},
            {"output_1": "hello", "output_2": "hello "},
            {"output_1": "hello.", "output_2": "hello"},
            {"output_1": "hello", "output_2": "hello."},
        ]
        result = by_exact_agreement(data, field_a="output_1", field_b="output_2")
        assert len(result) == 4
        assert data[4] not in result
        assert data[5] not in result

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output_1": "a"}]
        with pytest.raises(KeyError, match="missing required field 'output_2'"):
            by_exact_agreement(data, field_a="output_1", field_b="output_2")


class TestByExactDisagreement:
    """by_exact_agreement with keep='disagree' keeps items that differ."""

    def test_keeps_disagreeing_items(self):
        """Items where field_a != field_b are kept."""
        data = [
            {"output_1": "42", "output_2": "42"},
            {"output_1": "yes", "output_2": "no"},
        ]
        result = by_exact_agreement(
            data,
            field_a="output_1",
            field_b="output_2",
            disagree=True,
        )
        assert result == [data[1]]

    def test_removes_agreeing_items(self):
        """Items where field_a == field_b are removed."""
        data = [
            {"output_1": "yes", "output_2": "yes"},
            {"output_1": "no", "output_2": "no"},
        ]
        result = by_exact_agreement(
            data,
            field_a="output_1",
            field_b="output_2",
            disagree=True,
        )
        assert result == []

    def test_all_disagree(self):
        """When all items disagree, none are removed."""
        data = [
            {"output_1": "cat", "output_2": "dog"},
            {"output_1": "red", "output_2": "blue"},
        ]
        result = by_exact_agreement(
            data,
            field_a="output_1",
            field_b="output_2",
            disagree=True,
        )
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        result = by_exact_agreement([], field_a="a", field_b="b", disagree=True)
        assert result == []


class TestByRougeLAgreement:
    """by_rouge_l_agreement keeps items where two fields are similar."""

    def test_keeps_similar_items(self):
        """Items with high ROUGE-L between fields are kept."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "the cat sat on a mat",
            },
        ]
        result = by_rouge_l_agreement(
            data, threshold=0.7, field_a="out_a", field_b="out_b"
        )
        assert result == data

    def test_removes_dissimilar_items(self):
        """Items with low ROUGE-L between fields are removed."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "a quick brown fox jumped",
            },
        ]
        result = by_rouge_l_agreement(
            data, threshold=0.7, field_a="out_a", field_b="out_b"
        )
        assert result == []

    def test_exact_match_always_kept(self):
        """Identical fields have ROUGE-L of 1.0 and are always kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        result = by_rouge_l_agreement(
            data, threshold=0.99, field_a="out_a", field_b="out_b"
        )
        assert result == data

    def test_threshold_affects_filtering(self):
        """A stricter threshold removes more items."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "the cat sat on a mat",
            },
        ]
        strict = by_rouge_l_agreement(
            data, threshold=0.99, field_a="out_a", field_b="out_b"
        )
        lenient = by_rouge_l_agreement(
            data, threshold=0.3, field_a="out_a", field_b="out_b"
        )
        assert len(strict) <= len(lenient)

    def test_mixed_agreement(self):
        """Only items meeting the threshold are kept."""
        data = [
            {"out_a": "the cat sat on the mat", "out_b": "the cat sat on the mat"},
            {"out_a": "hello world", "out_b": "completely different text"},
        ]
        result = by_rouge_l_agreement(
            data, threshold=0.7, field_a="out_a", field_b="out_b"
        )
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        data = []
        result = by_rouge_l_agreement(data, field_a="a", field_b="b")
        assert result == []

    def test_single_item_agreeing(self):
        """Single agreeing item is kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        result = by_rouge_l_agreement(data, field_a="out_a", field_b="out_b")
        assert result == data

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"out_a": "hello"}]
        with pytest.raises(KeyError, match="missing required field 'out_b'"):
            by_rouge_l_agreement(data, field_a="out_a", field_b="out_b")


class TestByRougeLDisagreement:
    """by_rouge_l_agreement keeps items where two fields are dissimilar."""

    def test_removes_similar_items(self):
        """Items with high ROUGE-L between fields are removed."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "the cat sat on a mat",
            },
        ]
        result = by_rouge_l_agreement(
            data, threshold=0.7, field_a="out_a", field_b="out_b", disagree=True
        )
        assert result == []

    def test_keeps_dissimilar_items(self):
        """Items with low ROUGE-L between fields are kept."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "a quick brown fox jumped",
            },
        ]
        result = by_rouge_l_agreement(
            data, threshold=0.7, field_a="out_a", field_b="out_b", disagree=True
        )
        assert result == data


def _agree(data, **kwargs):
    return by_group_agreement(
        data, group_by_input_field="input", match_output_field="output", **kwargs
    )


class TestByGroupAgreement:
    """by_group_agreement drops groups whose outputs disagree."""

    def test_keeps_unanimous_group(self):
        """A group whose outputs all match is kept in full."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "42"},
        ]
        assert _agree(data) == data

    def test_drops_conflicting_group(self):
        """A group with disagreeing outputs is dropped."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "43"},
        ]
        assert _agree(data) == []

    def test_mixed_groups(self):
        """Unanimous groups kept, conflicting groups dropped."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "x", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "y", "output": "c"},
            {"input": "z", "output": "d"},
        ]
        assert [r["input"] for r in _agree(data)] == ["x", "x", "z"]

    def test_preserves_input_order(self):
        """Kept items stay in input order when groups are interleaved."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "x", "output": "a"},
        ]
        assert _agree(data) == data

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert _agree([]) == []

    def test_whitespace_ignored(self):
        """Surrounding whitespace is stripped from both fields."""
        data = [
            {"input": "x", "output": "yes"},
            {"input": " x ", "output": " yes "},
        ]
        assert _agree(data) == data

    def test_missing_field_raises(self):
        """A missing output field raises an error."""
        with pytest.raises(KeyError, match="missing required field 'output'"):
            _agree([{"input": "x"}])

    def test_does_not_stringify_input(self):
        """Inputs that differ only by type are separate groups."""
        data = [
            {"input": 1, "output": "a"},
            {"input": "1", "output": "b"},
        ]
        assert _agree(data) == data

    @pytest.mark.parametrize(
        "a, b", [(1, "1"), (True, 1), (1, 1.0), ([True], [1]), ({"k": 1}, {"k": "1"})]
    )
    def test_differently_typed_outputs_disagree(self, a, b):
        """Outputs that differ in type disagree, even if == or str() match."""
        data = [
            {"input": "x", "output": a},
            {"input": "x", "output": b},
        ]
        assert _agree(data) == []

    def test_compares_unhashable_outputs(self):
        """List and dict outputs are compared element-wise."""
        data = [
            {"input": "x", "output": [1, 2]},
            {"input": "x", "output": [1, 2]},
            {"input": "y", "output": {"k": 1}},
            {"input": "y", "output": {"k": 2}},
        ]
        assert _agree(data) == data[:2]

    @pytest.mark.parametrize(
        "a, b",
        [
            ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
            ({"n": {"a": 1, "b": 2}}, {"n": {"b": 2, "a": 1}}),
            ([1, 2], (1, 2)),
            ([{"a": 1, "b": 2}], ({"b": 2, "a": 1},)),
            ({1, 2}, {2, 1}),
        ],
    )
    def test_ignores_container_ordering(self, a, b):
        """Reordered dict keys and list-vs-tuple outputs agree."""
        data = [
            {"input": "x", "output": a},
            {"input": "x", "output": b},
        ]
        assert _agree(data) == data

    @pytest.mark.parametrize(
        "a, b",
        [
            ({"k": 1}, [("k", 1)]),
            ({"k": 1}, [["k", 1]]),
            ({}, []),
            ({1}, ["__set__", [1]]),
        ],
    )
    def test_mapping_and_sequence_disagree(self, a, b):
        """A mapping (or set) never agrees with a sequence of the same content."""
        data = [
            {"input": "x", "output": a},
            {"input": "x", "output": b},
        ]
        assert _agree(data) == []

    @pytest.mark.parametrize("reverse", [False, True])
    def test_consistent_across_cache(self, reverse):
        """Inputs sharing an @op cache key give the same result in any order.

        ``@op`` sorts dict entries in its cache key, so these two inputs share
        a cache entry; whichever call runs second receives the first call's
        result, which must match what it would have computed itself.
        """
        ordered = [
            {"input": "x", "output": {"a": 1, "b": 2}},
            {"input": "x", "output": {"a": 1, "b": 2}},
        ]
        mixed = [
            {"input": "x", "output": {"a": 1, "b": 2}},
            {"input": "x", "output": {"b": 2, "a": 1}},
        ]
        calls = [mixed, ordered] if reverse else [ordered, mixed]
        for data in calls:
            uncached = _agree(data, _no_cache=True)
            assert _agree(data) == uncached == data
        # The second cached call must have been served from the first's entry.
        hits = [
            r.cache_hit for r in trace() if r.op_name == "filter.by_group_agreement"
        ]
        assert hits[-1]

    def test_max_count_caps_agreeing_groups(self):
        """max_count caps kept items per group, using the same grouping key."""
        data = [
            {"input": "x", "output": "a"},
            {"input": " x ", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "x", "output": "a"},
        ]
        assert _agree(data, max_count=1) == data[:1] + data[2:3]

    @pytest.mark.parametrize("max_count", [0, -1])
    def test_nonpositive_max_count_keeps_nothing(self, max_count):
        """Nonpositive max_count keeps nothing."""
        data = [{"input": "x", "output": "a"}] * 3
        assert _agree(data, max_count=max_count) == []

    def test_unhashable_inputs_are_grouped(self):
        """List and dict inputs group structurally instead of raising."""
        data = [
            {"input": [1, 2], "output": "a"},
            {"input": (1, 2), "output": "a"},
            {"input": {"k": 1}, "output": "b"},
            {"input": {"k": 1}, "output": "c"},
        ]
        assert _agree(data) == data[:2]

    def test_differently_typed_inputs_are_separate_groups(self):
        """True and 1 (or 1 and 1.0) are separate groups, each judged alone."""
        data = [
            {"input": True, "output": "a"},
            {"input": 1, "output": "b"},
            {"input": 1.0, "output": "c"},
        ]
        assert _agree(data) == data

    def test_equal_outputs_agree_regardless_of_aliasing(self):
        """Equal outputs agree whether or not they share sub-objects."""
        s = "x" * 100
        copy_of_s = "".join(["x"] * 100)
        assert s == copy_of_s and s is not copy_of_s
        data = [
            {"input": "x", "output": [s, s]},
            {"input": "x", "output": [s, copy_of_s]},
        ]
        assert _agree(data) == data

"""Tests for diversity-based filter operations."""

import pytest

from pyforger.filter import by_exact_match, by_n_gram_overlap, by_rouge_l


class TestByExactMatch:
    """by_exact_match removes items with duplicate field values."""

    def test_removes_duplicates(self):
        """Duplicate instructions are removed."""
        data = [
            {"instruction": "hello", "output": "a"},
            {"instruction": "hello", "output": "b"},
            {"instruction": "world", "output": "c"},
        ]
        source = data
        result = by_exact_match(source, field="instruction")
        assert result == [data[0], data[2]]

    def test_keeps_first_occurrence(self):
        """The first occurrence of a duplicate is kept."""
        data = [
            {"instruction": "same", "output": "first"},
            {"instruction": "same", "output": "second"},
        ]
        source = data
        result = by_exact_match(source, field="instruction")
        assert result == [data[0]]

    def test_all_unique(self):
        """When all items are unique, none are removed."""
        data = [
            {"instruction": "a", "output": "1"},
            {"instruction": "b", "output": "2"},
        ]
        source = data
        result = by_exact_match(source, field="instruction")
        assert result == data

    def test_all_duplicates_returns_single(self):
        """When all items are identical, only the first is kept."""
        data = [
            {"instruction": "same", "output": "x"},
            {"instruction": "same", "output": "y"},
            {"instruction": "same", "output": "z"},
        ]
        source = data
        result = by_exact_match(source, field="instruction")
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_exact_match(source, field="instruction")
        assert result == []

    def test_max_count_allows_multiple(self):
        """max_count allows up to N occurrences of each value."""
        data = [
            {"instruction": "same", "response": "a"},
            {"instruction": "same", "response": "b"},
            {"instruction": "same", "response": "c"},
            {"instruction": "same", "response": "d"},
            {"instruction": "other", "response": "e"},
        ]
        source = data
        result = by_exact_match(source, field="instruction", max_count=3)
        assert result == [data[0], data[1], data[2], data[4]]

    def test_max_count_one_is_default(self):
        """Default max_count=1 keeps only first occurrence."""
        data = [
            {"instruction": "same", "response": "a"},
            {"instruction": "same", "response": "b"},
        ]
        source = data
        result = by_exact_match(source, field="instruction")
        assert result == [data[0]]

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'instruction'"):
            by_exact_match(source, field="instruction")

    def test_agree_on_keeps_unanimous_group(self):
        """With agree_on set, a group with matching agree_on values is kept."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "42"},
        ]
        source = data
        result = by_exact_match(source, field="input", agree_on="output")
        assert result == [data[0]]

    def test_agree_on_drops_conflicting_group(self):
        """With agree_on set, a group with disagreeing agree_on values is dropped."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "43"},
        ]
        source = data
        result = by_exact_match(source, field="input", agree_on="output")
        assert result == []

    def test_agree_on_mixed_groups(self):
        """Unanimous groups kept, conflicting groups dropped."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "x", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "y", "output": "c"},
            {"input": "z", "output": "d"},
        ]
        source = data
        result = by_exact_match(source, field="input", agree_on="output")
        assert [r["input"] for r in result] == ["x", "z"]

    def test_agree_on_with_max_count(self):
        """max_count still caps kept items from unanimous groups."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "x", "output": "a"},
            {"input": "x", "output": "a"},
        ]
        source = data
        result = by_exact_match(source, field="input", max_count=2, agree_on="output")
        assert len(result) == 2

    def test_agree_on_whitespace_ignored(self):
        """Leading/trailing whitespace is stripped for both field and agree_on."""
        data = [
            {"input": "x", "output": "yes"},
            {"input": " x ", "output": " yes "},
        ]
        source = data
        result = by_exact_match(source, field="input", agree_on="output")
        assert len(result) == 1

    def test_agree_on_missing_field_raises(self):
        """Missing agree_on field raises an error."""
        data = [{"input": "x"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'output'"):
            by_exact_match(source, field="input", agree_on="output")

    def test_agree_on_does_not_stringify_field(self):
        """Values that differ only by type are separate groups."""
        data = [
            {"input": 1, "output": "a"},
            {"input": "1", "output": "b"},
        ]
        result = by_exact_match(data, field="input", agree_on="output")
        assert result == data

    def test_agree_on_does_not_stringify_outputs(self):
        """Outputs that differ only by type count as disagreeing."""
        data = [
            {"input": "x", "output": 1},
            {"input": "x", "output": "1"},
        ]
        assert by_exact_match(data, field="input", agree_on="output") == []

    @pytest.mark.parametrize(
        "a, b",
        [(True, 1), (1, 1.0), ([True], [1]), ({"k": False}, {"k": 0})],
    )
    def test_agree_on_equal_but_differently_typed_outputs_disagree(self, a, b):
        """Outputs that are == but differ in type (e.g. True vs 1) disagree."""
        data = [
            {"input": "x", "output": a},
            {"input": "x", "output": b},
        ]
        assert by_exact_match(data, field="input", agree_on="output") == []

    def test_agree_on_compares_unhashable_outputs_by_value(self):
        """List and dict outputs are compared by value."""
        data = [
            {"input": "x", "output": [1, 2]},
            {"input": "x", "output": [1, 2]},
            {"input": "y", "output": {"k": 1}},
            {"input": "y", "output": {"k": 2}},
        ]
        result = by_exact_match(data, field="input", agree_on="output")
        assert result == [data[0]]

    def test_agree_on_preserves_input_order(self):
        """Kept items stay in input order when groups are interleaved."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "x", "output": "a"},
        ]
        result = by_exact_match(data, field="input", max_count=2, agree_on="output")
        assert result == data

    @pytest.mark.parametrize("max_count", [0, -1])
    def test_nonpositive_max_count_keeps_nothing(self, max_count):
        """Nonpositive max_count keeps nothing, with or without agree_on."""
        data = [{"input": "x", "output": "a"}] * 3
        assert by_exact_match(data, field="input", max_count=max_count) == []
        result = by_exact_match(
            data, field="input", max_count=max_count, agree_on="output"
        )
        assert result == []


class TestByRougeL:
    """by_rouge_l removes near-duplicates by ROUGE-L similarity."""

    def test_removes_near_duplicates(self):
        """Items very similar to a kept item are removed."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "the cat sat on a mat"},
        ]
        source = data
        result = by_rouge_l(source, field="instruction", threshold=0.7)
        assert result == [data[0]]

    def test_keeps_dissimilar_items(self):
        """Items below the threshold are kept."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "a quick brown fox"},
        ]
        source = data
        result = by_rouge_l(source, field="instruction", threshold=0.7)
        assert result == data

    def test_threshold_affects_filtering(self):
        """A higher threshold keeps more items."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "the cat sat on a mat"},
        ]
        source = data
        strict = by_rouge_l(source, field="instruction", threshold=0.5)
        lenient = by_rouge_l(source, field="instruction", threshold=0.99)
        assert len(strict) <= len(lenient)

    def test_exact_duplicates_removed(self):
        """Exact duplicates have ROUGE-L of 1.0 and are removed."""
        data = [
            {"instruction": "hello world"},
            {"instruction": "hello world"},
        ]
        source = data
        result = by_rouge_l(source, field="instruction", threshold=0.7)
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_rouge_l(source, field="instruction")
        assert result == []

    def test_single_item(self):
        """Single item is always kept."""
        data = [{"instruction": "hello world"}]
        source = data
        result = by_rouge_l(source, field="instruction")
        assert result == data

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'instruction'"):
            by_rouge_l(source, field="instruction")


class TestByNGramOverlap:
    """by_n_gram_overlap removes items with high n-gram overlap."""

    def test_removes_high_overlap(self):
        """Items with most n-grams already in pool are removed."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "the quick brown fox leaps"},
        ]
        source = data
        result = by_n_gram_overlap(source, field="instruction", threshold=0.5, n=2)
        assert result == [data[0]]

    def test_keeps_low_overlap(self):
        """Items with low n-gram overlap are kept."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "a lazy dog sleeps outside"},
        ]
        source = data
        result = by_n_gram_overlap(source, field="instruction", threshold=0.5, n=2)
        assert result == data

    def test_threshold_affects_filtering(self):
        """A higher threshold keeps more items."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "the quick brown fox leaps"},
        ]
        source = data
        strict = by_n_gram_overlap(source, field="instruction", threshold=0.2, n=2)
        lenient = by_n_gram_overlap(source, field="instruction", threshold=0.99, n=2)
        assert len(strict) <= len(lenient)

    def test_n_value_affects_filtering(self):
        """Different n values produce different results."""
        data = [
            {"instruction": "a b c d e f"},
            {"instruction": "a b c x y z"},
        ]
        source = data
        bigram = by_n_gram_overlap(source, field="instruction", threshold=0.3, n=2)
        trigram = by_n_gram_overlap(source, field="instruction", threshold=0.3, n=3)
        assert len(bigram) <= len(trigram)

    def test_short_text_padded(self):
        """Text shorter than n is padded and participates in overlap checking."""
        data = [
            {"instruction": "Yes"},
            {"instruction": "Yes"},
        ]
        source = data
        result = by_n_gram_overlap(source, field="instruction", threshold=0.5, n=2)
        assert result == [data[0]]

    def test_short_text_different_values_kept(self):
        """Different short texts are kept despite padding."""
        data = [
            {"instruction": "hello world foo bar baz"},
            {"instruction": "hi"},
        ]
        source = data
        result = by_n_gram_overlap(source, field="instruction", threshold=0.5, n=3)
        assert len(result) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_n_gram_overlap(source, field="instruction")
        assert result == []

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'instruction'"):
            by_n_gram_overlap(source, field="instruction")

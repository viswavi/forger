"""Tests for diversity-based filter operations."""

import pytest
from fixtures import make_source

from pyforger import run
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
        source = make_source(data)
        result = run(by_exact_match(source, field="instruction"))
        assert result == [data[0], data[2]]

    def test_keeps_first_occurrence(self):
        """The first occurrence of a duplicate is kept."""
        data = [
            {"instruction": "same", "output": "first"},
            {"instruction": "same", "output": "second"},
        ]
        source = make_source(data)
        result = run(by_exact_match(source, field="instruction"))
        assert result == [data[0]]

    def test_all_unique(self):
        """When all items are unique, none are removed."""
        data = [
            {"instruction": "a", "output": "1"},
            {"instruction": "b", "output": "2"},
        ]
        source = make_source(data)
        result = run(by_exact_match(source, field="instruction"))
        assert result == data

    def test_all_duplicates_returns_single(self):
        """When all items are identical, only the first is kept."""
        data = [
            {"instruction": "same", "output": "x"},
            {"instruction": "same", "output": "y"},
            {"instruction": "same", "output": "z"},
        ]
        source = make_source(data)
        result = run(by_exact_match(source, field="instruction"))
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_exact_match(source, field="instruction"))
        assert result == []

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'instruction'"):
            run(by_exact_match(source, field="instruction"))


class TestByRougeL:
    """by_rouge_l removes near-duplicates by ROUGE-L similarity."""

    def test_removes_near_duplicates(self):
        """Items very similar to a kept item are removed."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "the cat sat on a mat"},
        ]
        source = make_source(data)
        result = run(by_rouge_l(source, field="instruction", threshold=0.7))
        assert result == [data[0]]

    def test_keeps_dissimilar_items(self):
        """Items below the threshold are kept."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "a quick brown fox"},
        ]
        source = make_source(data)
        result = run(by_rouge_l(source, field="instruction", threshold=0.7))
        assert result == data

    def test_threshold_affects_filtering(self):
        """A higher threshold keeps more items."""
        data = [
            {"instruction": "the cat sat on the mat"},
            {"instruction": "the cat sat on a mat"},
        ]
        source = make_source(data)
        strict = run(by_rouge_l(source, field="instruction", threshold=0.5))
        lenient = run(by_rouge_l(source, field="instruction", threshold=0.99))
        assert len(strict) <= len(lenient)

    def test_exact_duplicates_removed(self):
        """Exact duplicates have ROUGE-L of 1.0 and are removed."""
        data = [
            {"instruction": "hello world"},
            {"instruction": "hello world"},
        ]
        source = make_source(data)
        result = run(by_rouge_l(source, field="instruction", threshold=0.7))
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_rouge_l(source, field="instruction"))
        assert result == []

    def test_single_item(self):
        """Single item is always kept."""
        data = [{"instruction": "hello world"}]
        source = make_source(data)
        result = run(by_rouge_l(source, field="instruction"))
        assert result == data

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'instruction'"):
            run(by_rouge_l(source, field="instruction"))


class TestByNGramOverlap:
    """by_n_gram_overlap removes items with high n-gram overlap."""

    def test_removes_high_overlap(self):
        """Items with most n-grams already in pool are removed."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "the quick brown fox leaps"},
        ]
        source = make_source(data)
        result = run(by_n_gram_overlap(source, field="instruction", threshold=0.5, n=2))
        assert result == [data[0]]

    def test_keeps_low_overlap(self):
        """Items with low n-gram overlap are kept."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "a lazy dog sleeps outside"},
        ]
        source = make_source(data)
        result = run(by_n_gram_overlap(source, field="instruction", threshold=0.5, n=2))
        assert result == data

    def test_threshold_affects_filtering(self):
        """A higher threshold keeps more items."""
        data = [
            {"instruction": "the quick brown fox jumps"},
            {"instruction": "the quick brown fox leaps"},
        ]
        source = make_source(data)
        strict = run(by_n_gram_overlap(source, field="instruction", threshold=0.2, n=2))
        lenient = run(
            by_n_gram_overlap(source, field="instruction", threshold=0.99, n=2)
        )
        assert len(strict) <= len(lenient)

    def test_n_value_affects_filtering(self):
        """Different n values produce different results."""
        data = [
            {"instruction": "a b c d e f"},
            {"instruction": "a b c x y z"},
        ]
        source = make_source(data)
        bigram = run(by_n_gram_overlap(source, field="instruction", threshold=0.3, n=2))
        trigram = run(
            by_n_gram_overlap(source, field="instruction", threshold=0.3, n=3)
        )
        assert len(bigram) <= len(trigram)

    def test_short_text_kept(self):
        """Text shorter than n produces no n-grams and is kept."""
        data = [
            {"instruction": "hello world foo bar baz"},
            {"instruction": "hi"},
        ]
        source = make_source(data)
        result = run(by_n_gram_overlap(source, field="instruction", threshold=0.5, n=3))
        assert len(result) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_n_gram_overlap(source, field="instruction"))
        assert result == []

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output": "a"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'instruction'"):
            run(by_n_gram_overlap(source, field="instruction"))

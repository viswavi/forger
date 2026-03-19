"""Tests for agreement-based filter operations."""

import pytest
from fixtures import make_source

from pyforger import DataSpec, run
from pyforger.filter import by_exact_agreement, by_rouge_l_agreement


class TestByExactAgreement:
    """by_exact_agreement keeps items where two fields are identical."""

    def test_keeps_matching_items(self):
        """Items where field_a == field_b are kept."""
        data = [
            {"output_1": "42", "output_2": "42"},
            {"output_1": "yes", "output_2": "no"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == [data[0]]

    def test_removes_disagreeing_items(self):
        """Items where field_a != field_b are removed."""
        data = [
            {"output_1": "cat", "output_2": "dog"},
            {"output_1": "red", "output_2": "blue"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == []

    def test_all_agree(self):
        """When all items agree, none are removed."""
        data = [
            {"output_1": "yes", "output_2": "yes"},
            {"output_1": "no", "output_2": "no"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_exact_agreement(source, field_a="a", field_b="b"))
        assert result == []

    def test_case_sensitive(self):
        """Comparison is case-sensitive."""
        data = [{"output_1": "Yes", "output_2": "yes"}]
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == []

    def test_whitespace_matters(self):
        """Trailing whitespace causes mismatch."""
        data = [{"output_1": "hello", "output_2": "hello "}]
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == []

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output_1": "a"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="output_2"):
            run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))

    def test_returns_dataspec(self):
        """by_exact_agreement returns a DataSpec node."""
        source = make_source([])
        assert isinstance(
            by_exact_agreement(source, field_a="a", field_b="b"), DataSpec
        )


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
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.7, field_a="out_a", field_b="out_b"
            )
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
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.7, field_a="out_a", field_b="out_b"
            )
        )
        assert result == []

    def test_exact_match_always_kept(self):
        """Identical fields have ROUGE-L of 1.0 and are always kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.99, field_a="out_a", field_b="out_b"
            )
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
        source = make_source(data)
        strict = run(
            by_rouge_l_agreement(
                source, threshold=0.99, field_a="out_a", field_b="out_b"
            )
        )
        lenient = run(
            by_rouge_l_agreement(
                source, threshold=0.3, field_a="out_a", field_b="out_b"
            )
        )
        assert len(strict) <= len(lenient)

    def test_mixed_agreement(self):
        """Only items meeting the threshold are kept."""
        data = [
            {"out_a": "the cat sat on the mat", "out_b": "the cat sat on the mat"},
            {"out_a": "hello world", "out_b": "completely different text"},
        ]
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.7, field_a="out_a", field_b="out_b"
            )
        )
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_rouge_l_agreement(source, field_a="a", field_b="b"))
        assert result == []

    def test_single_item_agreeing(self):
        """Single agreeing item is kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        source = make_source(data)
        result = run(by_rouge_l_agreement(source, field_a="out_a", field_b="out_b"))
        assert result == data

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"out_a": "hello"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="out_b"):
            run(by_rouge_l_agreement(source, field_a="out_a", field_b="out_b"))

    def test_returns_dataspec(self):
        """by_rouge_l_agreement returns a DataSpec node."""
        source = make_source([])
        assert isinstance(
            by_rouge_l_agreement(source, field_a="a", field_b="b"), DataSpec
        )

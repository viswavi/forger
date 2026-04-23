"""Tests for agreement-based filter operations."""

import pytest

from pyforger import run
from pyforger.acquire import from_list
from pyforger.filter import by_exact_agreement, by_rouge_l_agreement


class TestByExactAgreement:
    """by_exact_agreement keeps items where two fields are identical."""

    def test_keeps_matching_items(self):
        """Items where field_a == field_b are kept."""
        data = [
            {"output_1": "42", "output_2": "42"},
            {"output_1": "yes", "output_2": "no"},
        ]
        source = from_list(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == [data[0]]

    def test_removes_disagreeing_items(self):
        """Items where field_a != field_b are removed."""
        data = [
            {"output_1": "cat", "output_2": "dog"},
            {"output_1": "red", "output_2": "blue"},
        ]
        source = from_list(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == []

    def test_all_agree(self):
        """When all items agree, none are removed."""
        data = [
            {"output_1": "yes", "output_2": "yes"},
            {"output_1": "no", "output_2": "no"},
        ]
        source = from_list(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = from_list([])
        result = run(by_exact_agreement(source, field_a="a", field_b="b"))
        assert result == []

    def test_case_sensitive(self):
        """Comparison is case-sensitive."""
        data = [{"output_1": "Yes", "output_2": "yes"}]
        source = from_list(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
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
        source = from_list(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert len(result) == 4
        assert data[4] not in result
        assert data[5] not in result

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output_1": "a"}]
        source = from_list(data)
        with pytest.raises(RuntimeError, match="missing required field 'output_2'"):
            run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))


class TestByExactDisagreement:
    """by_exact_agreement with keep='disagree' keeps items that differ."""

    def test_keeps_disagreeing_items(self):
        """Items where field_a != field_b are kept."""
        data = [
            {"output_1": "42", "output_2": "42"},
            {"output_1": "yes", "output_2": "no"},
        ]
        source = make_source(data)
        result = run(
            by_exact_agreement(
                source, field_a="output_1", field_b="output_2", disagree=True
            )
        )
        assert result == [data[1]]

    def test_removes_agreeing_items(self):
        """Items where field_a == field_b are removed."""
        data = [
            {"output_1": "yes", "output_2": "yes"},
            {"output_1": "no", "output_2": "no"},
        ]
        source = make_source(data)
        result = run(
            by_exact_agreement(
                source, field_a="output_1", field_b="output_2", disagree=True
            )
        )
        assert result == []

    def test_all_disagree(self):
        """When all items disagree, none are removed."""
        data = [
            {"output_1": "cat", "output_2": "dog"},
            {"output_1": "red", "output_2": "blue"},
        ]
        source = make_source(data)
        result = run(
            by_exact_agreement(
                source, field_a="output_1", field_b="output_2", disagree=True
            )
        )
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(
            by_exact_agreement(source, field_a="a", field_b="b", disagree=True)
        )
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
        source = from_list(data)
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
        source = from_list(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.7, field_a="out_a", field_b="out_b"
            )
        )
        assert result == []

    def test_exact_match_always_kept(self):
        """Identical fields have ROUGE-L of 1.0 and are always kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        source = from_list(data)
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
        source = from_list(data)
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
        source = from_list(data)
        result = run(
            by_rouge_l_agreement(
                source, threshold=0.7, field_a="out_a", field_b="out_b"
            )
        )
        assert result == [data[0]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = from_list([])
        result = run(by_rouge_l_agreement(source, field_a="a", field_b="b"))
        assert result == []

    def test_single_item_agreeing(self):
        """Single agreeing item is kept."""
        data = [{"out_a": "hello world", "out_b": "hello world"}]
        source = from_list(data)
        result = run(by_rouge_l_agreement(source, field_a="out_a", field_b="out_b"))
        assert result == data

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"out_a": "hello"}]
        source = from_list(data)
        with pytest.raises(RuntimeError, match="missing required field 'out_b'"):
            run(by_rouge_l_agreement(source, field_a="out_a", field_b="out_b"))


class TestByRougeLDisagreement:
    """by_rouge_l_agreement with keep='disagree' keeps dissimilar items."""

    def test_keeps_dissimilar_items(self):
        """Items with ROUGE-L below threshold are kept."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "a quick brown fox jumped",
            },
        ]
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source,
                threshold=0.7,
                field_a="out_a",
                field_b="out_b",
                disagree=True,
            )
        )
        assert result == data

    def test_removes_similar_items(self):
        """Items with ROUGE-L at or above threshold are removed."""
        data = [
            {
                "out_a": "the cat sat on the mat",
                "out_b": "the cat sat on a mat",
            },
        ]
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source,
                threshold=0.7,
                field_a="out_a",
                field_b="out_b",
                disagree=True,
            )
        )
        assert result == []

    def test_mixed_keeps_only_disagreeing(self):
        """From a mix, only items below threshold survive."""
        data = [
            {"out_a": "the cat sat on the mat", "out_b": "the cat sat on the mat"},
            {"out_a": "hello world", "out_b": "completely different text"},
        ]
        source = make_source(data)
        result = run(
            by_rouge_l_agreement(
                source,
                threshold=0.7,
                field_a="out_a",
                field_b="out_b",
                disagree=True,
            )
        )
        assert result == [data[1]]

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(
            by_rouge_l_agreement(source, field_a="a", field_b="b", disagree=True)
        )
        assert result == []

"""Tests for agreement-based filter operations."""

import pytest
from fixtures import make_source

from pyforger import run
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
        source = make_source(data)
        result = run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))
        assert len(result) == 4
        assert data[4] not in result
        assert data[5] not in result

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"output_1": "a"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'output_2'"):
            run(by_exact_agreement(source, field_a="output_1", field_b="output_2"))

    def test_rejects_mixed_modes(self):
        """Passing both within-item and group params raises ValueError."""
        source = make_source([])
        with pytest.raises(ValueError, match="within-item mode"):
            by_exact_agreement(source, field_a="a", field_b="b", group_by="k")

    def test_rejects_no_mode(self):
        """Passing no params raises ValueError."""
        source = make_source([])
        with pytest.raises(ValueError, match="within-item mode"):
            by_exact_agreement(source)


class TestByExactAgreementGroupMode:
    """by_exact_agreement group mode drops groups with conflicting agree_on values."""

    def test_unanimous_group_keeps_one(self):
        """A group whose agree_on values all match yields one representative."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "42"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, group_by="input", agree_on="output"))
        assert result == [data[0]]

    def test_conflicting_group_is_dropped(self):
        """A group with disagreeing agree_on values is dropped entirely."""
        data = [
            {"input": "x", "output": "42"},
            {"input": "x", "output": "43"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, group_by="input", agree_on="output"))
        assert result == []

    def test_mixed_groups(self):
        """Unanimous groups are kept, conflicting ones dropped."""
        data = [
            {"input": "x", "output": "a"},
            {"input": "x", "output": "a"},
            {"input": "y", "output": "b"},
            {"input": "y", "output": "c"},
            {"input": "z", "output": "d"},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, group_by="input", agree_on="output"))
        assert len(result) == 2
        keys = {r["input"] for r in result}
        assert keys == {"x", "z"}

    def test_group_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_exact_agreement(source, group_by="input", agree_on="output"))
        assert result == []

    def test_group_whitespace_ignored(self):
        """Leading/trailing whitespace on agree_on values is stripped."""
        data = [
            {"input": "x", "output": "yes"},
            {"input": "x", "output": " yes "},
        ]
        source = make_source(data)
        result = run(by_exact_agreement(source, group_by="input", agree_on="output"))
        assert len(result) == 1


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
        with pytest.raises(RuntimeError, match="missing required field 'out_b'"):
            run(by_rouge_l_agreement(source, field_a="out_a", field_b="out_b"))

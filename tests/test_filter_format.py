"""Tests for format-based filter operations."""

import pytest
from fixtures import make_source

from pyforger import DataSpec, run
from pyforger.filter import by_invalid_response, by_keyword_blocklist, by_length


class TestByLength:
    """by_length filters items by instruction character count."""

    def test_keeps_items_within_bounds(self):
        """Items with instruction length in [min_len, max_len] are kept."""
        data = [
            {"instruction": "short", "output": "ok"},  # 5 chars
            {"instruction": "a" * 20, "output": "ok"},  # 20 chars
            {"instruction": "a" * 50, "output": "ok"},  # 50 chars
        ]
        source = make_source(data)
        result = run(by_length(source, min_len=10, max_len=30))
        assert result == [data[1]]

    def test_removes_too_short(self):
        """Items shorter than min_len are removed."""
        data = [{"instruction": "hi", "output": "ok"}]
        source = make_source(data)
        result = run(by_length(source, min_len=10))
        assert result == []

    def test_removes_too_long(self):
        """Items longer than max_len are removed."""
        data = [{"instruction": "a" * 200, "output": "ok"}]
        source = make_source(data)
        result = run(by_length(source, max_len=100))
        assert result == []

    def test_inclusive_bounds(self):
        """Boundary values (exactly min_len and max_len) are kept."""
        data = [
            {"instruction": "a" * 10, "output": "ok"},
            {"instruction": "a" * 50, "output": "ok"},
        ]
        source = make_source(data)
        result = run(by_length(source, min_len=10, max_len=50))
        assert len(result) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_length(source))
        assert result == []

    def test_custom_instruction_field(self):
        """Custom field name is used for length check."""
        data = [
            {"prompt": "a" * 20, "output": "ok"},
            {"prompt": "hi", "output": "ok"},
        ]
        source = make_source(data)
        result = run(by_length(source, min_len=10, instruction_field="prompt"))
        assert result == [data[0]]

    def test_missing_field_raises_key_error(self):
        """Missing expected field raises KeyError."""
        data = [{"output": "ok"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="instruction"):
            run(by_length(source))

    def test_returns_dataspec(self):
        """by_length returns a DataSpec node."""
        source = make_source([])
        assert isinstance(by_length(source), DataSpec)


class TestByInvalidResponse:
    """by_invalid_response removes items with copied or empty outputs."""

    def test_removes_exact_copy(self):
        """Items where output exactly matches instruction are removed."""
        data = [{"instruction": "Tell me a joke", "output": "Tell me a joke"}]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_removes_empty_output(self):
        """Items with empty output are removed."""
        data = [{"instruction": "Tell me a joke", "output": ""}]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_removes_whitespace_only_output(self):
        """Items with whitespace-only output are removed."""
        data = [{"instruction": "Tell me a joke", "output": "   \n\t  "}]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_removes_output_substring_of_instruction(self):
        """Items where output is a similar-length substring of instruction are removed."""
        data = [
            {"instruction": "Tell me a funny joke", "output": "me a funny joke"}
        ]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_removes_similar_length_substring(self):
        """Items where one is a substring of the other with similar length are removed."""
        data = [{"instruction": "Tell me a funny joke", "output": "Tell me a funny"}]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_keeps_short_instruction_in_long_output(self):
        """Short instruction appearing in a long output is not a false positive."""
        data = [
            {
                "instruction": "Hi",
                "output": "Hi there! Here is a detailed explanation of the topic.",
            }
        ]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == data

    def test_removes_high_similarity(self):
        """Items with similarity above threshold are removed."""
        data = [
            {"instruction": "Tell me a joke about cats", "output": "Tell me a joke about dogs"}
        ]
        source = make_source(data)
        result = run(by_invalid_response(source, similarity_threshold=0.8))
        assert result == []

    def test_keeps_valid_response(self):
        """Items with genuinely different output are kept."""
        data = [
            {
                "instruction": "Tell me a joke",
                "output": "Why did the chicken cross the road? To get to the other side!",
            }
        ]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == data

    def test_keeps_no_matches_removed(self):
        """When all items are valid, none are removed."""
        data = [
            {"instruction": "What is 2+2?", "output": "4"},
            {"instruction": "Name a color", "output": "Blue"},
        ]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_invalid_response(source))
        assert result == []

    def test_custom_field_names(self):
        """Custom field names work correctly."""
        data = [
            {"prompt": "Hello", "response": "Hello"},
            {"prompt": "Hello", "response": "World"},
        ]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="prompt", output_field="response"
            )
        )
        assert result == [data[1]]

    def test_removes_none_output(self):
        """Items with None output are removed."""
        data = [{"instruction": "Tell me a joke", "output": None}]
        source = make_source(data)
        result = run(by_invalid_response(source))
        assert result == []

    def test_missing_field_raises_key_error(self):
        """Missing expected field raises KeyError."""
        data = [{"instruction": "Tell me a joke"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="output"):
            run(by_invalid_response(source))

    def test_returns_dataspec(self):
        """by_invalid_response returns a DataSpec node."""
        source = make_source([])
        assert isinstance(by_invalid_response(source), DataSpec)


class TestByKeywordBlocklist:
    """by_keyword_blocklist removes items containing blocked phrases."""

    def test_removes_default_blocklist_match(self):
        """Items matching default blocklist phrases are removed."""
        data = [
            {"instruction": "Do something bad", "output": "As an AI, I cannot do that."},
            {"instruction": "Tell a joke", "output": "Why did the chicken cross the road?"},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source))
        assert result == [data[1]]

    def test_case_insensitive_by_default(self):
        """Matching is case-insensitive by default."""
        data = [{"instruction": "test", "output": "AS AN AI, I refuse."}]
        source = make_source(data)
        result = run(by_keyword_blocklist(source))
        assert result == []

    def test_case_sensitive_when_enabled(self):
        """Case-sensitive matching only blocks exact case matches."""
        data = [
            {"instruction": "test", "output": "AS AN AI, I refuse."},
            {"instruction": "test", "output": "as an ai, I refuse."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, case_sensitive=True))
        assert result == [data[0]]

    def test_custom_blocklist(self):
        """Custom blocklist replaces the default."""
        data = [
            {"instruction": "test", "output": "This contains foo in it."},
            {"instruction": "test", "output": "This is clean."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, blocklist=["foo"]))
        assert result == [data[1]]

    def test_no_matches_keeps_all(self):
        """When no items match blocklist, all are kept."""
        data = [
            {"instruction": "test", "output": "Perfectly fine answer."},
            {"instruction": "test", "output": "Another good response."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source))
        assert result == data

    def test_all_filtered_returns_empty(self):
        """When all items match blocklist, returns empty list."""
        data = [
            {"instruction": "test", "output": "I cannot fulfill your request."},
            {"instruction": "test", "output": "I'm sorry, I cannot help."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source))
        assert result == []

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_keyword_blocklist(source))
        assert result == []

    def test_custom_output_field(self):
        """Custom output field name works correctly."""
        data = [
            {"instruction": "test", "response": "As an AI, I cannot."},
            {"instruction": "test", "response": "Sure, here you go."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, output_field="response"))
        assert result == [data[1]]

    def test_returns_dataspec(self):
        """by_keyword_blocklist returns a DataSpec node."""
        source = make_source([])
        assert isinstance(by_keyword_blocklist(source), DataSpec)

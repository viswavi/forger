"""Tests for format-based filter operations."""

import pytest
from fixtures import make_source

from pyforger import run
from pyforger.filter import by_invalid_response, by_keyword_blocklist, by_length


class TestByLength:
    """by_length filters items by instruction word count."""

    def test_keeps_items_within_bounds(self):
        """Items with instruction word count in [min_len, max_len] are kept."""
        data = [
            {"instruction": "too short", "response": "ok"},  # 2 words
            {"instruction": " ".join(["word"] * 5), "response": "ok"},  # 5 words
            {"instruction": " ".join(["word"] * 15), "response": "ok"},  # 15 words
        ]
        source = make_source(data)
        result = run(
            by_length(source, instruction_field="instruction", min_len=3, max_len=10)
        )
        assert result == [data[1]]

    def test_removes_too_short(self):
        """Items with fewer words than min_len are removed."""
        data = [{"instruction": "hi", "response": "ok"}]
        source = make_source(data)
        result = run(by_length(source, instruction_field="instruction", min_len=5))
        assert result == []

    def test_removes_too_long(self):
        """Items with more words than max_len are removed."""
        data = [{"instruction": " ".join(["word"] * 20), "response": "ok"}]
        source = make_source(data)
        result = run(by_length(source, instruction_field="instruction", max_len=10))
        assert result == []

    def test_inclusive_bounds(self):
        """Boundary values (exactly min_len and max_len) are kept."""
        data = [
            {"instruction": " ".join(["word"] * 3), "response": "ok"},
            {"instruction": " ".join(["word"] * 7), "response": "ok"},
        ]
        source = make_source(data)
        result = run(
            by_length(source, instruction_field="instruction", min_len=3, max_len=7)
        )
        assert len(result) == 2

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_length(source, instruction_field="instruction"))
        assert result == []

    def test_missing_field_raises_key_error(self):
        """Missing expected field raises KeyError."""
        data = [{"response": "ok"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'instruction'"):
            run(by_length(source, instruction_field="instruction"))


class TestByInvalidResponse:
    """by_invalid_response removes items with copied or empty responses."""

    def test_removes_exact_copy(self):
        """Items where response exactly matches instruction are removed."""
        data = [{"instruction": "Tell me a joke", "response": "Tell me a joke"}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_removes_empty_response(self):
        """Items with empty response are removed."""
        data = [{"instruction": "Tell me a joke", "response": ""}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_removes_whitespace_only_response(self):
        """Items with whitespace-only response are removed."""
        data = [{"instruction": "Tell me a joke", "response": "   \n\t  "}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_removes_response_substring_of_instruction(self):
        """Response that is a substring of instruction is removed."""
        data = [{"instruction": "Tell me a funny joke", "response": "me a funny joke"}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_removes_similar_length_substring(self):
        """Instruction that is a substring of response is removed."""
        data = [{"instruction": "Tell me a funny joke", "response": "Tell me a funny"}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_keeps_short_instruction_in_long_response(self):
        """Short instruction appearing in a long response is not a false positive."""
        data = [
            {
                "instruction": "Hi",
                "response": "Hi there! Here is a detailed explanation of the topic.",
            }
        ]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == data

    def test_removes_high_similarity(self):
        """Items with similarity above threshold are removed."""
        data = [
            {
                "instruction": "Tell me a joke about cats",
                "response": "Tell me a joke about dogs",
            }
        ]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source,
                instruction_field="instruction",
                response_field="response",
                similarity_threshold=0.8,
            )
        )
        assert result == []

    def test_keeps_valid_response(self):
        """Items with genuinely different response are kept."""
        data = [
            {
                "instruction": "Tell me a joke",
                "response": "Why did the chicken cross the road?",
            }
        ]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == data

    def test_keeps_no_matches_removed(self):
        """When all items are valid, none are removed."""
        data = [
            {"instruction": "What is 2+2?", "response": "4"},
            {"instruction": "Name a color", "response": "Blue"},
        ]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == data

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_removes_none_response(self):
        """Items with None response are removed."""
        data = [{"instruction": "Tell me a joke", "response": None}]
        source = make_source(data)
        result = run(
            by_invalid_response(
                source, instruction_field="instruction", response_field="response"
            )
        )
        assert result == []

    def test_missing_field_raises_key_error(self):
        """Missing expected field raises KeyError."""
        data = [{"instruction": "Tell me a joke"}]
        source = make_source(data)
        with pytest.raises(RuntimeError, match="missing required field 'response'"):
            run(
                by_invalid_response(
                    source, instruction_field="instruction", response_field="response"
                )
            )


class TestByKeywordBlocklist:
    """by_keyword_blocklist removes items containing blocked phrases."""

    def test_removes_default_blocklist_match(self):
        """Items matching default blocklist phrases are removed."""
        data = [
            {
                "instruction": "Do something bad",
                "response": "As an AI, I cannot do that.",
            },
            {
                "instruction": "Tell a joke",
                "response": "Why did the chicken cross the road?",
            },
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, response_field="response"))
        assert result == [data[1]]

    def test_case_insensitive_by_default(self):
        """Matching is case-insensitive by default."""
        data = [{"instruction": "test", "response": "AS AN AI, I refuse."}]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, response_field="response"))
        assert result == []

    def test_case_sensitive_when_enabled(self):
        """Case-sensitive matching only blocks exact case matches."""
        data = [
            {"instruction": "test", "response": "AS AN AI, I refuse."},
            {"instruction": "test", "response": "as an ai, I refuse."},
        ]
        source = make_source(data)
        result = run(
            by_keyword_blocklist(source, response_field="response", case_sensitive=True)
        )
        assert result == [data[0]]

    def test_custom_blocklist(self):
        """Custom blocklist replaces the default."""
        data = [
            {"instruction": "test", "response": "This contains foo in it."},
            {"instruction": "test", "response": "This is clean."},
        ]
        source = make_source(data)
        result = run(
            by_keyword_blocklist(source, response_field="response", blocklist=["foo"])
        )
        assert result == [data[1]]

    def test_no_matches_keeps_all(self):
        """When no items match blocklist, all are kept."""
        data = [
            {"instruction": "test", "response": "Perfectly fine answer."},
            {"instruction": "test", "response": "Another good response."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, response_field="response"))
        assert result == data

    def test_all_filtered_returns_empty(self):
        """When all items match blocklist, returns empty list."""
        data = [
            {"instruction": "test", "response": "I cannot fulfill your request."},
            {"instruction": "test", "response": "I'm sorry, I cannot help."},
        ]
        source = make_source(data)
        result = run(by_keyword_blocklist(source, response_field="response"))
        assert result == []

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = make_source([])
        result = run(by_keyword_blocklist(source, response_field="response"))
        assert result == []

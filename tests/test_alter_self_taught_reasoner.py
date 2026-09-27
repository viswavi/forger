"""Tests for self_taught_reasoner ALTER building block."""
from __future__ import annotations

import pytest
from fixtures import MockModel

from pyforger.alter.generate import _normalized_equal, self_taught_reasoner


class TestSplitThinking:
    """thinking_field / answer_field derivation from the model's response."""

    def test_inline_think_tag_is_split(self):
        """No API reasoning: an inline <think> block is split out."""
        model = MockModel(responses=["<think>because 6*7</think>ANSWER: 42"])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}], "{prompt}", model, n=1
        )
        assert len(items) == 1
        assert items[0]["thinking"] == "because 6*7"
        assert items[0]["answer"] == "ANSWER: 42"

    def test_api_reasoning_preferred_over_inline_tag(self):
        """API reasoning is preferred over an inline <think> tag.

        When the model returns reasoning as a separate field, use it directly
        instead of trying to split <think> tags out of the content.
        """
        model = MockModel(
            responses=["ANSWER: 42"],
            reasoning=["because 6*7"],
        )
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}], "{prompt}", model, n=1
        )
        assert len(items) == 1
        assert items[0]["thinking"] == "because 6*7"
        assert items[0]["answer"] == "ANSWER: 42"

    def test_split_thinking_false_leaves_thinking_empty(self):
        """split_thinking=False stores the whole response as the answer."""
        model = MockModel(
            responses=["ANSWER: 42"],
            reasoning=["because 6*7"],
        )
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}],
            "{prompt}",
            model,
            n=1,
            split_thinking=False,
        )
        assert len(items) == 1
        assert items[0]["thinking"] == ""
        assert items[0]["answer"] == "ANSWER: 42"

    def test_missing_api_reasoning_falls_back_per_completion(self):
        """A completion without API reasoning falls back to its inline <think>."""
        model = MockModel(
            responses=[["ANSWER: 42", "<think>inline</think>ANSWER: 42"]],
            reasoning=[["from api", None]],
        )
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}], "{prompt}", model, n=2
        )
        assert [row["thinking"] for row in items] == ["from api", "inline"]
        assert [row["answer"] for row in items] == ["ANSWER: 42", "ANSWER: 42"]


class TestAnswerMatching:
    """Only responses whose extracted answer matches the reference are kept."""

    def test_matching_answer_is_kept(self):
        """A response with a matching final answer is kept."""
        model = MockModel(responses=["ANSWER: 42"])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}], "{prompt}", model, n=1
        )
        assert len(items) == 1

    def test_mismatched_answer_is_dropped(self):
        """A response with a non-matching final answer contributes no rows."""
        model = MockModel(responses=["ANSWER: 41"])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}], "{prompt}", model, n=1
        )
        assert items == []

    @pytest.mark.parametrize(
        "response, reference, answer",
        [
            ("ANSWER: 1000", "1,000", "1000"),
            ("ANSWER: 1,000", "1000", "1000"),
            ("ANSWER: 1,000", 1000, "1000"),
            ("#### 1,234,567", " 1,234,567 ", "1234567"),
        ],
    )
    def test_thousands_separators_match(self, response, reference, answer):
        """Commas in a numeric reference don't block a correct sample.

        The kept row's new ground truth is the extracted answer, without commas.
        """
        model = MockModel(responses=[response])
        items = self_taught_reasoner(
            [{"prompt": "q", "final_answer": reference}], "{prompt}", model, n=1
        )
        assert len(items) == 1
        assert items[0]["final_answer"] == answer

    @pytest.mark.parametrize(
        "response, reference",
        [("ANSWER: 12", "1,2"), ("\\boxed{34}", "3,4"), ("ANSWER: 35", "3,5")],
    )
    def test_list_or_decimal_comma_reference_rejects_merged_digits(
        self, response, reference
    ):
        """A wrong answer equal to a list's digits run together is dropped."""
        model = MockModel(responses=[response])
        items = self_taught_reasoner(
            [{"prompt": "q", "final_answer": reference}], "{prompt}", model, n=1
        )
        assert items == []


class TestNormalizedEqual:
    """The default match_fn normalizes both sides the same way."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ("42", "42"),
            ("42", 42),
            (" 42 ", "42"),
            ("1000", "1,000"),
            ("-1,000.5", "-1000.5"),
            ("Paris, France", "Paris, France"),
        ],
    )
    def test_equal(self, a, b):
        """Equal answers match regardless of whitespace, type or separators."""
        assert _normalized_equal(a, b)
        assert _normalized_equal(b, a)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("42", "41"),
            ("Paris, France", "Paris France"),  # commas only dropped from numbers
            ("1, 2", "1 2"),  # starts with a number but isn't one
            ("1,2", "12"),  # a list, not thousands grouping
            ("-2,1", "-21"),
            ("1,2,3", "123"),
            ("3,5", "35"),  # a decimal comma
            ("1,00,0", "1000"),  # malformed grouping is left alone
            ("1,000,2", "10002"),  # grouping only at the start doesn't count
            ("12,34", "1234"),  # 2-digit groups read as a list, not a number
            ("1,000", "1000.5"),
        ],
    )
    def test_not_equal(self, a, b):
        """Different answers, and text differing only by a comma, don't match."""
        assert not _normalized_equal(a, b)
        assert not _normalized_equal(b, a)


class TestCustomFilter:
    """extract_answer still fills the student answer when filter_fn is given."""

    @staticmethod
    def _spelled_out(text: str) -> str | None:
        """Extract a spelled-out answer the default extractor cannot read."""
        return "42" if "forty-two" in text else None

    def test_extract_answer_feeds_filter_fn(self, caplog):
        """filter_fn sees the custom extractor's answer, with a warning logged."""
        seen: list[dict] = []

        def keep_all(rows: list[dict]) -> list[dict]:
            seen.extend(rows)
            return rows

        model = MockModel(responses=["It is forty-two."])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "forty-two"}],
            "{prompt}",
            model,
            n=1,
            extract_answer=self._spelled_out,
            filter_fn=keep_all,
        )
        assert seen[0]["student_final_answer"] == "42"
        assert items[0]["final_answer"] == "42"  # student answer becomes the truth
        assert "extract_answer still runs when filter_fn is given" in caplog.text

    def test_no_warning_without_custom_extractor(self, caplog):
        """filter_fn alone uses the default extractor and logs no warning."""
        model = MockModel(responses=["ANSWER: 42"])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}],
            "{prompt}",
            model,
            n=1,
            filter_fn=lambda rows: rows,
        )
        assert items[0]["student_final_answer"] == "42"
        assert "extract_answer still runs" not in caplog.text

    def test_no_warning_without_filter_fn(self, caplog):
        """A custom extractor on the default matching path logs no warning."""
        model = MockModel(responses=["It is forty-two."])
        items = self_taught_reasoner(
            [{"prompt": "6*7?", "final_answer": "42"}],
            "{prompt}",
            model,
            n=1,
            extract_answer=self._spelled_out,
        )
        assert len(items) == 1
        assert "extract_answer still runs" not in caplog.text

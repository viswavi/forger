"""Tests for self_taught_reasoner ALTER building block."""
from __future__ import annotations

from fixtures import MockModel

from pyforger.alter.generate import self_taught_reasoner


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

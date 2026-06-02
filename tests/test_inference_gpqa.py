"""Tests for pyforger.inference.benchmarks.gpqa."""
from __future__ import annotations

from typing import Any

import pytest

from pyforger.inference.benchmarks import gpqa as gpqa_module
from pyforger.inference.benchmarks.gpqa import (
    GPQAExample,
    _row_to_example,
    format_prompt,
    load_gpqa,
    parse_answer,
)


class TestParseAnswer:
    """parse_answer recovers a letter from various completion formats."""

    def test_explicit_answer_uppercase(self):
        assert parse_answer("Answer: B") == "B"

    def test_explicit_answer_lowercase(self):
        assert parse_answer("answer: c") == "C"

    def test_explicit_answer_parenthesized(self):
        assert parse_answer("Answer: (A)") == "A"

    def test_explicit_answer_with_preamble(self):
        completion = (
            "The reasoning is X, Y, Z. Therefore the choice is D.\n"
            "Answer: D"
        )
        assert parse_answer(completion) == "D"

    def test_fallback_when_no_explicit_answer(self):
        assert parse_answer("So the answer is (D)") == "D"

    def test_fallback_picks_last_letter(self):
        # No "Answer:" prefix; fallback should grab the last A/B/C/D.
        assert parse_answer("I considered A, B, then settled on C") == "C"

    def test_no_letter_returns_none(self):
        assert parse_answer("no letter here, just words and 123") is None

    def test_empty_returns_none(self):
        assert parse_answer("") is None


class TestRender:
    """_render and format_prompt produce the expected shape."""

    def _ex(self, gold: str = "B") -> GPQAExample:
        return GPQAExample(
            question="What is 2+2?",
            choices=("3", "4", "5", "6"),
            gold=gold,
        )

    def test_format_prompt_zero_shot_omits_gold(self):
        prompt = format_prompt(self._ex(gold="B"))
        assert "Question: What is 2+2?" in prompt
        assert "A) 3" in prompt
        assert "B) 4" in prompt
        # Critically: the target example must NOT leak the gold answer.
        assert "Answer: B" not in prompt

    def test_format_prompt_with_few_shot_includes_their_gold(self):
        shot = GPQAExample(
            question="2+2?", choices=("1", "2", "3", "4"), gold="D"
        )
        target = GPQAExample(
            question="3+3?", choices=("5", "6", "7", "8"), gold="B"
        )
        prompt = format_prompt(target, few_shot=[shot])
        # The shot's gold answer should appear once.
        assert prompt.count("Answer: D") == 1
        # The target's gold should NOT appear.
        assert "Answer: B" not in prompt
        # Order: shot before target.
        assert prompt.index("2+2?") < prompt.index("3+3?")

    def test_format_prompt_multiple_shots(self):
        s1 = GPQAExample(question="q1", choices=("a", "b", "c", "d"), gold="A")
        s2 = GPQAExample(question="q2", choices=("a", "b", "c", "d"), gold="C")
        target = GPQAExample(
            question="tq", choices=("a", "b", "c", "d"), gold="B"
        )
        prompt = format_prompt(target, few_shot=[s1, s2])
        assert prompt.count("Answer: A") == 1
        assert prompt.count("Answer: C") == 1
        assert "Answer: B" not in prompt


class TestRowToExample:
    """_row_to_example produces correct gold letters and deterministic shuffles."""

    def _row(self) -> dict[str, Any]:
        return {
            "Question": "Q?",
            "Correct Answer": "CORRECT",
            "Incorrect Answer 1": "WRONG1",
            "Incorrect Answer 2": "WRONG2",
            "Incorrect Answer 3": "WRONG3",
        }

    def test_gold_points_at_correct_choice(self):
        for i in range(8):
            ex = _row_to_example(self._row(), i)
            gold_index = "ABCD".index(ex.gold)
            assert ex.choices[gold_index] == "CORRECT", (
                f"idx={i}: gold {ex.gold} points at {ex.choices[gold_index]!r}, "
                f"expected 'CORRECT' (choices={ex.choices})"
            )

    def test_shuffle_is_deterministic_per_index(self):
        a = _row_to_example(self._row(), 7)
        b = _row_to_example(self._row(), 7)
        assert a == b

    def test_shuffle_differs_across_indices(self):
        # Across enough indices, at least two should produce different
        # orderings — otherwise the shuffle is degenerate.
        orderings = {_row_to_example(self._row(), i).choices for i in range(8)}
        assert len(orderings) > 1

    def test_missing_column_raises_keyerror(self):
        bad_row = self._row()
        del bad_row["Correct Answer"]
        with pytest.raises(KeyError) as exc_info:
            _row_to_example(bad_row, 0)
        # The error message should list the available keys so a column
        # rename in HF is debuggable.
        assert "Correct Answer" in str(exc_info.value)
        assert "Question" in str(exc_info.value)


class TestLoadGPQA:
    """load_gpqa is @op-cached and converts retrieve_from_hf rows correctly."""

    def _synthetic_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "Question": f"Question {i}?",
                "Correct Answer": f"correct-{i}",
                "Incorrect Answer 1": f"w1-{i}",
                "Incorrect Answer 2": f"w2-{i}",
                "Incorrect Answer 3": f"w3-{i}",
            }
            for i in range(3)
        ]

    def test_load_gpqa_returns_examples(self, monkeypatch):
        rows = self._synthetic_rows()
        monkeypatch.setattr(
            gpqa_module, "retrieve_from_hf", lambda **kw: rows
        )
        out = load_gpqa(subset="gpqa_main", n=3)
        assert len(out) == 3
        assert all(isinstance(ex, GPQAExample) for ex in out)
        # Each example's gold letter must point at the row's correct answer.
        for i, ex in enumerate(out):
            gold_index = "ABCD".index(ex.gold)
            assert ex.choices[gold_index] == f"correct-{i}"

    def test_load_gpqa_passes_through_retrieve_kwargs(self, monkeypatch):
        captured: dict[str, Any] = {}

        def fake_retrieve(**kw):
            captured.update(kw)
            return self._synthetic_rows()

        monkeypatch.setattr(gpqa_module, "retrieve_from_hf", fake_retrieve)
        load_gpqa(subset="gpqa_diamond", n=3, seed=42)
        assert captured["dataset"] == "Idavidrein/gpqa"
        assert captured["config"] == "gpqa_diamond"
        assert captured["split"] == "train"
        assert captured["n"] == 3
        assert captured["seed"] == 42

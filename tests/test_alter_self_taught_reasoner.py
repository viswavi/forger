"""Tests for self_taught_reasoner ALTER building block."""
from __future__ import annotations

import pytest
from fixtures import MockModel

from pyforger.alter.generate import (
    _normalized_equal,
    _with_hint,
    default_extract_answer,
    self_taught_reasoner,
)


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


class TestDefaultExtractAnswer:
    """default_extract_answer keeps list commas but drops thousands separators."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("ANSWER: 1,000", "1000"),
            ("#### -1,234,567.5", "-1234567.5"),
            ("ANSWER: 1,2", "1,2"),  # a list, not one number
            ("The answer is 1000, so we are done.", "1000"),  # trailing comma
            ("\\boxed{1,000,}", "1000"),
            ("no digits here", None),
        ],
    )
    def test_extracts(self, text, expected):
        """Separators are normalized the same way as references."""
        assert default_extract_answer(text) == expected

    def test_list_answer_does_not_match_merged_digits(self):
        """A student "1,2" is not accepted for reference "12"."""
        model = MockModel(responses=["ANSWER: 1,2"])
        items = self_taught_reasoner(
            [{"prompt": "q", "final_answer": "12"}], "{prompt}", model, n=1
        )
        assert items == []


class _HintAware:
    """Answers 42 only when the prompt has a hint or is solvable; records prompts."""

    def __init__(self, solves: frozenset[str] = frozenset()):
        self.solves = solves
        self.prompts: list[str] = []

    def cache_key(self) -> tuple:
        """Stable identity for @op caching."""
        return ("hint-aware", tuple(sorted(self.solves)))

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """``n`` copies of ANSWER: 42 for hinted or solvable prompts, else ANSWER: 0."""
        self.prompts.extend(prompts)
        return [
            ["ANSWER: 42" if "hint" in p.lower() or p in self.solves else "ANSWER: 0"]
            * n
            for p in prompts
        ]


class TestRationalization:
    """hint_fn: a hinted second pass for items with no accepted response."""

    ITEMS = [
        {"prompt": "easy", "final_answer": "42"},
        {"prompt": "hard", "final_answer": "42"},
    ]

    def test_only_unsolved_items_get_the_hint(self):
        """The solved item keeps its own rows; the unsolved one is rationalized."""
        model = _HintAware(solves=frozenset({"easy"}))
        rows = self_taught_reasoner(
            self.ITEMS,
            "{prompt}",
            model,
            n=2,
            hint_fn="{prompt} (hint: {final_answer})",
            _no_cache=True,
        )
        assert model.prompts == ["easy", "hard", "hard (hint: 42)"]
        assert [(r["prompt"], r["rationalized"]) for r in rows] == [
            ("easy", False),
            ("easy", False),
            ("hard", True),
            ("hard", True),
        ]

    def test_still_unsolved_items_contribute_no_rows(self):
        """A hint that does not help leaves the item without rows."""
        model = _HintAware()
        rows = self_taught_reasoner(
            self.ITEMS, "{prompt}", model, n=1, hint_fn="{prompt} again", _no_cache=True
        )
        assert rows == []
        assert model.prompts == ["easy", "hard", "easy again", "hard again"]

    def test_verifier_judges_the_hinted_pass(self):
        """The hinted pass uses the same acceptance rule as the first."""
        rows = self_taught_reasoner(
            self.ITEMS[:1],
            "{prompt}",
            _HintAware(),
            n=1,
            verifier=lambda answer, row: answer.endswith("42"),
            hint_fn="{prompt} hint",
            _no_cache=True,
        )
        assert [r["rationalized"] for r in rows] == [True]

    def test_default_hint_is_the_reference(self):
        """By default the unsolved item is retried with its reference as the hint."""
        model = _HintAware(solves=frozenset({"easy"}))
        rows = self_taught_reasoner(self.ITEMS, "{prompt}", model, n=1, _no_cache=True)
        assert model.prompts == [
            "easy",
            "hard",
            "hard\n\nHint: a correct answer is:\n42",
        ]
        assert [(r["prompt"], r["rationalized"]) for r in rows] == [
            ("easy", False),
            ("hard", True),
        ]

    def test_items_without_a_reference_are_not_retried(self):
        """With a verifier and no reference there is no default hint to give."""
        model = _HintAware()
        rows = self_taught_reasoner(
            [{"prompt": "hard"}],
            "{prompt}",
            model,
            n=1,
            verifier=lambda answer, row: answer.endswith("42"),
            _no_cache=True,
        )
        assert rows == []
        assert model.prompts == ["hard"]

    def test_rationalize_false_skips_the_second_pass(self):
        """rationalize=False: one pass and no rationalized field."""
        model = _HintAware(solves=frozenset({"easy"}))
        rows = self_taught_reasoner(
            self.ITEMS, "{prompt}", model, n=1, rationalize=False, _no_cache=True
        )
        assert model.prompts == ["easy", "hard"]
        assert [r["prompt"] for r in rows] == ["easy"]
        assert "rationalized" not in rows[0]

    def test_hint_fn_without_rationalize_raises(self):
        """A hint_fn would be ignored with rationalize=False, so it is refused."""
        with pytest.raises(ValueError, match="rationalize=True"):
            self_taught_reasoner(
                self.ITEMS,
                "{prompt}",
                _HintAware(),
                rationalize=False,
                hint_fn="{prompt} hint",
                _no_cache=True,
            )

    def test_chat_hint_goes_after_the_last_user_message(self):
        """A chat prompt gets the hint appended to its last user message."""
        prompt = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "6*7?"},
        ]
        assert _with_hint(prompt, "Hint: 42") == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "6*7?\n\nHint: 42"},
        ]
        assert prompt[1]["content"] == "6*7?"

    def test_filter_fn_with_hint_fn_raises(self):
        """filter_fn sees all candidates at once, so unsolved items are unknown."""
        with pytest.raises(ValueError, match="hint_fn"):
            self_taught_reasoner(
                self.ITEMS,
                "{prompt}",
                _HintAware(),
                filter_fn=lambda rows: rows,
                hint_fn="{prompt} hint",
                _no_cache=True,
            )


class _Judge:
    """Answers YES when ``rule(prompt)`` is true, else NO; records every prompt.

    ``replies`` overrides the reply for a prompt containing a key, e.g. to
    return a thinking block or no verdict. ``cache_key`` leaves the record out.
    """

    def __init__(self, rule, replies=None, name="judge"):
        self.rule = rule
        self.replies = replies or {}
        self.name = name
        self.calls: list[list[str]] = []

    def cache_key(self) -> tuple:
        return ("_Judge", self.name, tuple(sorted(self.replies.items())))

    def generate_batch(self, prompts, n=1, **kwargs):
        self.calls.append(list(prompts))
        out = []
        for prompt in prompts:
            reply = next((r for k, r in self.replies.items() if k in prompt), None)
            out.append([reply or ("YES" if self.rule(prompt) else "NO")])
        return out


def _answers_shoes_off(prompt: str) -> bool:
    """The rule of a judge that accepts answers about taking shoes off."""
    answer = prompt.split("<student_answer>")[1].split("</student_answer>")[0]
    return "Remove" in answer or "Leave" in answer


_REFERENCE = "Take your shoes off at the door."


class TestAnswerJudge:
    """answer_judge= keeps samples the judge says reach the reference's conclusion."""

    def _run(self, judge, responses=None, items=None, **kwargs):
        model = kwargs.pop("model", None) or MockModel(
            responses=responses
            or [["Remove your shoes before going in.", "Keep your shoes on."]]
        )
        rows = self_taught_reasoner(
            items or [{"q": "Shoes indoors?", "final_answer": _REFERENCE}],
            "Question: {q}",
            model,
            n=2,
            answer_judge=judge,
            **kwargs,
        )
        return rows, model

    def test_keeps_answers_the_judge_accepts(self):
        """YES is kept with the reference unchanged; NO is dropped."""
        rows, _ = self._run(_Judge(_answers_shoes_off))
        assert [r["answer"] for r in rows] == ["Remove your shoes before going in."]
        assert rows[0]["student_final_answer"] == "Remove your shoes before going in."
        assert rows[0]["final_answer"] == _REFERENCE  # the reference is kept

    def test_prompt_shows_question_reference_and_answer_not_reasoning(self):
        """The judge sees the student's prompt and final answer, not its thinking."""
        judge = _Judge(_answers_shoes_off)
        self._run(
            judge, responses=[["<think>secret</think>Remove them.", "Keep them on."]]
        )
        prompt = judge.calls[0][0]
        assert "<question>\nQuestion: Shoes indoors?\n</question>" in prompt
        assert f"<reference_answer>\n{_REFERENCE}\n" in prompt
        assert "<student_answer>\nRemove them.\n" in prompt
        assert "secret" not in prompt

    def test_identical_answers_are_judged_once_in_one_call(self):
        """Distinct (question, reference, answer) prompts, all in one batch."""
        judge = _Judge(_answers_shoes_off)
        items = [
            {"q": "Shoes indoors?", "final_answer": _REFERENCE},
            {"q": "Shoes at a temple?", "final_answer": _REFERENCE},
        ]
        responses = [
            ["Remove your shoes.", "Remove your shoes."],
            ["Remove your shoes.", "Keep them on."],
        ]
        rows, _ = self._run(judge, responses=responses, items=items)
        assert len(judge.calls) == 1
        assert len(judge.calls[0]) == 3
        assert [r["q"] for r in rows] == ["Shoes indoors?"] * 2 + ["Shoes at a temple?"]

    def test_verdict_after_thinking_and_unclear_replies(self, caplog):
        """A draft verdict inside <think> does not count; no verdict rejects."""
        judge = _Judge(
            lambda prompt: True,
            replies={
                "Remove": "<think>maybe NO</think>**YES**",
                "Keep": "I cannot tell.",
            },
        )
        rows, _ = self._run(judge)
        assert [r["answer"] for r in rows] == ["Remove your shoes before going in."]
        assert "had no YES or NO" in caplog.text

    def test_empty_answer_is_rejected_without_asking(self):
        """An answer that is empty after the reasoning is never judged."""
        judge = _Judge(lambda prompt: True)
        rows, _ = self._run(
            judge, responses=[["<think>hmm</think>", "Keep your shoes on."]]
        )
        assert [r["answer"] for r in rows] == ["Keep your shoes on."]
        assert len(judge.calls[0]) == 1

    def test_extract_answer_chooses_the_text(self):
        """With extract_answer, its output is judged instead of the answer."""
        judge = _Judge(_answers_shoes_off)
        rows, _ = self._run(
            judge,
            responses=[["Verdict: Remove your shoes.", "Remove? Verdict: Keep them."]],
            extract_answer=lambda raw: raw.split("Verdict:")[-1],
        )
        assert [r["student_final_answer"] for r in rows] == [" Remove your shoes."]

    @pytest.mark.parametrize(
        "other",
        [
            {"match_fn": lambda a, b: True},
            {"verifier": lambda answer, row: True},
            {"filter_fn": lambda rows: rows},
        ],
    )
    def test_exclusive_with_other_checks(self, other):
        """answer_judge with match_fn, verifier, or filter_fn raises."""
        with pytest.raises(ValueError, match="answer_judge or"):
            self._run(_Judge(lambda prompt: True), **other)


class TestAnswerJudgeCache:
    """The judge is in the key; sampling is never redone for another judge."""

    def _run(self, model, judge):
        return self_taught_reasoner(
            [{"q": "Shoes indoors?", "final_answer": _REFERENCE}],
            "{q}",
            model,
            n=2,
            answer_judge=judge,
        )

    def _model(self):
        return MockModel(
            responses=[["Remove your shoes before going in.", "Keep your shoes on."]]
        )

    def test_same_call_is_a_cache_hit(self):
        """The same call again samples and judges nothing."""
        model, judge = self._model(), _Judge(_answers_shoes_off)
        first = self._run(model, judge)
        assert self._run(model, judge) == first
        assert model._call_count == 1
        assert len(judge.calls) == 1

    def test_new_judge_reuses_samples(self):
        """Another judge changes the result without sampling again."""
        model = self._model()
        assert len(self._run(model, _Judge(_answers_shoes_off))) == 1
        assert len(self._run(model, _Judge(lambda p: True, name="lenient"))) == 2
        assert model._call_count == 1


class _HintedResponses:
    """Gives ``first`` to unhinted prompts and ``hinted`` to hinted ones."""

    def __init__(self, first: list[str], hinted: list[str]) -> None:
        self.first, self.hinted = first, hinted
        self.prompts: list[str] = []
        self.calls = 0

    def cache_key(self) -> tuple:
        """Stable identity for @op caching; leaves out the call record."""
        return ("hinted-responses", tuple(self.first), tuple(self.hinted))

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """``first`` or ``hinted`` (the first ``n``) for each prompt."""
        self.calls += 1
        self.prompts.extend(prompts)
        return [list(self.hinted if "Hint:" in p else self.first)[:n] for p in prompts]


class TestRationalizationWithAnswerJudge:
    """The judge decides which items are unsolved; it never sees the hint."""

    ITEM = {"q": "Shoes indoors?", "final_answer": _REFERENCE}

    def _run(self, model, judge):
        return self_taught_reasoner([self.ITEM], "{q}", model, n=2, answer_judge=judge)

    def test_unsolved_by_the_judge_is_rationalized_and_cached(self):
        """Rejected answers are hinted; another judge re-draws nothing."""
        model = _HintedResponses(
            first=["Keep your shoes on.", "Keep them on."],
            hinted=["Remove your shoes before going in.", "Keep your shoes on."],
        )
        judge = _Judge(_answers_shoes_off)
        rows = self._run(model, judge)
        assert model.prompts == [
            "Shoes indoors?",
            f"Shoes indoors?\n\nHint: a correct answer is:\n{_REFERENCE}",
        ]
        assert [(r["answer"], r["rationalized"]) for r in rows] == [
            ("Remove your shoes before going in.", True)
        ]
        assert rows[0]["final_answer"] == _REFERENCE
        # One judge call per pass, each with the original question.
        assert len(judge.calls) == 2
        assert all("Hint:" not in p for call in judge.calls for p in call)

        # A lenient judge solves the item in the first pass, from cached samples.
        rows = self._run(model, _Judge(lambda p: True, name="lenient"))
        assert [(r["answer"], r["rationalized"]) for r in rows] == [
            ("Keep your shoes on.", False),
            ("Keep them on.", False),
        ]
        assert model.calls == 2

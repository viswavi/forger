"""Tests for self_taught_reasoner ALTER building block."""
from __future__ import annotations

import inspect
import math

import pytest
from fixtures import MockModel

from pyforger.alter.generate import (
    _normalized_equal,
    _with_hint,
    default_extract_answer,
    self_taught_reasoner,
)
from pyforger.inspect import cluster


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


class _AngleEmbedder:
    """Embeds each known text as a unit 2-D vector at a fixed angle.

    The cosine of two texts is then cos(angle difference), e.g. 0.9553 for
    0.3 radians. Records every call; ``cache_key`` leaves the record out.
    """

    def __init__(self, angles: dict[str, float], name: str = "angles") -> None:
        self.angles = angles
        self.name = name
        self.calls: list[list[str]] = []

    def cache_key(self) -> tuple:
        return ("_AngleEmbedder", self.name, tuple(sorted(self.angles.items())))

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[math.cos(self.angles[t]), math.sin(self.angles[t])] for t in texts]


_REFERENCE = "Take your shoes off at the door."
_ANGLES = {
    _REFERENCE: 0.0,
    "Remove your shoes before going in.": 0.3,  # cosine 0.9553
    "Keep your shoes on.": 1.2,  # cosine 0.3624
    "Leave shoes outside.": 0.5,  # cosine 0.8776
}


class TestAnswerSimilarity:
    """answer_similarity= keeps samples whose answer embeds close to the reference."""

    def _run(self, embedder, threshold=0.9, responses=None, items=None, **kwargs):
        model = kwargs.pop("model", None) or MockModel(
            responses=responses
            or [["Remove your shoes before going in.", "Keep your shoes on."]]
        )
        rows = self_taught_reasoner(
            items or [{"q": "Shoes indoors?", "final_answer": _REFERENCE}],
            "{q}",
            model,
            n=2,
            answer_similarity=threshold,
            embed_fn=embedder,
            **kwargs,
        )
        return rows, model

    def test_keeps_answers_at_or_above_the_threshold(self):
        """The close answer is kept with its cosine; the far one is dropped."""
        rows, _ = self._run(_AngleEmbedder(_ANGLES), threshold=0.9)
        assert [r["answer"] for r in rows] == ["Remove your shoes before going in."]
        assert rows[0]["answer_similarity"] == pytest.approx(0.9553)
        assert rows[0]["student_final_answer"] == "Remove your shoes before going in."
        assert rows[0]["final_answer"] == _REFERENCE  # the reference is kept

    def test_threshold_is_inclusive(self):
        """A cosine equal to the (rounded) threshold is kept."""
        rows, _ = self._run(_AngleEmbedder(_ANGLES), threshold=0.9553)
        assert len(rows) == 1
        rows, _ = self._run(_AngleEmbedder(_ANGLES), threshold=0.3)
        assert len(rows) == 2

    def test_custom_field_name(self):
        """similarity_field= renames the recorded cosine."""
        rows, _ = self._run(_AngleEmbedder(_ANGLES), similarity_field="sim")
        assert "sim" in rows[0] and "answer_similarity" not in rows[0]

    def test_one_embed_call_with_distinct_texts(self):
        """Every answer and reference is embedded once, in one call."""
        embedder = _AngleEmbedder(_ANGLES)
        items = [
            {"q": "Shoes indoors?", "final_answer": _REFERENCE},
            {"q": "Shoes at a temple?", "final_answer": _REFERENCE},
        ]
        responses = [
            ["Remove your shoes before going in.", "Keep your shoes on."],
            ["Leave shoes outside.", "Remove your shoes before going in."],
        ]
        rows, _ = self._run(embedder, threshold=0.85, responses=responses, items=items)
        assert len(embedder.calls) == 1
        embedded = embedder.calls[0]
        assert sorted(embedded) == sorted(_ANGLES)  # each distinct text once
        assert [r["answer"] for r in rows] == [
            "Remove your shoes before going in.",
            "Leave shoes outside.",
            "Remove your shoes before going in.",
        ]

    def test_empty_answer_is_rejected_without_embedding(self):
        """An answer that is empty after the reasoning never matches."""
        embedder = _AngleEmbedder(_ANGLES)
        rows, _ = self._run(
            embedder,
            threshold=-1.0,
            responses=[["<think>hmm</think>", "Keep your shoes on."]],
        )
        assert [r["answer"] for r in rows] == ["Keep your shoes on."]
        assert "" not in embedder.calls[0]

    def test_extract_answer_chooses_the_text(self):
        """With extract_answer, its output is embedded instead of the answer."""
        embedder = _AngleEmbedder(_ANGLES)
        rows, _ = self._run(
            embedder,
            responses=[["Verdict: Remove your shoes before going in.", "Verdict: "]],
            extract_answer=lambda raw: raw.split("Verdict:")[-1],
        )
        # Embedded stripped; the field keeps the extractor's output as is.
        assert len(rows) == 1
        assert rows[0]["student_final_answer"] == " Remove your shoes before going in."
        assert rows[0]["answer_similarity"] == pytest.approx(0.9553)

    @pytest.mark.parametrize(
        "other",
        [
            {"match_fn": lambda a, b: True},
            {"verifier": lambda answer, row: True},
            {"filter_fn": lambda rows: rows},
        ],
    )
    def test_exclusive_with_other_checks(self, other):
        """answer_similarity with match_fn, verifier, or filter_fn raises."""
        with pytest.raises(ValueError, match="answer_similarity or"):
            self._run(_AngleEmbedder(_ANGLES), **other)

    def test_embedding_args_need_a_threshold(self):
        """embed_fn or embedding_model without answer_similarity raises."""
        model = MockModel(responses=["x"])
        item = [{"q": "q", "final_answer": "x"}]
        with pytest.raises(ValueError, match="only with answer_similarity"):
            self_taught_reasoner(item, "{q}", model, n=1, embed_fn=lambda t: t)
        with pytest.raises(ValueError, match="only with answer_similarity"):
            self_taught_reasoner(item, "{q}", model, n=1, embedding_model="tfidf")

    def test_threshold_must_be_a_cosine(self):
        """A threshold outside [-1, 1] raises."""
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            self._run(_AngleEmbedder(_ANGLES), threshold=1.5)

    def test_missing_backend_raises_before_sampling(self, monkeypatch):
        """No silent TF-IDF fallback: a threshold does not carry over to it."""
        monkeypatch.setattr(cluster, "_sentence_transformers_importable", lambda: False)
        model = MockModel(responses=["x"])
        with pytest.raises(ImportError, match="sentence-transformers"):
            self_taught_reasoner(
                [{"q": "q", "final_answer": "x"}],
                "{q}",
                model,
                n=1,
                answer_similarity=0.8,
            )
        assert model._call_count == 0

    def test_tfidf_backend(self):
        """embedding_model="tfidf" needs no embed_fn; an identical answer scores 1."""
        model = MockModel(responses=[["Bow when greeting.", "Shake hands firmly."]])
        rows = self_taught_reasoner(
            [{"q": "Greeting?", "final_answer": "Bow when greeting."}],
            "{q}",
            model,
            n=2,
            answer_similarity=0.99,
            embedding_model="tfidf",
        )
        assert [r["answer"] for r in rows] == ["Bow when greeting."]
        assert rows[0]["answer_similarity"] == pytest.approx(1.0)

    def test_default_embedding_model_matches_inspect(self):
        """The literal default (a cycle prevents importing it) is cluster's default."""
        default = inspect.signature(self_taught_reasoner).parameters["embedding_model"]
        assert default.default == cluster._DEFAULT_EMBEDDING_MODEL


class TestAnswerSimilarityCache:
    """The threshold and the backend are in the key; sampling is never redone."""

    def _run(self, model, embedder, threshold):
        return self_taught_reasoner(
            [{"q": "Shoes indoors?", "final_answer": _REFERENCE}],
            "{q}",
            model,
            n=2,
            answer_similarity=threshold,
            embed_fn=embedder,
        )

    def _model(self):
        return MockModel(
            responses=[["Remove your shoes before going in.", "Keep your shoes on."]]
        )

    def test_same_call_is_a_cache_hit(self):
        """The same call again samples and embeds nothing."""
        model, embedder = self._model(), _AngleEmbedder(_ANGLES)
        first = self._run(model, embedder, 0.9)
        assert self._run(model, embedder, 0.9) == first
        assert model._call_count == 1
        assert len(embedder.calls) == 1

    def test_new_threshold_reuses_samples_and_embeddings(self):
        """Another threshold changes the result without sampling or embedding again."""
        model, embedder = self._model(), _AngleEmbedder(_ANGLES)
        assert len(self._run(model, embedder, 0.9)) == 1
        assert len(self._run(model, embedder, 0.3)) == 2
        assert model._call_count == 1
        assert len(embedder.calls) == 1

    def test_new_backend_embeds_again_but_reuses_samples(self):
        """Another embedder is another key: it embeds, and its cosines are used."""
        model = self._model()
        first, second = _AngleEmbedder(_ANGLES), _AngleEmbedder(
            {**_ANGLES, "Keep your shoes on.": 0.1}, name="other"
        )
        assert len(self._run(model, first, 0.9)) == 1
        assert len(self._run(model, second, 0.9)) == 2
        assert len(first.calls) == 1 and len(second.calls) == 1
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


class TestRationalizationWithAnswerSimilarity:
    """Similarity decides which items are unsolved; hinted samples are cached too."""

    ITEM = {"q": "Shoes indoors?", "final_answer": _REFERENCE}

    def _run(self, model, embedder, threshold):
        return self_taught_reasoner(
            [self.ITEM],
            "{q}",
            model,
            n=2,
            answer_similarity=threshold,
            embed_fn=embedder,
        )

    def test_unsolved_by_similarity_is_rationalized_and_cached(self):
        """Below the threshold the item is hinted; a new threshold re-draws nothing."""
        model = _HintedResponses(
            first=["Keep your shoes on.", "Leave shoes outside."],  # 0.36, 0.88
            hinted=["Remove your shoes before going in.", "Keep your shoes on."],
        )
        embedder = _AngleEmbedder(_ANGLES)
        rows = self._run(model, embedder, 0.9)
        assert model.prompts == [
            "Shoes indoors?",
            f"Shoes indoors?\n\nHint: a correct answer is:\n{_REFERENCE}",
        ]
        # The rationalized row records its own cosine and keeps the reference.
        assert [(r["answer"], r["rationalized"]) for r in rows] == [
            ("Remove your shoes before going in.", True)
        ]
        assert rows[0]["answer_similarity"] == pytest.approx(0.9553)
        assert rows[0]["final_answer"] == _REFERENCE
        assert len(embedder.calls) == 2  # one per pass

        # The same item is still unsolved at 0.95: both passes come from cache.
        rows = self._run(model, embedder, 0.95)
        assert [r["rationalized"] for r in rows] == [True]
        assert model.calls == 2 and len(embedder.calls) == 2

        # At 0.85 the first pass solves it, so no hinted row is kept.
        rows = self._run(model, embedder, 0.85)
        assert [(r["answer"], r["rationalized"]) for r in rows] == [
            ("Leave shoes outside.", False)
        ]
        assert model.calls == 2 and len(embedder.calls) == 2

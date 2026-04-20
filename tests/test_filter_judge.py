"""Tests for LLM-as-a-judge filter operations."""

import hashlib
import json
import random

import pytest
from pyforger import run
from pyforger.acquire import from_list
from pyforger.filter import by_llm_score


class _FakeJudge:
    """Deterministic fake judge for testing.

    Returns JSON with a random score and explanation, seeded by prompt
    content for reproducibility. Matches the Model.generate interface.
    """

    _WORD_POOL = [
        "good",
        "bad",
        "clear",
        "unclear",
        "relevant",
        "irrelevant",
        "complete",
        "incomplete",
    ]

    def __init__(self):
        """Initialize call tracking."""
        self.call_count = 0
        self.last_temperature = None
        self.last_n = None

    def generate(
        self,
        prompt,
        *,
        n=1,
        temperature=None,
        max_tokens=None,
    ):
        """Return n JSON strings with random scores."""
        self.call_count += 1
        self.last_temperature = temperature
        self.last_n = n
        results = []
        for i in range(n):
            digest = hashlib.md5(f"{prompt}|{i}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = random.Random(seed)
            score = rng.randint(1, 5)
            words = rng.choices(self._WORD_POOL, k=2)
            results.append(json.dumps({"score": score, "explanation": " ".join(words)}))
        return results


class _BadJudge:
    """Judge that returns malformed responses for testing parse failures."""

    def __init__(self, responses):
        """Store canned responses to return in order.

        Args:
            responses: List of strings to return, one per generate call.
        """
        self._responses = list(responses)
        self._idx = 0

    def generate(self, prompt, *, n=1, temperature=None, max_tokens=None):
        """Return canned responses."""
        results = []
        for _ in range(n):
            results.append(self._responses[self._idx % len(self._responses)])
            self._idx += 1
        return results


class TestByLlmScore:
    """by_llm_score filters items by LLM judge score."""

    def test_keeps_high_scoring_items(self):
        """Items scoring at or above threshold are kept."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source = from_list(data)
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(result) == len(data)

    def test_removes_low_scoring_items(self):
        """Items scoring below threshold are removed."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source = from_list(data)
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=5,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(result) < len(data)

    def test_threshold_affects_filtering(self):
        """A stricter threshold keeps fewer items."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source = from_list(data)
        strict = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=5,
                input_field="instruction",
                output_field="response",
            )
        )
        lenient = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(strict) <= len(lenient)

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = from_list([])
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_reproducibility(self):
        """Same input produces identical results."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source1 = from_list(data)
        source2 = from_list(data)
        result1 = run(
            by_llm_score(
                source1,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        result2 = run(
            by_llm_score(
                source2,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        assert result1 == result2

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"instruction": "Explain gravity"}]
        source = from_list(data)
        with pytest.raises(RuntimeError, match="missing required field 'response'"):
            run(
                by_llm_score(
                    source,
                    model=_FakeJudge(),
                    input_field="instruction",
                    output_field="response",
                )
            )

    def test_threshold_exceeding_max_score_raises(self):
        """Threshold greater than max_score raises ValueError."""
        source = from_list([])
        with pytest.raises(ValueError, match="threshold.*must not exceed.*max_score"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=6,
                max_score=5,
                input_field="instruction",
                output_field="response",
            )

    def test_forwards_temperature(self):
        """Temperature is forwarded to the model."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _FakeJudge()
        run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                temperature=0.0,
                input_field="instruction",
                output_field="response",
            )
        )
        assert judge.last_temperature == 0.0


class TestByLlmAgreementMode:
    """by_llm_score filters items by multi-judge LLM consensus."""

    def test_keeps_items_with_sufficient_agreement(self):
        """Items where enough judges agree are kept."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
        ]
        source = from_list(data)
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=1,
                agreement_threshold=0.01,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(result) == len(data)

    def test_removes_items_with_low_agreement(self):
        """Items failing consensus are removed."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
        ]
        source = from_list(data)
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=5,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(result) < len(data)

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = from_list([])
        result = run(
            by_llm_score(
                source,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_reproducibility(self):
        """Same input produces identical results."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source1 = from_list(data)
        source2 = from_list(data)
        result1 = run(
            by_llm_score(
                source1,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        result2 = run(
            by_llm_score(
                source2,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )
        )
        assert result1 == result2

    def test_num_judges_respected(self):
        """Agreement filter requests n=num_judges from the model."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
        ]
        source = from_list(data)
        judge = _FakeJudge()
        run(
            by_llm_score(
                source,
                model=judge,
                num_judges=5,
                score_threshold=1,
                agreement_threshold=0.01,
                temperature=0.7,
                input_field="instruction",
                output_field="response",
            )
        )
        assert judge.last_n == 5

    def test_forwards_temperature(self):
        """Temperature is forwarded to the model."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _FakeJudge()
        run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                agreement_threshold=0.01,
                temperature=0.7,
                input_field="instruction",
                output_field="response",
            )
        )
        assert judge.last_temperature == 0.7

    def test_invalid_num_judges_raises(self):
        """num_judges < 1 raises ValueError."""
        source = from_list([])
        with pytest.raises(ValueError, match="num_judges"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                num_judges=0,
                input_field="instruction",
                output_field="response",
            )

    def test_invalid_agreement_threshold_raises(self):
        """agreement_threshold out of range raises ValueError."""
        source = from_list([])
        with pytest.raises(ValueError, match="agreement_threshold"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                agreement_threshold=0.0,
                input_field="instruction",
                output_field="response",
            )
        with pytest.raises(ValueError, match="agreement_threshold"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                agreement_threshold=1.5,
                input_field="instruction",
                output_field="response",
            )

    def test_score_threshold_exceeding_max_score_raises(self):
        """score_threshold greater than max_score raises ValueError."""
        source = from_list([])
        with pytest.raises(
            ValueError, match="score_threshold.*must not exceed.*max_score"
        ):
            by_llm_score(
                source,
                model=_FakeJudge(),
                score_threshold=6,
                max_score=5,
                input_field="instruction",
                output_field="response",
            )

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"instruction": "Explain gravity"}]
        source = from_list(data)
        with pytest.raises(RuntimeError, match="missing required field 'response'"):
            run(
                by_llm_score(
                    source,
                    model=_FakeJudge(),
                    input_field="instruction",
                    output_field="response",
                )
            )


class TestMalformedResponses:
    """Judge handles malformed model responses gracefully."""

    def test_plain_text_response_drops_item(self):
        """Non-JSON response causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _BadJudge(["this is not json"])
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_missing_score_key_drops_item(self):
        """JSON without a score key causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _BadJudge([json.dumps({"explanation": "no score here"})])
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_non_numeric_score_drops_item(self):
        """JSON with a non-numeric score causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _BadJudge([json.dumps({"score": "high", "explanation": "good"})])
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_out_of_range_score_drops_item(self):
        """Score outside [1, max_score] causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _BadJudge([json.dumps({"score": 999, "explanation": "off scale"})])
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                max_score=5,
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_zero_score_drops_item(self):
        """Score of 0 (below minimum 1) causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = from_list(data)
        judge = _BadJudge([json.dumps({"score": 0, "explanation": "zero"})])
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert result == []

    def test_valid_items_kept_alongside_malformed(self):
        """Valid responses are kept even when others are malformed."""
        data = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        source = from_list(data)
        judge = _BadJudge(
            [
                "not json",
                json.dumps({"score": 5, "explanation": "great"}),
            ]
        )
        result = run(
            by_llm_score(
                source,
                model=judge,
                score_threshold=1,
                input_field="instruction",
                output_field="response",
            )
        )
        assert len(result) == 1
        assert result[0]["instruction"] == "Q2"

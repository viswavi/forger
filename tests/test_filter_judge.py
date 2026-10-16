"""Tests for LLM-as-a-judge filter operations."""

import hashlib
import json
import random
import re

import pytest

from pyforger import _tracking
from pyforger.filter import by_comparative_judge, by_exact_match, by_llm_score
from pyforger.filter.judge import (
    _borda_points,
    _label,
    _parse_ranking_response,
    _parse_score_response,
)
from pyforger.op import trace
from pyforger.steps import pipeline, step


class _FakeJudge:
    """Deterministic fake judge for testing.

    Returns JSON with a random score and explanation, seeded by prompt
    content for reproducibility. Matches the Model.generate_batch interface;
    the k-th repeat of a prompt within one batch is the k-th judge.
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
        self.call_count = 0  # prompts sent
        self.batch_calls = 0  # generate_batch calls
        self.last_temperature = None
        self.last_n = None
        self.prompts = []

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """Return n JSON strings with random scores per prompt."""
        self.batch_calls += 1
        self.call_count += len(prompts)
        self.last_temperature = temperature
        self.last_n = n
        self.prompts.extend(prompts)
        repeats: dict[str, int] = {}
        results = []
        for prompt in prompts:
            judge = repeats.get(str(prompt), 0)
            repeats[str(prompt)] = judge + 1
            results.append([self._response(prompt, judge * n + i) for i in range(n)])
        return results

    def _response(self, prompt, i):
        digest = hashlib.md5(f"{prompt}|{i}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        score = rng.randint(1, 5)
        words = rng.choices(self._WORD_POOL, k=2)
        return json.dumps({"score": score, "explanation": " ".join(words)})


class _BadJudge:
    """Judge that returns canned responses in order, for testing parsing."""

    def __init__(self, responses):
        """Store canned responses to return in order.

        Args:
            responses: List of strings to return, one per request, cycling.
        """
        self._responses = list(responses)
        self._idx = 0
        self.prompts = []

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """Return canned responses."""
        self.prompts.extend(prompts)
        results = []
        for _ in prompts:
            completions = []
            for _ in range(n):
                completions.append(self._responses[self._idx % len(self._responses)])
                self._idx += 1
            results.append(completions)
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
        source = data
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert len(result) == len(data)

    def test_removes_low_scoring_items(self):
        """Items scoring below threshold are removed."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source = data
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=5,
            input_field="instruction",
            output_field="response",
        )
        assert len(result) < len(data)

    def test_threshold_affects_filtering(self):
        """A stricter threshold keeps fewer items."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source = data
        strict = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=5,
            input_field="instruction",
            output_field="response",
        )
        lenient = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert len(strict) <= len(lenient)

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_reproducibility(self):
        """Same input produces identical results."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source1 = list(data)
        source2 = list(data)
        result1 = by_llm_score(
            source1,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        result2 = by_llm_score(
            source2,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        assert result1 == result2

    def test_missing_field_raises_error(self):
        """Missing expected field raises an error."""
        data = [{"instruction": "Explain gravity"}]
        source = data
        with pytest.raises(KeyError, match="missing required field 'response'"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )

    def test_threshold_exceeding_max_score_raises(self):
        """Threshold greater than max_score raises ValueError."""
        source = []
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
        source = data
        judge = _FakeJudge()
        by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            temperature=0.0,
            input_field="instruction",
            output_field="response",
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
        source = data
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=1,
            agreement_threshold=0.01,
            input_field="instruction",
            output_field="response",
        )
        assert len(result) == len(data)

    def test_removes_items_with_low_agreement(self):
        """Items failing consensus are removed."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
        ]
        source = data
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            score_threshold=5,
            input_field="instruction",
            output_field="response",
        )
        assert len(result) < len(data)

    def test_empty_input(self):
        """Empty input returns empty list."""
        source = []
        result = by_llm_score(
            source,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_reproducibility(self):
        """Same input produces identical results."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
            {"instruction": "What is 2+2?", "response": "The answer is 4."},
            {"instruction": "Write a poem", "response": "Roses are red..."},
        ]
        source1 = data
        source2 = data
        result1 = by_llm_score(
            source1,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        result2 = by_llm_score(
            source2,
            model=_FakeJudge(),
            input_field="instruction",
            output_field="response",
        )
        assert result1 == result2

    def test_num_judges_respected(self):
        """Each of num_judges judges is its own n=1 request in one batch."""
        data = [
            {"instruction": "Explain gravity", "response": "Gravity is a force..."},
        ]
        source = data
        judge = _FakeJudge()
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
        assert (judge.last_n, judge.call_count, judge.batch_calls) == (1, 5, 1)

    def test_forwards_temperature(self):
        """Temperature is forwarded to the model."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _FakeJudge()
        by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            agreement_threshold=0.01,
            temperature=0.7,
            input_field="instruction",
            output_field="response",
        )
        assert judge.last_temperature == 0.7

    def test_invalid_num_judges_raises(self):
        """num_judges < 1 raises ValueError."""
        source = []
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
        source = []
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
        source = []
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
        source = data
        with pytest.raises(KeyError, match="missing required field 'response'"):
            by_llm_score(
                source,
                model=_FakeJudge(),
                input_field="instruction",
                output_field="response",
            )


class TestMalformedResponses:
    """Judge handles malformed model responses gracefully."""

    def test_plain_text_response_drops_item(self):
        """Non-JSON response causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _BadJudge(["this is not json"])
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_missing_score_key_drops_item(self):
        """JSON without a score key causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _BadJudge([json.dumps({"explanation": "no score here"})])
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_non_numeric_score_drops_item(self):
        """JSON with a non-numeric score causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _BadJudge([json.dumps({"score": "high", "explanation": "good"})])
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_out_of_range_score_drops_item(self):
        """Score outside [1, max_score] causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _BadJudge([json.dumps({"score": 999, "explanation": "off scale"})])
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            max_score=5,
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_zero_score_drops_item(self):
        """Score of 0 (below minimum 1) causes the item to be dropped."""
        data = [{"instruction": "Q", "response": "A"}]
        source = data
        judge = _BadJudge([json.dumps({"score": 0, "explanation": "zero"})])
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert result == []

    def test_valid_items_kept_alongside_malformed(self):
        """Valid responses are kept even when others are malformed."""
        data = [
            {"instruction": "Q1", "response": "A1"},
            {"instruction": "Q2", "response": "A2"},
        ]
        source = data
        judge = _BadJudge(
            [
                "not json",
                json.dumps({"score": 5, "explanation": "great"}),
            ]
        )
        result = by_llm_score(
            source,
            model=judge,
            score_threshold=1,
            input_field="instruction",
            output_field="response",
        )
        assert len(result) == 1
        assert result[0]["instruction"] == "Q2"


class _KeyedJudge(_FakeJudge):
    """_FakeJudge with a stable cache key, so call_count does not change the key."""

    def cache_key(self):
        """Exclude the mutable call counters from the @op cache key."""
        return ("_KeyedJudge",)


_THRESHOLD_DATA = [
    {"instruction": "Explain gravity", "response": "Gravity is a force..."},
    {"instruction": "What is 2+2?", "response": "The answer is 4."},
    {"instruction": "Write a poem", "response": "Roses are red..."},
    {"instruction": "Name a color", "response": "Blue."},
]


class TestByLlmScoreCacheAndScores:
    """by_llm_score caches judge scores apart from thresholds and records them."""

    def test_changing_threshold_reuses_judge_scores(self):
        """A new score_threshold or agreement_threshold makes no new judge calls."""
        judge = _KeyedJudge()
        kwargs = dict(model=judge, input_field="instruction", output_field="response")
        lenient = by_llm_score(_THRESHOLD_DATA, score_threshold=1, **kwargs)
        assert judge.call_count == len(_THRESHOLD_DATA)
        strict = by_llm_score(_THRESHOLD_DATA, score_threshold=5, **kwargs)
        assert judge.call_count == len(_THRESHOLD_DATA)
        assert len(strict) <= len(lenient) == len(_THRESHOLD_DATA)

    def test_scores_recorded_in_step(self, monkeypatch):
        """Inside a step, the judge's per-row scores land on the op's record."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        judge = _KeyedJudge()

        @step
        def judged(rows):
            return by_llm_score(
                rows,
                model=judge,
                score_threshold=3,
                input_field="instruction",
                output_field="response",
            )

        kept = judged(_THRESHOLD_DATA)
        scores = _tracking.step_records()["judged"].ops[0].scores
        assert scores is not None
        assert (scores.label, scores.keep, scores.threshold) == ("score", "ge", 3)
        assert len(scores.values) == len(_THRESHOLD_DATA)
        assert sum(1 for v in scores.values if v >= 3) == len(kept)

    def test_unparsed_scores_recorded_as_none(self, monkeypatch):
        """An unparseable judge response is recorded as a None score."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        judge = _BadJudge(["not json", json.dumps({"score": 4})])

        @step
        def judged_bad(rows):
            return by_llm_score(
                rows, model=judge, input_field="instruction", output_field="response"
            )

        judged_bad(_THRESHOLD_DATA[:2])
        scores = _tracking.step_records()["judged_bad"].ops[0].scores
        assert scores is not None and scores.values == [None, 4]

    def test_no_cache_asks_the_judges_again(self):
        """_no_cache=True re-runs the judge calls, not only the threshold step."""
        judge = _KeyedJudge()
        kwargs = dict(model=judge, input_field="instruction", output_field="response")
        by_llm_score(_THRESHOLD_DATA, **kwargs)
        assert judge.call_count == len(_THRESHOLD_DATA)
        by_llm_score(_THRESHOLD_DATA, _no_cache=True, **kwargs)
        assert judge.call_count == 2 * len(_THRESHOLD_DATA)

    def test_several_judges_record_the_approval_fraction(self, monkeypatch):
        """With num_judges > 1 the recorded score is what agreement_threshold sees."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")
        judge = _BadJudge([json.dumps({"score": 5}), json.dumps({"score": 2})])

        @step
        def split_vote(rows):
            return by_llm_score(
                rows,
                model=judge,
                input_field="instruction",
                output_field="response",
                score_threshold=4,
                num_judges=2,
                agreement_threshold=0.5,
                temperature=0.7,
            )

        assert len(split_vote(_THRESHOLD_DATA[:1])) == 1
        scores = _tracking.step_records()["split_vote"].ops[0].scores
        assert scores is not None
        assert (scores.label, scores.keep, scores.threshold) == (
            "approval >=4",
            "ge",
            0.5,
        )
        assert scores.values == [0.5]


# --------------------------------------------------------------- custom prompts

_ROWS = [
    {"q": "What is 2+2?", "a": "4", "topic": "math"},
    {"q": "Capital of France?", "a": "Paris", "topic": "geography"},
]


class TestByLlmScorePromptTemplate:
    """prompt_template replaces the default judge prompt."""

    def test_default_prompt_unchanged(self):
        """Without a template the prompt is the original fixed text."""
        judge = _FakeJudge()
        by_llm_score(_ROWS[:1], model=judge, input_field="q", output_field="a")
        assert judge.prompts == [
            "Rate the quality of the following output on a scale of 1 to 5.\n"
            "Criteria: overall quality, relevance, and correctness\n\n"
            "Input: What is 2+2?\n\n"
            "Output: 4\n\n"
            'Respond with JSON: {"score": <integer>, "explanation": "<text>"}'
        ]

    def test_string_template_fills_placeholders_and_appends_format(self):
        """Placeholders and row fields are filled; the JSON instruction is added."""
        judge = _FakeJudge()
        by_llm_score(
            _ROWS[:1],
            model=judge,
            input_field="q",
            output_field="a",
            criteria="accuracy",
            max_score=10,
            score_threshold=1,
            prompt_template="[{topic}] Grade {output} for {input} by {criteria} "
            "out of {max_score}.",
        )
        assert judge.prompts == [
            "[math] Grade 4 for What is 2+2? by accuracy out of 10.\n\n"
            'Respond with JSON: {"score": <integer from 1 to 10>, '
            '"explanation": "<text>"}'
        ]

    def test_string_template_with_score_key_is_left_alone(self):
        """A template that already shows the "score" key gets nothing appended."""
        judge = _FakeJudge()
        by_llm_score(
            _ROWS[:1],
            model=judge,
            input_field="q",
            output_field="a",
            prompt_template='Grade {output}. Answer {{"score": N}}.',
        )
        assert judge.prompts == ['Grade 4. Answer {"score": N}.']

    def test_callable_template_gets_row_and_placeholders(self):
        """A callable receives the row plus input/output/criteria/max_score."""
        seen = []

        def template(row):
            seen.append(row)
            return [
                {"role": "system", "content": "You are strict."},
                {"role": "user", "content": f"Grade {row['output']}"},
            ]

        judge = _FakeJudge()
        by_llm_score(
            _ROWS[:1], model=judge, input_field="q", output_field="a", criteria="c"
        )
        by_llm_score(
            _ROWS[:1],
            model=judge,
            input_field="q",
            output_field="a",
            criteria="c",
            prompt_template=template,
        )
        assert seen == [
            {**_ROWS[0], "input": "What is 2+2?", "output": "4", "criteria": "c"}
            | {"max_score": 5}
        ]
        messages = judge.prompts[-1]
        assert messages[0] == {"role": "system", "content": "You are strict."}
        assert messages[1]["content"].startswith("Grade 4\n\nRespond with JSON:")

    def test_template_changes_the_cache_key(self):
        """A different template re-asks the judge."""
        judge = _KeyedJudge()
        kwargs = dict(model=judge, input_field="q", output_field="a")
        by_llm_score(_ROWS, prompt_template="A {output}", **kwargs)
        by_llm_score(_ROWS, prompt_template="B {output}", **kwargs)
        assert judge.call_count == 2 * len(_ROWS)


class TestByLlmScoreBatching:
    """All judge calls go through one generate_batch call."""

    def test_one_batch_for_all_rows_and_judges(self):
        """4 rows x 3 judges is one batch of 12 n=1 requests."""
        judge = _FakeJudge()
        by_llm_score(
            _THRESHOLD_DATA,
            model=judge,
            input_field="instruction",
            output_field="response",
            num_judges=3,
            agreement_threshold=0.5,
            temperature=0.7,
        )
        assert (judge.batch_calls, judge.call_count, judge.last_n) == (1, 12, 1)
        # Each row's prompt is repeated once per judge, rows in order.
        assert judge.prompts[0] == judge.prompts[1] == judge.prompts[2]
        assert judge.prompts[2] != judge.prompts[3]

    def test_responses_are_assigned_to_their_rows(self):
        """Judge k of row i gets the (i * num_judges + k)-th response."""
        judge = _BadJudge(
            [
                json.dumps({"score": 5}),
                json.dumps({"score": 5}),
                json.dumps({"score": 1}),
                json.dumps({"score": 5}),
            ]
        )
        kept = by_llm_score(
            _THRESHOLD_DATA[:2],
            model=judge,
            input_field="instruction",
            output_field="response",
            num_judges=2,
            temperature=0.7,
        )
        assert kept == _THRESHOLD_DATA[:1]

    def test_empty_input_makes_no_batch_call(self):
        """No rows, no generate_batch call."""
        judge = _FakeJudge()
        by_llm_score([], model=judge, input_field="q", output_field="a")
        assert judge.batch_calls == 0


class TestLenientParsing:
    """Judge scores are read from JSON in fences, prose, or a score pattern."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ('{"score": 4, "explanation": "ok"}', (4, "ok")),
            ('```json\n{"score": 3, "explanation": "fine"}\n```', (3, "fine")),
            (
                'Here is my verdict: {"score": 5, "explanation": "great"} Thanks!',
                (5, "great"),
            ),
            ('Note {"draft": true} then {"score": 2}', (2, None)),
            ('{"result": {"score": 4, "explanation": "nested"}}', (4, "nested")),
            ('{"explanation": "cut off", "score": 4', (4, "cut off")),
            ("{'score': 3, 'explanation': 'python dict'}", (3, None)),
            ('{"score": "4"}', (4, None)),
            ('{"score": 4.7}', (4.7, None)),
            ('{"score": "high"}', (None, None)),
            ('{"score": 9}', (None, None)),
            ('{"score": true}', (None, None)),
            ("I would give it four out of five.", (None, None)),
            ("", (None, None)),
        ],
    )
    def test_parse_cases(self, raw, expected):
        """Each response parses to the expected (score, explanation)."""
        assert _parse_score_response(raw, 5) == expected

    def test_fenced_response_keeps_the_row(self):
        """A fenced JSON answer counts as a valid score in the filter."""
        judge = _BadJudge(['Sure.\n```json\n{"score": 5, "explanation": "x"}\n```'])
        kept = by_llm_score(
            _THRESHOLD_DATA[:1],
            model=judge,
            input_field="instruction",
            output_field="response",
            score_threshold=4,
        )
        assert kept == _THRESHOLD_DATA[:1]


class TestScoreAndExplanationFields:
    """score_field and explanation_field annotate kept rows."""

    def test_default_rows_unchanged(self):
        """Without the fields, kept rows are the input rows."""
        judge = _BadJudge([json.dumps({"score": 5, "explanation": "good"})])
        kept = by_llm_score(
            _THRESHOLD_DATA[:1],
            model=judge,
            input_field="instruction",
            output_field="response",
        )
        assert kept == _THRESHOLD_DATA[:1]

    def test_kept_rows_carry_mean_score_and_explanations(self):
        """The mean of parsed scores and the parsed explanations are added."""
        judge = _BadJudge(
            [
                json.dumps({"score": 5, "explanation": "clear"}),
                "garbage",
                json.dumps({"score": 4, "explanation": "mostly right"}),
            ]
        )
        rows = [dict(r) for r in _THRESHOLD_DATA[:1]]
        kept = by_llm_score(
            rows,
            model=judge,
            input_field="instruction",
            output_field="response",
            num_judges=3,
            agreement_threshold=0.5,
            temperature=0.7,
            score_field="judge_score",
            explanation_field="judge_why",
        )
        assert kept == [
            {
                **_THRESHOLD_DATA[0],
                "judge_score": 4.5,
                "judge_why": ["clear", "mostly right"],
            }
        ]
        assert rows == _THRESHOLD_DATA[:1]  # input rows are not mutated

    def test_dropped_rows_get_no_fields(self):
        """Only kept rows are annotated."""
        judge = _BadJudge([json.dumps({"score": 1}), json.dumps({"score": 5})])
        kept = by_llm_score(
            _THRESHOLD_DATA[:2],
            model=judge,
            input_field="instruction",
            output_field="response",
            score_field="s",
        )
        assert kept == [{**_THRESHOLD_DATA[1], "s": 5.0}]


# ---------------------------------------------------------- comparative judge

_CANDIDATE_RE = re.compile(r"Candidate ([A-Z]+):\n(.*?)(?=\n\n|$)", re.S)


def _shown(prompt):
    """Return ``{label: candidate text}`` as shown in a comparative prompt."""
    return dict(_CANDIDATE_RE.findall(prompt))


class _RankingJudge:
    """Comparative judge that ranks candidates by a preference list.

    ``preferences[k]`` orders candidate texts best first for the k-th request
    (cycling); by default it prefers longer texts. Stable cache key.
    """

    def __init__(self, preferences=None, wrap="{}"):
        """Store the preference lists and the response wrapper."""
        self.preferences = preferences
        self.wrap = wrap
        self.prompts = []
        self.batch_calls = 0
        self.last_n = None

    def cache_key(self):
        """Exclude mutable call records from the @op cache key."""
        return ("_RankingJudge", repr(self.preferences), self.wrap)

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """Answer each prompt with a JSON ranking of its labels."""
        self.batch_calls += 1
        self.last_n = n
        results = []
        for prompt in prompts:
            k = len(self.prompts)
            self.prompts.append(prompt)
            shown = _shown(prompt)
            if self.preferences is None:
                order = sorted(shown, key=lambda lab: -len(shown[lab]))
            else:
                pref = self.preferences[k % len(self.preferences)]
                order = sorted(shown, key=lambda lab: pref.index(shown[lab]))
            body = json.dumps({"ranking": order, "explanation": "because"})
            results.append([self.wrap.format(body)] * n)
        return results


_GROUPED = [
    {"q": "q1", "a": "short"},
    {"q": "q1", "a": "a much longer answer"},
    {"q": "q2", "a": "only one"},
    {"q": "q1", "a": "medium answer"},
    {"q": "q3", "a": "x"},
    {"q": "q3", "a": "xyz"},
]


def _comparative(rows, judge, **kwargs):
    kwargs.setdefault("keep", 1)
    if kwargs.get("num_judges", 1) > 1:
        kwargs.setdefault("temperature", 0.7)
    return by_comparative_judge(
        rows, group_by="q", model=judge, input_field="q", output_field="a", **kwargs
    )


class TestByComparativeJudge:
    """by_comparative_judge keeps the best candidates of each group."""

    def test_keeps_best_per_group_in_input_order(self):
        """The judge's favorite of each group survives; singletons are kept."""
        judge = _RankingJudge()
        kept = _comparative(_GROUPED, judge)
        assert kept == [_GROUPED[1], _GROUPED[2], _GROUPED[5]]

    def test_keep_two(self):
        """keep=2 keeps the top two of each group."""
        kept = _comparative(_GROUPED, _RankingJudge(), keep=2)
        assert kept == [_GROUPED[1], _GROUPED[2], _GROUPED[3], _GROUPED[4], _GROUPED[5]]

    def test_one_batch_one_prompt_per_group_and_judge(self):
        """Groups larger than keep get num_judges n=1 prompts in one batch."""
        judge = _RankingJudge()
        _comparative(_GROUPED, judge, num_judges=3)
        assert (judge.batch_calls, len(judge.prompts), judge.last_n) == (1, 6, 1)
        q1_prompts = [p for p in judge.prompts if "Input: q1" in p]
        assert len(q1_prompts) == 3
        assert all("Input: q1" in p and p.count("Input:") == 1 for p in q1_prompts)
        assert all(set(_shown(p)) == {"A", "B", "C"} for p in q1_prompts)

    def test_only_single_rows_skip_the_call(self):
        """Groups of two or more are judged even when keep covers them."""
        judge = _RankingJudge()
        kept = _comparative(_GROUPED, judge, keep=3)
        assert kept == _GROUPED
        assert [p.split("Input: ")[1].split("\n")[0] for p in judge.prompts] == [
            "q1",
            "q3",
        ]

    def test_all_single_rows_make_no_calls(self):
        """Rows with distinct group values are kept without any judge call."""
        judge = _RankingJudge()
        rows = [{"q": f"q{i}", "a": "x"} for i in range(3)]
        assert _comparative(rows, judge) == rows
        assert judge.prompts == []

    def test_borda_combines_judges(self):
        """Borda points: x 2+1+0, y 1+2+2, z 0+0+1, so y then x."""
        rows = [{"q": "q", "a": t} for t in ("x", "y", "z")]
        prefs = [["x", "y", "z"], ["y", "x", "z"], ["y", "z", "x"]]
        assert _comparative(rows, _RankingJudge(prefs), num_judges=3) == [rows[1]]
        kept = _comparative(rows, _RankingJudge(prefs), num_judges=3, keep=2)
        assert kept == [rows[0], rows[1]]

    def test_ties_go_to_the_earlier_row(self):
        """Two judges with opposite rankings tie; the first row wins."""
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        prefs = [["y", "x"], ["x", "y"]]
        assert _comparative(rows, _RankingJudge(prefs), num_judges=2) == [rows[0]]

    def test_shuffle_is_seeded(self):
        """The same seed shows the same orders; judges and seeds differ."""
        rows = [{"q": "q", "a": f"answer {i}"} for i in range(6)]

        def orders(seed):
            judge = _RankingJudge()
            _comparative(rows, judge, num_judges=3, seed=seed, _no_cache=True)
            return [list(_shown(p).values()) for p in judge.prompts]

        first = orders(0)
        assert orders(0) == first
        assert len({tuple(o) for o in first}) > 1  # judges see different orders
        assert orders(1) != first
        assert all(sorted(o) == sorted(r["a"] for r in rows) for o in first)

    def test_position_bias_is_spread_by_shuffling(self):
        """A judge that always prefers label A no longer always picks row 0."""
        rows = [
            {"q": f"q{g}", "a": f"g{g} answer {i}"} for g in range(8) for i in range(3)
        ]

        class _FirstLabelJudge(_RankingJudge):
            def generate_batch(self, prompts, *, n=1, **kwargs):
                body = json.dumps({"ranking": ["A", "B", "C"]})
                return [[body] * n for _ in prompts]

        kept = _comparative(rows, _FirstLabelJudge())
        assert len(kept) == 8
        assert any(not r["a"].endswith("answer 0") for r in kept)

    def test_changing_keep_reuses_judge_calls(self):
        """A new keep makes no new calls."""
        rows = [{"q": f"q{g}", "a": "a" * (i + 1)} for g in range(3) for i in range(4)]
        judge = _RankingJudge()
        one = _comparative(rows, judge, keep=1)
        assert len(judge.prompts) == 3
        two = _comparative(rows, judge, keep=2)
        assert len(judge.prompts) == 3
        assert len(one) == 3 and len(two) == 6
        hits = [r.cache_hit for r in trace() if r.op_name.endswith(".rankings")]
        assert hits == [False, True]

    def test_changing_keep_with_mixed_group_sizes_reuses_judge_calls(self):
        """Groups of 2, 3, and 4 rows: keep 1, 2, 3, 4 all reuse the first calls."""
        rows = [
            {"q": f"q{size}", "a": "a" * (i + 1)}
            for size in (2, 3, 4)
            for i in range(size)
        ]
        judge = _RankingJudge()
        kept = [len(_comparative(rows, judge, keep=keep)) for keep in (1, 2, 3, 4)]
        assert kept == [3, 6, 8, 9]
        assert len(judge.prompts) == 3
        hits = [r.cache_hit for r in trace() if r.op_name.endswith(".rankings")]
        assert hits == [False, True, True, True]

    def test_rank_and_explanation_fields(self):
        """Kept rows carry their rank and the group's judge explanations."""
        judge = _RankingJudge()
        kept = _comparative(
            _GROUPED,
            judge,
            keep=2,
            num_judges=2,
            rank_field="rank",
            explanation_field="why",
        )
        assert kept == [
            {**_GROUPED[1], "rank": 1, "why": ["because", "because"]},
            {**_GROUPED[2], "rank": 1, "why": []},
            {**_GROUPED[3], "rank": 2, "why": ["because", "because"]},
            {**_GROUPED[4], "rank": 2, "why": ["because", "because"]},
            {**_GROUPED[5], "rank": 1, "why": ["because", "because"]},
        ]

    def test_explanations_skip_unparsed_judges(self):
        """Only judges whose ranking parsed contribute an explanation."""
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        judge = _BadJudge(
            ['{"ranking": ["B", "A"], "explanation": "B is right"}', "garbage"]
        )
        kept = _comparative(rows, judge, num_judges=2, explanation_field="why")
        assert len(kept) == 1 and kept[0]["why"] == ["B is right"]

    def test_default_rows_unchanged(self):
        """Without rank_field or explanation_field, kept rows are input rows."""
        kept = _comparative(_GROUPED, _RankingJudge())
        assert all(any(row is k for row in _GROUPED) for k in kept)

    def test_unparseable_group_is_dropped(self):
        """A group whose judges all answer garbage is dropped and recorded None."""
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        assert _comparative(rows, _BadJudge(["no idea"])) == []

    def test_lenient_ranking_response(self):
        """Rankings in fences with label variants are understood."""
        rows = [{"q": "q", "a": t} for t in ("a", "bbb", "cc")]
        judge = _RankingJudge(wrap="Ranking below.\n```json\n{}\n```")
        assert _comparative(rows, judge) == [rows[1]]

    def test_custom_prompt_template(self):
        """A template gets input/candidates/labels; the format line is appended."""
        judge = _RankingJudge()
        rows = [{"q": "q", "a": t, "lang": "en"} for t in ("x", "yy")]
        _comparative(
            rows,
            judge,
            prompt_template="[{lang}] Which of {labels} answers {input} best?\n\n"
            "{candidates}",
        )
        (prompt,) = judge.prompts
        assert prompt.startswith("[en] Which of A, B answers q best?\n\nCandidate ")
        assert prompt.endswith('"explanation": "<text>"}')
        assert '"ranking"' in prompt

    def test_callable_prompt_template(self):
        """A callable receives the first row plus the placeholders."""
        seen = []

        def template(ctx):
            seen.append(ctx)
            return (
                f'Rank {ctx["labels"]}; reply {{"ranking": [...]}}\n{ctx["candidates"]}'
            )

        rows = [{"q": "q", "a": t} for t in ("x", "yy")]
        judge = _RankingJudge()
        assert _comparative(rows, judge, prompt_template=template) == [rows[1]]
        assert seen[0]["num_candidates"] == 2 and seen[0]["input"] == "q"
        assert "Respond with JSON" not in judge.prompts[0]

    def test_records_rank_scores_in_step(self, monkeypatch):
        """The step report gets each row's rank, kept when at most keep."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def best(rows):
            return _comparative(rows, _RankingJudge(), keep=1, min_yield=0.1)

        kept = best(_GROUPED)
        record = _tracking.step_records()["best"].ops[0]
        assert record.op_name == "filter.by_comparative_judge"
        assert record.scores is not None
        assert (record.scores.label, record.scores.keep, record.scores.threshold) == (
            "rank in group",
            "le",
            1.0,
        )
        assert record.scores.values == [3, 1, 1, 2, 2, 1]
        assert len(kept) == 3

    def test_counts_as_quality_filter(self, monkeypatch):
        """A pipeline using it gets no "no quality filter" warning."""
        monkeypatch.setenv("FORGER_STEP_REPORT", "0")

        @step
        def best(rows):
            return _comparative(rows, _RankingJudge())

        @step
        def dedupe(rows):
            return by_exact_match(rows, field="a")

        @pipeline
        def run(rows):
            return dedupe(best(rows))

        run(_GROUPED)
        rec = _tracking.last_pipeline()
        assert rec is not None
        assert not any("quality filter" in w for w in rec.warnings)

    def test_invalid_arguments(self):
        """Both keep and num_judges must be at least 1."""
        with pytest.raises(ValueError, match="keep"):
            _comparative(_GROUPED, _RankingJudge(), keep=0)
        with pytest.raises(ValueError, match="num_judges"):
            _comparative(_GROUPED, _RankingJudge(), num_judges=0)

    def test_missing_group_field_raises(self):
        """A row without the group_by field raises a KeyError."""
        with pytest.raises(KeyError, match="missing required field 'q'"):
            _comparative([{"a": "x"}], _RankingJudge())


class TestRankingHelpers:
    """Ranking parsing, Borda points, and labels."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ('{"ranking": ["B", "A", "C"]}', [1, 0, 2]),
            ('{"ranking": ["Candidate C", "(a)", "b."]}', [2, 0, 1]),
            ('{"ranking": "B > C > A"}', [1, 2, 0]),
            ('{"ranking": ["B", "B", "A"]}', [1, 0]),
            ('{"ranking": ["B", "Z", "A"]}', None),
            ('{"explanation": "x", "ranking": ["C", "A"', [2, 0]),
            ('{"ranking": []}', None),
            ("B is best", None),
        ],
    )
    def test_parse_ranking(self, raw, expected):
        """Labels map to candidates; unknown and repeated labels are skipped."""
        assert _parse_ranking_response(raw, [0, 1, 2]) == expected

    def test_parse_ranking_follows_the_shown_order(self):
        """Label k means the candidate shown k-th, not the k-th row."""
        assert _parse_ranking_response('{"ranking": ["A", "B"]}', [1, 0]) == [1, 0]

    def test_borda_points(self):
        """Full rankings score m-1-p; left-out candidates share the rest."""
        assert _borda_points([1, 0, 2], 3) == {1: 2.0, 0: 1.0, 2: 0.0}
        assert _borda_points([2], 3) == {2: 2.0, 0: 0.5, 1: 0.5}

    def test_labels(self):
        """Labels run A..Z then AA, AB."""
        assert [_label(k) for k in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


# ------------------------------------------------------- review regressions


class _AnnotatingJudge(_RankingJudge):
    """Ranks the candidate "BEST" first, labelled with an annotation."""

    def __init__(self, annotation):
        """Store the annotation format, e.g. ``"{} (best)"``."""
        super().__init__()
        self.annotation = annotation

    def cache_key(self):
        """Stable key per annotation."""
        return ("_AnnotatingJudge", self.annotation)

    def generate_batch(self, prompts, *, n=1, temperature=None, max_tokens=None):
        """Put BEST first with an annotated label, then the rest bare."""
        results = []
        for prompt in prompts:
            self.prompts.append(prompt)
            shown = _shown(prompt)
            best = next(lab for lab, text in shown.items() if text == "BEST")
            rest = [lab for lab in shown if lab != best]
            ranking = [self.annotation.format(best), *rest]
            results.append([json.dumps({"ranking": ranking})] * n)
        return results


class TestAnnotatedLabels:
    """#20: labels with words after them map to their label."""

    @pytest.mark.parametrize(
        "entry", ["A (best)", "Candidate A:", "A - clearest", "a) most accurate"]
    )
    def test_annotated_label_maps(self, entry):
        """The annotated first entry still names candidate A."""
        raw = json.dumps({"ranking": [entry, "C", "B"]})
        assert _parse_ranking_response(raw, [2, 0, 1]) == [2, 1, 0]

    @pytest.mark.parametrize(
        "annotation", ["{} (best)", "Candidate {}:", "{} - clearest"]
    )
    def test_top_pick_is_kept(self, annotation):
        """The judge's annotated top pick wins its group."""
        rows = [{"q": "q", "a": t} for t in ("meh-1", "BEST", "meh-2")]
        kept = _comparative(rows, _AnnotatingJudge(annotation), rank_field="rank")
        assert kept == [{"q": "q", "a": "BEST", "rank": 1}]

    def test_unmappable_entry_makes_the_answer_unparseable(self):
        """An entry naming no label never demotes the top pick; it voids the answer."""
        raw = json.dumps({"ranking": ["the first one", "C", "B"]})
        assert _parse_ranking_response(raw, [2, 0, 1]) is None
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        assert _comparative(rows, _BadJudge([raw])) == []


class TestThinkBlocks:
    """#21: only the text after a closed </think> block is parsed."""

    def test_score_draft_in_think_is_ignored(self):
        """A draft score inside <think> does not count, in either parsing path."""
        raw = '<think>Draft: {"score": 2, "explanation": "first"}</think>\n'
        assert _parse_score_response(
            raw + '{"score": 5, "explanation": "final"}', 5
        ) == (
            5,
            "final",
        )
        assert _parse_score_response(
            '<think>maybe "score": 2</think> "score": 5', 5
        ) == (
            5,
            None,
        )

    def test_unclosed_think_is_unparseable(self):
        """An unclosed <think> block gives no score and no ranking."""
        assert _parse_score_response('<think>{"score": 5}', 5) == (None, None)
        assert _parse_ranking_response('<think>{"ranking": ["A"]}', [0, 1]) is None

    def test_ranking_draft_in_think_is_ignored(self):
        """The final ranking after </think> is used, not the draft."""
        raw = '<think>{"ranking": ["C","B","A"]}</think>{"ranking": ["A","B","C"]}'
        assert _parse_ranking_response(raw, [2, 0, 1]) == [2, 0, 1]

    def test_last_json_object_wins(self):
        """Outside think blocks the last object with the key counts, like the regex."""
        assert _parse_score_response('draft {"score": 2} final {"score": 5}', 5)[0] == 5
        assert _parse_score_response('{"score": 4, "detail": {"score": 2}}', 5)[0] == 4

    def test_by_llm_score_uses_the_final_answer(self):
        """The repro: final score 5 after a draft of 1 keeps the row."""
        judge = _BadJudge(
            [
                '<think>A first guess would be {"score": 1, "explanation": "x"}'
                '</think>\n{"score": 5, "explanation": "final"}'
            ]
        )
        kept = by_llm_score(
            _THRESHOLD_DATA[:1],
            model=judge,
            input_field="instruction",
            output_field="response",
            score_threshold=4,
            explanation_field="why",
        )
        assert kept == [{**_THRESHOLD_DATA[0], "why": ["final"]}]

    def test_comparative_unclosed_think_drops_the_group(self):
        """A comparative answer with an unclosed <think> counts as unparseable."""
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        assert _comparative(rows, _BadJudge(['<think>{"ranking": ["A", "B"]}'])) == []


class TestComparativeJudgesDiffer:
    """#37: several comparative judges never get identical prompts at temperature 0."""

    def test_several_judges_need_temperature(self):
        """num_judges > 1 at temperature 0 raises, like by_llm_score."""
        rows = [{"q": "q", "a": t} for t in ("x", "y")]
        with pytest.raises(ValueError, match="temperature must be > 0.0"):
            by_comparative_judge(
                rows,
                group_by="q",
                model=_RankingJudge(),
                input_field="q",
                output_field="a",
                num_judges=3,
            )

    @pytest.mark.parametrize("size", [2, 3, 4])
    def test_rotations_balance_positions(self, size):
        """With size judges, every candidate is shown once in every position."""
        rows = [{"q": "q", "a": f"c{i}"} for i in range(size)]
        judge = _RankingJudge()
        _comparative(rows, judge, num_judges=size)
        orders = [list(_shown(p).values()) for p in judge.prompts]
        assert len(set(map(tuple, orders))) == size
        for position in range(size):
            assert sorted(o[position] for o in orders) == sorted(r["a"] for r in rows)


class TestScoreValues:
    """#38: both parsing paths read scores the same way; fractions are kept."""

    @pytest.mark.parametrize(
        "json_raw, text_raw, expected",
        [
            ('{"score": 4.9}', 'score is "score": 4.9 here', 4.9),
            ('{"score": "4/5"}', '"score": "4/5"', 4),
            ('{"score": "4.5 out of 5"}', '"score": "4.5 out of 5"', 4.5),
            ('{"score": 5}', '"score": 5', 5),
            ('{"score": 0.5}', '"score": 0.5', None),
        ],
    )
    def test_json_and_regex_paths_agree(self, json_raw, text_raw, expected):
        """The JSON object and the regex fallback give the same score."""
        assert _parse_score_response(json_raw, 5)[0] == expected
        assert _parse_score_response(text_raw, 5)[0] == expected

    def test_fraction_compared_unrounded(self):
        """4.9 is below score_threshold 5; score_field keeps the fractional mean."""
        judge = _BadJudge([json.dumps({"score": 4.9}), json.dumps({"score": 4.5})])
        kept = by_llm_score(
            _THRESHOLD_DATA[:2],
            model=judge,
            input_field="instruction",
            output_field="response",
            score_threshold=5,
        )
        assert kept == []
        judge = _BadJudge([json.dumps({"score": 4.5}), json.dumps({"score": 5})])
        kept = by_llm_score(
            _THRESHOLD_DATA[:1],
            model=judge,
            input_field="instruction",
            output_field="response",
            num_judges=2,
            temperature=0.7,
            score_threshold=4,
            score_field="s",
        )
        assert kept == [{**_THRESHOLD_DATA[0], "s": 4.75}]

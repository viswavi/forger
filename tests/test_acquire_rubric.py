"""Contract, retry, batching, and cache tests for textual rubric generation."""

from __future__ import annotations

import json

import pytest
from fixtures import RecordingMockModel

from pyforger.acquire import generate_custom_rubric_criteria
from pyforger.op import trace


class ScriptedModel(RecordingMockModel):
    """Consume a batch script so retries can return different responses."""

    def __init__(self, batches):
        """Initialize scripted batches and call recording."""
        super().__init__(responses=[])
        self.batches = batches
        self.settings = []

    def generate_batch(self, prompts, **kwargs):
        """Record the request and return the next scripted batch."""
        self.recorded_prompts.append(prompts)
        self.settings.append(kwargs)
        result = self.batches[self._call_count]
        self._call_count += 1
        return result


def encoded(*pairs):
    """Encode criterion pairs in the model's expected JSON format."""
    return json.dumps({"rubric": pairs})


def test_context_candidates_order_and_output_types():
    """Preserve input context and return ordered tuples with float weights."""
    model = RecordingMockModel(
        responses=[
            encoded([" Is the advice actionable? ", 100], ["Is it empathetic?", 75.5]),
            encoded(["Does it cite relevant evidence?", 100]),
        ]
    )
    inputs = ["Career advice {context}", 'Research this topic\n"quoted"']
    candidates = [["Consider mentoring.", "Quit immediately."], []]
    result = generate_custom_rubric_criteria(
        inputs,
        model,
        meta_description="Evaluate for a general audience",
        candidate_responses=candidates,
    )
    assert result == [
        [("Is the advice actionable?", 100.0), ("Is it empathetic?", 75.5)],
        [("Does it cite relevant evidence?", 100.0)],
    ]
    assert all(
        isinstance(pair, tuple) and isinstance(pair[1], float)
        for rubric in result
        for pair in rubric
    )
    for i, prompt in enumerate(model.recorded_prompts[0]):
        assert json.loads(prompt[1]["content"]) == {
            "input": inputs[i],
            "candidate_responses": candidates[i],
            "meta_description": "Evaluate for a general audience",
        }
    assert inputs[0] == "Career advice {context}"
    assert candidates[0] == ["Consider mentoring.", "Quit immediately."]


def test_optional_context_empty_rubric_and_empty_inputs():
    """Accept absent context and empty rubrics without calling on empty inputs."""
    model = RecordingMockModel(responses=[encoded()])
    assert generate_custom_rubric_criteria([], model) == []
    assert model._call_count == 0
    assert generate_custom_rubric_criteria(["A document"], model) == [[]]
    assert json.loads(model.recorded_prompts[0][0][1]["content"]) == {
        "input": "A document"
    }


def test_only_failed_inputs_retried_in_order_with_batch_limit():
    """Retry failed inputs while preserving order and respecting batch size."""
    model = ScriptedModel(
        [
            [[encoded(["First?", 100])], [encoded(["Second?", -1])]],
            [[encoded()]],
            [[encoded(["Second?", 50])]],
        ]
    )
    result = generate_custom_rubric_criteria(
        ["first", "second", "third"],
        model,
        batch_size=2,
        max_retries=1,
        temperature=0.25,
    )
    assert result == [[("First?", 100.0)], [("Second?", 50.0)], []]
    assert [len(batch) for batch in model.recorded_prompts] == [2, 1, 1]
    assert json.loads(model.recorded_prompts[-1][0][1]["content"])["input"] == "second"
    assert all(s["n"] == 1 and s["temperature"] == 0.25 for s in model.settings)


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "null",
        "[]",
        "{}",
        '{"rubric": null}',
        '{"rubric": "text"}',
        encoded(["Criterion?", True]),
        encoded(["Criterion?", "50"]),
        encoded(["Criterion?", -1]),
        encoded(["Criterion?", 101]),
        encoded(["Criterion?", float("nan")]),
        encoded(["Criterion?", float("inf")]),
        encoded(["Criterion?", 10**400]),
        encoded(["  ", 50]),
        encoded([None, 50]),
        encoded(["Criterion?"]),
        encoded(["Criterion?", 50, "extra"]),
        encoded({"criterion": "Criterion?", "weight": 50}),
        encoded(["Valid?", 50], ["Invalid?", -1]),
    ],
)
def test_malformed_rubric_gets_fresh_retry(bad):
    """Reject malformed pairs and request fresh output on retry."""
    model = ScriptedModel([[[bad]], [[encoded(["Recovered?", 0])]]])
    assert generate_custom_rubric_criteria(["input"], model, max_retries=1) == [
        [("Recovered?", 0.0)]
    ]
    assert model._call_count == 2


@pytest.mark.parametrize("missing", [[], [[]]])
def test_missing_model_outputs_are_retried(missing):
    """Retry missing batches or completions without losing input alignment."""
    model = ScriptedModel([missing, [[encoded(["Recovered?", 100])]]])
    assert generate_custom_rubric_criteria(["input"], model, max_retries=1) == [
        [("Recovered?", 100.0)]
    ]


def test_exhaustion_reports_indices_and_does_not_cache_failure():
    """Identify exhausted inputs and avoid caching generation failures."""
    model = RecordingMockModel(responses=[encoded(), "bad"])
    for _ in range(2):
        with pytest.raises(ValueError, match=r"input indices \[1\].*1 attempts"):
            generate_custom_rubric_criteria(["first", "second"], model, max_retries=0)
    assert model._call_count == 2


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"inputs": "not a list"}, TypeError),
        ({"inputs": [42]}, TypeError),
        ({"meta_description": []}, TypeError),
        ({"candidate_responses": ["response"]}, TypeError),
        ({"candidate_responses": [[None]]}, TypeError),
        ({"candidate_responses": []}, ValueError),
        ({"candidate_responses": [[], []]}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 1.5}, ValueError),
        ({"batch_size": True}, ValueError),
        ({"max_retries": -1}, ValueError),
        ({"max_retries": 0.5}, ValueError),
    ],
)
def test_invalid_arguments_fail_before_generation(kwargs, error):
    """Reject invalid arguments before issuing any model requests."""
    model = RecordingMockModel(responses=[])
    with pytest.raises(error):
        generate_custom_rubric_criteria(
            **{"inputs": ["input"], "model": model, **kwargs}
        )
    assert model._call_count == 0


def test_cache_preserves_tuples_and_bypass_reaches_model():
    """Cache valid results while honoring bypass and changed context."""
    model = RecordingMockModel(responses=[encoded(["Criterion?", 100])])
    expected = [[("Criterion?", 100.0)]]
    assert generate_custom_rubric_criteria(["input"], model) == expected
    assert generate_custom_rubric_criteria(["input"], model) == expected
    assert trace()[-1].cache_hit
    assert model._call_count == 1
    assert generate_custom_rubric_criteria(["input"], model, _no_cache=True) == expected
    assert model._call_count == 2
    generate_custom_rubric_criteria(["input"], model, meta_description="new context")
    generate_custom_rubric_criteria(["input"], model, candidate_responses=[["example"]])
    assert model._call_count == 4

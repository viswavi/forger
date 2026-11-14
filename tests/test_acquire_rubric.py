"""Contract, retry, batching, and cache tests for textual rubric generation."""

from __future__ import annotations

import json

import pytest
from fixtures import RecordingMockModel

from pyforger.acquire import (
    convert_rubric_criteria,
    generate_custom_rubric,
    generate_custom_rubric_criteria,
)
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
        encoded(["Criterion?", -1]),  # no positive weight
        encoded(["Criterion?", 0]),
        encoded(["Zero?", 0], ["Mistake?", -50]),
        encoded(["Criterion?", -101]),
        encoded(["Criterion?", 101]),
        encoded(["Criterion?", float("nan")]),
        encoded(["Criterion?", float("inf")]),
        encoded(["Criterion?", 10**400]),
        encoded(["  ", 50]),
        encoded([None, 50]),
        encoded(["Criterion?"]),
        encoded(["Criterion?", 50, "extra"]),
        encoded({"criterion": "Criterion?", "weight": 50}),
        encoded(["Valid?", 50], ["Invalid?", -101]),
    ],
)
def test_malformed_rubric_gets_fresh_retry(bad):
    """Reject malformed pairs and request fresh output on retry."""
    model = ScriptedModel([[[bad]], [[encoded(["Recovered?", 1], ["Zero?", 0])]]])
    assert generate_custom_rubric_criteria(["input"], model, max_retries=1) == [
        [("Recovered?", 1.0), ("Zero?", 0.0)]
    ]
    assert model._call_count == 2


def test_negative_weights_parse_as_mistakes():
    """A mistake keeps its negative weight; the prompt asks for presence questions."""
    model = RecordingMockModel(
        responses=[
            encoded(
                ["Does it give a weight-based dose?", 100],
                ["Does it recommend aspirin for a child?", -75],
                ["Does it use jargon?", -100],
            )
        ]
    )
    assert generate_custom_rubric_criteria(["Child has a fever."], model) == [
        [
            ("Does it give a weight-based dose?", 100.0),
            ("Does it recommend aspirin for a child?", -75.0),
            ("Does it use jargon?", -100.0),
        ]
    ]
    system = " ".join(model.recorded_prompts[0][0][0]["content"].split())
    assert "weight from -100 to 100" in system
    assert "never as a question about its absence with a positive weight" in system
    assert 'without asking about pain history?" Split it' in system


def test_negative_criteria_false_asks_for_requirements_only():
    """Opting out sends weights 0 to 100 and retries a negative weight."""
    model = ScriptedModel(
        [
            [[encoded(["Is it kind?", 100], ["Is it rude?", -50])]],
            [[encoded(["Is it kind?", 100])]],
        ]
    )
    result = generate_custom_rubric_criteria(
        ["Comfort a friend."], model, negative_criteria=False, max_retries=1
    )
    assert result == [[("Is it kind?", 100.0)]]
    assert model._call_count == 2
    system = model.recorded_prompts[0][0][0]["content"]
    assert "weight from 0 to 100" in system
    assert "-100" not in system and "negative weight" not in system
    assert "A yes answer must indicate better response quality." in system


def test_negative_criteria_false_reaches_mixed_entry_point():
    """generate_custom_rubric passes the flag to the criteria stage."""
    model = ScriptedModel(
        [
            [[encoded(["Is it kind?", 100])]],
            [[converted("prompt", "Is it kind?")]],
        ]
    )
    result = generate_custom_rubric(
        ["Comfort a friend."], model, negative_criteria=False
    )
    assert result == [[[{"prompt": "Is it kind?"}, 100.0]]]
    assert "weight from 0 to 100" in model.recorded_prompts[0][0][0]["content"]


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


CODE = "def grade(prompt, response, ground_truth):\n    return 1.0"


def converted(kind, source=""):
    """Encode one criterion conversion in the model's expected JSON format."""
    return json.dumps({"kind": kind, "source": source})


def test_code_criteria_only_drops_uncheckable_criteria():
    """Keep code criteria, drop omitted ones, and tell both stages."""
    model = ScriptedModel(
        [
            [[encoded(["Is the final answer 42?", 100], ["Is it clear?", 50])]],
            [[converted("code", CODE)], [converted("omit")]],
        ]
    )
    result = generate_custom_rubric(["What is 6 * 7?"], model, code_criteria_only=True)
    assert result == [[[{"code": CODE}, 100.0]]]
    assert all(
        "Code-only mode" in batch[0][0]["content"] for batch in model.recorded_prompts
    )


def test_code_criteria_only_rejects_prompt_kind_and_empty_rubrics():
    """A prompted judge is retried; an all-omitted rubric raises."""
    model = ScriptedModel(
        [
            [[encoded(["Is it kind?", 100])]],
            [[converted("prompt", "Is it kind?")]],
            [[converted("omit")]],
        ]
    )
    with pytest.raises(ValueError, match=r"input indices \[0\] have no code"):
        generate_custom_rubric(
            ["Comfort a friend."], model, code_criteria_only=True, max_retries=1
        )
    assert model._call_count == 3


def test_default_mode_keeps_prompted_judges():
    """Without the flag, "omit" is invalid and prompts are kept verbatim."""
    model = ScriptedModel(
        [
            [[encoded(["Is it kind?", 100])]],
            [[converted("omit")]],
            [[converted("prompt", "Is it kind?")]],
        ]
    )
    result = generate_custom_rubric(["Comfort a friend."], model, max_retries=1)
    assert result == [[[{"prompt": "Is it kind?"}, 100.0]]]
    assert "Code-only mode" not in model.recorded_prompts[0][0][0]["content"]


GUIDELINES = "grade() may call run_program(code, stdin); no names starting with '_'."


def test_code_guidelines_reach_both_stages_system_prompts():
    """generate_custom_rubric forwards the guidelines to both stages, retries too."""
    model = ScriptedModel(
        [
            [[encoded(["  ", 100])]],  # malformed, so the criteria stage retries
            [[encoded(["Does the program print 42?", 100])]],
            [[converted("code", "x = 1")]],  # no grade(), so conversion retries
            [[converted("code", CODE)]],
        ]
    )
    result = generate_custom_rubric(
        ["Print 42."],
        model,
        code_criteria_only=True,
        code_guidelines=GUIDELINES,
        max_retries=1,
    )
    assert result == [[[{"code": CODE}, 100.0]]]
    systems = [batch[0][0]["content"] for batch in model.recorded_prompts]
    assert len(systems) == 4
    assert all(system.endswith(GUIDELINES) for system in systems)
    assert all(
        "What grading code can check in this environment" in s for s in systems[:2]
    )
    assert all("take precedence over the general" in s for s in systems[2:])
    # The guidelines are instructions, not part of the per-row user data.
    assert all(
        GUIDELINES not in batch[0][1]["content"] for batch in model.recorded_prompts
    )


def test_prompts_unchanged_without_code_guidelines():
    """Leaving out the guidelines, or passing None, sends the default prompts."""
    script = [[[encoded(["Is it kind?", 100])]], [[converted("prompt", "Is it kind?")]]]
    default, explicit = ScriptedModel(script), ScriptedModel(script)
    generate_custom_rubric(["Comfort a friend."], default)
    generate_custom_rubric(
        ["Comfort a friend."], explicit, code_guidelines=None, _no_cache=True
    )
    assert len(default.recorded_prompts) == 2
    assert default.recorded_prompts == explicit.recorded_prompts
    assert all(
        "from the caller" not in batch[0][0]["content"]
        for batch in default.recorded_prompts
    )


def test_conversion_preserves_negative_weights():
    """A mistake keeps its weight; the prompt says grade scores the mistake itself."""
    model = ScriptedModel(
        [
            [
                [
                    encoded(
                        ["Is the final answer 42?", 100],
                        ["Does it print a stack trace?", -50],
                    )
                ]
            ],
            [[converted("code", CODE)], [converted("code", CODE)]],
        ]
    )
    result = generate_custom_rubric(["Print 6 * 7."], model)
    assert result == [[[{"code": CODE}, 100.0], [{"code": CODE}, -50.0]]]
    conversion = model.recorded_prompts[1][0][0]["content"]
    assert "never invert its score" in conversion
    assert "1.0 when the response does what" in conversion


def test_convert_rubric_criteria_keeps_negative_prompt_criteria():
    """Converting a text rubric alone keeps a negative weight as given."""
    model = ScriptedModel([[[converted("prompt", "A?")], [converted("prompt", "B?")]]])
    result = convert_rubric_criteria([[("A?", 10.0), ("B?", -10.0)]], model)
    assert result == [[[{"prompt": "A?"}, 10.0], [{"prompt": "B?"}, -10.0]]]


@pytest.mark.parametrize("rubric", [[("A?", -10.0)], [("A?", 0.0), ("B?", -1.0)]])
def test_convert_rejects_rubric_without_positive_weight(rubric):
    """No response could score above 0, so conversion refuses before any call."""
    model = RecordingMockModel(responses=[])
    with pytest.raises(ValueError, match="invalid textual rubric at index 0"):
        convert_rubric_criteria([rubric], model)
    assert model._call_count == 0


def test_code_only_rubric_left_with_only_mistakes_raises():
    """Omitting the only requirement leaves nothing to score above 0."""
    model = ScriptedModel(
        [
            [[encoded(["Is it clear?", 100], ["Does it print a stack trace?", -50])]],
            [[converted("omit")], [converted("code", CODE)]],
        ]
    )
    with pytest.raises(ValueError, match=r"input indices \[0\] have no code"):
        generate_custom_rubric(["Print 42."], model, code_criteria_only=True)


def test_convert_rubric_criteria_takes_code_guidelines():
    """The conversion stage alone also accepts guidelines."""
    model = ScriptedModel([[[converted("code", CODE)]]])
    result = convert_rubric_criteria(
        [[("Does it print 42?", 100.0)]], model, code_guidelines=GUIDELINES
    )
    assert result == [[[{"code": CODE}, 100.0]]]
    assert model.recorded_prompts[0][0][0]["content"].endswith(GUIDELINES)


@pytest.mark.parametrize("bad", [["no _"], 3, {"rule": "no _"}])
@pytest.mark.parametrize(
    "call",
    [
        lambda m, kw: generate_custom_rubric_criteria(["x"], m, **kw),
        lambda m, kw: convert_rubric_criteria([[("x?", 1.0)]], m, **kw),
        lambda m, kw: generate_custom_rubric(["x"], m, **kw),
    ],
)
def test_non_string_code_guidelines_raise_type_error(call, bad):
    """Reject non-string guidelines before any model request."""
    model = RecordingMockModel(responses=[])
    with pytest.raises(TypeError, match="code_guidelines"):
        call(model, {"code_guidelines": bad})
    assert model._call_count == 0


OPTIONS = {"answer_only": True, "judge_sees": ["response"]}


def test_prompt_criterion_options_reach_prompt_criteria_only():
    """Options are merged into prompt criteria; code criteria stay single-key."""
    model = ScriptedModel(
        [
            [[encoded(["Is the answer 42?", 100], ["Is it kind?", 50])]],
            [[converted("code", CODE)], [converted("prompt", "Is it kind?")]],
        ]
    )
    result = generate_custom_rubric(
        ["What is 6 * 7?"], model, prompt_criterion_options=OPTIONS
    )
    assert result == [
        [
            [{"code": CODE}, 100.0],
            [
                {
                    "prompt": "Is it kind?",
                    "answer_only": True,
                    "judge_sees": ["response"],
                },
                50.0,
            ],
        ]
    ]
    # Options are judge settings, not instructions to the rubric writer.
    assert all(
        "answer_only" not in message["content"]
        for batch in model.recorded_prompts
        for prompt in batch
        for message in prompt
    )


def test_scale_option_reaches_every_prompt_criterion():
    """A judge scale passes through to each prompt criterion, not to code."""
    model = ScriptedModel(
        [
            [
                [
                    encoded(
                        ["Is the answer 42?", 100],
                        ["Is it kind?", 50],
                        ["Is it short?", 10],
                    )
                ]
            ],
            [
                [converted("code", CODE)],
                [converted("prompt", "Is it kind?")],
                [converted("prompt", "Is it short?")],
            ],
        ]
    )
    result = generate_custom_rubric(
        ["What is 6 * 7?"], model, prompt_criterion_options={"scale": "likert", "n": 3}
    )
    assert result == [
        [
            [{"code": CODE}, 100.0],
            [{"prompt": "Is it kind?", "scale": "likert", "n": 3}, 50.0],
            [{"prompt": "Is it short?", "scale": "likert", "n": 3}, 10.0],
        ]
    ]
    conversion = model.recorded_prompts[1][0][0]["content"]
    assert (
        "answers it as a yes/no question, scored 1.0 for yes and 0.0 for no"
        in conversion
    )


def test_prompt_criterion_options_are_copied_per_criterion():
    """Editing one criterion's options leaves the other criteria alone."""
    model = ScriptedModel([[[converted("prompt", "A?")], [converted("prompt", "B?")]]])
    result = convert_rubric_criteria(
        [[("A?", 1.0), ("B?", 1.0)]], model, prompt_criterion_options=OPTIONS
    )
    result[0][0][0]["judge_sees"].append("ground_truth")
    assert result[0][1][0]["judge_sees"] == ["response"]
    assert OPTIONS["judge_sees"] == ["response"]


def test_changing_prompt_criterion_options_reuses_cached_conversion():
    """New options never return stale rows and never ask the model again."""
    model = ScriptedModel([[[converted("prompt", "A?")]]])
    rubrics = [[("A?", 1.0)]]
    assert convert_rubric_criteria(rubrics, model) == [[[{"prompt": "A?"}, 1.0]]]
    assert convert_rubric_criteria(
        rubrics, model, prompt_criterion_options={"answer_only": True}
    ) == [[[{"prompt": "A?", "answer_only": True}, 1.0]]]
    assert convert_rubric_criteria(rubrics, model, _cache_only=True) == [
        [[{"prompt": "A?"}, 1.0]]
    ]
    assert model._call_count == 1


@pytest.mark.parametrize(
    "bad, error",
    [
        (["answer_only"], TypeError),
        ({1: True}, TypeError),
        ({"prompt": "other"}, ValueError),
        ({"code": CODE}, ValueError),
    ],
)
def test_invalid_prompt_criterion_options_fail_before_generation(bad, error):
    """Reject options that are not extra keys before any model request."""
    model = RecordingMockModel(responses=[])
    with pytest.raises(error, match="prompt_criterion_options"):
        generate_custom_rubric(["x"], model, prompt_criterion_options=bad)
    with pytest.raises(error, match="prompt_criterion_options"):
        convert_rubric_criteria([[("x?", 1.0)]], model, prompt_criterion_options=bad)
    assert model._call_count == 0

"""Tests for generate_from_context ALTER building block."""
from __future__ import annotations

import pytest
from fixtures import MockModel, RecordingMockModel

from pyforger.alter import generate_from_context
from pyforger.alter.generate import _resolve_prompt

# ---------------------------------------------------------------------------
# _resolve_prompt
# ---------------------------------------------------------------------------


class TestResolvePrompt:
    """_resolve_prompt maps a template or callable to a prompt string."""

    def test_string_template_fills_keys(self):
        """Format string keys are replaced by matching values from the item dict."""
        result = _resolve_prompt(
            "Hello {name}, you are {age}.", {"name": "Alice", "age": 30}
        )
        assert result == "Hello Alice, you are 30."

    def test_callable_called_with_item(self):
        """Callable prompt_fn receives the item dict and its return value is used."""
        result = _resolve_prompt(lambda item: f"Q: {item['q']}", {"q": "What is AI?"})
        assert result == "Q: What is AI?"

    def test_callable_returning_message_list(self):
        """Callable may return a message list (OpenAI format) unchanged."""
        messages = [{"role": "user", "content": "hi"}]
        result = _resolve_prompt(lambda item: messages, {})
        assert result is messages

    def test_string_template_missing_key_raises(self):
        """A format key absent from the item dict raises KeyError."""
        with pytest.raises(KeyError):
            _resolve_prompt("Hello {missing}", {"name": "Alice"})


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestGenerateFromContextValidation:
    """generate_from_context validates its arguments before calling the model."""

    def test_invalid_batch_size_raises(self):
        """batch_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            generate_from_context([], "p", MockModel(responses=["ok"]), batch_size=0)

    def test_invalid_samples_per_item_raises(self):
        """samples_per_item <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="samples_per_item"):
            generate_from_context(
                [], "p", MockModel(responses=["ok"]), samples_per_item=0
            )


# ---------------------------------------------------------------------------
# Execution — basic
# ---------------------------------------------------------------------------


class TestGenerateFromContextExecution:
    """generate_from_context returns a list of output dicts derived from inputs."""

    def test_no_schema_returns_text_key(self):
        """Without output_schema each response is stored as {"text": raw_response}."""
        model = MockModel(responses=["world"])
        result = generate_from_context([{"text": "hello"}], "prompt: {text}", model)
        assert result == [{"text": "world"}]

    def test_one_output_per_input_item_by_default(self):
        """With samples_per_item=1 (default), output length equals input length."""
        items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        model = MockModel(responses=["ok"] * 3)
        result = generate_from_context(items, "p: {text}", model)
        assert len(result) == 3

    def test_samples_per_item_multiplies_output(self):
        """Total outputs equal len(inputs) * samples_per_item."""
        items = [{"text": "a"}, {"text": "b"}]
        model = MockModel(responses=[["ok"] * 3, ["ok"] * 3])
        result = generate_from_context(items, "p: {text}", model, samples_per_item=3)
        assert len(result) == 6

    def test_empty_inputs_returns_empty(self):
        """Empty input produces no outputs and no model calls."""
        model = MockModel(responses=["ok"])
        result = generate_from_context([], "p", model)
        assert result == []
        assert model._call_count == 0

    def test_reasoning_attached_when_model_returns_it(self):
        """A completion's reasoning trace is carried as a 'thinking_trace' key."""
        model = MockModel(responses=["world"], reasoning=["thinking about it"])
        result = generate_from_context([{"text": "hello"}], "prompt: {text}", model)
        assert result == [{"text": "world", "thinking_trace": "thinking about it"}]

    def test_no_thinking_trace_key_when_model_has_none(self):
        """Without reasoning from the model, no 'thinking_trace' key is added."""
        model = MockModel(responses=["world"])
        result = generate_from_context([{"text": "hello"}], "prompt: {text}", model)
        assert result == [{"text": "world"}]

    def test_reasoning_schema_field_coexists_with_thinking_trace(self):
        """A parsed 'reasoning' field is left intact alongside the trace."""
        model = MockModel(
            responses=['{"reasoning": ["six times seven"], "answer": "42"}'],
            reasoning=["thinking about it"],
        )
        result = generate_from_context(
            [{"text": "hello"}],
            "prompt: {text}",
            model,
            output_schema={"reasoning": list, "answer": str},
        )
        assert result == [
            {
                "reasoning": ["six times seven"],
                "answer": "42",
                "thinking_trace": "thinking about it",
            }
        ]

    def test_thinking_trace_does_not_overwrite_schema_field(self, caplog):
        """A parsed 'thinking_trace' field is kept; the provider trace is dropped."""
        model = MockModel(
            responses=['{"thinking_trace": ["six times seven"], "answer": "42"}'],
            reasoning=["thinking about it"],
        )
        result = generate_from_context(
            [{"text": "hello"}],
            "prompt: {text}",
            model,
            output_schema={"thinking_trace": list, "answer": str},
        )
        assert result == [{"thinking_trace": ["six times seven"], "answer": "42"}]
        assert "already has a 'thinking_trace' field" in caplog.text


# ---------------------------------------------------------------------------
# Execution — prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Prompts sent to the model are constructed correctly from input items."""

    def test_string_template_is_filled_per_item(self):
        """Each input item fills the format string to produce a distinct prompt."""
        items = [{"topic": "gravity"}, {"topic": "photosynthesis"}]
        model = RecordingMockModel(responses=["ok"] * 2)
        generate_from_context(items, "Explain {topic}.", model)
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert "Explain gravity." in all_prompts
        assert "Explain photosynthesis." in all_prompts

    def test_callable_prompt_fn_is_called_per_item(self):
        """Callable prompt_fn is invoked once per input item."""
        items = [{"q": "Q1"}, {"q": "Q2"}]
        model = RecordingMockModel(responses=["ok"] * 2)
        generate_from_context(
            items,
            lambda item: f"Answer: {item['q']}",
            model,
        )
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert "Answer: Q1" in all_prompts
        assert "Answer: Q2" in all_prompts

    def test_schema_instructions_appended_to_string_prompt(self):
        """When output_schema is set, schema instructions are baked into prompt."""
        items = [{"text": "hi"}]
        model = RecordingMockModel(responses=['{"answer": "yes"}'])
        generate_from_context(
            items,
            "Question: {text}",
            model,
            output_schema={"answer": str},
        )
        sent_prompt = model.recorded_prompts[0][0]
        assert isinstance(sent_prompt, str)
        # Check that the final prompt sent to the model includes schema instructions.
        assert "Required keys" in sent_prompt
        assert "answer" in sent_prompt

    def test_schema_instructions_appended_to_message_list_prompt(self):
        """Schema instructions are still added when prompt is a list of messages."""
        items = [{"text": "hi"}]
        model = RecordingMockModel(responses=['{"answer": "yes"}'])
        generate_from_context(
            items,
            lambda item: [
                {"role": "system", "content": "Generate an answer to the question."},
                {"role": "user", "content": f"Question: {item['text']}"},
            ],
            model,
            output_schema={"answer": str},
        )
        sent_prompt = model.recorded_prompts[0][0]
        assert isinstance(sent_prompt, list)
        assert len(sent_prompt) > 2  # Original messages + schema instructions
        # Last message should be a user message with schema instructions
        last_message = sent_prompt[-1]
        assert last_message["role"] == "user"
        assert "Required keys" in last_message["content"]
        assert "answer" in last_message["content"]

    def test_samples_per_item_requests_n_completions(self):
        """samples_per_item completions of the filled prompt are requested."""
        items = [{"text": "x"}]
        model = RecordingMockModel(responses=[["ok"] * 4])
        result = generate_from_context(items, "p: {text}", model, samples_per_item=4)
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert all(p == "p: x" for p in all_prompts)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Execution — output_schema parsing
# ---------------------------------------------------------------------------


class TestOutputSchemaHandling:
    """output_schema triggers JSON parsing and key validation of model responses."""

    def test_valid_json_with_schema_is_parsed(self):
        """Valid JSON with all required keys is accepted and returned as a dict."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"answer": "42", "reason": "math"}'])
        result = generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"answer": str, "reason": str},
        )
        assert len(result) == 1
        assert result[0]["answer"] == "42"
        assert result[0]["reason"] == "math"

    def test_invalid_json_yields_none_slot_after_retries(self):
        """Non-JSON responses leave a None slot once retries are exhausted."""
        items = [{"text": "a"}]
        model = MockModel(responses=["not json"])
        result = generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"answer": str},
            max_retries=1,
        )
        assert result == [None]

    def test_json_missing_required_key_yields_none_slot(self):
        """JSON that omits a required schema key leaves a None slot."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"answer": "42"}'])  # missing "reason"
        result = generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"answer": str, "reason": str},
            max_retries=0,
        )
        assert result == [None]


class TestTolerantSchemaParsing:
    """A JSON object wrapped in a fence or prose parses; extra keys are dropped."""

    @pytest.mark.parametrize(
        "response",
        [
            '```json\n{"answer": "42"}\n```',
            '```\n{"answer": "42"}\n```',
            'Here is the JSON:\n```json\n{"answer": "42"}\n```\nHope this helps.',
            'Sure! {"answer": "42"} Let me know if you need more.',
            '<think>maybe {"answer": "41"}?</think>\n{"answer": "42"}',
            # The first object lacks "answer"; the second has it.
            'Draft: {"note": "x"} Final: {"answer": "42"} Also {"answer": "43"}',
        ],
    )
    def test_object_is_found(self, response):
        """The first object with every schema key is parsed, without a retry."""
        model = MockModel(responses=[response])
        result = generate_from_context(
            [{"text": "a"}], "p: {text}", model, output_schema={"answer": str}
        )
        assert result == [{"answer": "42"}]
        assert model._call_count == 1

    def test_raw_newline_in_string(self):
        """A value with unescaped newlines (multi-line code) parses."""
        model = MockModel(
            responses=['```json\n{"code": "def f():\n    return 1"}\n```']
        )
        result = generate_from_context(
            [{"text": "a"}], "p: {text}", model, output_schema={"code": str}
        )
        assert result == [{"code": "def f():\n    return 1"}]

    def test_nested_object_is_not_a_match(self):
        """An object nested inside another is not taken as the answer."""
        model = MockModel(responses=['{"result": {"answer": "42"}}'])
        result = generate_from_context(
            [{"text": "a"}],
            "p: {text}",
            model,
            output_schema={"answer": str},
            max_retries=0,
        )
        assert result == [None]

    def test_extra_keys_are_dropped(self):
        """Keys outside the schema do not reach the output."""
        model = MockModel(responses=['{"answer": "42", "crafted_by": "me"}'])
        result = generate_from_context(
            [{"text": "a"}], "p: {text}", model, output_schema={"answer": str}
        )
        assert result == [{"answer": "42"}]

    def test_extra_keys_are_dropped_with_merge(self):
        """merge=True attaches only the schema's keys to the input row."""
        model = MockModel(responses=['{"answer": "42", "crafted_by": "me"}'])
        result = generate_from_context(
            [{"text": "a"}],
            "p: {text}",
            model,
            output_schema={"answer": str},
            merge=True,
        )
        assert result == [{"text": "a", "answer": "42"}]

    def test_value_types_are_not_checked(self):
        """A value of another type than the schema names is kept as is."""
        model = MockModel(responses=['{"answer": 42}'])
        result = generate_from_context(
            [{"text": "a"}], "p: {text}", model, output_schema={"answer": str}
        )
        assert result == [{"answer": 42}]

    def test_missing_key_is_retried(self):
        """An object without a schema key fails to parse and is asked again."""
        model = MockModel(responses=['```json\n{"reason": "math"}\n```'])
        result = generate_from_context(
            [{"text": "a"}],
            "p: {text}",
            model,
            output_schema={"answer": str},
            max_retries=2,
        )
        assert result == [None]
        assert model._call_count == 3

    @pytest.mark.parametrize("response", ['[{"answer": "42"}]', '"42"', "42"])
    def test_non_object_json_fails(self, response):
        """JSON that is not an object (a list, a string, a number) fails."""
        model = MockModel(responses=[response])
        result = generate_from_context(
            [{"text": "a"}],
            "p: {text}",
            model,
            output_schema={"answer": str},
            max_retries=0,
        )
        assert result == [None]


# ---------------------------------------------------------------------------
# Execution — retries
# ---------------------------------------------------------------------------


class TestRetries:
    """Failed parses are retried up to max_retries additional times."""

    def test_failed_parse_retried_up_to_max_retries(self):
        """A prompt that always fails is attempted 1 + max_retries times."""
        items = [{"text": "a"}]
        model = MockModel(responses=["bad json"])
        generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=2,
        )
        # 1 initial attempt + 2 retries = 3 calls (each with 1 prompt)
        assert model._call_count == 3

    def test_successful_parse_stops_retries(self):
        """If a prompt parses successfully, it is not retried."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"key": "val"}'])
        result = generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=3,
        )
        assert len(result) == 1
        assert model._call_count == 1

    def test_max_retries_zero_means_one_attempt(self):
        """No retries are allowed if max_retries=0 is provided."""
        items = [{"text": "a"}]
        model = MockModel(responses=["bad json"])
        generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=0,
        )
        assert model._call_count == 1

    def test_only_failed_prompt_is_retried(self):
        """When one prompt parses and another fails, only the failure is retried."""
        items = [{"text": "a"}, {"text": "b"}]
        # First call: prompt 0 valid, prompt 1 invalid. The retry batch holds only
        # prompt 1, so MockModel serves it the (valid) responses[0].
        model = RecordingMockModel(responses=['{"key": "ok"}', "bad json"])
        result = generate_from_context(
            items,
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=1,
        )
        assert result == [{"key": "ok"}, {"key": "ok"}]
        assert model._call_count == 2
        assert len(model.recorded_prompts[0]) == 2  # both prompts on first pass
        assert model.recorded_prompts[1] == [model.recorded_prompts[0][1]]  # only #1


# ---------------------------------------------------------------------------
# Execution — batching
# ---------------------------------------------------------------------------


class TestBatching:
    """batch_size controls how many prompts are sent per generate_batch call."""

    def test_batch_size_splits_calls(self):
        """If there are 5 prompts with batch_size of 2, 3 batched calls are made."""
        items = [{"text": str(i)} for i in range(5)]
        model = MockModel(responses=["ok"] * 5)
        generate_from_context(items, "p: {text}", model, batch_size=2)
        assert model._call_count == 3  # ceil(5/2) = 3


# ---------------------------------------------------------------------------
# merge / output_field
# ---------------------------------------------------------------------------


class TestMerge:
    """merge=True returns the input rows with each output attached."""

    def test_merge_one_sample(self):
        """Each output row is {**item, **parsed}; no sample_idx for one sample."""
        model = MockModel(responses=['{"answer": "4"}', '{"answer": "9"}'])
        rows = generate_from_context(
            [{"q": "2+2"}, {"q": "3*3"}],
            "Solve {q}",
            model,
            output_schema={"answer": str},
            merge=True,
        )
        assert rows == [{"q": "2+2", "answer": "4"}, {"q": "3*3", "answer": "9"}]

    def test_merge_several_samples(self):
        """With samples_per_item > 1, each sample is a row with its sample_idx."""
        model = MockModel(responses=[["x0", "x1"], ["y0", "y1"]])
        rows = generate_from_context(
            [{"q": "a"}, {"q": "b"}], "{q}", model, samples_per_item=2, merge=True
        )
        assert rows == [
            {"q": "a", "text": "x0", "sample_idx": 0},
            {"q": "a", "text": "x1", "sample_idx": 1},
            {"q": "b", "text": "y0", "sample_idx": 0},
            {"q": "b", "text": "y1", "sample_idx": 1},
        ]

    def test_output_field_nests_output(self):
        """output_field keeps the input's keys and nests the output under it."""
        model = MockModel(responses=["world"], reasoning=["thought"])
        rows = generate_from_context(
            [{"text": "hello"}], "{text}", model, merge=True, output_field="reply"
        )
        assert rows == [
            {"text": "hello", "reply": {"text": "world", "thinking_trace": "thought"}}
        ]

    def test_key_clash_output_wins_and_warns_once(self, caplog):
        """Clashing keys take the output's value; one warning names them."""
        model = MockModel(responses=["new a", "new b"])
        rows = generate_from_context(
            [{"text": "a"}, {"text": "b"}], "{text}", model, merge=True
        )
        assert rows == [{"text": "new a"}, {"text": "new b"}]
        clash = [r.getMessage() for r in caplog.records if "replace" in r.msg]
        assert clash == [
            "alter.generate_from_context: merge=True replaces the input's "
            "['text']; pass another output_field= (or rename the input keys) to "
            "keep both"
        ]

    def test_output_field_clash_warns(self, caplog):
        """An output_field that already is an input key is replaced with a warning."""
        model = MockModel(responses=["model answer"])
        rows = generate_from_context(
            [{"problem": "1+1", "solution": "gold"}],
            "{problem}",
            model,
            merge=True,
            output_field="solution",
        )
        assert rows == [{"problem": "1+1", "solution": {"text": "model answer"}}]
        assert "replaces the input's ['solution']" in caplog.text

    def test_sample_idx_clash_warns(self, caplog):
        """A sample_idx from an earlier merge is replaced with a warning."""
        model = MockModel(responses=[["x0", "x1"]])
        rows = generate_from_context(
            [{"q": "a", "sample_idx": 3}],
            "{q}",
            model,
            samples_per_item=2,
            merge=True,
            output_field="out",
        )
        assert [row["sample_idx"] for row in rows] == [0, 1]
        assert "replaces the input's ['sample_idx']" in caplog.text

    def test_merge_drops_parse_failures(self):
        """Slots that failed to parse are left out of the merged rows."""
        model = MockModel(responses=['{"k": "v"}', "not json"])
        rows = generate_from_context(
            [{"q": 1}, {"q": 2}],
            "{q}",
            model,
            output_schema={"k": str},
            max_retries=0,
            merge=True,
        )
        assert rows == [{"q": 1, "k": "v"}]

    def test_output_field_needs_merge(self):
        """output_field without merge=True raises."""
        with pytest.raises(ValueError, match="merge=True"):
            generate_from_context(
                [], "p", MockModel(responses=["ok"]), output_field="out"
            )

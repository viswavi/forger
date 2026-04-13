"""Tests for generate_from_context ALTER building block."""
from __future__ import annotations

import pytest
from fixtures import MockModel, RecordingMockModel

from pyforger import DataSpec, NodeType
from pyforger.alter import generate_from_context
from pyforger.alter.generate import _resolve_prompt

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_source(items: list[dict]) -> DataSpec:
    """Build a minimal upstream DataSpec with fixed output."""
    return DataSpec(NodeType.ACQUIRE, [], lambda: items, "source")


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
# DataSpec structure
# ---------------------------------------------------------------------------


class TestGenerateFromContextDataSpec:
    """generate_from_context builds a well-formed ALTER DataSpec."""

    def test_returns_dataspec_with_alter_node_type(self):
        """Return value is a DataSpec with node_type ALTER."""
        source = _make_source([{"text": "a"}])
        spec = generate_from_context(source, "prompt: {text}", MockModel())
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ALTER

    def test_has_upstream_dependency(self):
        """The upstream inputs DataSpec is registered as a dependency."""
        source = _make_source([{"text": "a"}])
        spec = generate_from_context(source, "prompt: {text}", MockModel())
        assert spec.get_dependencies() == [source]

    def test_no_execution_on_build(self):
        """Building the node does not call model.generate_batch."""
        model = MockModel()
        source = _make_source([{"text": "a"}])
        generate_from_context(source, "prompt: {text}", model)
        assert model._call_count == 0

    def test_invalid_batch_size_raises(self):
        """batch_size <= 0 raises ValueError at build time."""
        source = _make_source([])
        with pytest.raises(ValueError, match="batch_size"):
            generate_from_context(source, "p", MockModel(), batch_size=0)

    def test_invalid_samples_per_item_raises(self):
        """samples_per_item <= 0 raises ValueError at build time."""
        source = _make_source([])
        with pytest.raises(ValueError, match="samples_per_item"):
            generate_from_context(source, "p", MockModel(), samples_per_item=0)

    def test_list_of_dataspecs_raises(self):
        """Passing a list of DataSpecs raises TypeError; use forger.zip to combine."""
        a = _make_source([{"x": 1}])
        b = _make_source([{"y": 2}])
        with pytest.raises(TypeError):
            generate_from_context([a, b], "p", MockModel())


# ---------------------------------------------------------------------------
# Execution — basic
# ---------------------------------------------------------------------------


class TestGenerateFromContextExecution:
    """execute_fn produces output dicts from the upstream input items."""

    def test_no_schema_returns_text_key(self):
        """Without output_schema each response is stored as {"text": raw_response}."""
        source = _make_source([{"text": "hello"}])
        model = MockModel(responses=["world"])
        spec = generate_from_context(source, "prompt: {text}", model)
        result = spec.execute_fn([{"text": "hello"}])
        assert result == [{"text": "world"}]

    def test_one_output_per_input_item_by_default(self):
        """With samples_per_item=1 (default), output length equals input length."""
        items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        model = MockModel(responses=["ok"])
        spec = generate_from_context(_make_source(items), "p: {text}", model)
        result = spec.execute_fn(items)
        assert len(result) == 3

    def test_samples_per_item_multiplies_output(self):
        """Total outputs equal len(inputs) * samples_per_item."""
        items = [{"text": "a"}, {"text": "b"}]
        model = MockModel(responses=["ok"])
        spec = generate_from_context(
            _make_source(items), "p: {text}", model, samples_per_item=3
        )
        result = spec.execute_fn(items)
        assert len(result) == 6

    def test_empty_inputs_returns_empty(self):
        """Empty upstream data produces no outputs and no model calls."""
        model = MockModel()
        spec = generate_from_context(_make_source([]), "p", model)
        result = spec.execute_fn([])
        assert result == []
        assert model._call_count == 0


# ---------------------------------------------------------------------------
# Execution — prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Prompts sent to the model are constructed correctly from input items."""

    def test_string_template_is_filled_per_item(self):
        """Each input item fills the format string to produce a distinct prompt."""
        items = [{"topic": "gravity"}, {"topic": "photosynthesis"}]
        model = RecordingMockModel(responses=["ok"])
        spec = generate_from_context(_make_source(items), "Explain {topic}.", model)
        spec.execute_fn(items)
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert "Explain gravity." in all_prompts
        assert "Explain photosynthesis." in all_prompts

    def test_callable_prompt_fn_is_called_per_item(self):
        """Callable prompt_fn is invoked once per input item."""
        items = [{"q": "Q1"}, {"q": "Q2"}]
        model = RecordingMockModel(responses=["ok"])
        spec = generate_from_context(
            _make_source(items),
            lambda item: f"Answer: {item['q']}",
            model,
        )
        spec.execute_fn(items)
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert "Answer: Q1" in all_prompts
        assert "Answer: Q2" in all_prompts

    def test_schema_instructions_appended_to_string_prompt(self):
        """When output_schema is set, schema instructions are baked into prompt."""
        items = [{"text": "hi"}]
        model = RecordingMockModel(responses=['{"answer": "yes"}'])
        spec = generate_from_context(
            _make_source(items),
            "Question: {text}",
            model,
            output_schema={"answer": str},
        )
        spec.execute_fn(items)
        sent_prompt = model.recorded_prompts[0][0]
        assert isinstance(sent_prompt, str)
        # Check that the final prompt sent to the model includes schema instructions.
        assert "Required keys" in sent_prompt
        assert "answer" in sent_prompt

    def test_samples_per_item_sends_duplicate_prompts(self):
        """samples_per_item copies of the same filled prompt are sent to the model."""
        items = [{"text": "x"}]
        model = RecordingMockModel(responses=["ok"])
        spec = generate_from_context(
            _make_source(items), "p: {text}", model, samples_per_item=4
        )
        spec.execute_fn(items)
        all_prompts = [p for batch in model.recorded_prompts for p in batch]
        assert len(all_prompts) == 4
        assert all(p == "p: x" for p in all_prompts)


# ---------------------------------------------------------------------------
# Execution — output_schema parsing
# ---------------------------------------------------------------------------


class TestOutputSchemaHandling:
    """output_schema triggers JSON parsing and key validation of model responses."""

    def test_valid_json_with_schema_is_parsed(self):
        """Valid JSON with all required keys is accepted and returned as a dict."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"answer": "42", "reason": "math"}'])
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"answer": str, "reason": str},
        )
        result = spec.execute_fn(items)
        assert len(result) == 1
        assert result[0]["answer"] == "42"
        assert result[0]["reason"] == "math"

    def test_invalid_json_is_dropped_after_retries(self):
        """Non-JSON responses are rejected and dropped once retries are exhausted."""
        items = [{"text": "a"}]
        model = MockModel(responses=["not json"])
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"answer": str},
            max_retries=1,
        )
        result = spec.execute_fn(items)
        assert result == []

    def test_json_missing_required_key_is_dropped(self):
        """JSON that omits a required schema key is rejected."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"answer": "42"}'])  # missing "reason"
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"answer": str, "reason": str},
            max_retries=0,
        )
        result = spec.execute_fn(items)
        assert result == []


# ---------------------------------------------------------------------------
# Execution — retries
# ---------------------------------------------------------------------------


class TestRetries:
    """Failed parses are retried up to max_retries additional times."""

    def test_failed_parse_retried_up_to_max_retries(self):
        """A prompt that always fails is attempted 1 + max_retries times."""
        items = [{"text": "a"}]
        model = MockModel(responses=["bad json"])
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=2,
        )
        spec.execute_fn(items)
        # 1 initial attempt + 2 retries = 3 calls (each with 1 prompt)
        assert model._call_count == 3

    def test_successful_parse_stops_retries(self):
        """If a prompt parses successfully, it is not retried."""
        items = [{"text": "a"}]
        model = MockModel(responses=['{"key": "val"}'])
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=3,
        )
        result = spec.execute_fn(items)
        assert len(result) == 1
        assert model._call_count == 1

    def test_max_retries_zero_means_one_attempt(self):
        """No retries are allowed if max_retries=0 is provided."""
        items = [{"text": "a"}]
        model = MockModel(responses=["bad json"])
        spec = generate_from_context(
            _make_source(items),
            "p: {text}",
            model,
            output_schema={"key": str},
            max_retries=0,
        )
        spec.execute_fn(items)
        assert model._call_count == 1


# ---------------------------------------------------------------------------
# Execution — batching
# ---------------------------------------------------------------------------


class TestBatching:
    """batch_size controls how many prompts are sent per generate_batch call."""

    def test_batch_size_splits_calls(self):
        """If there are 5 prompts with batch_size of 2, 3 batched calls are made."""
        items = [{"text": str(i)} for i in range(5)]
        model = MockModel(responses=["ok"])
        spec = generate_from_context(
            _make_source(items), "p: {text}", model, batch_size=2
        )
        spec.execute_fn(items)
        assert model._call_count == 3  # ceil(5/2) = 3

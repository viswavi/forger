"""Tests for sample_from_LLM and related acquire building block."""
from __future__ import annotations

from fixtures import MockModel, RecordingMockModel

from pyforger import DataSpec, NodeType
from pyforger.acquire import sample_from_LLM
from pyforger.acquire.sample import _schema_format_instructions


class TestSchemaFormatInstructions:
    """_schema_format_instructions produces prompt-ready schema instructions."""

    def test_contains_required_keys_and_json_instruction(self):
        """Output contains required keys list and asks for JSON only."""
        out = _schema_format_instructions({"answer": str, "reason": str})
        assert "Required keys" in out
        assert "answer" in out
        assert "reason" in out
        assert "JSON object" in out or "json" in out.lower()

    def test_example_format_has_placeholders(self):
        """Example format uses type placeholders like <str>."""
        out = _schema_format_instructions({"answer": str})
        assert "<str>" in out
        assert '"answer"' in out

    def test_string_value_uses_generic_placeholder(self):
        """When schema value is a string (description), use <value>."""
        out = _schema_format_instructions({"answer": "The final answer"})
        assert "<value>" in out
        assert "answer" in out


class TestSchemaPromptInjection:
    """Prompt includes schema instructions when output_schema is set."""

    def test_string_prompt_includes_schema_instructions(self):
        """String prompt passed to generate_batch has Required keys."""
        model = RecordingMockModel(responses=['{"answer": "42"}'])
        spec = sample_from_LLM(
            "Generate an answer",
            model,
            n=1,
            output_schema={"answer": str},
        )
        list(spec.execute_fn())
        assert len(model.recorded_prompts[0]) == 1
        prompt_text = model.recorded_prompts[0][0]
        assert isinstance(prompt_text, str)
        assert "Required keys" in prompt_text
        assert "answer" in prompt_text
        assert "JSON object" in prompt_text or "json" in prompt_text.lower()

    def test_message_list_prompt_appends_schema_as_user_message(self):
        """Message-list prompt gets an extra user message with schema instructions."""
        model = RecordingMockModel(responses=['{"answer": "42"}'])
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 6*7?"},
        ]
        spec = sample_from_LLM(
            messages,
            model,
            n=1,
            output_schema={"answer": str},
        )
        list(spec.execute_fn())
        assert len(model.recorded_prompts[0]) == 1
        recorded = model.recorded_prompts[0][0]
        assert isinstance(recorded, list)
        assert len(recorded) == 3
        assert recorded[0] == messages[0]
        assert recorded[1] == messages[1]
        assert recorded[2]["role"] == "user"
        assert "Required keys" in recorded[2]["content"]
        assert "answer" in recorded[2]["content"]

    def test_no_schema_string_prompt_unchanged(self):
        """Without output_schema, string prompt is passed through as-is."""
        model = RecordingMockModel(responses=["hello"])
        spec = sample_from_LLM("Generate", model, n=1)
        list(spec.execute_fn())
        assert model.recorded_prompts[0][0] == "Generate"

    def test_no_schema_message_list_unchanged(self):
        """Without output_schema, message list is passed through as-is."""
        model = RecordingMockModel(responses=["hello"])
        messages = [{"role": "user", "content": "Hi"}]
        spec = sample_from_LLM(messages, model, n=1)
        list(spec.execute_fn())
        assert model.recorded_prompts[0][0] == messages


class TestSampleFromLLMReturnsDataSpec:
    """sample_from_LLM returns a DataSpec with node_type ACQUIRE."""

    def test_returns_dataspec(self):
        """Return value is a DataSpec with ACQUIRE node type."""
        model = MockModel(responses=['{"answer": "x"}'])
        spec = sample_from_LLM(
            "Generate something",
            model,
            n=2,
            output_schema={"answer": str},
        )
        assert isinstance(spec, DataSpec)
        assert spec.node_type == NodeType.ACQUIRE

    def test_no_execution_on_build(self):
        """Building sample_from_LLM does not call model.generate_batch."""
        model = MockModel(responses=['{"text": "a"}'])
        sample_from_LLM("Hi", model, n=1)
        assert model._call_count == 0

    def test_has_no_dependencies(self):
        """Sample-from-LLM node has no upstream dependencies."""
        model = MockModel(responses=['{"text": "a"}'])
        spec = sample_from_LLM("Hi", model, n=1)
        assert spec.get_dependencies() == []


class TestSampleFromLLMExecution:
    """Execute_fn runs the generation loop and yields accepted samples."""

    def test_execute_yields_accepted_samples(self):
        """Execute returns an iterable of parsed samples up to n."""
        model = MockModel(responses=['{"text": "one"}', '{"text": "two"}'])
        spec = sample_from_LLM(
            "Generate",
            model,
            n=2,
            batch_size=2,
            output_schema={"text": str},
        )
        gen = spec.execute_fn()
        items = list(gen)
        assert len(items) == 2
        assert items[0]["text"] == "one"
        assert items[1]["text"] == "two"
        assert model._call_count == 1

    def test_execute_respects_n(self):
        """Stops after n samples even if batch could return more."""
        model = MockModel(responses=['{"text": "x"}'])
        spec = sample_from_LLM("Generate", model, n=3, batch_size=10)
        items = list(spec.execute_fn())
        assert len(items) == 3
        assert model._call_count == 1

    def test_no_schema_returns_text_key(self):
        """Without output_schema, each sample is {"text": raw_response}."""
        model = MockModel(responses=["plain text"])
        spec = sample_from_LLM("Generate", model, n=1)
        items = list(spec.execute_fn())
        assert items == [{"text": "plain text"}]

    def test_message_list_prompt_executes(self):
        """Message-list prompts run through the generation loop."""
        model = MockModel(responses=['{"text": "result"}'])
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Generate data"},
        ]
        spec = sample_from_LLM(messages, model, n=2, batch_size=2)
        items = list(spec.execute_fn())
        assert len(items) == 2
        assert model._call_count == 1

    def test_message_list_with_schema(self):
        """Message-list prompt with output_schema parses structured output."""
        model = MockModel(responses=['{"answer": "42"}'])
        messages = [{"role": "user", "content": "What is 6*7?"}]
        spec = sample_from_LLM(
            messages,
            model,
            n=1,
            output_schema={"answer": str},
        )
        items = list(spec.execute_fn())
        assert len(items) == 1
        assert items[0]["answer"] == "42"


class TestParseAndSchema:
    """Parsing and schema validation behavior."""

    def test_schema_validation_accepts_valid_json(self):
        """Valid JSON with required keys is accepted."""
        model = MockModel(responses=['{"answer": "42", "reason": "math"}'])
        spec = sample_from_LLM(
            "Generate",
            model,
            n=1,
            output_schema={"answer": str, "reason": str},
        )
        items = list(spec.execute_fn())
        assert len(items) == 1
        assert items[0]["answer"] == "42"
        assert items[0]["reason"] == "math"

    def test_schema_rejects_missing_keys(self):
        """JSON missing required keys is rejected."""
        model = MockModel(responses=['{"answer": "42"}'])  # missing "reason"
        spec = sample_from_LLM(
            "Generate",
            model,
            n=1,
            output_schema={"answer": str, "reason": str},
            max_retries=1,
        )
        items = list(spec.execute_fn())
        assert len(items) == 0

    def test_invalid_json_rejected(self):
        """Non-JSON output is rejected (parse returns None)."""
        model = MockModel(responses=["not json at all"])
        spec = sample_from_LLM(
            "Generate",
            model,
            n=1,
            output_schema={"x": str},
            max_retries=1,
        )
        items = list(spec.execute_fn())
        assert len(items) == 0


class TestRetriesAndBatch:
    """max_retries and batch_size behavior."""

    def test_invalid_batch_size_rejected(self):
        """batch_size <= 0 raises ValueError."""
        model = MockModel(responses=['{"text": "x"}'])
        try:
            sample_from_LLM("Hi", model, n=1, batch_size=0)
        except ValueError as e:
            assert "batch_size" in str(e)
        else:
            raise AssertionError("Expected ValueError")

    def test_invalid_n_rejected(self):
        """Zero or negative n raises ValueError."""
        model = MockModel(responses=['{"text": "x"}'])
        try:
            sample_from_LLM("Hi", model, n=0)
        except ValueError as e:
            assert "n" in str(e)
        else:
            raise AssertionError("Expected ValueError")

    def test_batch_size_splits_rounds(self):
        """Multiple rounds when n > batch_size."""
        model = MockModel(responses=['{"text": "x"}'])
        spec = sample_from_LLM("Generate", model, n=5, batch_size=2)
        items = list(spec.execute_fn())
        assert len(items) == 5
        # 3 rounds: 2 + 2 + 1
        assert model._call_count == 3

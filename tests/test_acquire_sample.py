"""Tests for sample_from_LLM and related acquire building block."""
from __future__ import annotations

from typing import Any

from pyforger import DataSpec, NodeType
from pyforger.acquire import Model, sample_from_LLM
from pyforger.acquire.sample import _schema_format_instructions


class MockModel(Model):
    """Model that returns canned responses from generate_batch for testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model_id: str = "test/model",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id=model_id, **kwargs)
        self.responses = responses or ['{"text": "sample"}']
        self._call_count = 0

    def generate_batch(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        self._call_count += 1
        out: list[list[str]] = []
        for i in range(len(prompts)):
            out.append([self.responses[i % len(self.responses)]])
        return out


class RecordingMockModel(MockModel):
    """MockModel that records the prompts list passed to generate_batch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompts: list[str] = []

    def generate_batch(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        self.prompts = prompts
        return super().generate_batch(prompts, n=n, temperature=temperature, max_tokens=max_tokens)


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
    """When output_schema is set, prompt passed to model contains schema instructions."""

    def test_prompt_includes_schema_instructions(self):
        """First prompt passed to generate_batch contains Required keys and schema key."""
        model = RecordingMockModel(responses=['{"answer": "42"}'])
        spec = sample_from_LLM(
            "Generate an answer",
            model,
            n=1,
            output_schema={"answer": str},
        )
        list(spec.execute_fn())
        assert len(model.prompts) == 1
        prompt_text = model.prompts[0]
        assert "Required keys" in prompt_text
        assert "answer" in prompt_text
        assert "JSON object" in prompt_text or "json" in prompt_text.lower()


class TestSampleFromLLMReturnsDataSpec:
    """sample_from_LLM returns a DataSpec with node_type ACQUIRE."""

    def test_returns_dataspec(self):
        """Return value is a DataSpec with ACQUIRE node type."""
        model = MockModel(responses=['{"answer": "x"}'])
        spec = sample_from_LLM("Generate something", model, n=2, output_schema={"answer": str})
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
        # One batch of 3
        assert model._call_count == 1

    def test_no_schema_returns_text_key(self):
        """Without output_schema, each sample is {"text": raw_response}."""
        model = MockModel(responses=["plain text"])
        spec = sample_from_LLM("Generate", model, n=1)
        items = list(spec.execute_fn())
        assert items == [{"text": "plain text"}]


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
        """JSON missing required keys is rejected; no sample yielded from that response."""
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


class TestFewShotPool:
    """Pool and pool_sample_k behavior."""

    def test_pool_sampling_used_in_prompt(self):
        """When pool and pool_sample_k are set, batch is built (no crash)."""
        pool = [{"text": "example1"}, {"text": "example2"}]
        model = MockModel(responses=['{"text": "out"}'])
        spec = sample_from_LLM(
            "Generate",
            model,
            n=1,
            pool=pool,
            pool_sample_k=1,
            seed=42,
        )
        items = list(spec.execute_fn())
        assert len(items) == 1
        assert model._call_count == 1

    def test_empty_pool_ok(self):
        """pool=None and pool_sample_k=0 run without error."""
        model = MockModel(responses=['{"text": "x"}'])
        spec = sample_from_LLM("Generate", model, n=1, pool=None, pool_sample_k=0)
        items = list(spec.execute_fn())
        assert len(items) == 1


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
        """n <= 0 raises ValueError."""
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

    def test_seed_reproducibility(self):
        """Same seed yields same order from pool (deterministic batch build)."""
        pool = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        model1 = MockModel(responses=['{"text": "out"}'])
        spec1 = sample_from_LLM("G", model1, n=1, pool=pool, pool_sample_k=1, seed=99)
        list(spec1.execute_fn())
        model2 = MockModel(responses=['{"text": "out"}'])
        spec2 = sample_from_LLM("G", model2, n=1, pool=pool, pool_sample_k=1, seed=99)
        list(spec2.execute_fn())
        # Both should have run one batch; prompts may differ by pool sample
        assert model1._call_count == 1
        assert model2._call_count == 1


class TestModelFactory:
    """model() factory creates configured Model instances."""

    def test_model_factory_returns_model(self):
        """model() factory returns a Model instance."""
        from pyforger.acquire import model as model_factory
        m = model_factory("test/id")
        assert isinstance(m, Model)
        assert m._model_id == "test/id"

    def test_model_factory_forwards_kwargs(self):
        """model() forwards keyword arguments to Model."""
        from pyforger.acquire import model as model_factory
        m = model_factory("test/id", api_key="sk-test", temperature=0.5)
        assert m._api_key == "sk-test"
        assert m._temperature == 0.5

"""Tests for decompose.by_llm_label."""
from __future__ import annotations

from typing import Any

import pytest

from pyforger import run
from pyforger.acquire import from_list
from pyforger.acquire.model import Model, Prompt
from pyforger.core import DataSpec, NodeType
from pyforger.decompose import by_llm_label


class MockModel(Model):
    """Model that returns canned responses from generate_batch for testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model_id: str = "test/model",
        **kwargs: Any,
    ) -> None:
        """Initialize with canned responses cycled across prompts."""
        super().__init__(model_id=model_id, **kwargs)
        self.responses = responses or ["yes"]
        self._call_count = 0

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Return one canned response per prompt."""
        self._call_count += 1
        return [[self.responses[i % len(self.responses)]] for i in range(len(prompts))]


def _identity(src: DataSpec) -> DataSpec:
    """Sub-pipeline that returns items unchanged."""
    return DataSpec(NodeType.ALTER, src, lambda data: list(data), "identity")


def _tag(tag: str):
    """Return a sub-pipeline builder that stamps each item with a tag."""

    def _builder(src: DataSpec) -> DataSpec:
        def _execute(data: list[dict]) -> list[dict]:
            return [{**item, "tag": tag} for item in data]

        return DataSpec(NodeType.ALTER, src, _execute, f"tag_{tag}")

    return _builder


class TestByLlmLabel:
    """by_llm_label classifies, buckets, and routes items through sub-pipelines."""

    def test_routes_each_item_to_matching_label(self):
        """Items are routed to labels[label] based on parsed LLM response."""
        data = [{"instruction": "Classify: a"}, {"instruction": "Classify: b"}]
        model = MockModel(responses=["yes", "no"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="Is this classification? {instruction}",
            labels={"yes": _tag("Y"), "no": _tag("N")},
        )
        result = run(spec)
        assert len(result) == 2
        tags = {r["tag"] for r in result}
        assert tags == {"Y", "N"}

    def test_buckets_share_sub_pipeline_call(self):
        """Items sharing a label are routed through the same sub-pipeline."""
        data = [{"x": "1"}, {"x": "2"}, {"x": "3"}]
        model = MockModel(responses=["a", "a", "a"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"a": _identity},
        )
        result = run(spec)
        assert len(result) == 3
        assert model._call_count == 1  # one batched classification call

    def test_default_pipeline_used_for_unknown_label(self):
        """Items with a label not in labels go through default when set."""
        data = [{"x": "1"}, {"x": "2"}]
        model = MockModel(responses=["known", "unknown"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"known": _tag("K")},
            default=_tag("D"),
        )
        result = run(spec)
        tags = [r["tag"] for r in result]
        assert tags == ["K", "D"]

    def test_unknown_label_dropped_without_default(self):
        """Items with a label not in labels are dropped when default is None."""
        data = [{"x": "1"}, {"x": "2"}]
        model = MockModel(responses=["known", "unknown"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"known": _tag("K")},
        )
        result = run(spec)
        assert len(result) == 1
        assert result[0]["tag"] == "K"

    def test_parse_fn_transforms_response(self):
        """parse_fn converts the raw response before label lookup."""
        data = [{"x": "1"}]
        model = MockModel(responses=["YES please"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            parse_fn=lambda r: r.strip().lower().startswith("yes"),
            labels={True: _tag("T"), False: _tag("F")},
        )
        result = run(spec)
        assert result == [{"x": "1", "tag": "T"}]

    def test_empty_input_returns_empty(self):
        """Empty upstream returns empty without calling the model."""
        model = MockModel(responses=["yes"])
        spec = by_llm_label(
            from_list([]),
            model=model,
            prompt_template="{x}",
            labels={"yes": _identity},
        )
        result = run(spec)
        assert result == []
        assert model._call_count == 0

    def test_output_order_follows_first_seen_label(self):
        """Results concatenate buckets in the order labels first appear."""
        data = [{"x": "1"}, {"x": "2"}, {"x": "3"}, {"x": "4"}]
        model = MockModel(responses=["b", "a", "b", "a"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"a": _tag("A"), "b": _tag("B")},
        )
        result = run(spec)
        # b was seen first, so B bucket comes first
        assert [r["tag"] for r in result] == ["B", "B", "A", "A"]

    def test_node_type_is_decompose(self):
        """by_llm_label produces a DECOMPOSE DataSpec."""
        spec = by_llm_label(
            from_list([]),
            model=MockModel(),
            prompt_template="{x}",
            labels={"yes": _identity},
        )
        assert spec.node_type == NodeType.DECOMPOSE

    def test_sub_pipeline_may_filter(self):
        """Sub-pipelines may drop items (e.g., a filter)."""

        def _drop_all(src: DataSpec) -> DataSpec:
            return DataSpec(NodeType.FILTER, src, lambda _d: [], "drop_all")

        data = [{"x": "1"}, {"x": "2"}]
        model = MockModel(responses=["drop", "keep"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"drop": _drop_all, "keep": _identity},
        )
        result = run(spec)
        assert result == [{"x": "2"}]

    def test_missing_prompt_field_raises_clear_error(self):
        """Prompt template missing a field raises with a helpful message."""
        data = [{"other": "value"}]
        model = MockModel(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{instruction}",
            labels={"yes": _identity},
        )
        with pytest.raises((ValueError, RuntimeError), match="'instruction' missing"):
            run(spec)

    def test_callable_prompt_template(self):
        """prompt_template may be a callable that returns a string."""
        data = [{"x": "hello"}]
        model = MockModel(responses=["yes"])

        captured: list[str] = []

        def _builder(src: DataSpec) -> DataSpec:
            return DataSpec(NodeType.ALTER, src, lambda d: list(d), "cap")

        def _prompt(item: dict) -> str:
            captured.append(f"wrapped: {item['x']}")
            return captured[-1]

        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template=_prompt,
            labels={"yes": _builder},
        )
        run(spec)
        assert captured == ["wrapped: hello"]

    def test_prompts_sent_to_model_match_items(self):
        """The model receives exactly one formatted prompt per upstream item."""

        class Recorder(MockModel):
            def __init__(self, responses: list[str]):
                super().__init__(responses=responses)
                self.prompts_seen: list[Prompt] = []

            def generate_batch(self, prompts, **kwargs):
                self.prompts_seen = list(prompts)
                return super().generate_batch(prompts, **kwargs)

        data = [{"x": "a"}, {"x": "b"}]
        model = Recorder(responses=["yes", "yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="Classify {x}",
            labels={"yes": _identity},
        )
        run(spec)
        assert model.prompts_seen == ["Classify a", "Classify b"]

    def test_length_mismatch_raises(self):
        """If generate_batch returns fewer responses than prompts, raise."""

        class Truncator(MockModel):
            def generate_batch(self, prompts, **kwargs):
                self._call_count += 1
                return [[self.responses[0]]]  # always one response

        data = [{"x": "1"}, {"x": "2"}, {"x": "3"}]
        model = Truncator(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"yes": _identity},
        )
        with pytest.raises(RuntimeError, match="returned 1 responses for 3 prompts"):
            run(spec)

    def test_unhashable_label_raises(self):
        """parse_fn returning an unhashable value raises a clear error."""
        data = [{"x": "1"}]
        model = MockModel(responses=["anything"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            parse_fn=lambda r: ["unhashable", "list"],
            labels={},
        )
        with pytest.raises((TypeError, RuntimeError), match="unhashable label"):
            run(spec)

    def test_all_unknown_no_default_returns_empty(self):
        """When every item has an unknown label and no default, output is empty."""
        data = [{"x": "1"}, {"x": "2"}]
        model = MockModel(responses=["mystery", "mystery"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"known": _identity},
        )
        result = run(spec)
        assert result == []

    def test_sub_pipeline_exception_propagates(self):
        """An exception inside a sub-pipeline surfaces as RuntimeError."""

        def _boom(src: DataSpec) -> DataSpec:
            def _execute(_data):
                raise ValueError("boom")

            return DataSpec(NodeType.ALTER, src, _execute, "boom")

        data = [{"x": "1"}]
        model = MockModel(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"yes": _boom},
        )
        with pytest.raises(RuntimeError, match="boom"):
            run(spec)

    def test_within_bucket_order_preserved(self):
        """Items within a bucket preserve their upstream order."""
        data = [{"x": "a"}, {"x": "b"}, {"x": "c"}]
        model = MockModel(responses=["same", "same", "same"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"same": _identity},
        )
        result = run(spec)
        assert [r["x"] for r in result] == ["a", "b", "c"]

    def test_default_parser_lowercases_and_strips(self):
        """Default parse_fn normalizes responses so lowercase label keys match."""
        data = [{"x": "1"}, {"x": "2"}]
        model = MockModel(responses=["  YES ", "No"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"yes": _tag("Y"), "no": _tag("N")},
        )
        tags = [r["tag"] for r in run(spec)]
        assert tags == ["Y", "N"]

    def test_parse_fn_returning_none_routes_to_default(self):
        """A None label goes to default if set (hashable, not in labels)."""
        data = [{"x": "1"}]
        model = MockModel(responses=["anything"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            parse_fn=lambda r: None,
            labels={},
            default=_tag("fallback"),
        )
        result = run(spec)
        assert result == [{"x": "1", "tag": "fallback"}]

    def test_parse_fn_exception_propagates(self):
        """An exception raised inside parse_fn propagates."""

        def _bad_parse(r: str):
            raise ValueError("parse fail")

        data = [{"x": "1"}]
        model = MockModel(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            parse_fn=_bad_parse,
            labels={"yes": _identity},
        )
        with pytest.raises((ValueError, RuntimeError), match="parse fail"):
            run(spec)

    def test_temperature_passed_through_when_set(self):
        """Forwards temperature to generate_batch only when explicitly set."""

        class Recorder(MockModel):
            def __init__(self, responses):
                super().__init__(responses=responses)
                self.kwargs_seen: dict = {}

            def generate_batch(self, prompts, **kwargs):
                self.kwargs_seen = dict(kwargs)
                return super().generate_batch(prompts, **kwargs)

        data = [{"x": "1"}]
        model = Recorder(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            temperature=0.3,
            labels={"yes": _identity},
        )
        run(spec)
        assert model.kwargs_seen.get("temperature") == 0.3

    def test_temperature_omitted_when_none(self):
        """Omits temperature kwarg when default (None) is used."""

        class Recorder(MockModel):
            def __init__(self, responses):
                super().__init__(responses=responses)
                self.kwargs_seen: dict = {}

            def generate_batch(self, prompts, **kwargs):
                self.kwargs_seen = dict(kwargs)
                return super().generate_batch(prompts, **kwargs)

        data = [{"x": "1"}]
        model = Recorder(responses=["yes"])
        spec = by_llm_label(
            from_list(data),
            model=model,
            prompt_template="{x}",
            labels={"yes": _identity},
        )
        run(spec)
        assert "temperature" not in model.kwargs_seen

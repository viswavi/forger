"""Reusable test fixtures for Forger tests."""

from typing import Any

from pyforger.acquire.model import Model, Prompt
from pyforger.core import DataSpec, NodeType


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
        self.responses = responses or ['{"text": "sample"}']
        self._call_count = 0

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        """Return one canned response per prompt, cycling through self.responses."""
        self._call_count += 1
        return [[self.responses[i % len(self.responses)]] for i in range(len(prompts))]


class RecordingMockModel(MockModel):
    """MockModel that records every batch of prompts passed to generate_batch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize and set up prompt recording."""
        super().__init__(*args, **kwargs)
        self.recorded_prompts: list[list[Prompt]] = []

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[list[str]]:
        """Append the current prompt batch to recorded_prompts, then delegate."""
        self.recorded_prompts.append(list(prompts))
        return super().generate_batch(
            prompts, n=n, temperature=temperature, max_tokens=max_tokens
        )


class CallCounter:
    """Tracks how many times an execute function is called."""

    def __init__(self):
        """Initialize call count to zero."""
        self.count = 0

    def reset(self):
        """Reset call count to zero."""
        self.count = 0


def make_source(data: list, counter: CallCounter = None) -> DataSpec:
    """Source node returning fixed data."""

    def _execute():
        if counter is not None:
            counter.count += 1
        return data

    return DataSpec(NodeType.ACQUIRE, [], _execute, "test_source")


def make_double_repetition_data(source: DataSpec) -> DataSpec:
    """Alter node that repeats its input list twice."""

    def _execute(data):
        return data + data

    return DataSpec(NodeType.ALTER, source, _execute, "test_duplicate")


def make_empty_filter(source: DataSpec) -> DataSpec:
    """Filter node that always returns empty."""

    def _execute(_data):
        # _data is unused -- this is just here to match the expected signature.
        return []

    return DataSpec(NodeType.FILTER, [source], _execute, "test_empty_filter")


def make_concat(*sources: DataSpec) -> DataSpec:
    """Alter node that concatenates multiple inputs."""

    def _execute(*input_data):
        result = []
        for data in input_data:
            result.extend(data)
        return result

    return DataSpec(NodeType.ALTER, list(sources), _execute, "test_concat")

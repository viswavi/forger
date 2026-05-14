"""Reusable test fixtures for Forger tests."""

from __future__ import annotations

from typing import Any

from pyforger.acquire.model import Model, Prompt


class MockModel(Model):
    """Model that returns canned responses from generate_batch for testing."""

    def __init__(
        self,
        responses: list[str],
        model_id: str = "test/model",
        **kwargs: Any,
    ) -> None:
        """Build a MockModel that cycles ``responses`` across prompts."""
        super().__init__(model_id=model_id, **kwargs)
        self.responses = responses
        self._call_count = 0

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching. Excludes mutable _call_count."""
        return (type(self).__name__, self._model_id, tuple(self.responses))

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Return one canned response per prompt, cycling through self.responses."""
        self._call_count += 1
        return [
            [self.responses[i % len(self.responses)]] * n for i in range(len(prompts))
        ]


class RecordingMockModel(MockModel):
    """MockModel that records every batch of prompts passed to generate_batch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build a RecordingMockModel and initialize the recording buffer."""
        super().__init__(*args, **kwargs)
        self.recorded_prompts: list[list[Prompt]] = []

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Append the current prompt batch to recorded_prompts, then delegate."""
        self.recorded_prompts.append(list(prompts))
        return super().generate_batch(
            prompts, n=n, temperature=temperature, max_tokens=max_tokens, stop=stop
        )


class CallCounter:
    """Tracks how many times a function is called. Used to assert call counts."""

    def __init__(self):
        """Initialize call count to zero."""
        self.count = 0

    def reset(self):
        """Reset call count to zero."""
        self.count = 0

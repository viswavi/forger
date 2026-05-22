"""Reusable test fixtures for Forger tests."""

from __future__ import annotations

from typing import Any

from pyforger.acquire.model import Model, Prompt


class MockModel(Model):
    """Model that returns canned responses from generate_batch for testing."""

    def __init__(
        self,
        responses: list[str] | list[list[str]],
        model_id: str = "test/model",
        **kwargs: Any,
    ) -> None:
        """Build a MockModel with canned per-prompt completion lists.

        ``responses[i]`` is the list of completions returned for the i-th prompt of
        each generate_batch call. A flat ``list[str]`` is auto-promoted by wrapping
        each string in a single-element list.  Otherwise, if a list of lists is given,
        the inner lists are used directly as the per-prompt completions (e.g. lists of
        4 responses for n=4 completions per prompt).
        """
        super().__init__(model_id=model_id, **kwargs)
        normalized: list[list[str]] = [
            [r] if isinstance(r, str) else list(r) for r in responses
        ]
        self.responses = normalized
        self._call_count = 0

    def cache_key(self) -> tuple:
        """Stable hash key for @op caching. Excludes mutable _call_count."""
        return (
            type(self).__name__,
            self._model_id,
            tuple(tuple(r) for r in self.responses),
        )

    def generate_batch(
        self,
        prompts: list[Prompt],
        *,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """Return self.responses[i] for the i-th prompt; require len == n."""
        self._call_count += 1
        canned = self.responses[: len(prompts)]
        if any(len(r) != n for r in canned):
            shapes = [len(r) for r in canned]
            raise ValueError(
                f"MockModel canned response shapes {shapes} do not match n={n}"
            )
        return [list(r) for r in canned]


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

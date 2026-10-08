"""Concurrent LiteLLM calls with per-problem, per-stage @op caching."""
from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from litellm import acompletion

from pyforger import op

Message = dict[str, str]

# litellm caches its async HTTP client per event loop, so calls must share one
# long-lived loop rather than each opening/closing its own via asyncio.run().
_shared_loop = None
_loop_lock = threading.Lock()


def shared_loop():
    """Return the shared background event loop used by API requests."""
    global _shared_loop
    with _loop_lock:
        if _shared_loop is None:
            _shared_loop = asyncio.new_event_loop()
            threading.Thread(target=_shared_loop.run_forever, daemon=True).start()
        return _shared_loop


async def _acomplete_one(
    model_id: str,
    api_key: str,
    api_base: str | None,
    messages: list[Message],
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    kwargs: dict = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = await acompletion(
        model=model_id,
        api_key=api_key,
        api_base=api_base,
        messages=messages,
        max_tokens=max_tokens,
        num_retries=2,
        **kwargs,
    )
    return response.choices[0].message.content or ""


async def _acomplete_n(
    model_id: str,
    api_key: str,
    api_base: str | None,
    messages: list[Message],
    n: int,
    temperature: float | None,
    max_tokens: int | None,
) -> list[str | None]:
    """Fetch n completions with concurrent single-completion calls.

    Anthropic does not support n-choice requests. Omit temperature when None
    for providers that reject it.
    """
    results = await asyncio.gather(
        *[
            _acomplete_one(
                model_id, api_key, api_base, messages, temperature, max_tokens
            )
            for _ in range(n)
        ],
        return_exceptions=True,
    )
    return [None if isinstance(item, BaseException) else item for item in results]


class PartialCompletionError(RuntimeError):
    """Carry successful completions from a partially failed sampling batch."""

    def __init__(self, completions):
        """Store configuration and initialize instance state."""
        super().__init__("Some completion requests failed")
        self.completions = completions


@op(name="llm_async.sample_sync")
def sample_sync(
    model_id: str,
    messages: list[Message],
    *,
    n: int,
    temperature: float | None,
    max_tokens: int | None = None,
    stage_tag: str = "",
) -> list[str]:
    """Run async sampling on a shared loop through a synchronous @op entry point.

    stage_tag distinguishes equal-sized sampling stages in the cache key;
    otherwise a later stage would replay earlier completions.
    """
    coro = _acomplete_n(
        model_id,
        os.environ["LITELLM_API_KEY"],
        os.environ.get("LITELLM_BASE_URL"),
        messages,
        n,
        temperature,
        max_tokens,
    )
    completions = asyncio.run_coroutine_threadsafe(coro, shared_loop()).result()
    if any(item is None for item in completions):
        raise PartialCompletionError(completions)  # never cache a transient failure
    return [item for item in completions if item is not None]


def sample_many(
    jobs: list[dict[str, Any]],
    *,
    max_workers: int = 20,
) -> list[list[str | None] | None]:
    """Run cached sample_sync jobs concurrently, preserving input order.

    Return None for jobs that exhaust retries without aborting the batch.
    """
    if not jobs:
        return []
    results: list[list[str | None] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(sample_sync, **job): i for i, job in enumerate(jobs)
        }
        for future in future_to_idx:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad job shouldn't sink batch
                print(f"[llm_async] job {idx} failed: {type(exc).__name__}")
                results[idx] = getattr(exc, "completions", None)
    return results

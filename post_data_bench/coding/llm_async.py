"""Async LiteLLM calling helper for gold-generation and perturbation-authoring.

Deliberately bypasses pyforger.acquire.Model / generate_batch / generate_from_context:
those are synchronous under the hood (litellm.completion / litellm.batch_completion)
and, in generate_from_context's case, cached at the whole-batch granularity. This
module uses litellm.acompletion directly for genuine concurrent network I/O across
problems, while keeping per-problem/per-stage @op caching.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from litellm import acompletion

from pyforger import op

Message = dict[str, str]


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
) -> list[str]:
    """Fetch n completions via n concurrent single-completion calls.

    Originally this issued one litellm.acompletion(..., n=n, ...) call, relying on
    the `n=` batched-completions parameter -- but confirmed empirically (against the
    CMU gateway) that Anthropic's API doesn't support `n` at all:
    `litellm.UnsupportedParamsError: anthropic does not support parameters: ['n']`.
    Fanning out n independent single-completion calls works universally across
    providers and, as a bonus, gives more real concurrency than a single n-choices
    request would have.

    `temperature` is omitted from each request entirely when None: some models (e.g.
    Claude Sonnet 5 via the CMU gateway) reject an explicit temperature param
    outright ("temperature is deprecated for this model") rather than just
    clamping/ignoring an out-of-range value.
    """
    return await asyncio.gather(
        *[
            _acomplete_one(
                model_id, api_key, api_base, messages, temperature, max_tokens
            )
            for _ in range(n)
        ]
    )


@op(name="llm_async.sample_sync")
def sample_sync(
    model_id: str,
    api_key: str,
    api_base: str | None,
    messages: list[Message],
    *,
    n: int,
    temperature: float | None,
    max_tokens: int | None = None,
    stage_tag: str = "",
) -> list[str]:
    """Sync, @op-cacheable entry point wrapping one async acompletion call.

    @op's wrapper (pyforger/op.py) is plain synchronous code that pickles the return
    value, so the unit it wraps must itself be sync; asyncio.run opens and closes its
    own event loop per call, so this is safe to invoke concurrently from multiple
    threads (each thread gets its own loop, no shared state).

    `stage_tag` exists purely to vary the @op cache key: gold_generation.py's
    rejection-sampling schedule (1, 1, 2, 4, ...) issues two stages with the same
    `n`, and without a differentiator the second stage would just replay the first
    stage's cached completions instead of drawing a genuinely new sample. `stage_tag`
    is never read in the body -- its only job is being part of the bound arguments
    @op hashes.
    """
    return asyncio.run(
        _acomplete_n(model_id, api_key, api_base, messages, n, temperature, max_tokens)
    )


def sample_many(
    jobs: list[dict[str, Any]],
    *,
    max_workers: int = 20,
) -> list[list[str] | None]:
    """Run many sample_sync calls concurrently across a ThreadPoolExecutor.

    Each job dict is the kwargs for one sample_sync call. Each worker thread blocks
    on its own asyncio.run(acompletion(...)), so many requests are genuinely in
    flight at once (the GIL releases during network I/O) while sample_sync's @op
    cache still gives per-job granularity and resumability -- unlike caching a whole
    batch as one @op call, which would invalidate every job whenever any single one
    of them changes.

    Returns one entry per job, in the same order as `jobs`; a job whose call fails
    (after sample_sync's own retries) gets None rather than aborting the rest of the
    batch.
    """
    if not jobs:
        return []
    results: list[list[str] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(sample_sync, **job): i for i, job in enumerate(jobs)
        }
        for future in future_to_idx:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad job shouldn't sink batch
                print(f"[llm_async] job {idx} failed: {type(exc).__name__}: {exc}")
                results[idx] = None
    return results

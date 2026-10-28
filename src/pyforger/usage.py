"""JSON-safe per-invocation accounting shared by execution backends."""

from __future__ import annotations

import uuid
from typing import Any

TOKENS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
)


def usage_record(
    value: dict[str, Any] | None = None, *, zero: bool = False
) -> dict[str, Any]:
    value = value or {}
    cost = value.get("cost_usd", 0.0 if zero else None)
    return {
        **{key: value.get(key, 0 if zero else None) for key in TOKENS},
        "cost_usd": cost,
        "known_cost_usd": value.get("known_cost_usd", cost or 0.0),
        "cost_complete": value.get("cost_complete", cost is not None),
        "call_id": value.get("call_id") or uuid.uuid4().hex,
        "scope": "call",
    }


def combine_usage(
    values: list[dict[str, Any]], *, call_id: str | None = None
) -> dict[str, Any]:
    result = usage_record(zero=True)
    if call_id is not None:
        result["call_id"] = call_id
    for value in values:
        for key in TOKENS:
            result[key] = (
                result[key] + value[key]
                if result[key] is not None and value.get(key) is not None
                else None
            )
        result["known_cost_usd"] += value.get(
            "known_cost_usd", value.get("cost_usd") or 0.0
        )
        result["cost_complete"] &= value.get(
            "cost_complete", value.get("cost_usd") is not None
        )
    result["cost_usd"] = result["known_cost_usd"] if result["cost_complete"] else None
    return result


def _sdk_metrics(session: Any) -> dict[str, Any]:
    stats = getattr(getattr(session, "state", None), "stats", None)
    metrics = dict(stats.usage_to_metrics) if stats is not None else {}
    agent = getattr(session, "agent", None)
    for llm in (
        getattr(agent, "llm", None),
        getattr(getattr(agent, "condenser", None), "llm", None),
    ):
        if llm is not None and llm.metrics is not None:
            metrics.setdefault(llm.usage_id, llm.metrics)
    # A registry and agent may expose the very same metric under different IDs.
    return {str(id(value)): value for value in metrics.values()}


def sdk_snapshot(session: Any) -> dict[str, Any]:
    return {
        key: {
            "identity": id(metrics),
            "tokens": len(metrics.token_usages),
            "costs": len(metrics.costs),
            "responses": len(metrics.response_latencies),
        }
        for key, metrics in _sdk_metrics(session).items()
    }


def sdk_usage(
    session: Any, baseline: dict[str, Any] | None = None, *, incomplete: bool = False
) -> dict[str, Any]:
    baseline = baseline or {}
    parts = []
    for key, metrics in _sdk_metrics(session).items():
        previous = baseline.get(key, {})
        if previous.get("identity") != id(metrics):
            previous = {}
        tokens = metrics.token_usages[previous.get("tokens", 0) :]
        costs = metrics.costs[previous.get("costs", 0) :]
        responses = len(metrics.response_latencies) - previous.get("responses", 0)
        if not tokens and not costs and responses == 0:
            continue
        # The SDK drops unpriced AND free costs. Separate response counts keep
        # complementary missing token/cost rows from appearing fully measured.
        measured = responses > 0 and len(tokens) == responses
        complete = not incomplete and responses > 0 and len(costs) == responses
        parts.append(
            usage_record(
                {
                    "prompt_tokens": sum(row.prompt_tokens for row in tokens)
                    if measured
                    else None,
                    "completion_tokens": sum(row.completion_tokens for row in tokens)
                    if measured
                    else None,
                    "reasoning_tokens": sum(row.reasoning_tokens for row in tokens)
                    if measured
                    else None,
                    "cached_tokens": sum(row.cache_read_tokens for row in tokens)
                    if measured
                    else None,
                    "cache_write_tokens": sum(row.cache_write_tokens for row in tokens)
                    if measured
                    else None,
                    "known_cost_usd": sum(row.cost for row in costs),
                    "cost_usd": sum(row.cost for row in costs) if complete else None,
                    "cost_complete": complete,
                }
            )
        )
    return combine_usage(parts) if parts else usage_record()

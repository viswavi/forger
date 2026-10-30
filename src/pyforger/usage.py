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


def _token_value(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _cost_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") and number >= 0 else None


def _normalize_usage(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    cost = _cost_value(source.get("cost_usd"))
    known = _cost_value(source.get("known_cost_usd"))
    if known is None:
        known = cost or 0.0
    raw_complete = source.get("cost_complete")
    complete = cost is not None if raw_complete is None else raw_complete is True
    return {
        **{key: _token_value(source.get(key)) for key in TOKENS},
        "cost_usd": cost,
        "known_cost_usd": known,
        "cost_complete": complete and cost is not None,
        "call_id": source.get("call_id"),
        "scope": "call",
    }


def usage_record(
    value: dict[str, Any] | None = None, *, zero: bool = False
) -> dict[str, Any]:
    normalized = _normalize_usage(value)
    if zero:
        for key in TOKENS:
            normalized[key] = 0 if normalized[key] is None else normalized[key]
        normalized["cost_usd"] = 0.0 if normalized["cost_usd"] is None else normalized["cost_usd"]
        source = value if isinstance(value, dict) else {}
        normalized["cost_complete"] = source.get("cost_complete") is not False
    return {
        **normalized,
        "call_id": normalized.get("call_id") or uuid.uuid4().hex,
    }


def combine_usage(
    values: list[dict[str, Any]], *, call_id: str | None = None
) -> dict[str, Any]:
    result = usage_record(zero=True)
    if call_id is not None:
        result["call_id"] = call_id
    for raw in values:
        value = _normalize_usage(raw)
        for key in TOKENS:
            result[key] = (
                result[key] + value[key]
                if result[key] is not None and value[key] is not None
                else None
            )
        result["known_cost_usd"] += value["known_cost_usd"]
        result["cost_complete"] &= value["cost_complete"]
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

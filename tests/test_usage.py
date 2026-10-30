"""Accounting against real SDK metrics and native Harbor result models."""

from types import SimpleNamespace

from openhands.sdk.conversation.conversation_stats import ConversationStats, Metrics

from pyforger.usage import combine_usage, sdk_snapshot, sdk_usage, usage_record


def test_sdk_missing_and_complementary_metadata():
    metrics = Metrics(model_name="unknown")
    session = SimpleNamespace(state=SimpleNamespace(stats=ConversationStats()))
    session.state.stats.usage_to_metrics["agent"] = metrics
    assert sdk_usage(session)["cost_usd"] is None
    metrics.add_response_latency(0.1, "a")
    metrics.add_token_usage(10, 5, 2, 3, 100, "a", 1)
    metrics.add_response_latency(0.1, "b")
    metrics.add_cost(0.2)
    usage = sdk_usage(session)
    assert usage["cost_usd"] is None
    assert usage["prompt_tokens"] is None
    assert usage["known_cost_usd"] == 0.2


def test_sdk_deltas_include_condenser_and_preserve_failed_subtotal():
    stats = ConversationStats()
    session = SimpleNamespace(state=SimpleNamespace(stats=stats))
    stats.usage_to_metrics["agent"] = Metrics(model_name="a")
    baseline = sdk_snapshot(session)
    for key in ("agent", "condenser"):
        metrics = stats.usage_to_metrics.setdefault(key, Metrics(model_name=key))
        metrics.add_response_latency(0.1, key)
        metrics.add_token_usage(10, 5, 2, 3, 100, key, 1)
        metrics.add_cost(0.2)
    first = sdk_usage(session, baseline)
    assert first["prompt_tokens"] == 20
    assert first["cached_tokens"] == 4
    assert first["cache_write_tokens"] == 6
    assert first["reasoning_tokens"] == 2
    assert first["cost_usd"] == 0.4
    baseline = sdk_snapshot(session)
    metrics.add_response_latency(0.1, "next")
    metrics.add_token_usage(7, 3, 0, 0, 100, "next")
    metrics.add_cost(0.1)
    second = sdk_usage(session, baseline)
    assert second["prompt_tokens"] == 7
    assert second["cost_usd"] == 0.1
    failed = sdk_usage(session, baseline, incomplete=True)
    assert failed["cost_usd"] is None
    assert failed["known_cost_usd"] == 0.1
    assert first["call_id"] != second["call_id"]


def test_combine_usage_tolerates_non_numeric_provider_usage():
    combined = combine_usage(
        [
            {"prompt_tokens": "123", "known_cost_usd": "0.01"},
            {"prompt_tokens": 7, "cost_usd": "bad", "cached_tokens": "x"},
            usage_record({"prompt_tokens": 3, "cost_usd": 0.2}),
        ]
    )
    # A malformed token/cost field means that field is unknown, not fatal.
    assert combined["prompt_tokens"] is None
    assert combined["cached_tokens"] is None
    assert combined["known_cost_usd"] == 0.2
    assert combined["cost_usd"] is None


def test_usage_record_normalizes_malformed_values():
    record = usage_record(
        {"prompt_tokens": "10", "cost_usd": "0.5", "cost_complete": True}
    )
    assert record["prompt_tokens"] is None
    assert record["cost_usd"] is None
    assert record["cost_complete"] is False


def test_unknown_attempt_never_becomes_free():
    priced = usage_record({"cost_usd": 0.2, "prompt_tokens": 10})
    combined = combine_usage([priced, usage_record()])
    assert combined["cost_usd"] is None
    assert combined["known_cost_usd"] == 0.2
    assert combined["scope"] == "call"


def test_native_harbor_mixed_model_cost():
    from harbor.models.agent.context import AgentContext, ModelUsage
    from harbor.models.trial.result import TrialResult

    from pyforger.harbor import _trial_usage

    context = AgentContext(
        n_input_tokens=10,
        n_output_tokens=5,
        cost_usd=0.1,
        model_usage={
            "priced": ModelUsage(cost_usd=0.1),
            "unknown": ModelUsage(cost_usd=None),
        },
    )
    result = TrialResult.model_construct(agent_result=context)
    usage = _trial_usage(result)
    assert usage["cost_usd"] is None
    assert usage["known_cost_usd"] == 0.1
    assert usage["prompt_tokens"] == 10

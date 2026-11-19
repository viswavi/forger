"""Token accounting for remote and local generation."""

import copy
import logging
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyforger.acquire.model import LocalModel, Model
from pyforger.op import op
from pyforger.token_units import TokenUnitQuotaExceeded

DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4.1-Flash"
GLM = "litellm_proxy/azure_ai/FW-GLM-5.3"


def response(prompt_tokens=10, completion_tokens=20, *, reasoning_tokens=15):
    """Build a response with cached inputs and reasoning usage."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=prompt_tokens),
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=reasoning_tokens
            ),
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content="thought")
            )
        ],
        _hidden_params={"response_cost": 0.01},
    )


@pytest.fixture(autouse=True)
def isolate_model(monkeypatch):
    """Isolate quota configuration and provider parameter checks."""
    monkeypatch.delenv("FORGER_TOKEN_USAGE_FILE", raising=False)
    monkeypatch.delenv("FORGER_TOKEN_UNIT_LIMIT", raising=False)
    monkeypatch.setattr(
        "pyforger.acquire.model.litellm.get_supported_openai_params",
        lambda **kwargs: [],
    )


@pytest.fixture
def complete(monkeypatch):
    """Replace remote completions with metered fixture responses."""
    call = Mock(side_effect=lambda **kwargs: response())
    monkeypatch.setattr("pyforger.acquire.model.litellm.completion", call)
    return call


def enable_budget(monkeypatch, tmp_path, limit=100):
    """Configure a shared ledger for the test."""
    monkeypatch.setenv("FORGER_TOKEN_USAGE_FILE", str(tmp_path / "usage.json"))
    monkeypatch.setenv("FORGER_TOKEN_UNIT_LIMIT", str(limit))


@pytest.mark.parametrize("model_id,expected", [(DEEPSEEK, 90), (GLM, 424.9)])
def test_weights_include_cached_inputs_and_do_not_add_reasoning(
    complete, model_id, expected
):
    """Weight aggregate usage once and reset instance totals."""
    model = Model(model_id)
    assert model.total_token_units == 0
    result = model.generate("Hello")
    assert result.usage["token_units"] == expected
    assert result.usage["reasoning_tokens"] == 15
    model.generate("Again")
    assert model.total_token_units == expected * 2
    assert model.total_cost == 0.02
    model.reset_history()
    assert model.total_token_units == 0


def test_unknown_model_usage_is_unknown(complete):
    """Unrecognized model families have unknown token units."""
    model = Model("gpt-4o")
    assert math.isnan(model.total_token_units)
    assert math.isnan(model.generate("Hello").usage["token_units"])
    assert math.isnan(model.total_token_units)
    assert model.total_cost == 0.01
    model.reset_history()
    assert math.isnan(model.total_token_units)
    assert model.total_cost == 0


@pytest.mark.parametrize("model_id", [DEEPSEEK, "gpt-4o"])
def test_logs_units_and_dollars_on_first_call_and_milestones(
    complete, caplog, model_id
):
    """Log completed costs alongside units, using raw-token cadence if unknown."""
    complete.side_effect = [response(600_000, 0)] * 3
    model = Model(model_id)
    with caplog.at_level(logging.INFO, logger="pyforger.acquire.model"):
        model.generate("First")
        model.generate("Second")
        model.generate("Third")
    logs = [r.message for r in caplog.records if "cumulative usage" in r.message]
    assert len(logs) == 2
    assert "$0.010000 USD" in logs[0]
    assert "$0.020000 USD" in logs[1]
    assert ("1200000.0" if model_id == DEEPSEEK else "NaN") + " token units" in logs[1]
    assert model.total_cost == 0.03


def test_shared_ledger_records_model_dollars(monkeypatch, tmp_path, complete):
    """Model history and the persistent ledger expose the same known dollars."""
    enable_budget(monkeypatch, tmp_path, limit=1000)
    model = Model(DEEPSEEK)
    model.generate("Hello")
    model.generate("Again")
    state = model._token_budget.snapshot()
    assert state["token_units"] == model.total_token_units == 180
    assert state["cost_usd"] == model.total_cost == 0.02
    assert state["models"][DEEPSEEK]["cost_usd"] == 0.02
    assert state["unpriced_calls"] == 0


def test_unavailable_pricing_preserves_units_and_marks_ledger(
    monkeypatch, tmp_path, complete
):
    """A missing price remains distinguishable from a known free response."""
    enable_budget(monkeypatch, tmp_path)
    raw = response()
    raw._hidden_params = {}
    complete.side_effect = None
    complete.return_value = raw
    monkeypatch.setattr("litellm.completion_cost", Mock(return_value=None))
    monkeypatch.setattr("pyforger.acquire.model._COST_WARNING_SEEN", set())
    model = Model(DEEPSEEK)
    with pytest.warns(UserWarning, match="Cost unavailable"):
        model.generate("Hello")
    state = model._token_budget.snapshot()
    assert state["token_units"] == model.total_token_units == 90
    assert state["cost_usd"] == model.total_cost == 0
    assert state["unpriced_calls"] == 1


def test_missing_usage_is_unknown(complete):
    """Missing usage makes cumulative units unknown."""
    raw = response()
    raw.usage = None
    complete.side_effect = None
    complete.return_value = raw
    model = Model(DEEPSEEK)
    assert math.isnan(model.generate("Hello").usage["token_units"])
    assert math.isnan(model.total_token_units)


@pytest.mark.parametrize("progress", [False, True])
def test_partial_batch_failure_charges_all_successes(monkeypatch, progress):
    """Retain successful charges when another batch request fails."""
    error = RuntimeError("one call failed")

    def completion(messages, **kwargs):
        if messages[0]["content"] == "bad":
            raise error
        return response()

    monkeypatch.setattr("pyforger.acquire.model.litellm.completion", completion)
    model = Model(DEEPSEEK, show_progress=progress)
    with pytest.raises(RuntimeError, match="one call failed"):
        model.generate_batch(["good", "bad", "also good"])
    assert model.total_token_units == 180
    assert model.total_cost == 0.02
    # history holds only results of calls that return; the totals keep the spend.
    assert model.history == []


def test_parse_failure_still_charges_response(complete):
    """Account for returned responses before parsing their choices."""
    raw = response()
    raw.choices = None
    complete.side_effect = None
    complete.return_value = raw
    model = Model(DEEPSEEK)
    with pytest.raises(TypeError):
        model.generate("Hello")
    assert model.total_token_units == 90
    assert model.total_cost == 0.01


def test_concurrent_calls_have_consistent_totals(complete):
    """Concurrent updates preserve every response's usage."""
    model = Model(GLM)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(model.generate, ["Hello"] * 100))
    assert model.total_token_units == pytest.approx(42_490)  # 100 x 424.9
    assert model.total_usage["prompt_tokens"] == 1_000
    assert len(model.history) == 100


@pytest.mark.parametrize("effort", ["none", "low", "high"])
def test_reasoning_dict_passes_through(complete, effort):
    """Forward the requested reasoning effort without renaming it."""
    Model(DEEPSEEK, reasoning={"effort": effort}).generate("Hello")
    assert complete.call_args.kwargs["reasoning"] == {"effort": effort}
    assert "reasoning_effort" not in complete.call_args.kwargs


@pytest.mark.parametrize("with_budget", [False, True])
@pytest.mark.parametrize(
    "clone", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_model_serialization_preserves_accounting(
    monkeypatch, tmp_path, complete, clone, with_budget
):
    """Copied models preserve totals and gain independent instance locks."""
    if with_budget:
        enable_budget(monkeypatch, tmp_path, limit=1000)
    model = Model(DEEPSEEK)
    model.generate("Hello")
    restored = clone(model)
    assert restored.total_token_units == 90
    restored.generate("Again")
    assert restored.total_token_units == 180
    assert model.total_token_units == 90


def test_shared_budget_keeps_successful_calls_from_failed_batch(
    monkeypatch, tmp_path, complete
):
    """Persist partial batch usage for subsequent model instances."""
    enable_budget(monkeypatch, tmp_path, limit=180)

    def completion(messages, **kwargs):
        if messages[0]["content"] == "bad":
            raise RuntimeError("failed request")
        return response()

    complete.side_effect = completion
    model = Model(DEEPSEEK)
    with pytest.raises(RuntimeError, match="failed request"):
        model.generate_batch(["first", "bad", "last"])
    assert model.total_token_units == 180
    with pytest.raises(TokenUnitQuotaExceeded):
        Model(DEEPSEEK).generate("more")


def test_cached_op_does_not_charge_again(complete):
    """Reusing an op result neither calls the model nor adds usage."""

    @op
    def generate(model, prompt):
        return model.generate(prompt)

    model = Model(DEEPSEEK)
    original_key = model.cache_key()
    assert generate(model, "Hello").usage["token_units"] == 90
    assert generate(model, "Hello").usage["token_units"] == 90
    assert model.cache_key() == original_key
    assert model.total_token_units == 90
    complete.assert_called_once()


def test_shared_budget_across_instances_survives_history_reset(
    monkeypatch, tmp_path, complete
):
    """Clearing instance history cannot reset the shared quota."""
    enable_budget(monkeypatch, tmp_path)
    first = Model(DEEPSEEK)
    second = Model(DEEPSEEK)
    first.generate("Hello")
    first.reset_history()
    second.generate("Hello")
    with pytest.raises(TokenUnitQuotaExceeded):
        first.generate("Hello")
    assert complete.call_count == 2
    assert first.total_token_units == 0
    assert second.total_token_units == 90


def test_shared_budget_blocks_unknown_models_before_api(
    monkeypatch, tmp_path, complete
):
    """Reject unweighted models before spending shared quota."""
    enable_budget(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        Model("gpt-4o").generate("Hello")
    complete.assert_not_called()


@pytest.mark.parametrize("counts", [(None, 1), (1, None), (-1, 1)])
def test_shared_budget_requires_valid_usage(monkeypatch, tmp_path, complete, counts):
    """Invalid usage blocks all subsequent requests against the ledger."""
    enable_budget(monkeypatch, tmp_path)
    complete.side_effect = None
    complete.return_value = response(*counts)
    model = Model(DEEPSEEK)
    with pytest.raises(ValueError, match="token usage"):
        model.generate("Hello")
    assert math.isnan(model.total_token_units)
    with pytest.raises(ValueError, match="accounting is incomplete"):
        Model(DEEPSEEK).generate("Hello")
    complete.assert_called_once()


def test_shared_budget_records_usage_before_parsing(monkeypatch, tmp_path, complete):
    """Persist charges even when response parsing raises."""
    enable_budget(monkeypatch, tmp_path, limit=90)
    raw = response()
    raw.choices = None
    complete.side_effect = None
    complete.return_value = raw
    with pytest.raises(TypeError):
        Model(DEEPSEEK).generate("Hello")
    with pytest.raises(TokenUnitQuotaExceeded):
        Model(DEEPSEEK).generate("Hello")
    complete.assert_called_once()


def test_cost_metadata_failure_does_not_lose_token_charge(
    monkeypatch, tmp_path, complete
):
    """Cost metadata failures cannot erase token charges."""
    enable_budget(monkeypatch, tmp_path, limit=90)
    monkeypatch.setattr(
        "pyforger.acquire.model._extract_cost", Mock(side_effect=ValueError("bad cost"))
    )
    model = Model(DEEPSEEK)
    with pytest.raises(ValueError, match="bad cost"):
        model.generate("Hello")
    assert model.total_token_units == 90
    with pytest.raises(TokenUnitQuotaExceeded):
        Model(DEEPSEEK).generate("Hello")
    complete.assert_called_once()


def test_shared_budget_checks_each_queued_batch_request(
    monkeypatch, tmp_path, complete
):
    """Stop queued requests once completed calls exhaust the budget."""
    enable_budget(monkeypatch, tmp_path, limit=90)
    real_executor = ThreadPoolExecutor
    monkeypatch.setattr(
        "pyforger.acquire.model.ThreadPoolExecutor",
        lambda **kwargs: real_executor(max_workers=1),
    )
    model = Model(DEEPSEEK)
    with pytest.raises(TokenUnitQuotaExceeded):
        model.generate_batch(["first", "second", "third"])
    assert model.total_token_units == 90
    complete.assert_called_once()


def test_local_generation_is_free_even_with_budget(monkeypatch, tmp_path):
    """Local inference is quota-exempt and reports unknown weights as NaN."""
    enable_budget(monkeypatch, tmp_path, limit=1)
    output = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[SimpleNamespace(text="local", token_ids=[3, 4, 5])],
    )
    vllm = SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)
    monkeypatch.setattr("pyforger.acquire.model._import_vllm", lambda: vllm)
    model = LocalModel("Qwen/Qwen3-4B")
    model._llm = SimpleNamespace(chat=lambda *args, **kwargs: [output])
    result = model.generate("Hello")
    assert math.isnan(result.usage["token_units"])
    assert math.isnan(model.total_token_units)
    assert model.total_cost == 0
    assert not (tmp_path / "usage.json").exists()


def test_pipeline_token_unit_cap_stops_queued_requests(complete, monkeypatch):
    """max_token_units stops the batch at the cap; finished requests stay charged."""
    from pyforger import BudgetExceededError, pipeline, step
    from pyforger.alter import generate_from_context

    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    model = Model(DEEPSEEK, max_concurrency=1)  # 90 units per request

    @step
    def write(rows, model):
        return generate_from_context(rows, "Say {q}.", model)

    @pipeline(max_token_units=100)
    def main(rows, model):
        return write(rows, model)

    with pytest.raises(BudgetExceededError, match="max_token_units=100"):
        main([{"q": str(i)} for i in range(5)], model)
    assert complete.call_count == 2  # 90, then 180 >= 100: the rest never left
    assert model.total_token_units == 180


def test_process_token_unit_cap(complete, monkeypatch):
    """FORGER_MAX_TOKEN_UNITS caps all units spent in the process."""
    from pyforger import BudgetExceededError, _tracking

    model = Model(DEEPSEEK)
    spent = _tracking.usage_snapshot()[3]
    monkeypatch.setenv("FORGER_MAX_TOKEN_UNITS", str(spent + 90))
    model.generate("Hello")
    with pytest.raises(BudgetExceededError, match="FORGER_MAX_TOKEN_UNITS"):
        model.generate("Again")
    monkeypatch.setenv("FORGER_MAX_TOKEN_UNITS", "lots")
    with pytest.raises(ValueError, match="FORGER_MAX_TOKEN_UNITS must be a number"):
        model.generate("Again")


def test_without_weights_only_dollars_are_tracked(complete):
    """Forger's default: no weights, so units are NaN and dollars still add up."""
    from pyforger import token_units

    token_units.set_weights(None)
    model = Model(DEEPSEEK)
    result = model.generate("Hello")
    assert math.isnan(result.usage["token_units"])
    assert math.isnan(model.total_token_units)
    assert model.total_cost == pytest.approx(0.01)


@pytest.mark.parametrize("where", ["process", "pipeline"])
def test_token_unit_caps_refuse_models_without_weights(complete, monkeypatch, where):
    """A unit cap cannot count an unweighted model's spend, so it refuses the call."""
    from pyforger import _tracking, pipeline, step

    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    model = Model("anthropic/claude-haiku-5-5")
    if where == "process":
        spent = _tracking.usage_snapshot()[3]  # earlier tests' units
        monkeypatch.setenv("FORGER_MAX_TOKEN_UNITS", str(spent + 1000))
        with pytest.raises(ValueError, match="No token-unit weights"):
            model.generate("Hello")
    else:

        @step
        def ask(rows, model):
            return [{**r, "a": model.generate(r["q"]).texts[0]} for r in rows]

        @pipeline(max_token_units=1000)
        def main(rows, model):
            return ask(rows, model)

        with pytest.raises(ValueError, match="No token-unit weights"):
            main([{"q": "Hello"}], model)
    complete.assert_not_called()
    # Without a unit cap the same model runs, priced in dollars only.
    monkeypatch.delenv("FORGER_MAX_TOKEN_UNITS", raising=False)
    model.generate("Hello")
    assert complete.call_count == 1


def test_pipeline_rejects_negative_token_unit_cap():
    """max_token_units must be a number >= 0."""
    from pyforger import pipeline

    with pytest.raises(ValueError, match="max_token_units"):
        pipeline(max_token_units=-1)


def test_units_reach_step_report_and_spend_ledger(complete, monkeypatch, tmp_path):
    """Each response's token units reach its step and one ledger line each."""
    from pyforger import _spend, _tracking, pipeline, step
    from pyforger.alter import generate_from_context
    from pyforger.inspect import spend

    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.setenv("FORGER_SPEND_LEDGER", str(tmp_path / "spend.jsonl"))

    @step
    def write(rows, model):
        return generate_from_context(rows, "Say {q}.", model)

    @step
    def again(rows, model):
        return generate_from_context(rows, "Again {text}.", model)

    @pipeline
    def main(rows, model):
        return again(write(rows, model), model)

    main([{"q": "a"}, {"q": "b"}], Model(GLM))
    # 424.9 units per GLM-5.3 call.
    assert _tracking.step_records()["write"].token_units == pytest.approx(849.8)
    assert _tracking.last_pipeline().token_units == pytest.approx(1699.6)
    records = _spend.read(tmp_path / "spend.jsonl")
    assert [r["token_units"] for r in records] == [424.9] * 4
    assert spend(tmp_path / "spend.jsonl").token_units == pytest.approx(1699.6)


def test_agent_usage_gets_token_units(monkeypatch):
    """dynamic_step's agent usage is weighted like Model usage."""
    from pyforger.alter import dynamic

    seen = []
    monkeypatch.setattr(
        dynamic._tracking, "add_usage", lambda *a, **k: seen.append((a, k))
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    metrics = SimpleNamespace(accumulated_token_usage=usage, accumulated_cost=0.5)
    conversation = SimpleNamespace(
        conversation_stats=SimpleNamespace(get_combined_metrics=lambda: metrics)
    )
    dynamic._record_agent_usage(conversation, DEEPSEEK)
    assert seen == [((10, 20, 0.5, 90.0), {"model_id": DEEPSEEK})]


def test_cap_on_part_of_a_fan_out_is_raised_not_short(monkeypatch):
    """A cap that refuses one fan-out copy fails the prompt, not a short result."""
    from pyforger import BudgetExceededError, pipeline, step
    from pyforger.alter import generate_from_context

    monkeypatch.setenv("FORGER_STEP_REPORT", "0")

    def priced(**kwargs):
        raw = response()
        raw._hidden_params = {"response_cost": 1.0}
        return raw

    monkeypatch.setattr("pyforger.acquire.model.litellm.completion", priced)
    model = Model("azure_ai/gpt-5.6-terra", max_concurrency=1)  # fans out n

    @step
    def write(rows, model):
        return generate_from_context(
            rows, "Say {q}.", model, samples_per_item=3, max_retries=0
        )

    @pipeline(max_cost_usd=1.5)
    def main(rows, model):
        return write(rows, model)

    with pytest.raises(BudgetExceededError):
        main([{"q": "a"}], model)
    assert model.total_cost == 2.0


def test_dynamic_step_agents_use_the_token_unit_quota(monkeypatch, tmp_path):
    """Agent usage is charged to the shared quota, which then stops new agents."""
    from pyforger.alter import dynamic

    enable_budget(monkeypatch, tmp_path, limit=100)
    model = Model(DEEPSEEK)
    usage = dynamic._AgentUsage(DEEPSEEK, model._token_budget)
    usage.check()  # 0 of 100 units: fine
    counts = SimpleNamespace(prompt_tokens=10, completion_tokens=20)  # 90 units
    metrics = SimpleNamespace(accumulated_token_usage=counts, accumulated_cost=0.01)
    usage.record(
        SimpleNamespace(
            conversation_stats=SimpleNamespace(get_combined_metrics=lambda: metrics)
        )
    )
    assert model._token_budget.snapshot()["token_units"] == 90
    counts.prompt_tokens, counts.completion_tokens = 20, 40  # 180 units in all
    usage.record(
        SimpleNamespace(
            conversation_stats=SimpleNamespace(get_combined_metrics=lambda: metrics)
        )
    )
    with pytest.raises(TokenUnitQuotaExceeded):
        usage.check()
    with pytest.raises(ValueError):
        dynamic._AgentUsage("gpt-5.6-luna", model._token_budget).check()


def test_timed_out_attempts_stay_charged(monkeypatch, tmp_path, complete):
    """Each attempt that times out (and may still be billed) keeps its hold."""
    enable_budget(monkeypatch, tmp_path, limit=10_000)

    class Timeout(Exception):
        status_code = 408

    complete.side_effect = [Timeout("timed out"), response()]
    monkeypatch.setattr("pyforger.acquire.model._sleep", lambda delay: None)
    model = Model(DEEPSEEK, num_retries=1, max_tokens=100)
    model.generate("Hello")
    state = model._token_budget.snapshot()
    # The timed-out attempt: 2 prompt tokens (5 characters / 3) + 100 x 4.
    assert state["unreturned_calls"] == 1
    assert state["token_units"] == 402 + 90
    assert state["held_token_units"] == 0


def _cut_off(**kwargs):
    """A response whose single completion stopped at max_tokens."""
    cut = response()
    cut.choices[0].finish_reason = "length"
    return cut


def test_batch_stops_when_most_completions_are_cut_off(monkeypatch, complete):
    """After 20 finished completions, 80% of them truncated stops the batch."""
    from pyforger.acquire.model import TruncationStopError

    complete.side_effect = _cut_off
    model = Model(DEEPSEEK, max_concurrency=1, max_tokens=50)
    with pytest.raises(TruncationStopError, match="20 of 20 finished completions"):
        model.generate_batch([f"p{i}" for i in range(25)])
    assert complete.call_count == 20


def test_truncation_stop_can_be_turned_off(monkeypatch, complete):
    """FORGER_ALLOW_TRUNCATION=1 sends the whole batch."""
    monkeypatch.setenv("FORGER_ALLOW_TRUNCATION", "1")
    complete.side_effect = _cut_off
    results = Model(DEEPSEEK, max_concurrency=1).generate_batch(
        [f"p{i}" for i in range(25)]
    )
    assert complete.call_count == 25 and sum(r.truncated for r in results) == 25

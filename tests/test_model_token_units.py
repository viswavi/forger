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

DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
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


@pytest.mark.parametrize("model_id,expected", [(DEEPSEEK, 70), (GLM, 290)])
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
    assert state["token_units"] == model.total_token_units == 140
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
    assert state["token_units"] == model.total_token_units == 70
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
    monkeypatch.setattr(
        "pyforger.acquire.model.litellm.batch_completion",
        lambda **kwargs: [response(), error, response()],
    )
    model = Model(DEEPSEEK, show_progress=progress)
    with pytest.raises(RuntimeError, match="one call failed"):
        model.generate_batch(["good", "bad", "also good"])
    assert model.total_token_units == 140
    assert model.total_cost == 0.02
    assert len(model.history) == 2


def test_parse_failure_still_charges_response(complete):
    """Account for returned responses before parsing their choices."""
    raw = response()
    raw.choices = None
    complete.side_effect = None
    complete.return_value = raw
    model = Model(DEEPSEEK)
    with pytest.raises(TypeError):
        model.generate("Hello")
    assert model.total_token_units == 70
    assert model.total_cost == 0.01


def test_concurrent_calls_have_consistent_totals(complete):
    """Concurrent updates preserve every response's usage."""
    model = Model(GLM)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(model.generate, ["Hello"] * 100))
    assert model.total_token_units == 29_000
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
    assert restored.total_token_units == 70
    restored.generate("Again")
    assert restored.total_token_units == 140
    assert model.total_token_units == 70


def test_shared_budget_keeps_successful_calls_from_failed_batch(
    monkeypatch, tmp_path, complete
):
    """Persist partial batch usage for subsequent model instances."""
    enable_budget(monkeypatch, tmp_path, limit=140)

    def completion(messages, **kwargs):
        if messages[0]["content"] == "bad":
            raise RuntimeError("failed request")
        return response()

    complete.side_effect = completion
    model = Model(DEEPSEEK)
    with pytest.raises(RuntimeError, match="failed request"):
        model.generate_batch(["first", "bad", "last"])
    assert model.total_token_units == 140
    with pytest.raises(TokenUnitQuotaExceeded):
        Model(DEEPSEEK).generate("more")


def test_cached_op_does_not_charge_again(complete):
    """Reusing an op result neither calls the model nor adds usage."""

    @op
    def generate(model, prompt):
        return model.generate(prompt)

    model = Model(DEEPSEEK)
    original_key = model.cache_key()
    assert generate(model, "Hello").usage["token_units"] == 70
    assert generate(model, "Hello").usage["token_units"] == 70
    assert model.cache_key() == original_key
    assert model.total_token_units == 70
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
    assert second.total_token_units == 70


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
    enable_budget(monkeypatch, tmp_path, limit=70)
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
    enable_budget(monkeypatch, tmp_path, limit=70)
    monkeypatch.setattr(
        "pyforger.acquire.model._extract_cost", Mock(side_effect=ValueError("bad cost"))
    )
    model = Model(DEEPSEEK)
    with pytest.raises(ValueError, match="bad cost"):
        model.generate("Hello")
    assert model.total_token_units == 70
    with pytest.raises(TokenUnitQuotaExceeded):
        Model(DEEPSEEK).generate("Hello")
    complete.assert_called_once()


def test_shared_budget_checks_each_queued_batch_request(
    monkeypatch, tmp_path, complete
):
    """Stop queued requests once completed calls exhaust the budget."""
    enable_budget(monkeypatch, tmp_path, limit=70)
    real_executor = ThreadPoolExecutor
    monkeypatch.setattr(
        "pyforger.acquire.model.ThreadPoolExecutor",
        lambda **kwargs: real_executor(max_workers=1),
    )
    model = Model(DEEPSEEK)
    with pytest.raises(TokenUnitQuotaExceeded):
        model.generate_batch(["first", "second", "third"])
    assert model.total_token_units == 70
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

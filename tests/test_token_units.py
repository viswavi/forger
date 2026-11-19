"""Fixed weighting and shared quota behavior across calls and processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyforger.token_units import (
    DEFAULT_GENERATION_LIMIT,
    DEFAULT_JUDGE_MODEL,
    TokenUnitBudget,
    TokenUnitQuotaExceeded,
    default_generation_limit,
    default_rl_limit,
    held_units,
    model_weights,
    response_cost_usd,
    response_token_counts,
    token_units,
    validate_judge_model,
    validate_reasoning_effort,
)

DEEPSEEK = "litellm_proxy/azure_ai/DeepSeek-V4.1-Flash"


@pytest.mark.parametrize(
    "model,prompt,output,expected",
    [
        ("litellm_proxy/azure_ai/DeepSeek-V4.1-Flash", 100, 20, 180),
        ("custom/DEEPSEEK-v4.1-flash-prod", 1, 10, 41),
        ("litellm_proxy/azure_ai/FW-GLM-5.3", 100, 0, 583),
        ("custom/GLM-5.3", 0, 1, 18.33),
    ],
)
def test_fixed_weights(model, prompt, output, expected):
    """Provider prices and alias routing cannot change the fixed units."""
    assert token_units(model, prompt, output) == expected


def test_usage_counts_cached_input_and_reasoning_once():
    """Input cache details and output reasoning subsets do not change totals."""
    response = {
        "usage": {
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 90},
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 15},
        }
    }
    assert token_units("DeepSeek-V4.1-Flash", *response_token_counts(response)) == 180
    assert response_token_counts(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
    ) == (5, 2)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"usage": {}},
        {"usage": {"prompt_tokens": -1, "completion_tokens": 5}},
        {"usage": {"prompt_tokens": 1, "completion_tokens": True}},
    ],
)
def test_missing_or_invalid_counts_rejected(response):
    """An unmetered response cannot silently count as free."""
    with pytest.raises(ValueError):
        response_token_counts(response)


@pytest.mark.parametrize("cost", [0, 0.25])
@pytest.mark.parametrize("as_dict", [False, True])
def test_response_cost_uses_provider_cost(cost, as_dict):
    """Use provider pricing, including zero, before consulting the registry."""
    response = {"_hidden_params": {"response_cost": cost}}
    if not as_dict:
        response = SimpleNamespace(**response)
    with patch("litellm.completion_cost") as calculate:
        assert response_cost_usd(response, "custom/DeepSeek-V4.1-Flash") == cost
        calculate.assert_not_called()


@pytest.mark.parametrize("hidden", [None, [], {"response_cost": "unknown"}])
def test_response_cost_falls_back_to_litellm(hidden):
    """Recover pricing when provider cost metadata is missing or malformed."""
    response = SimpleNamespace(_hidden_params=hidden)
    with patch("litellm.completion_cost", return_value=0.5) as calculate:
        assert response_cost_usd(response, "custom/DeepSeek-V4.1-Flash") == 0.5
    calculate.assert_called_once_with(
        completion_response=response, model="custom/DeepSeek-V4.1-Flash"
    )


@pytest.mark.parametrize("cost", [None, -1, float("nan"), float("inf"), True, "1", {}])
def test_invalid_response_cost_is_unavailable(cost):
    """Reject negative, nonfinite, and nonnumeric pricing values."""
    response = {"_hidden_params": {"response_cost": cost}}
    with patch("litellm.completion_cost", return_value=cost):
        assert response_cost_usd(response, "DeepSeek-V4.1-Flash") is None


def test_response_pricing_errors_are_unavailable():
    """Pricing errors do not propagate into token accounting."""
    with patch("litellm.completion_cost", side_effect=RuntimeError("unpriced")):
        assert response_cost_usd({}, "DeepSeek-V4.1-Flash") is None


def test_unknown_models_rejected():
    """Unspecified models have no implicit unit weight."""
    with pytest.raises(ValueError):
        token_units("gpt-example", 10, 20)
    with pytest.raises(ValueError):
        token_units("DeepSeek-V4.1-Flash-GLM-5.3", 10, 20)
    # DeepSeek-V4-Flash is a judge fallback only, not a generation model.
    with pytest.raises(ValueError):
        token_units("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731", 10, 20)


def test_judge_policy():
    """Only the specified judge family and reasoning levels are accepted."""
    assert validate_judge_model("account/DeepSeek-V4.1-Flash-custom")
    for model in (
        "DeepSeek-V4-Pro",
        "FW-GLM-5.3",
        "DeepSeek-R1",
        "DeepSeek-V4-Flash-0731",
    ):
        with pytest.raises(ValueError):
            validate_judge_model(model)
    for effort in ("none", "low", "high"):
        assert validate_reasoning_effort(effort) == effort
    with pytest.raises(ValueError):
        validate_reasoning_effort("medium")


def test_judge_weights_are_separate_from_generation(tmp_path):
    """The judge quota keeps its old units; generation has its own weights."""
    judge = "litellm_proxy/azure_ai/DeepSeek-V4.1-Flash"
    assert DEFAULT_JUDGE_MODEL == judge
    assert model_weights(judge, judge=True) == (1, 3)
    assert token_units(judge, 10, 2, judge=True) == 16
    assert model_weights(judge) == (1, 4)
    assert token_units(judge, 10, 2) == 18
    # DeepSeek-V4-Flash: a judge fallback at 1/3, refused for generation; the
    # 0731 name does not collide with V4.1.
    old = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
    assert model_weights(old, judge=True) == (1, 3)
    with pytest.raises(ValueError):
        model_weights(old)
    glm = "litellm_proxy/azure_ai/FW-GLM-5.3"
    assert model_weights(glm, judge=True) == (4, 12.5)
    assert model_weights(glm) == (5.83, 18.33)
    ledger = TokenUnitBudget(tmp_path / "judge.json", 100)
    ledger.record(judge, 10, 2, judge=True)
    assert ledger.check()["token_units"] == 16


def test_record_and_hold_price_with_the_same_table(tmp_path):
    """Generation records V4.1-Flash at 1/4 and the judge at 1/3, as holds do."""
    model = "litellm_proxy/azure_ai/DeepSeek-V4.1-Flash"
    messages = [{"role": "user", "content": "x" * 30}]  # 10 held prompt tokens
    for judge, output_weight in ((False, 4), (True, 3)):
        ledger = TokenUnitBudget(tmp_path / f"judge-{judge}.json", 1000)
        held = held_units(model, messages, max_tokens=10, judge=judge)
        assert (
            held == token_units(model, 10, 10, judge=judge) == 10 + 10 * output_weight
        )
        with ledger.hold(model, held) as hold:
            assert ledger.snapshot()["held_token_units"] == held
        state = ledger.record(model, 10, 10, hold=hold, judge=judge)
        assert (state["token_units"], state["held_token_units"]) == (held, 0)
    # DeepSeek-V4-Flash is only a judge model: generation records refuse it.
    old = "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
    ledger = TokenUnitBudget(tmp_path / "old.json", 1000)
    with pytest.raises(ValueError):
        ledger.record(old, 10, 10)
    assert ledger.record(old, 10, 10, judge=True)["token_units"] == 40


def test_task_quotas():
    """The policy covers the launcher task names."""
    assert DEFAULT_GENERATION_LIMIT == 264_000_000
    for task in ("coding", "culture", " Coding "):
        assert default_rl_limit(task) == 50_000_000
    assert default_rl_limit("algorithmic") == 150_000_000
    with pytest.raises(ValueError):
        default_rl_limit("unknown")
    for task in ("coding", "culture"):
        assert default_generation_limit(task) == DEFAULT_GENERATION_LIMIT
    assert default_generation_limit("algorithmic") == 176_000_000
    assert default_generation_limit("clinical") == 293_000_000
    with pytest.raises(ValueError):
        default_generation_limit("unknown")


def test_quota_survives_restarts_and_records_inflight_calls(tmp_path):
    """Later processes share usage, including responses already in flight."""
    path = tmp_path / "usage.json"
    first = TokenUnitBudget(path, 700)
    assert first.check()["token_units"] == 0
    first.record("GLM-5.3", 100, 0, cost_usd=0.25)  # 100 x 5.83 = 583 units
    second = TokenUnitBudget(path, 700)
    assert second.check()["remaining_token_units"] == 117
    second.record("DeepSeek-V4.1-Flash", 10, 30, cost_usd=0.5)  # 130: 713 >= 700
    with pytest.raises(TokenUnitQuotaExceeded):
        first.check()
    state = first.record("DeepSeek-V4.1-Flash", 1, 10)  # 41
    assert state["token_units"] == 754
    assert state["calls"] == 3
    assert state["cost_usd"] == 0.75
    assert state["unpriced_calls"] == 1
    assert state["models"]["GLM-5.3"]["cost_usd"] == 0.25
    assert state["models"]["GLM-5.3"]["unpriced_calls"] == 0
    assert state["models"]["DeepSeek-V4.1-Flash"]["cost_usd"] == 0.5
    assert state["models"]["DeepSeek-V4.1-Flash"]["unpriced_calls"] == 1
    assert json.loads(path.read_text())["remaining_token_units"] == 0
    with pytest.raises(ValueError, match="differs"):
        TokenUnitBudget(path, 100).check()


def test_old_ledger_migration_preserves_usage(tmp_path):
    """Preserve prior units while marking historical pricing as unavailable."""
    path = tmp_path / "usage.json"
    legacy = {
        "version": 1,
        "limit": 100,
        "token_units": 12.0,
        "calls": 3,
        "unmetered_calls": 0,
        "models": {
            "DeepSeek-V4.1-Flash": {
                "prompt_tokens": 3,
                "completion_tokens": 3,
                "token_units": 12.0,
                "calls": 3,
            }
        },
    }
    path.write_text(json.dumps(legacy))
    budget = TokenUnitBudget(path, 100)
    migrated = budget.check()
    assert migrated["version"] == 1
    assert migrated["limit"] == 100
    assert migrated["token_units"] == 12
    assert migrated["cost_usd"] == 0
    assert migrated["unpriced_calls"] == 3
    assert migrated["models"]["DeepSeek-V4.1-Flash"]["unpriced_calls"] == 3
    assert json.loads(path.read_text()) == migrated
    state = budget.record("DeepSeek-V4.1-Flash", 1, 1, cost_usd=0.25)  # 5 units
    assert state["token_units"] == 17
    assert state["cost_usd"] == 0.25
    assert state["unpriced_calls"] == 3
    assert state["models"]["DeepSeek-V4.1-Flash"]["cost_usd"] == 0.25


@pytest.mark.parametrize("cost", [None, -1, float("nan"), float("inf"), True, "1", {}])
def test_invalid_cost_still_charges_token_usage(tmp_path, cost):
    """Record token usage when optional cost metadata is unusable."""
    budget = TokenUnitBudget(tmp_path / "usage.json", 100)
    state = budget.record("DeepSeek-V4.1-Flash", 1, 1, cost_usd=cost)
    assert state["token_units"] == 5
    assert state["cost_usd"] == 0
    assert state["unpriced_calls"] == 1
    assert state["models"]["DeepSeek-V4.1-Flash"]["unpriced_calls"] == 1
    assert budget.check()["remaining_token_units"] == 95


def test_malformed_ledger_cost_metadata_does_not_block_tokens(tmp_path):
    """Recover damaged dollar metadata without losing token usage."""
    path = tmp_path / "usage.json"
    budget = TokenUnitBudget(path, 100)
    state = budget.record("DeepSeek-V4.1-Flash", 2, 2, cost_usd=0.5)  # 10 units
    for entry in (state, state["models"]["DeepSeek-V4.1-Flash"]):
        entry["cost_usd"] = "invalid"
        entry["unpriced_calls"] = []
    path.write_text(json.dumps(state))
    result = budget.record("DeepSeek-V4.1-Flash", 1, 1, cost_usd=0.25)  # 5 units
    assert result["token_units"] == 15
    assert result["cost_usd"] == 0.25
    assert result["unpriced_calls"] == 1
    assert result["models"]["DeepSeek-V4.1-Flash"]["cost_usd"] == 0.25
    assert result["models"]["DeepSeek-V4.1-Flash"]["unpriced_calls"] == 1


def test_missing_usage_stops_future_calls(tmp_path):
    """A successful but unmetered call poisons the shared ledger, not just a worker."""
    path = tmp_path / "usage.json"
    TokenUnitBudget(path, 100).mark_unmetered()
    with pytest.raises(ValueError, match="incomplete"):
        TokenUnitBudget(path, 100).check()


def test_concurrent_processes_do_not_lose_usage(tmp_path):
    """Independent workers serialize updates to a shared ledger."""
    source = Path(__file__).resolve().parents[1] / "src/pyforger/token_units.py"
    program = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('units', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ledger = module.TokenUnitBudget(sys.argv[2], 100000)
for _ in range(10):
    ledger.record('GLM-5.3', 100, 100, cost_usd=0.125)
"""
    path = tmp_path / "usage.json"
    processes = [
        subprocess.Popen([sys.executable, "-c", program, str(source), str(path)])
        for _ in range(4)
    ]
    for process in processes:
        assert process.wait(timeout=30) == 0
    state = TokenUnitBudget(path, 100000).snapshot()
    assert state["calls"] == 40
    assert state["token_units"] == pytest.approx(40 * (583 + 1833))
    assert state["models"]["GLM-5.3"]["completion_tokens"] == 4000
    assert state["cost_usd"] == 5
    assert state["unpriced_calls"] == 0
    assert state["models"]["GLM-5.3"]["cost_usd"] == 5


def test_symlink_ledger_never_reads_or_overwrites_target(tmp_path):
    """Reject a symlink without modifying its target."""
    target = tmp_path / "target"
    target.write_text("untouched")
    link = tmp_path / "ledger"
    link.symlink_to(target)
    with pytest.raises(OSError):
        TokenUnitBudget(link, 100).record("DeepSeek-V4.1-Flash", 1, 1)
    assert target.read_text() == "untouched"


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_hold_counts_as_used_until_recorded(tmp_path):
    """An in-flight request's worst case blocks the quota; its response replaces it."""
    ledger = TokenUnitBudget(tmp_path / "usage.json", 1000)
    with ledger.hold(DEEPSEEK, 1000) as hold:
        assert ledger.snapshot()["remaining_token_units"] == 0
        with pytest.raises(TokenUnitQuotaExceeded):
            ledger.check()
    state = ledger.record(DEEPSEEK, 10, 20, hold=hold)
    assert (state["token_units"], state["held_token_units"], state["held"]) == (
        90,
        0,
        {},
    )


@pytest.mark.parametrize(
    "error,charged",
    [
        (TimeoutError("read timed out"), True),
        (_Status(500), True),
        (_Status(504), True),
        (_Status(502), False),
        (_Status(503), False),
        (_Status(429), False),
    ],
)
def test_failed_request_keeps_hold_when_it_may_be_billed(tmp_path, error, charged):
    """Timeouts, 500 and 504 stay charged; refusals and a down gateway are free."""
    ledger = TokenUnitBudget(tmp_path / "usage.json", 1000)
    with pytest.raises(type(error)):
        with ledger.hold(DEEPSEEK, 300):
            raise error
    state = ledger.snapshot()
    assert state["held_token_units"] == 0
    assert state["token_units"] == (300 if charged else 0)
    assert state["unreturned_calls"] == int(charged)


def test_killed_process_releases_its_hold(tmp_path):
    """A process killed mid-request frees its hold on the next read: not spent."""
    path = tmp_path / "usage.json"
    code = (
        "import os, sys; from pyforger.token_units import TokenUnitBudget\n"
        f"ledger = TokenUnitBudget({str(path)!r}, 1000)\n"
        f"with ledger.hold({DEEPSEEK!r}, 1000):\n"
        "    os._exit(1)\n"
    )
    import pyforger

    src = str(Path(pyforger.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": src}
    subprocess.run([sys.executable, "-c", code], check=False, env=env)
    state = TokenUnitBudget(path, 1000).check()
    assert (state["held_token_units"], state["token_units"]) == (0, 0)
    assert (state["released_calls"], state["released_token_units"]) == (1, 1000)
    assert state["unreturned_calls"] == 0


def test_live_process_keeps_its_hold(tmp_path):
    """Another live process's hold stays held: only dead owners are released."""
    path = tmp_path / "usage.json"
    ready = tmp_path / "ready"
    code = (
        "import time; from pathlib import Path\n"
        "from pyforger.token_units import TokenUnitBudget\n"
        f"ledger = TokenUnitBudget({str(path)!r}, 1000)\n"
        f"with ledger.hold({DEEPSEEK!r}, 400):\n"
        f"    Path({str(ready)!r}).touch()\n"
        "    time.sleep(30)\n"
    )
    import pyforger

    env = {**os.environ, "PYTHONPATH": str(Path(pyforger.__file__).parents[1])}
    child = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        for _ in range(600):
            if ready.exists():
                break
            time.sleep(0.05)
        state = TokenUnitBudget(path, 1000).snapshot()
        assert (state["held_token_units"], state["released_calls"]) == (400, 0)
    finally:
        child.kill()
        child.wait()
    assert TokenUnitBudget(path, 1000).snapshot()["released_calls"] == 1


def test_interrupt_releases_the_hold(tmp_path):
    """Ctrl-C inside a request is the process stopping, not a billed failure."""
    ledger = TokenUnitBudget(tmp_path / "usage.json", 1000)
    with pytest.raises(KeyboardInterrupt):
        with ledger.hold(DEEPSEEK, 300):
            raise KeyboardInterrupt
    state = ledger.snapshot()
    assert (state["token_units"], state["held_token_units"]) == (0, 0)
    assert (state["released_calls"], state["unreturned_calls"]) == (1, 0)


def test_held_units_assume_long_prompts_and_full_completions():
    """3 characters per prompt token and max_tokens per completion."""
    messages = [{"role": "user", "content": "x" * 101}]
    assert held_units(DEEPSEEK, messages, n=2, max_tokens=10) == 34 + 4 * 20
    assert held_units(DEEPSEEK, [], max_tokens=None) == 4 * 65_536
    assert held_units(DEEPSEEK, [], max_tokens=None, judge=True) == 3 * 65_536


def test_stale_holds_are_released(tmp_path, monkeypatch):
    """After two hours a hold can only belong to a dead process: it is released.

    This covers owners whose death cannot be seen, e.g. on another host.
    """
    ledger = TokenUnitBudget(tmp_path / "usage.json", 1000)
    with ledger.hold(DEEPSEEK, 400):
        assert ledger.snapshot()["held_token_units"] == 400
        now = time.time()
        monkeypatch.setattr("pyforger.token_units.time.time", lambda: now + 3 * 3600)
        state = ledger.snapshot()
    assert (state["held_token_units"], state["token_units"]) == (0, 0)
    assert (state["released_calls"], state["unreturned_calls"]) == (1, 0)

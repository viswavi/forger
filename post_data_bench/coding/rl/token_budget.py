"""Meter direct LiteLLM calls, including agents with Forger hidden."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# The launcher exports the standalone module beside this file in the sandbox.
_source = Path(__file__).with_name("token_units.py")
if not _source.is_file():
    _source = Path(__file__).resolve().parents[3] / "src/pyforger/token_units.py"
_spec = importlib.util.spec_from_file_location("pyforger.token_units", _source)
assert _spec is not None and _spec.loader is not None
_units = sys.modules.get("pyforger.token_units")
if _units is None:
    _units = importlib.util.module_from_spec(_spec)
    sys.modules["pyforger.token_units"] = _units
    _spec.loader.exec_module(_units)

DEFAULT_GENERATION_LIMIT = _units.DEFAULT_GENERATION_LIMIT
default_generation_limit = _units.default_generation_limit
DEFAULT_JUDGE_MODEL = _units.DEFAULT_JUDGE_MODEL
DEFAULT_REASONING_EFFORT = _units.DEFAULT_REASONING_EFFORT
TokenUnitBudget = _units.TokenUnitBudget
TokenUnitQuotaExceeded = _units.TokenUnitQuotaExceeded
default_rl_limit = _units.default_rl_limit
held_units = _units.held_units
model_weights = _units.model_weights
response_token_counts = _units.response_token_counts
response_cost_usd = _units.response_cost_usd
token_units = _units.token_units
validate_judge_model = _units.validate_judge_model
validate_reasoning_effort = _units.validate_reasoning_effort


def _prepare(budget, kwargs):
    if budget not in ("generation", "judge"):
        raise ValueError("budget must be generation or judge")
    judge = budget == "judge"
    prefix = "JUDGE" if judge else "FORGER"
    path = os.environ.get(prefix + "_TOKEN_USAGE_FILE")
    limit = os.environ.get(prefix + "_TOKEN_UNIT_LIMIT")
    if not path or not limit:
        raise ValueError(f"Set {prefix}_TOKEN_USAGE_FILE and {prefix}_TOKEN_UNIT_LIMIT")
    ledger = TokenUnitBudget(path, limit)
    if judge:
        model = kwargs.setdefault(
            "model", os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        )
        validate_judge_model(model)
        reasoning = kwargs.setdefault(
            "reasoning",
            {
                "effort": os.environ.get(
                    "JUDGE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
                )
            },
        )
        if not isinstance(reasoning, dict) or set(reasoning) != {"effort"}:
            raise ValueError("Judge reasoning must specify only effort")
        validate_reasoning_effort(reasoning["effort"])
        key = os.environ.get("LITELLM_JUDGE_API_KEY")
        if not key:
            raise ValueError("Set the separate LITELLM_JUDGE_API_KEY")
        kwargs["api_key"] = key
        kwargs.setdefault(
            "api_base",
            os.environ.get("LITELLM_JUDGE_BASE_URL")
            or os.environ.get("LITELLM_BASE_URL"),
        )
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("n", 1)
    else:
        kwargs.setdefault("api_key", os.environ.get("LITELLM_API_KEY"))
        kwargs.setdefault("api_base", os.environ.get("LITELLM_BASE_URL"))
    model_weights(kwargs.get("model"))
    if kwargs.get("stream"):
        raise ValueError("Token-budget helpers require non-streaming responses")
    if kwargs.get("num_retries"):
        # A retried attempt the gateway still bills would never be recorded.
        raise ValueError(
            "Token-budget helpers count one attempt per call; retry yourself"
        )
    kwargs.setdefault("max_tokens", 4096)
    ledger.check()
    return ledger


def _hold(ledger, kwargs):
    """Hold the call's worst case while it is in flight (`TokenUnitBudget.hold`)."""
    units = held_units(
        kwargs["model"],
        kwargs.get("messages") or [],
        kwargs.get("n") or 1,
        kwargs.get("max_tokens"),
    )
    return ledger.hold(kwargs["model"], units)


def _record(ledger, model, result, hold):
    try:
        prompt, completion = response_token_counts(result)
    except ValueError:
        ledger.mark_unmetered(hold)
        raise
    ledger.record(
        model,
        prompt,
        completion,
        cost_usd=response_cost_usd(result, model),
        hold=hold,
    )
    return result


def completion(*, budget="generation", **kwargs):
    """Call LiteLLM synchronously and persist usage before returning its response."""
    import litellm

    ledger = _prepare(budget, kwargs)
    with _hold(ledger, kwargs) as hold:
        result = litellm.completion(**kwargs)
    return _record(ledger, kwargs["model"], result, hold)


async def acompletion(*, budget="generation", **kwargs):
    """Call LiteLLM asynchronously against the same shared ledger."""
    import litellm

    ledger = _prepare(budget, kwargs)
    with _hold(ledger, kwargs) as hold:
        result = await litellm.acompletion(**kwargs)
    return _record(ledger, kwargs["model"], result, hold)

"""A batch of requests whose rest would not fit a cap stops early."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from litellm import ModelResponse

import pyforger
from pyforger import _tracking, alter
from pyforger.acquire import Model
from pyforger.steps import step


@pytest.fixture(autouse=True)
def _quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("FORGER_SPEND_LEDGER", "0")
    monkeypatch.delenv("FORGER_MAX_COST_USD", raising=False)
    monkeypatch.delenv("FORGER_PROJECT_SPEND", raising=False)
    _tracking.clear_step_records()


class _Fake:
    """litellm.completion that costs $0.01 per completion and counts requests."""

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, messages, n=1, **kwargs):
        self.requests += 1
        response = ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                    "index": i,
                }
                for i in range(n)
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10 * n,
                "total_tokens": 20,
            },
            model="gpt-4o-mini",
        )
        response._hidden_params["response_cost"] = 0.01 * n
        return response


def _spend_offset() -> float:
    """What this process already spent, so a cap means the same in every test."""
    return _tracking.usage_snapshot()[2]


def test_batch_that_would_pass_the_cap_stops_early(monkeypatch):
    """400 requests at $0.01 need $4; with $1 left it stops at 5% (20 requests)."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 1.0))
    fake = _Fake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        with pytest.raises(pyforger.SpendProjectionError) as info:
            model.generate_batch([f"p{i}" for i in range(400)])
    assert fake.requests == 20
    message = str(info.value)
    assert "the first 20 of 400 completions cost $0.20" in message
    assert "the other 380 would cost about $3.80" in message
    assert "FORGER_MAX_COST_USD" in message
    assert isinstance(info.value, pyforger.BudgetExceededError)


def test_batch_that_fits_runs_to_the_end(monkeypatch):
    """A batch that fits its cap is never stopped."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 10.0))
    fake = _Fake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        results = model.generate_batch([f"p{i}" for i in range(400)])
    assert len(results) == 400 and fake.requests == 400


class _PricedFake(_Fake):
    """Costs $0.10 per completion for prompts starting "hard", else $0.001."""

    def __call__(self, messages, n=1, **kwargs):
        response = super().__call__(messages, n=n, **kwargs)
        hard = messages[-1]["content"].startswith("hard")
        response._hidden_params["response_cost"] = (0.10 if hard else 0.001) * n
        return response


def test_expensive_rows_first_do_not_stop_a_batch_that_fits(monkeypatch):
    """40 hard prompts first, then 360 easy: $4.36 in all, under a $6 cap.

    At the average of the first 20 sent in input order ($0.10) the rest would
    look like $38; the shuffled send order and the lower bound keep it running.
    """
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 6.0))
    prompts = [f"hard {i}" for i in range(40)] + [f"easy {i}" for i in range(360)]
    fake = _PricedFake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        results = model.generate_batch(prompts)
    assert len(results) == 400 and fake.requests == 400
    assert [r[0] for r in results[:2]] == ["ok", "ok"]  # in input order


def test_mixed_costs_that_clearly_overrun_still_stop(monkeypatch):
    """Half hard, half easy: ~$20 needed with $3 left stops well before the end."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 3.0))
    prompts = [f"{'hard' if i % 2 else 'easy'} {i}" for i in range(400)]
    fake = _PricedFake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        with pytest.raises(pyforger.SpendProjectionError, match="at least"):
            model.generate_batch(prompts)
    assert fake.requests < 40


def test_opt_out_and_small_batches_use_only_the_hard_cap(monkeypatch):
    """FORGER_PROJECT_SPEND=0 and batches under 40 completions stop only at the cap."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 0.25))
    monkeypatch.setenv("FORGER_PROJECT_SPEND", "0")
    fake = _Fake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        with pytest.raises(pyforger.BudgetExceededError) as info:
            model.generate_batch([f"p{i}" for i in range(400)])
    assert not isinstance(info.value, pyforger.SpendProjectionError)
    assert fake.requests in (25, 26)  # the hard cap, at $0.25 (float sums)

    monkeypatch.delenv("FORGER_PROJECT_SPEND")
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 0.05))
    fake = _Fake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        with pytest.raises(pyforger.BudgetExceededError) as info:
            model.generate_batch(
                [f"p{i}" for i in range(30)]
            )  # under 40: no projection
    assert not isinstance(info.value, pyforger.SpendProjectionError)


def test_message_names_the_op_and_step_and_rerun_reuses_finished_requests(monkeypatch):
    """The error names the op and step; a rerun sends only the unsent requests."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 1.0))
    rows = [{"q": f"question {i}"} for i in range(400)]

    @step
    def answer(rows: list[dict], model) -> list[dict]:
        return alter.generate_from_context(rows, "Answer: {q}", model)

    fake = _Fake()
    with patch("litellm.completion", fake):
        model = Model("gpt-4o-mini", api_key="x", max_concurrency=1)
        with pytest.raises(pyforger.SpendProjectionError, match="in step answer"):
            answer(rows, model)
        sent = fake.requests
        # More room: the rerun sends only the requests the first run did not.
        monkeypatch.setenv("FORGER_MAX_COST_USD", str(_spend_offset() + 10.0))
        answer(rows, model)
    assert fake.requests == 400
    assert sent < 40

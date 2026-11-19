"""Tests for ``my_step.submit(...)``: steps that run at the same time in a pipeline."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pyforger
from pyforger import _tracking, filter, op, steps
from pyforger.acquire import Model
from pyforger.steps import PipelineContractError, StepContractError, pipeline, step

_WAIT = 10  # seconds; only reached when the steps do not overlap


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports and forget earlier step records."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_MAX_COST_USD", raising=False)
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


class _Event(threading.Event):
    def cache_key(self) -> str:
        return "event"


class _Barrier(threading.Barrier):
    def cache_key(self) -> str:
        return "barrier"


def _rows(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


def _keep_all(rows):
    return filter.by_length(rows, instruction_field="text", min_len=0)


def _response(cost: float) -> SimpleNamespace:
    message = SimpleNamespace(content="hi", reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, completion_tokens_details=None
        ),
        _hidden_params={"response_cost": cost},
    )


def test_submitted_step_overlaps_a_direct_step():
    """Each step waits for the other to start, so they only finish if they overlap."""
    a_started, b_started = _Event(), _Event()
    met = []

    @op(name="alter.test_meet_a", category="alter")
    def meet_a(rows):
        a_started.set()
        met.append(b_started.wait(_WAIT))
        return list(rows)

    @op(name="alter.test_meet_b", category="alter")
    def meet_b(rows):
        b_started.set()
        met.append(a_started.wait(_WAIT))
        return list(rows)

    @step
    def step_a(rows):
        return _keep_all(meet_a(rows))

    @step
    def step_b(rows):
        return _keep_all(meet_b(rows))

    @pipeline
    def main(rows):
        a = step_a.submit(rows[:1])
        b = step_b(rows[1:])
        return a.result() + b

    out = main(_rows("x", "y"))
    assert met == [True, True]
    assert [r["text"] for r in out] == ["x", "y"]


def test_submit_returns_a_future_with_the_rows_or_the_error():
    """``.result()`` returns the step's rows or re-raises its exception."""

    @op(name="alter.test_explode", category="alter")
    def explode(rows):
        raise RuntimeError("bad batch")

    @step
    def fragile(rows):
        return _keep_all(explode(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        good = keep.submit(rows)
        bad = fragile.submit(rows)
        assert isinstance(good, Future)
        with pytest.raises(RuntimeError, match="bad batch"):
            bad.result()
        return good.result()

    assert main(_rows("x")) == _rows("x")


def test_summary_lists_steps_in_submission_order():
    """A step submitted first and finished last is still listed first."""
    b_done = _Event()

    @op(name="alter.test_after_b", category="alter")
    def after_b(rows):
        assert b_done.wait(_WAIT)
        return list(rows)

    @step
    def slow(rows):
        return _keep_all(after_b(rows))

    @step
    def fast(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        first = slow.submit(rows[:1])
        second = fast(rows[1:])
        b_done.set()  # fast has joined the pipeline; let slow finish
        return first.result() + second

    main(_rows("x", "y"))
    rec = _tracking.last_pipeline()
    assert rec is not None
    assert [s.name for s in rec.steps] == ["slow", "fast"]


def test_step_reports_print_whole(capsys, monkeypatch):
    """Reports from two threads come out one after the other, not interleaved."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "1")
    both = _Barrier(2)

    @op(name="alter.test_together", category="alter")
    def together(rows):
        both.wait(_WAIT)
        return list(rows)

    @step
    def left(rows):
        return _keep_all(together(rows))

    @step
    def right(rows):
        return _keep_all(together(rows))

    @pipeline
    def main(rows):
        a, b = left.submit(rows[:1]), right.submit(rows[1:])
        return a.result() + b.result()

    main(_rows("x", "y"))
    out = capsys.readouterr().out
    blocks = [b for b in out.split("\n\n") if b.strip()]
    for name in ("left", "right"):
        (block,) = [b for b in blocks if f"step {name}" in b]
        other = "right" if name == "left" else "left"
        assert f"step {other}" not in block
    summary = out[out.index("pipeline main") :]
    assert summary.index("left") < summary.index("right")


def test_submit_inside_a_step_raises_like_a_nested_call():
    """``.submit`` inside a running step raises at once, as a direct call would."""

    @step
    def inner(rows):
        return _keep_all(rows)

    @step
    def outer(rows):
        inner.submit(rows)
        return _keep_all(rows)

    with pytest.raises(StepContractError, match="inner was called inside step outer"):
        outer(_rows("x"))


def test_pipeline_waits_for_a_step_it_never_awaited():
    """A submitted step still running when the pipeline returns is waited for."""
    release = _Event()
    ran = []

    @op(name="alter.test_late", category="alter")
    def late(rows):
        assert release.wait(_WAIT)
        ran.append(True)
        return list(rows)

    @step
    def background(rows):
        return _keep_all(late(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        background.submit(rows)
        out = keep(rows)
        release.set()
        return out

    main(_rows("x"))
    assert ran == [True]
    rec = _tracking.last_pipeline()
    assert rec is not None and [s.name for s in rec.steps] == ["background", "keep"]


def test_pipeline_raises_the_error_of_a_step_it_never_awaited():
    """An un-awaited step's exception is raised from the pipeline, not lost."""

    @op(name="alter.test_fail_late", category="alter")
    def fail_late(rows):
        raise ValueError("judge timed out")

    @step
    def background(rows):
        return _keep_all(fail_late(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        background.submit(rows)
        return keep(rows)

    with pytest.raises(ValueError, match="judge timed out") as err:
        main(_rows("x"))
    assert "raised by step background" in "\n".join(err.value.__notes__)


def test_pipeline_error_cancels_queued_steps_and_does_not_wait(monkeypatch):
    """When the pipeline raises, queued steps never start and running ones are left."""
    monkeypatch.setattr(steps, "_POOL", steps._StepPool(1))
    release = _Event()
    started = []

    @op(name="alter.test_block", category="alter")
    def block(rows):
        started.append(rows[0]["text"])
        release.wait(_WAIT)
        return list(rows)

    @step
    def blocked(rows):
        return _keep_all(block(rows))

    futures = []

    @pipeline
    def main(rows):
        futures.append(blocked.submit(rows[:1]))
        futures.append(blocked.submit(rows[1:]))  # queued behind the first
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        main(_rows("x", "y"))
    running, queued = futures
    assert queued.cancelled() and not running.done()
    release.set()
    running.result(_WAIT)
    assert started == ["x"]


def test_concurrent_steps_each_count_only_their_own_usage():
    """Two steps spending at once keep their own totals; the pipeline sums them."""
    both = _Barrier(2)

    @op(name="alter.test_spend_big", category="alter")
    def spend_big(rows):
        both.wait(_WAIT)
        _tracking.add_usage(100, 10, 0.5)
        both.wait(_WAIT)  # the other step has spent too
        return list(rows)

    @op(name="alter.test_spend_small", category="alter")
    def spend_small(rows):
        both.wait(_WAIT)
        _tracking.add_usage(7, 1, 0.02)
        both.wait(_WAIT)
        return list(rows)

    @step
    def big(rows):
        return _keep_all(spend_big(rows))

    @step
    def small(rows):
        return _keep_all(spend_small(rows))

    @pipeline
    def main(rows):
        a = big.submit(rows[:1])
        b = small(rows[1:])
        return a.result() + b

    main(_rows("x", "y"))
    records = _tracking.step_records()
    assert (records["big"].prompt_tokens, records["big"].cost_usd) == (100, 0.5)
    assert (records["small"].prompt_tokens, records["small"].cost_usd) == (7, 0.02)
    rec = _tracking.last_pipeline()
    assert rec is not None and rec.prompt_tokens == 107
    assert rec.cost_usd == pytest.approx(0.52)
    assert rec.unattributed.is_zero()


def test_pipeline_cap_stops_requests_in_both_threads():
    """$0.40 per call and a $1 cap: both steps raise, and at most 4 calls are sent."""
    model = Model("openai/test-model")

    @op(name="alter.test_ask_five", category="alter")
    def ask_five(rows, tag):
        for _ in range(5):
            model.generate(f"hello {tag}")
        return list(rows)

    @step
    def ask(rows, tag):
        return _keep_all(ask_five(rows, tag))

    @step
    def keep(rows):
        return _keep_all(rows)

    stopped = []

    @pipeline(max_cost_usd=1.0)
    def main(rows):
        background = ask.submit(rows, "a")
        for wait in (lambda: ask(rows, "b"), background.result):
            try:
                wait()
            except pyforger.BudgetExceededError:
                stopped.append(True)
        return keep(rows)

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = _response(cost=0.4)
        main(_rows("x"))
        assert stopped == [True, True]
        # 3 calls reach the cap; a call each thread checked before then may finish.
        assert 3 <= mock_litellm.completion.call_count <= 4


def test_submit_without_a_pipeline_runs_the_step():
    """Outside a pipeline, ``.submit`` just runs the step in the background."""

    @step
    def keep(rows):
        return _keep_all(rows)

    assert keep.submit(_rows("x")).result(_WAIT) == _rows("x")
    assert "keep" in _tracking.step_records()


def test_strict_pipeline_still_checks_submitted_steps():
    """Submitted steps join the pipeline's checks like direct ones."""

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline(strict=True)
    def main(rows):
        return keep.submit(rows).result()

    with pytest.raises(PipelineContractError, match="only one step ran"):
        main(_rows("x"))

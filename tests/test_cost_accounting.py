"""Tests for usage accumulators, cached-step cost, the spend ledger, and caps."""

from __future__ import annotations

import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pyforger
from pyforger import _spend, _tracking, filter, op
from pyforger.acquire import Model
from pyforger.inspect import spend
from pyforger.steps import pipeline, render_pipeline, render_step, step


@pytest.fixture(autouse=True)
def _quiet_and_fresh(monkeypatch):
    """Silence reports, forget step records, and use the default ledger path."""
    monkeypatch.setenv("FORGER_STEP_REPORT", "0")
    monkeypatch.delenv("FORGER_SPEND_LEDGER", raising=False)
    monkeypatch.delenv("FORGER_MAX_COST_USD", raising=False)
    _tracking.clear_step_records()
    yield
    _tracking.clear_step_records()


def _rows(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


def _keep_all(rows):
    return filter.by_length(rows, instruction_field="text", min_len=0)


def _ledger(tmp_path: Path) -> list[dict]:
    path = tmp_path / "spend.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _response(text: str = "hi", *, cost: float = 0.4) -> SimpleNamespace:
    message = SimpleNamespace(content=text, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, completion_tokens_details=None
        ),
        _hidden_params={"response_cost": cost},
    )


@op(name="alter.test_ask", category="alter")
def _ask(rows: list[dict], model, times: int = 1) -> list[dict]:
    """Make ``times`` single-prompt model calls, one batch each."""
    for _ in range(times):
        model.generate("hello")
    return list(rows)


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #

_N_WORKERS = 5


@op(name="alter.test_chunk", category="alter")
def _chunk(rows: list[dict], i: int, barrier: threading.Barrier) -> list[dict]:
    """Add usage i+1 times 100 tokens, then wait until every chunk has added."""
    _tracking.add_usage(100 * (i + 1), i + 1, 0.01 * (i + 1))
    barrier.wait(timeout=10)
    return list(rows)


class _Barrier(threading.Barrier):
    def cache_key(self) -> str:
        return "barrier"


def test_concurrent_ops_each_count_only_their_own_usage():
    """Ops run at the same time under copy_context do not see each other's usage.

    With one process-wide counter diffed at start and end, each chunk's window
    held the others' spend, so totals came out cumulative.
    """

    @step
    def chunks(rows):
        barrier = _Barrier(_N_WORKERS)
        with ThreadPoolExecutor(_N_WORKERS) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, _chunk, rows, i, barrier)
                for i in range(_N_WORKERS)
            ]
            for future in futures:
                future.result()
        return _keep_all(rows)

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        return keep(chunks(rows))

    main(_rows("a b"))
    rec = _tracking.step_records()["chunks"]
    calls = sorted(
        (c for c in rec.ops if c.op_name == "alter.test_chunk"),
        key=lambda c: c.prompt_tokens,
    )
    assert [c.prompt_tokens for c in calls] == [100, 200, 300, 400, 500]
    assert [c.completion_tokens for c in calls] == [1, 2, 3, 4, 5]
    assert [c.cost_usd for c in calls] == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert rec.prompt_tokens == 1500
    assert rec.cost_usd == pytest.approx(0.15)
    pipe = _tracking.last_pipeline()
    assert pipe.prompt_tokens == 1500
    assert pipe.unattributed.is_zero()


def test_usage_from_a_thread_without_context_is_unattributed():
    """A thread that did not copy the context reaches no step or pipeline total.

    With one pipeline running, the pipeline report lists it on its own line.
    """

    @op(name="alter.test_detached", category="alter")
    def detached(rows):
        thread = threading.Thread(target=_tracking.add_usage, args=(70, 30, 0.5))
        thread.start()
        thread.join()
        _tracking.add_usage(1, 1, 0.25)  # in the op's own context
        return list(rows)

    @step
    def first(rows):
        return _keep_all(detached(rows))

    @step
    def second(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        return second(first(rows))

    main(_rows("a b"))
    rec = _tracking.step_records()["first"]
    assert (rec.prompt_tokens, rec.cost_usd) == (1, pytest.approx(0.25))
    pipe = _tracking.last_pipeline()
    assert (pipe.unattributed.prompt_tokens, pipe.unattributed.cost_usd) == (
        70,
        pytest.approx(0.5),
    )
    assert pipe.prompt_tokens == 1
    assert pipe.cost_usd == pytest.approx(0.25)
    text = render_pipeline(pipe)
    assert "unattributed: 70 in / 30 out tokens · $0.5, from threads" in text


def test_no_unattributed_line_when_every_call_is_attributed():
    """Usage inside a step but outside its ops is the step's, not unattributed."""

    @step
    def costly(rows):
        _tracking.add_usage(5, 5, 0.1)  # not in an op, but in the step
        return _keep_all(rows)

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        return keep(costly(rows))

    main(_rows("a b"))
    pipe = _tracking.last_pipeline()
    assert pipe.cost_usd == pytest.approx(0.1)
    assert "unattributed" not in render_pipeline(pipe)


def test_step_in_a_worker_thread_counts_toward_its_pipeline():
    """A step called from a thread without context still adds to the pipeline."""

    @op(name="alter.test_spend", category="alter")
    def spend_op(rows):
        _tracking.add_usage(10, 10, 0.2)
        return list(rows)

    @step
    def work(rows):
        return _keep_all(spend_op(rows))

    @pipeline
    def main(rows):
        with ThreadPoolExecutor(2) as pool:
            parts = list(pool.map(work, [rows[:1], rows[1:]]))
        return parts[0] + parts[1]

    main(_rows("a b", "c d"))
    pipe = _tracking.last_pipeline()
    assert pipe.cost_usd == pytest.approx(0.4)
    assert pipe.unattributed.is_zero()


# --------------------------------------------------------------------------- #
# Cached steps
# --------------------------------------------------------------------------- #


@op(name="alter.test_costly", category="alter")
def _costly(rows: list[dict]) -> list[dict]:
    _tracking.add_usage(100, 50, 0.25)
    return list(rows)


def test_cached_step_keeps_the_cost_of_the_run_that_cached_it():
    """A cache hit spends nothing and reports the earlier run's tokens and cost."""

    @step
    def costly(rows):
        return _keep_all(_costly(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        return keep(costly(rows))

    main(_rows("a b"))
    first = _tracking.step_records()["costly"]
    assert not first.cache_hit and first.original is None

    main(_rows("a b"))
    rec = _tracking.step_records()["costly"]
    assert rec.cache_hit
    assert (rec.prompt_tokens, rec.cost_usd) == (0, 0.0)
    assert rec.original is not None
    assert (rec.original.prompt_tokens, rec.original.completion_tokens) == (100, 50)
    assert rec.original.cost_usd == pytest.approx(0.25)
    assert rec.original.duration_s > 0
    # The step report is one line; inspect_step's full report has the details.
    assert "cached · originally" in render_step(rec)
    assert "$0.25" in render_step(rec)
    assert "cached (originally" in render_step(rec, full=True)
    assert "100 in / 50 out tokens · $0.25)" in render_step(rec, full=True)

    pipe = _tracking.last_pipeline()
    assert pipe.cost_usd == 0.0
    assert pipe.cached_cost_usd == pytest.approx(0.25)
    text = render_pipeline(pipe)
    assert "nothing spent this run" in text
    assert "cached (originally $0.25)" in text
    assert "($0.25)" in text  # the cached step's cost cell


def test_step_metadata_without_usage_still_loads():
    """Steps cached before usage was stored show no earlier cost."""
    rec = _tracking.StepRecord(name="old")
    token = _tracking.enter_step(rec, lambda result: None)
    try:
        _tracking.restore_step_meta(json.dumps({"version": 1, "ops": []}))
    finally:
        _tracking.exit_step(token)
    assert rec.original is None and rec.ops == []
    rec.cache_hit = True
    assert "originally" not in render_step(rec)


# --------------------------------------------------------------------------- #
# Spend ledger
# --------------------------------------------------------------------------- #


def test_ledger_line_is_written_per_call_and_survives_an_exception(tmp_path):
    """Each result is in the ledger before the next request; a crash keeps them."""
    model = Model("openai/test-model")
    lines_seen: list[int] = []

    def completion(**kwargs):
        lines_seen.append(len(_ledger(tmp_path)))
        return _response(cost=0.1)

    @step
    def ask(rows):
        return _keep_all(_ask(rows, model, times=3))

    @step
    def crash(rows):
        _keep_all(rows)
        raise RuntimeError("killed")

    @pipeline
    def main(rows):
        return crash(ask(rows))

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = completion
        with pytest.raises(RuntimeError, match="killed"):
            main(_rows("a b"))
    assert lines_seen == [0, 1, 2]
    records = _ledger(tmp_path)
    assert len(records) == 3
    first = records[0]
    assert set(first) == {
        "timestamp",
        "pid",
        "run_id",
        "pipeline",
        "step",
        "op",
        "model_id",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "token_units",
    }
    assert first["token_units"] is None  # openai/test-model has no fixed weights
    assert (first["pipeline"], first["step"], first["op"]) == (
        "main",
        "ask",
        "alter.test_ask",
    )
    assert first["model_id"] == "openai/test-model"
    assert (first["prompt_tokens"], first["completion_tokens"]) == (10, 5)
    assert first["cost_usd"] == pytest.approx(0.1)
    assert first["run_id"] == _spend.run_id()
    assert len({r["run_id"] for r in records}) == 1


def test_ledger_path_override_and_off_switch(tmp_path, monkeypatch):
    """FORGER_SPEND_LEDGER moves the ledger; FORGER_SPEND_LEDGER=0 turns it off."""
    custom = tmp_path / "ledgers" / "mine.jsonl"
    monkeypatch.setenv("FORGER_SPEND_LEDGER", str(custom))
    _tracking.add_usage(1, 2, 0.3, model_id="m")
    assert len(custom.read_text().splitlines()) == 1
    assert not (tmp_path / "spend.jsonl").exists()

    monkeypatch.setenv("FORGER_SPEND_LEDGER", "0")
    _tracking.add_usage(1, 2, 0.3, model_id="m")
    assert len(custom.read_text().splitlines()) == 1
    assert not (tmp_path / "spend.jsonl").exists()


def test_unwritable_ledger_does_not_fail_the_call(tmp_path, monkeypatch):
    """A ledger that cannot be written is logged, never raised."""
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("FORGER_SPEND_LEDGER", str(blocker / "spend.jsonl"))
    _tracking.add_usage(1, 2, 0.3)  # parent is a file: logged, not raised


def _line(run: str, pipe, step_name, cost: float, when: str) -> str:
    return json.dumps(
        {
            "timestamp": when,
            "pid": 1,
            "run_id": run,
            "pipeline": pipe,
            "step": step_name,
            "op": "alter.x",
            "model_id": "m",
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "cost_usd": cost,
        }
    )


def test_spend_totals_by_run_pipeline_and_step(tmp_path, capsys):
    """spend() groups by run, pipeline, and step; skips a cut-off last line."""
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n".join(
            [
                _line("r1", "main", "gen", 0.5, "2026-10-15T10:00:00+00:00"),
                _line("r1", "main", "gen", 0.25, "2026-10-15T10:01:00+00:00"),
                _line("r1", "main", "judge", 1.0, "2026-10-15T10:02:00+00:00"),
                _line("r2", "main", "gen", 2.0, "2026-10-16T09:00:00+00:00"),
                _line("r2", None, None, 0.125, "2026-10-16T09:05:00+00:00"),
                '{"timestamp": "2026-10-16T09:06:00+00:00", "run_id": "r2", "cos',
            ]
        )
        + "\n"
    )
    report = spend(path)
    assert [(t.run_id, t.pipeline, t.step) for t in report.rows] == [
        ("r1", "main", "gen"),
        ("r1", "main", "judge"),
        ("r2", "main", "gen"),
        ("r2", None, None),
    ]
    gen = report.rows[0]
    assert (gen.calls, gen.prompt_tokens, gen.completion_tokens) == (2, 20, 2)
    assert gen.cost_usd == pytest.approx(0.75)
    assert gen.last == "2026-10-15T10:01:00+00:00"
    assert report.calls == 5
    assert report.cost_usd == pytest.approx(3.875)
    out = capsys.readouterr().out
    assert "spend ledger" in out and "5 LLM calls" in out and "$3.88" in out
    assert "main · judge" in out and "(none) · (none)" in out

    later = spend(path, since="2026-10-16T00:00:00+00:00")
    assert [t.run_id for t in later.rows] == ["r2", "r2"]
    assert later.cost_usd == pytest.approx(2.125)


def test_spend_reads_the_default_ledger_and_handles_a_missing_file(tmp_path):
    """Without a path spend() reads $FORGER_CACHE/spend.jsonl, empty if absent."""
    assert spend().calls == 0  # nothing written yet
    _tracking.add_usage(3, 4, 0.5, model_id="m")
    report = spend()
    assert report.path == str(tmp_path / "spend.jsonl")
    assert (report.calls, report.prompt_tokens) == (1, 3)


def test_spend_needs_a_path_when_the_ledger_is_off(monkeypatch):
    """With the ledger off there is no default file to read."""
    monkeypatch.setenv("FORGER_SPEND_LEDGER", "0")
    with pytest.raises(ValueError, match="ledger is off"):
        spend()


# --------------------------------------------------------------------------- #
# Spending caps
# --------------------------------------------------------------------------- #


def test_pipeline_cap_raises_before_the_next_request(tmp_path):
    """$0.40 per call and a $1 cap: the 4th request is never sent."""
    model = Model("openai/test-model")

    @step
    def ask(rows):
        return _keep_all(_ask(rows, model, times=5))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline(max_cost_usd=1.0)
    def main(rows):
        return keep(ask(rows))

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = _response(cost=0.4)
        with pytest.raises(pyforger.BudgetExceededError, match="max_cost_usd=\\$1"):
            main(_rows("a b"))
        assert mock_litellm.completion.call_count == 3
    assert len(_ledger(tmp_path)) == 3  # the calls before the cap are recorded


def test_unattributed_usage_does_not_trip_the_pipeline_cap(monkeypatch):
    """Spend from a thread outside the pipeline's context is not charged to it.

    Repro: pipeline A had max_cost_usd=2 and raised before any call of its own
    because an unrelated thread spent $5. That spend still counts toward
    FORGER_MAX_COST_USD.
    """
    model = Model("openai/test-model")

    @op(name="alter.test_detached_then_ask", category="alter")
    def detached_then_ask(rows):
        thread = threading.Thread(target=_tracking.add_usage, args=(1, 1, 5.0))
        thread.start()
        thread.join()
        model.generate("hello")
        return list(rows)

    @step
    def ask(rows):
        return _keep_all(detached_then_ask(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline(max_cost_usd=2.0)
    def main(rows):
        return keep(ask(rows))

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = _response(cost=0.4)
        main(_rows("a b"))
        assert mock_litellm.completion.call_count == 1
        pipe = _tracking.last_pipeline()
        assert pipe.cost_usd == pytest.approx(0.4)
        assert pipe.unattributed.cost_usd == pytest.approx(5.0)

        monkeypatch.setenv(
            "FORGER_MAX_COST_USD", str(_tracking.usage_snapshot()[2] - 1.0)
        )
        with pytest.raises(pyforger.BudgetExceededError, match="FORGER_MAX"):
            model.generate("hello")


def test_concurrent_pipelines_do_not_share_unattributed_usage():
    """Repro: two pipelines spending from non-copied workers each showed both.

    With two pipelines running, a thread without context belongs to neither.
    """
    started = _Barrier(2)

    def spend_detached(cost):
        thread = threading.Thread(target=_tracking.add_usage, args=(1, 1, cost))
        thread.start()
        thread.join()

    def make(cost):
        @op(name=f"alter.test_detached_{int(cost)}")
        def detached(rows):
            started.wait(timeout=10)  # both pipelines are running now
            _tracking.add_usage(1, 1, cost)
            spend_detached(cost)
            started.wait(timeout=10)  # neither has finished yet
            return list(rows)

        @step(name=f"spend_{int(cost)}")
        def spend_step(rows):
            return _keep_all(detached(rows))

        @step(name=f"keep_{int(cost)}")
        def keep(rows):
            return _keep_all(rows)

        @pipeline(name=f"p{int(cost)}")
        def main(rows):
            records[cost] = _tracking.active_pipeline()
            return keep(spend_step(rows))

        return main

    records = {}
    threads = [
        threading.Thread(target=make(c), args=(_rows("a b"),)) for c in (1.0, 5.0)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for cost in (1.0, 5.0):
        assert records[cost].cost_usd == pytest.approx(cost)
        assert records[cost].unattributed.is_zero()


def test_process_cap_from_the_environment(monkeypatch):
    """FORGER_MAX_COST_USD caps all spend in the process, in or out of pipelines."""
    model = Model("openai/test-model")
    already = _tracking.usage_snapshot()[2]  # earlier tests spent in this process
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(already + 1.0))
    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = _response(cost=0.6)
        model.generate("one")
        model.generate("two")
        with pytest.raises(pyforger.BudgetExceededError, match="FORGER_MAX_COST_USD"):
            model.generate("three")
        assert mock_litellm.completion.call_count == 2


@pytest.mark.parametrize("bad", ["ten dollars", "nan", "-1"])
def test_bad_env_caps_are_rejected(monkeypatch, bad):
    """Repro: FORGER_MAX_COST_USD=nan silently turned the cap off."""
    monkeypatch.setenv("FORGER_MAX_COST_USD", bad)
    with pytest.raises(ValueError, match="FORGER_MAX_COST_USD must be a number"):
        _tracking.check_budget()


@pytest.mark.parametrize("bad", [-1, float("nan")])
def test_bad_pipeline_caps_are_rejected(bad):
    """A negative or NaN max_cost_usd raises when the pipeline is decorated."""
    with pytest.raises(ValueError, match="max_cost_usd must be a number"):
        pipeline(max_cost_usd=bad)


# --------------------------------------------------------------------------- #
# Review regressions
# --------------------------------------------------------------------------- #


def test_pipeline_cap_holds_for_a_step_in_a_worker_thread():
    """Repro: a step in a plain thread pool made 5 $1 calls past a $2 cap.

    The step joins the running pipeline, so its spend counts toward the cap as
    it arrives, not only when the step returns.
    """
    model = Model("openai/test-model")

    @step
    def ask(rows):
        return _keep_all(_ask(rows, model, times=5))

    @pipeline(max_cost_usd=2.0)
    def main(rows):
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(ask, rows).result()

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.return_value = _response(cost=1.0)
        with pytest.raises(pyforger.BudgetExceededError, match="pipeline main"):
            main(_rows("a b"))
        assert mock_litellm.completion.call_count == 2


def test_worker_step_that_raises_is_still_charged_to_its_pipeline(tmp_path):
    """Repro: a worker-thread step spent $3 and raised; the pipeline showed $0.

    Its ledger lines also name the pipeline (they had pipeline=None).
    """

    @op(name="alter.test_spend_then_fail", category="alter")
    def spend_then_fail(rows):
        _tracking.add_usage(10, 10, 3.0, model_id="m")
        raise RuntimeError("bad batch")

    @step
    def fragile(rows):
        return _keep_all(spend_then_fail(rows))

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        with ThreadPoolExecutor(1) as pool:
            try:
                pool.submit(fragile, rows).result()
            except RuntimeError:
                pass
        return keep(rows)

    main(_rows("a b"))
    assert _tracking.last_pipeline().cost_usd == pytest.approx(3.0)
    (line,) = _ledger(tmp_path)
    assert (line["pipeline"], line["step"]) == ("main", "fragile")


def test_op_that_raised_inside_a_step_reports_its_cost():
    """Repro: a step that caught an op's error showed $2 with no op row for it."""

    @op(name="alter.test_parse_fails", category="alter")
    def parse_fails(rows):
        _tracking.add_usage(20, 5, 2.0)
        raise ValueError("bad JSON")

    @step
    def tolerant(rows):
        try:
            parse_fails(rows)
        except ValueError:
            pass
        return _keep_all(rows)

    tolerant(_rows("a b"))
    rec = _tracking.step_records()["tolerant"]
    failed, length = rec.ops
    assert failed.op_name == "alter.test_parse_fails"
    assert failed.error == "ValueError: bad JSON"
    assert failed.n_out is None
    assert failed.cost_usd == pytest.approx(2.0)
    assert sum(c.cost_usd for c in rec.ops) == pytest.approx(rec.cost_usd)
    assert "alter.test_parse_fails raised ValueError: bad JSON" in rec.warnings
    assert "RAISED" in render_step(rec)


def test_op_that_raised_outside_a_step_is_not_a_bare_step():
    """A failed op called directly in a pipeline adds no bare step record."""

    @op(name="alter.test_fails", category="alter")
    def fails(rows):
        raise ValueError("nope")

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline
    def main(rows):
        try:
            fails(rows)
        except ValueError:
            pass
        return keep(keep(rows))

    main(_rows("a b"))
    assert [s.name for s in _tracking.last_pipeline().steps] == ["keep", "keep"]


def test_progress_batch_writes_the_ledger_per_completed_request(tmp_path):
    """Repro: a batch interrupted mid-way left no ledger lines for paid requests.

    On the progress path each response is recorded as it completes, so the
    ledger has the fast requests while the slow one is still running.
    """
    model = Model("openai/test-model", show_progress=True)
    seen_while_slow: list[int] = []

    def completion(*, messages, **kwargs):
        if messages[0]["content"] == "slow":
            for _ in range(500):
                if len(_ledger(tmp_path)) >= 3:
                    break
                threading.Event().wait(0.01)
            seen_while_slow.append(len(_ledger(tmp_path)))
        return _response(cost=1.0)

    @op(name="alter.test_ask_batch", category="alter")
    def ask_batch(rows):
        model.generate_batch(["a", "b", "c", "slow"])
        return list(rows)

    @step
    def ask(rows):
        return _keep_all(ask_batch(rows))

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        mock_litellm.completion.side_effect = completion
        ask(_rows("a b"))
    assert seen_while_slow == [3]
    records = _ledger(tmp_path)
    assert len(records) == 4
    assert {(r["step"], r["op"]) for r in records} == {("ask", "alter.test_ask_batch")}
    rec = _tracking.step_records()["ask"]
    assert rec.ops[0].cost_usd == pytest.approx(4.0)


class _FakeMetrics:
    """Stands in for OpenHands combined metrics that grow with each run."""

    def __init__(self) -> None:
        self.accumulated_cost = 0.0
        self.accumulated_token_usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0
        )


def _fake_sdk(monkeypatch, *, cost_per_run: float, accept_on_run: int | None):
    """Replace the OpenHands SDK with a conversation that spends per run."""
    sdk = pytest.importorskip("openhands.sdk")
    from pyforger.alter import _dynamic_tool as tool
    from pyforger.alter import dynamic

    started: list[int] = []

    class FakeConversation:
        def __init__(self, *, agent, callbacks, **kwargs):
            started.append(1)
            self.agent = agent
            self.callbacks = callbacks
            self.metrics = _FakeMetrics()
            self.conversation_stats = SimpleNamespace(
                get_combined_metrics=lambda: self.metrics
            )
            self.state = SimpleNamespace(events=[])
            self.runs = 0

        def send_message(self, text):
            pass

        def run(self):
            self.runs += 1
            self.metrics.accumulated_cost += cost_per_run
            self.metrics.accumulated_token_usage.prompt_tokens += 10
            event = SimpleNamespace(source="agent", llm_response_id=str(self.runs))
            self.state.events.append(event)
            for callback in self.callbacks:
                callback(event)  # OpenHands lets callback errors propagate
            if self.runs == accept_on_run:
                session_id = self.agent.tools[0].params["session_id"]
                tool._session(session_id).accepted[:] = [
                    "def run(item): return [item['text']]"
                ]

        def close(self):
            pass

    monkeypatch.setattr(sdk, "Conversation", FakeConversation)
    monkeypatch.setattr(sdk, "LLM", lambda **kwargs: object())
    monkeypatch.setattr(sdk, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dynamic, "prepare_input", lambda item: item)
    monkeypatch.setattr(tool, "prepare_input", lambda item: item)
    monkeypatch.setattr(dynamic, "_execute", lambda code, item, timeout: ["ok"])
    return started


def _agent_model():
    return SimpleNamespace(
        _model_id="test", _temperature=0, _kwargs={}, _api_key=None, _max_tokens=None
    )


def test_dynamic_step_checks_the_cap_before_each_agent(monkeypatch, tmp_path):
    """Repro: 6 agents started under a $1 cap, each spending $1."""
    from pyforger import alter

    started = _fake_sdk(monkeypatch, cost_per_run=1.0, accept_on_run=1)

    @step
    def keep(rows):
        return _keep_all(rows)

    @pipeline(max_cost_usd=1.0)
    def main(rows):
        alter.dynamic_step(rows, "goal", _agent_model(), max_parallel=1)
        return keep(rows)

    with pytest.raises(pyforger.BudgetExceededError, match="pipeline main"):
        main([{"text": str(i)} for i in range(6)])
    assert len(started) == 1
    (line,) = _ledger(tmp_path)
    assert (line["pipeline"], line["op"], line["cost_usd"]) == (
        "main",
        "alter.dynamic_step",
        1.0,
    )


def test_dynamic_step_stops_an_agent_at_the_cap(monkeypatch, tmp_path):
    """Usage is recorded per agent event and the run stops once a cap is hit."""
    from pyforger.alter import dynamic

    _fake_sdk(monkeypatch, cost_per_run=1.0, accept_on_run=None)
    already = _tracking.usage_snapshot()[2]
    monkeypatch.setenv("FORGER_MAX_COST_USD", str(already + 2.5))
    with pytest.raises(pyforger.BudgetExceededError, match="FORGER_MAX_COST_USD"):
        dynamic._generate("goal", {"text": "x"}, _agent_model(), 20, 5)
    records = _ledger(tmp_path)
    assert [r["cost_usd"] for r in records] == [1.0, 1.0, 1.0]  # 3 runs, not 20
    assert _tracking.usage_snapshot()[2] == pytest.approx(already + 3.0)

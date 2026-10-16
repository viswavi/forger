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


@op(name="alter.test_ask")
def _ask(rows: list[dict], model, times: int = 1) -> list[dict]:
    """Make ``times`` single-prompt model calls, one batch each."""
    for _ in range(times):
        model.generate("hello")
    return list(rows)


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #

_N_WORKERS = 5


@op(name="alter.test_chunk")
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
    """A thread that did not copy the context adds to the pipeline, not a step."""

    @op(name="alter.test_detached")
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
    assert pipe.prompt_tokens == 71
    assert pipe.cost_usd == pytest.approx(0.75)
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

    @op(name="alter.test_spend")
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


@op(name="alter.test_costly")
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
    assert "cached (originally" in render_step(rec)
    assert "$0.25" in render_step(rec)

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
    }
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


def test_pipeline_cap_counts_unattributed_usage():
    """Unattributed spend counts toward the pipeline's cap."""
    model = Model("openai/test-model")

    @op(name="alter.test_detached_then_ask")
    def detached_then_ask(rows):
        thread = threading.Thread(target=_tracking.add_usage, args=(1, 1, 2.0))
        thread.start()
        thread.join()
        model.generate("hello")
        return list(rows)

    @step
    def ask(rows):
        return _keep_all(detached_then_ask(rows))

    @pipeline(max_cost_usd=1.0)
    def main(rows):
        return ask(rows)

    with patch("pyforger.acquire.model.litellm") as mock_litellm:
        with pytest.raises(pyforger.BudgetExceededError):
            main(_rows("a b"))
        mock_litellm.completion.assert_not_called()


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


def test_bad_caps_are_rejected(monkeypatch):
    """A negative max_cost_usd or a non-number FORGER_MAX_COST_USD raises."""
    with pytest.raises(ValueError, match="max_cost_usd"):
        pipeline(max_cost_usd=-1)
    monkeypatch.setenv("FORGER_MAX_COST_USD", "ten dollars")
    with pytest.raises(ValueError, match="FORGER_MAX_COST_USD"):
        _tracking.check_budget()

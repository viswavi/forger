"""Tests for pyforger._progress: the shared progress display."""
from __future__ import annotations

import logging

from pyforger import _progress
from pyforger._progress import Progress, done_line, vllm_tqdm


def test_done_line():
    """Done, elapsed, and the time left at the current rate."""
    assert done_line(120, 500, 31) == "120/500 done, 31s, ~1m 38s left"
    assert done_line(500, 500, 90) == "500/500 done, 1m 30s"


def test_time_left_ignores_the_wait_before_the_first_unit(monkeypatch, caplog):
    """Loading time before the first unit does not inflate the estimate."""
    clock = [0.0]
    monkeypatch.setattr(_progress.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(_progress, "LOG_INTERVAL_S", 0.0)
    caplog.set_level(logging.INFO)
    p = Progress("embed", 100, "log")
    clock[0] = 60.0  # a minute of model loading
    p.update(10)  # first unit: no rate yet, so no estimate
    clock[0] = 70.0
    p.update(10)  # 10 units in 10 s: 80 left -> ~1m 20s
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "embed: 10/100 done, 1m",
        "embed: 20/100 done, 1m 10s, ~1m 20s left",
    ]


def test_log_mode_logs_every_interval_and_at_the_end(monkeypatch, caplog):
    """Redirected output gets periodic lines and a final one; off is silent."""
    monkeypatch.setattr(_progress, "LOG_INTERVAL_S", 0.0)
    caplog.set_level(logging.INFO)
    with Progress("model: completions", 4, "log") as p:
        p.update(1)
        p.update(1)
    lines = [r.getMessage() for r in caplog.records]
    assert lines[0].startswith("model: completions: 1/4 done, ")
    assert lines[-1].startswith("model: completions: 4/4 done, ")
    caplog.clear()
    with Progress("quiet", 2, "off") as p:
        p.update(2)
    assert not caplog.records


def test_vllm_tqdm_forwards_counts():
    """Counts from vLLM's bar reach the sink; other tqdm calls do nothing."""
    seen = []
    bar = vllm_tqdm(seen.append)(total=3, desc="Processed prompts", postfix="")
    bar.update(2)
    bar.postfix = "speed"
    bar.refresh()
    bar.close()
    assert seen == [2] and bar.n == 2 and bar.format_dict["elapsed"] > 0
    # vLLM 0.16 wraps its request loop: the items pass through, uncounted.
    assert list(vllm_tqdm(seen.append)(range(3), desc="Adding requests")) == [0, 1, 2]
    assert seen == [2]

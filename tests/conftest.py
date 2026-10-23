"""Pytest fixtures shared across tests.

The autouse ``_isolate_op_cache`` fixture redirects @op's disk cache to a
per-test tmp dir. Without it, two tests calling the same op with the same
inputs (e.g. two ``sample_from_LLM`` calls in test_acquire_sample.py)
collide on the persistent cache: the second hits cache, the model is
never called, and assertions like ``model._call_count == 1`` fail.
"""
from __future__ import annotations

import pytest

from pyforger.acquire import model as _model
from pyforger.op import clear_trace


@pytest.fixture(autouse=True)
def _isolate_op_cache(tmp_path, monkeypatch):
    """Point @op's disk cache at a fresh per-test tmp dir; clear the trace.

    This means that every test starts with a clean cache.
    """
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    clear_trace()
    yield


@pytest.fixture(autouse=True)
def _forget_in_flight_limits():
    """A 429 in one test must not throttle the next test's requests."""
    _model._IN_FLIGHT.clear()
    yield
    _model._IN_FLIGHT.clear()


@pytest.fixture(autouse=True)
def _forget_learned_no_n():
    """Forget models that generate_batch learned to fan out in an earlier test."""
    _model._LEARNED_NO_N.clear()
    yield
    _model._LEARNED_NO_N.clear()

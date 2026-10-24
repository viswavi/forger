"""Pytest fixtures shared across tests.

The autouse ``_isolate_op_cache`` fixture redirects @op's disk cache to a
per-test tmp dir. Without it, two tests calling the same op with the same
inputs (e.g. two ``sample_from_LLM`` calls in test_acquire_sample.py)
collide on the persistent cache: the second hits cache, the model is
never called, and assertions like ``model._call_count == 1`` fail.
"""

from __future__ import annotations

import os

import pytest

# Use the versioned bundled metadata, not a network fetch on every SDK import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from pyforger.op import clear_trace  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_op_cache(tmp_path, monkeypatch):
    """Point @op's disk cache at a fresh per-test tmp dir; clear the trace.

    This means that every test starts with a clean cache.
    """
    monkeypatch.setenv("FORGER_CACHE", str(tmp_path))
    clear_trace()
    yield

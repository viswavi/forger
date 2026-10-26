"""Generic ordered calls and environment-credential redaction."""

from functools import partial
from threading import Barrier, Lock

import pytest

from pyforger.utils import redact, run_concurrently


@pytest.mark.parametrize("concurrency", [None, 1, 2])
def test_run_concurrently_preserves_order(concurrency):
    """Call each job once and retain its input association."""
    assert run_concurrently(
        (partial(str, i) for i in range(5)), concurrency=concurrency
    ) == ["0", "1", "2", "3", "4"]
    assert run_concurrently([], concurrency=concurrency) == []


def test_run_concurrently_bounds_active_jobs():
    """Real workers overlap without exceeding the explicit worker bound."""
    barrier = Barrier(2, timeout=5)
    lock = Lock()
    active = 0
    peak = 0

    def job():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        barrier.wait()
        with lock:
            active -= 1
        return "done"

    assert run_concurrently([job] * 4, concurrency=2) == ["done"] * 4
    assert peak == 2


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5, "2"])
def test_run_concurrently_rejects_invalid_limits(concurrency):
    """Validate limits even for an empty job sequence."""
    with pytest.raises(ValueError, match="concurrency"):
        run_concurrently([], concurrency=concurrency)


def test_run_concurrently_propagates_errors():
    """A failing callable is not converted to a successful result."""

    def fail():
        raise RuntimeError("job failed")

    for concurrency in (1, 2):
        with pytest.raises(RuntimeError, match="job failed"):
            run_concurrently([fail], concurrency=concurrency)


def test_redact_environment_credentials(monkeypatch):
    """Remove credential values but retain ordinary and short environment values."""
    monkeypatch.setenv("TEST_API_KEY", "example-api-credential")
    monkeypatch.setenv("TEST_token", "example-token-credential")
    monkeypatch.setenv("TEST_SECRET", "example-secret-credential")
    monkeypatch.setenv("TEST_PASSWORD", "example-password-credential")
    monkeypatch.setenv("TEST_NAME", "example-ordinary-value")
    monkeypatch.setenv("TEST_SHORT_KEY", "short")
    assert (
        redact(
            "example-api-credential example-token-credential example-secret-credential "
            "example-password-credential example-ordinary-value short"
        )
        == "[REDACTED] [REDACTED] [REDACTED] [REDACTED] example-ordinary-value short"
    )


def test_run_concurrently_default_starts_all_jobs():
    """The default does not impose ThreadPoolExecutor's usual worker limit."""
    barrier = Barrier(40, timeout=10)

    def job(index):
        barrier.wait()
        return index

    assert run_concurrently(partial(job, i) for i in range(40)) == list(range(40))

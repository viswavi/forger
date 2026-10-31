"""Exercise real provisioning retry policy without any remote resources."""

import json
import sys
import threading
from functools import partial
from types import SimpleNamespace

import pytest

from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.reusable_session import _DeadlineBudget
from pyforger.sail_backend import SailBackend
from pyforger.sail_provision import create_sailbox
from pyforger.session_budget import SessionBudget

errors = pytest.importorskip("sail.errors")
ApiError = errors.ApiError
SailboxCreationError = errors.SailboxCreationError
TransportError = errors.TransportError

REFRESH = "Sailbox concurrency limits are refreshing; retry shortly"


def budget_for(clock, seconds):
    budget = SessionBudget(3, seconds)
    budget._started = clock.now
    return budget


class Clock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    return clock


def test_refresh_retries_inside_permit_and_preserves_budget(clock):
    budget = budget_for(clock, 100)
    permit = threading.Semaphore(1)
    calls = []
    box = object()

    def create(**kwargs):
        assert not permit.acquire(blocking=False)
        calls.append(kwargs)
        clock.now += 1
        if len(calls) < 3:
            raise SailboxCreationError(REFRESH, status_code=429)
        return box

    with permit:
        assert (
            create_sailbox(
                create,
                budget=budget,
                sleep=clock.sleep,
                jitter=lambda a, b: b,
                name="same-name",
                setup_times=[],
            )
            is box
        )
    assert clock.delays == [2, 4]
    assert [call["timeout"] for call in calls] == [100, 97, 92]
    assert [call["image_build_timeout"] for call in calls] == [100, 97, 92]
    assert {call["name"] for call in calls} == {"same-name"}
    assert budget.remaining_s() == 92  # Only successful provisioning is exempt.
    assert budget.iterations == 0


@pytest.mark.parametrize(
    "error",
    [
        SailboxCreationError("insufficient credits", status_code=402, retryable=True),
        SailboxCreationError(REFRESH, status_code=401),
        SailboxCreationError(REFRESH, status_code=403),
        SailboxCreationError(REFRESH, status_code=402),
        SailboxCreationError(REFRESH, status_code=400),
        SailboxCreationError(REFRESH, status_code=503, retryable=True),
        SailboxCreationError(
            "hard concurrency quota exceeded", status_code=429, retryable=True
        ),
        SailboxCreationError("creation timed out", retryable=True),
        SailboxCreationError(REFRESH + "."),
        SailboxCreationError(REFRESH, body=json.dumps({"message": "invalid config"})),
        TransportError(REFRESH),
        ApiError(REFRESH, status_code=429, retryable=True),
        RuntimeError(REFRESH),
        ValueError("invalid config"),
    ],
)
def test_unsafe_errors_are_not_retried(clock, error):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise error

    with pytest.raises(type(error)) as caught:
        create_sailbox(create, budget=budget_for(clock, 100), sleep=clock.sleep)
    assert caught.value is error
    assert len(calls) == 1
    assert clock.delays == []


@pytest.mark.parametrize(
    "body", ["", json.dumps({"message": REFRESH}), json.dumps({"error": REFRESH})]
)
def test_exact_legacy_or_structured_rejection_is_bounded(clock, body):
    calls = []
    error = SailboxCreationError(REFRESH, body=body)

    def create(**kwargs):
        calls.append(kwargs)
        raise error

    with pytest.raises(SailboxCreationError) as caught:
        create_sailbox(
            create,
            budget=budget_for(clock, 100),
            sleep=clock.sleep,
            jitter=lambda a, b: b,
        )
    assert caught.value is error
    assert len(calls) == 5
    assert clock.delays == [2, 4, 8, 10]


def test_no_attempt_after_sleep_exhausts_deadline(clock):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise SailboxCreationError(REFRESH)

    with pytest.raises(TimeoutError):
        create_sailbox(
            create,
            budget=budget_for(clock, 1),
            sleep=clock.sleep,
            jitter=lambda a, b: b,
        )
    assert len(calls) == 1
    assert clock.delays == [1]


def test_expired_budget_never_creates(clock):
    budget = budget_for(clock, 1)
    clock.now = 2
    with pytest.raises(TimeoutError):
        create_sailbox(lambda **kwargs: pytest.fail("must not create"), budget=budget)


def test_absolute_deadline_is_not_extended_by_setup_exemption(clock):
    budget = _DeadlineBudget(
        AgentSpec(kind="oracle", solution="true"),
        WorkspaceTask("test", timeout_s=10),
        5,
    )

    def create(**kwargs):
        clock.now += 2
        return object()

    assert create_sailbox(create, budget=budget, setup_times=[]) is not None
    assert budget.remaining_s() == 3
    assert budget.deadline == 5


@pytest.mark.parametrize("reusable", [False, True])
def test_backend_creation_retry_keeps_cleanup_owner(monkeypatch, reusable):
    """Replace only billable Sail boundary; run real backend and retry code."""
    calls = []
    delays = []

    class Box:
        sailbox_id = "local-test"
        terminated = 0

        def run(self, *args, **kwargs):
            return SimpleNamespace(exit_code=1, timed_out=False)

        def terminate(self):
            self.terminated += 1

    box = Box()

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise SailboxCreationError(REFRESH, status_code=429)
        return box

    monkeypatch.setitem(
        sys.modules,
        "sail",
        SimpleNamespace(
            Image=SimpleNamespace(from_registry=lambda _: object()),
            App=SimpleNamespace(find=lambda *a, **k: object()),
            Sailbox=SimpleNamespace(create=create),
        ),
    )
    module = "pyforger.reusable_session" if reusable else "pyforger.sail_backend"
    monkeypatch.setattr(
        module + ".create_sailbox", partial(create_sailbox, sleep=delays.append)
    )
    backend = SailBackend()
    task = WorkspaceTask("test", timeout_s=30)
    spec = AgentSpec(kind="oracle", solution="true")
    with pytest.raises(RuntimeError, match="Cannot initialize Sail workspace"):
        if reusable:
            with backend.open_session() as session:
                session.run_agent(task, spec)
        else:
            backend.run_agent(task, spec)
    assert len(calls) == 2
    assert len(delays) == 1
    assert 1 <= delays[0] <= 2
    assert box.terminated == 1

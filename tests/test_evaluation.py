"""Independent panels execute real TaskSpecs with caller-owned storage."""

import threading
import time
from collections import UserDict
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from pyforger.agent_backend import AgentSpec
from pyforger.agent_tasks import TaskSpec
from pyforger.evaluation import (
    PanelReport,
    evaluate_panel,
    evaluate_panels,
    validate_panel,
)
from pyforger.local_backend import LocalBackend
from pyforger.task_runs import run_recorded_task


def test_panel_actual_tasks_and_unique_identities(tmp_path):
    """Panel callbacks receive stable unique IDs and real task outcomes."""
    seen = []

    def evaluate(model, identity):
        seen.append((model, identity))
        index = int(identity.rsplit("-", 1)[1])
        task = TaskSpec("verify", verifier=f"echo {int(index == 0)} > reward.txt")
        return run_recorded_task(
            task,
            agent=AgentSpec(kind="nop"),
            backend=LocalBackend(),
            output=tmp_path / identity,
        )

    report = evaluate_panel(
        ["provider/model-a", "model-b"], evaluate, attempts=2, concurrency=2
    )
    assert report.accepted is True
    assert len(seen) == len({identity for _, identity in seen}) == 4
    assert {(row.model, row.trial_id) for row in report.trials} == set(seen)
    assert report.models["provider/model-a"]["success_rate"] == 0.5
    assert PanelReport.model_validate_json(report.model_dump_json()) == report


def test_null_excluded_false_counted_and_no_cache():
    """Repeated panels execute every callback and exclude only missing grades."""
    calls = []

    def evaluate(model, identity):
        calls.append(identity)
        return {"passed": [True, False, None][int(identity.rsplit("-", 1)[1])]}

    for _ in range(2):
        report = evaluate_panel(["a"], evaluate, attempts=3)
        assert report.accepted is False
        assert report.models["a"] == dict(
            attempted=3, valid=2, passed=1, excluded=1, success_rate=0.5
        )
    assert len(calls) == 6
    explicit = evaluate_panel(["a"], evaluate, attempts=3, min_valid=2)
    assert explicit.accepted is True
    assert explicit.models["a"]["success_rate"] == 0.5
    assert evaluate_panel(["a"], lambda *_: {"passed": None}).accepted is False
    assert evaluate_panel(["a"], lambda *_: {"passed": False}).accepted is False


@pytest.mark.parametrize(
    "models,options",
    [
        ([], {}),
        (["a", "a"], {}),
        ([""], {}),
        ([" "], {}),
        ("abc", {}),
        ([1], {}),
        (["a"], {"attempts": 0}),
        (["a"], {"attempts": True}),
        (["a"], {"concurrency": 0}),
        (["a"], {"concurrency": True}),
        (["a"], {"min_valid": 0}),
        (["a"], {"min_valid": True}),
        (["a"], {"min_valid": 1.5}),
        (["a"], {"min_rate": 0.9, "max_rate": 0.2}),
        (["a"], {"min_rate": -0.1}),
        (["a"], {"max_rate": float("nan")}),
    ],
)
def test_invalid_panel_fails_before_launch(models, options):
    """All panel configuration errors fail before invoking any callback."""

    def forbidden(*args):
        raise AssertionError("Callback launched")

    with pytest.raises(ValueError):
        evaluate_panel(models, forbidden, **options)


def test_panel_bounded_executor():
    """Threaded callback concurrency is explicitly bounded."""
    lock = threading.Lock()
    active = peak = 0

    def evaluate(model, identity):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"passed": False}

    evaluate_panel(["a", "b"], evaluate, attempts=3, concurrency=2)
    assert peak == 2


def test_callback_errors_propagate():
    """Panel machinery never silently converts callback exceptions into grades."""

    def evaluate(*args):
        raise RuntimeError("provider offline")

    with pytest.raises(RuntimeError, match="provider offline"):
        evaluate_panel(["a"], evaluate)
    with pytest.raises(ValueError, match="passed"):
        evaluate_panel(["a"], lambda *_: {})
    with pytest.raises(ValueError):
        evaluate_panel(["a"], lambda *_: {"passed": "false"})


def test_default_panel_concurrency_and_standalone_validation():
    """Default parallelism starts every trial and shared validation is callable."""
    validate_panel(("a", "b"), attempts=2)
    barrier = threading.Barrier(4)

    def evaluate(model, identity):
        barrier.wait(timeout=5)
        return {"passed": False}

    report = evaluate_panel(["a", "b"], evaluate, attempts=2)
    assert len(report.trials) == 4


class _TypedTrial(BaseModel):
    """A typed panel callback result carrying its own grade."""

    passed: bool | None
    note: str = ""


def test_panel_accepts_model_results():
    """Model-typed callback results normalize into the typed report."""
    report = evaluate_panel(
        ["a"],
        lambda model, identity: _TypedTrial(passed=True, note=identity),
        attempts=2,
    )
    assert report.accepted is False  # rate 1.0 is outside the default window
    assert report.models["a"]["success_rate"] == 1.0
    assert [t.result["passed"] for t in report.trials] == [True, True]
    assert all(t.result["note"].startswith("model-0-") for t in report.trials)


def test_evaluate_panels_runs_every_panel_in_order():
    """Panel fan-out returns one report per panel name without filtering."""
    evaluated = []

    def run(name: str) -> PanelReport:
        evaluated.append(name)
        return evaluate_panel(
            ["m"],
            lambda *_: {"passed": name == "pass"},
            attempts=1,
            **(
                dict(min_rate=0.0, max_rate=1.0)
                if name == "pass"
                else dict(min_rate=0.2, max_rate=0.8)
            ),
        )

    reports = evaluate_panels({"pass": ["m"], "fail": ["m"]}, run, concurrency=None)
    assert list(reports) == ["pass", "fail"]
    assert reports["pass"].accepted is True
    assert reports["fail"].accepted is False
    assert sorted(evaluated) == ["fail", "pass"]

    with pytest.raises(TypeError):
        evaluate_panels({"bad": ["m"]}, lambda name: {"accepted": True})
    with pytest.raises(ValueError):
        evaluate_panels({"empty": []}, run)
    with pytest.raises(ValueError):
        evaluate_panels({"dup": ["m", "m"]}, run)


@dataclass
class _DataclassTrial:
    passed: bool
    note: str
    score: float
    evidence: dict[str, str]


@pytest.mark.parametrize("result_type", [UserDict, _DataclassTrial])
def test_panel_preserves_mapping_and_dataclass_diagnostics(result_type):
    """All supported result containers preserve arbitrary diagnostic fields."""
    payload = dict(passed=True, note="verified", score=0.75, evidence={"file": "out"})
    result = (
        result_type(**payload) if result_type is _DataclassTrial else UserDict(payload)
    )
    report = evaluate_panel(["a"], lambda *_: result, attempts=1, max_rate=1)
    assert report.accepted
    assert report.trials[0].result == payload


def test_panel_refuses_unpublished_private_evidence():
    from pyforger.evidence_journal import EvidenceJournal

    journal = EvidenceJournal()
    journal.append({"text": "evidence"})
    with pytest.raises(ValueError, match="Publish private evidence"):
        evaluate_panel(
            ["model"],
            lambda *_: {"passed": True, "trajectory": journal.trajectory()},
            attempts=1,
        )

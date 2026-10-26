import threading
import time

import pytest
from pyforger.evaluation import evaluate_panel


def test_bounded_per_model_replacements_and_concurrency():
    lock = threading.Lock()
    active = peak = 0

    def run(model, identity):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"passed": None, "status": "infrastructure_error", "execution": identity}

    result = evaluate_panel(
        ["a", "b"],
        run,
        concurrency=2,
        max_replacements=2,
        replacement_predicate=lambda row: row["passed"] is None,
    )
    assert peak <= 2
    assert len(result["trials"]) == 10
    assert len(result["execution_history"]) == 14
    assert len({r["execution_id"] for r in result["execution_history"]}) == 14
    assert [r["execution_id"] for r in result["execution_history"][10:]] == [
        "model-0-trial-0-replacement-1",
        "model-0-trial-1-replacement-2",
        "model-1-trial-0-replacement-1",
        "model-1-trial-1-replacement-2",
    ]
    assert not result["accepted"]


def test_single_replacement_keeps_logical_id_and_original_evidence():
    def run(model, identity):
        failed = identity == "model-0-trial-0"
        return {
            "passed": None if failed else False,
            "status": "infra" if failed else "done",
            "partial": identity,
        }

    result = evaluate_panel(
        ["a"],
        run,
        concurrency=1,
        min_rate=0,
        max_rate=1,
        max_replacements=2,
        replacement_predicate=lambda r: r["status"] == "infra",
    )
    assert result["accepted"] and len(result["trials"]) == 5
    assert result["trials"][0]["trial_id"] == "model-0-trial-0"
    assert result["trials"][0]["result"]["partial"] == "model-0-trial-0-replacement-1"
    assert result["execution_history"][0]["result"]["partial"] == "model-0-trial-0"
    assert len(result["execution_history"]) == 6


def test_default_shape_and_no_predicate_calls():
    result = evaluate_panel(
        ["a"],
        lambda *_: {"passed": False},
        min_rate=0,
        replacement_predicate=lambda _: pytest.fail("disabled"),
    )
    assert set(result) == {"accepted", "models", "trials"}


@pytest.mark.parametrize("value", [{}, {"passed": "false"}])
def test_malformed_grade_not_retried(value):
    calls = []

    def run(*args):
        calls.append(args)
        return value

    with pytest.raises(ValueError):
        evaluate_panel(
            ["a"],
            run,
            concurrency=1,
            max_replacements=2,
            replacement_predicate=lambda _: True,
        )
    assert calls


def test_callback_exception_propagates_not_reclassified():
    def run(*_):
        raise TimeoutError("caller deadline")

    with pytest.raises(TimeoutError):
        evaluate_panel(
            ["a"],
            run,
            concurrency=1,
            max_replacements=2,
            replacement_predicate=lambda _: True,
        )


def test_replacement_exception_retains_prior_partial_evidence():
    def run(model, identity):
        if "replacement" in identity:
            error = TimeoutError("deadline exhausted")
            error.diagnostics = {"current_trajectory": {"steps": ["partial-current"]}}
            raise error
        return {"passed": None, "partial": ["original"]}

    with pytest.raises(TimeoutError) as caught:
        evaluate_panel(
            ["a"],
            run,
            concurrency=1,
            max_replacements=2,
            replacement_predicate=lambda _: True,
        )
    assert caught.value.diagnostics["current_trajectory"]["steps"] == [
        "partial-current"
    ]
    assert len(caught.value.diagnostics["panel_execution_history"]) == 5

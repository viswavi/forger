"""Real deterministic sessions exercise host validation and shared repair budgets."""

from __future__ import annotations

import time

import pytest

from pyforger.agent_backend import AgentSpec, WorkspaceTask
from pyforger.agents import generate_files
from pyforger.local_backend import LocalBackend
from pyforger.measurements import Measurement
from pyforger.structured import GenerationInconclusive, ValidationExhausted


def test_same_workspace_repair_and_history():
    """An oracle preserves its marker and receives host feedback across repairs."""
    result = LocalBackend().run_agent_validated(
        WorkspaceTask("write output", outputs=("x",)),
        AgentSpec(
            kind="oracle",
            solution=(
                'if test -f marker; then printf "%s" "$FORGER_FEEDBACK" > x; '
                "else touch marker; echo bad > x; fi"
            ),
        ),
        validate=lambda files: Measurement(
            passed=b"expected-content" in files["x"], reason="expected-content"
        ),
    )
    assert len(result.validation_attempts) == 2
    assert result.validation_attempts[0].output == {"x": b"bad\n"}
    assert result.validation_attempts[1].measurement.passed is True
    assert result.diagnostics["iterations"] == 2


def test_missing_output_is_repairable():
    """Missing files receive feedback before the host validator is called."""
    seen = []

    def validate(files):
        seen.append(files)
        return Measurement(passed=True)

    result = LocalBackend().run_agent_validated(
        WorkspaceTask("write output", outputs=("x",)),
        AgentSpec(
            kind="oracle",
            solution="if test -f marker; then echo repaired > x; else touch marker; fi",
        ),
        validate=validate,
    )
    assert len(result.validation_attempts) == 2
    assert result.validation_attempts[0].output == {}
    assert "Missing" in result.validation_attempts[0].measurement.reason
    assert seen == [{"x": b"repaired\n"}]


@pytest.mark.parametrize(
    "passed,error,count",
    [(False, ValidationExhausted, 3), (None, GenerationInconclusive, 1)],
)
def test_rejection_and_inconclusive_retain_outputs(passed, error, count):
    """Only rejection consumes repairs; both stop conditions preserve output."""
    with pytest.raises(error) as caught:
        generate_files(
            "write",
            {},
            ("x",),
            agent=AgentSpec(kind="oracle", solution="echo bad > x"),
            validate=lambda _: Measurement(passed=passed),
        )
    assert caught.value.output == {"x": b"bad\n"}
    assert len(caught.value.attempts) == count


def test_execution_errors_propagate(tmp_path):
    """A failed shell command is not retried as a rejected output."""
    marker = tmp_path / "calls"
    with pytest.raises(RuntimeError, match="script failed"):
        generate_files(
            "write",
            {},
            ("x",),
            agent=AgentSpec(kind="oracle", solution=f"echo call >> {marker}; exit 7"),
            validate=lambda _: Measurement(passed=False),
        )
    assert marker.read_text() == "call\n"


def test_validator_error_propagates():
    """Host validator failures are not converted to repair feedback."""
    calls = []

    def validate(files):
        calls.append(files)
        raise OSError("validator unavailable")

    with pytest.raises(OSError, match="unavailable"):
        generate_files(
            "write",
            {},
            ("x",),
            agent=AgentSpec(kind="oracle", solution="echo bad > x"),
            validate=validate,
        )
    assert len(calls) == 1


def test_shared_iteration_budget():
    """A second repair cannot reset the original iteration allowance."""
    with pytest.raises(RuntimeError, match="iteration budget"):
        generate_files(
            "write",
            {},
            ("x",),
            agent=AgentSpec(kind="oracle", max_iterations=1, solution="echo bad > x"),
            validate=lambda _: Measurement(passed=False),
        )


def test_walltime_cancels_shell_children(tmp_path):
    """The timeout kills the shell process group rather than orphaning a child."""
    marker = tmp_path / "late"
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        generate_files(
            "write",
            {},
            ("x",),
            timeout_s=0.1,
            agent=AgentSpec(
                kind="oracle", solution=f"sleep 0.5; echo orphan > {marker}"
            ),
            validate=lambda _: Measurement(passed=True),
        )
    assert time.monotonic() - started < 1
    time.sleep(0.6)
    assert not marker.exists()


@pytest.mark.parametrize("repairs", [-1, True, 1.5])
def test_invalid_repairs(repairs):
    """Invalid repair counts are rejected before execution."""
    with pytest.raises(ValueError):
        generate_files(
            "write",
            {},
            (),
            agent=AgentSpec(kind="nop"),
            validate=lambda _: Measurement(passed=True),
            max_repairs=repairs,
        )


def test_sdk_budget_attachment():
    """A frozen SDK agent accepts private budget state without field mutation."""
    from openhands.sdk import LLM

    from pyforger.agent_session import BudgetedAgent, SessionBudget

    agent = BudgetedAgent(llm=LLM(model="openai/test"))
    budget = SessionBudget(2, 10)
    agent.attach_budget(budget)
    assert agent.budget is budget
    assert agent.static_system_message
    budget.consume_step()
    budget.consume_step()
    with pytest.raises(RuntimeError, match="iteration budget"):
        budget.consume_step()


def test_validation_time_counts_toward_deadline():
    """A bounded synchronous validator cannot reset the shared wall-time budget."""

    def validate(files):
        time.sleep(0.15)
        return Measurement(passed=True)

    with pytest.raises(TimeoutError, match="wall-time"):
        generate_files(
            "write",
            {},
            ("x",),
            timeout_s=0.1,
            agent=AgentSpec(kind="oracle", solution="echo good > x"),
            validate=validate,
        )


def test_zero_repairs_validates_once():
    """Zero repairs still produces and evaluates one retained output."""
    with pytest.raises(ValidationExhausted) as caught:
        generate_files(
            "write",
            {},
            ("x",),
            max_repairs=0,
            agent=AgentSpec(kind="oracle", solution="echo bad > x"),
            validate=lambda _: Measurement(passed=False),
        )
    assert len(caught.value.attempts) == 1


def test_unlimited_steps_still_share_deadline():
    """Unlimited steps still enforce the shared wall-time deadline."""
    from pyforger.session_budget import SessionBudget

    budget = SessionBudget(None, 10)
    for _ in range(1001):
        budget.consume_step()
    assert budget.iterations == 1001
    budget._started -= 11
    with pytest.raises(TimeoutError, match="wall-time"):
        budget.consume_step()
    assert budget.iterations == 1001


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_optional_iteration_validation(limit):
    """Invalid optional iteration limits are rejected."""
    from pyforger.session_budget import SessionBudget

    with pytest.raises(ValueError):
        SessionBudget(limit, 10)
    with pytest.raises(ValueError):
        AgentSpec(kind="nop", max_iterations=limit)


def test_unlimited_spec_preserves_none():
    """An unlimited spec preserves None through adaptation."""
    assert AgentSpec(kind="nop", max_iterations=None).max_iterations is None
    assert (
        AgentSpec(kind="nop", max_iterations=None).cache_key()
        != AgentSpec(kind="nop").cache_key()
    )


@pytest.mark.parametrize("timeout", [float("inf"), float("nan"), 0, -1])
def test_unlimited_steps_require_finite_wall_deadline(timeout):
    """Unlimited steps still require a finite wall-time deadline."""
    from pyforger.agent_backend import WorkspaceTask
    from pyforger.session_budget import SessionBudget

    with pytest.raises(ValueError, match="finite"):
        SessionBudget(None, timeout)
    with pytest.raises(ValueError, match="finite"):
        WorkspaceTask(instruction="test", timeout_s=timeout)

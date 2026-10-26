"""Opt-in backend handles retaining a conversation and its workspace."""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Self

from pyforger.agent_backend import AgentRunResult, AgentSpec, WorkspaceTask
from pyforger.agent_backends import require_learner_runtime
from pyforger.sail_provision import create_sailbox
from pyforger.session_budget import SessionBudget
from pyforger.structured import GenerationInconclusive, ValidationExhausted


class _DeadlineBudget(SessionBudget):
    def __init__(
        self, agent: AgentSpec, task: WorkspaceTask, deadline: float | None
    ) -> None:
        super().__init__(agent.max_iterations, task.timeout_s)
        self.deadline = (
            deadline if deadline is not None else self._started + task.timeout_s
        )
        self.turn_deadline = self.deadline

    def remaining_s(self) -> float:
        remaining = min(self.deadline, self.turn_deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Agent session wall-time budget exhausted")
        return remaining


class ReusableSession:
    """Backend-compatible, single-owner generation session.

    First use binds the agent kind, tools, packages and network policy.
    SDK models/credentials can change without replacing the conversation.
    Files supplied on later turns overwrite named inputs, leaving all other
    workspace state intact. Verifier/solution material is never accepted: use
    independent fresh backend calls for evaluation. The first task's timeout
    and iteration limit are cumulative when no absolute deadline is supplied.
    With an absolute deadline, each task timeout caps its active call while
    idle time is charged only to the enclosing deadline. Synchronous
    validation callbacks must bound their own execution.

    ValidationExhausted/GenerationInconclusive leave the conversation available
    for outer feedback. Other execution errors close it rather than silently
    starting a new conversation. Always close or use a with statement.
    """

    supports_inner_loop_validation: bool
    supports_sdk_condensation: bool
    supports_science_rl_tools: bool

    def __init__(self, backend: Any, *, deadline: float | None = None) -> None:
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be a finite absolute monotonic time")
        self._backend = backend
        self._deadline = deadline
        self._stack = ExitStack()
        self._lock = threading.Lock()
        self._closed = False
        self._agent: AgentSpec | None = None
        self._policy: Any = None
        self._budget: Any = None
        self._runner: Any = None
        self._diagnostics: dict[str, Any] = {}
        self._credential_values: set[str] = set()
        self.name = backend.name
        for capability in (
            "supports_inner_loop_validation",
            "supports_sdk_condensation",
            "supports_science_rl_tools",
        ):
            setattr(self, capability, getattr(backend, capability, False))

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("Agent session is closed")
        return self

    def __exit__(self, *args: object) -> None:
        if isinstance(args[1], BaseException):
            self._close_after_error(args[1])
        else:
            self.close()

    def cache_key(self) -> object:
        """Never cache live session state as if it were a fresh backend."""
        raise TypeError("Reusable agent sessions cannot be cached")

    def execute_program(
        self, code: str, item: dict[str, Any], timeout_s: float
    ) -> list[str]:
        """Replay independently; retained author state must not enter evaluation."""
        return self._backend.execute_program(code, item, timeout_s)

    def close(self) -> None:
        """Release owned resources once; do not race an active generation call."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Agent session is already in use")
        try:
            self._close()
        finally:
            self._lock.release()

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stack.close()

    def _close_after_error(self, exc: BaseException) -> None:
        try:
            self._close()
        except BaseException as cleanup_error:
            self._diagnostics.update(
                cleanup_error_type=type(cleanup_error).__name__,
                cleanup_error=str(cleanup_error),
                cleanup_traceback=traceback.format_exc(),
            )
        existing = getattr(exc, "diagnostics", {})
        own = dict(self._diagnostics)
        if own.get("current_trajectory") is not None:
            own["current_trajectory"]["captured_at"] = time.time()
        current_id = existing.get("current_trajectory", {}).get("call_id")
        own_id = own.get("current_trajectory", {}).get("call_id")
        if current_id is not None and current_id != own_id:
            retained = dict(existing)
            retained["enclosing_sessions"] = [
                *existing.get("enclosing_sessions", []),
                own,
            ]
        else:
            if own.get("current_trajectory") is not None:
                own["current_trajectory"]["incomplete"] = True
            retained = {**existing, **own}
        if retained.get("current_trajectory") is not None:
            retained["trajectory"] = retained["current_trajectory"]
        from pyforger.agent_diagnostics import sanitizer

        retained = sanitizer([], secrets=self._credential_values)(retained)
        exc.diagnostics = retained

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Continue the retained conversation with this task's inputs."""
        return self._run(task, agent, None, 0)

    def run_agent_validated(
        self,
        task: WorkspaceTask,
        agent: AgentSpec,
        *,
        validate: Any,
        max_repairs: int = 2,
    ) -> AgentRunResult:
        """Continue generation and host-validated repairs in the retained session."""
        return self._run(task, agent, validate, max_repairs)

    def _compatible(self, agent: AgentSpec) -> bool:
        assert self._agent is not None
        return (agent.kind, agent.solution, dict(agent.kwargs), agent.runtime) == (
            self._agent.kind,
            self._agent.solution,
            dict(self._agent.kwargs),
            self._agent.runtime,
        )

    def _run(
        self, task: WorkspaceTask, agent: AgentSpec, validate: Any, max_repairs: int
    ) -> AgentRunResult:
        if type(max_repairs) is not int or max_repairs < 0:
            raise ValueError("max_repairs must be a nonnegative integer")
        if task.verifier is not None or task.verifier_files or task.solution_files:
            raise ValueError(
                "Reusable generation sessions reject hidden verifier/solution material"
            )
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Agent session is already in use")
        try:
            if self._closed:
                raise RuntimeError("Agent session is closed")
            require_learner_runtime(self, agent)
            policy = (task.packages, task.network)
            if self._agent is not None and (
                not self._compatible(agent) or policy != self._policy
            ):
                raise ValueError(
                    "Reusable session requires the same agent kind/tools, packages and network"
                )
            from pyforger.agent_diagnostics import credential_values

            self._credential_values.update(credential_values([agent]))
            call_started = time.monotonic()
            previous = self._diagnostics.get("current_trajectory")
            if previous is not None:
                self._diagnostics.setdefault("session_calls", []).append(previous)
            current = {
                "call_id": uuid.uuid4().hex,
                "turn_id": None,
                "conversation_id": None,
                "steps": [],
                "incomplete": True,
                "phase": "call_setup",
                "started_at": time.time(),
                "no_current_history_reason": "turn_not_started",
            }
            self._diagnostics["current_trajectory"] = current
            self._diagnostics["trajectory"] = current
            try:
                if self._agent is None:
                    self._agent, self._policy = agent, policy
                    self._budget = _DeadlineBudget(agent, task, self._deadline)
                    self._budget.remaining_s()
                    self._runner = self._initialize(task, agent)
                self._budget.turn_deadline = min(
                    self._budget.deadline, call_started + task.timeout_s
                )
                self._budget.remaining_s()
                result = self._runner(task, agent, validate, max_repairs)
                self._agent = agent
                self._budget.remaining_s()
                return result
            except (ValidationExhausted, GenerationInconclusive):
                try:
                    self._budget.remaining_s()
                except BaseException as exc:
                    self._close_after_error(exc)
                    raise
                raise
            except BaseException as exc:
                self._close_after_error(exc)
                raise
        finally:
            self._lock.release()

    def _initialize(self, task: WorkspaceTask, agent: AgentSpec) -> Any:
        raise NotImplementedError


class LocalSession(ReusableSession):
    def _initialize(self, task: WorkspaceTask, agent: AgentSpec) -> Any:
        from pyforger.agent_session import run_validated_session
        from pyforger.local_backend import _collect, _materialize, _ScriptSession
        from pyforger.measurements import Measurement

        root = Path(
            self._stack.enter_context(
                tempfile.TemporaryDirectory(prefix="forger-local-session-")
            )
        )
        _materialize(root, dict(task.files))
        if task.packages:
            self._backend._install(
                root, replace(task, timeout_s=self._budget.remaining_s())
            )

        def llm_for(spec: AgentSpec) -> Any:
            from pydantic import SecretStr

            from pyforger.agent_session import deadline_llm
            from pyforger.local_agent import agent_llm

            assert spec.model is not None
            llm = agent_llm(spec.model)
            if spec.api_key_env:
                llm = llm.model_copy(
                    update={"api_key": SecretStr(os.environ[spec.api_key_env])}
                )
            return deadline_llm(llm, self._budget)

        session = (
            self._backend._sdk_session(root, task, agent, budget=self._budget)
            if agent.kind == "openhands-sdk"
            else _ScriptSession(root, agent, self._budget)
        )
        self._stack.callback(session.close)

        def run(
            task: WorkspaceTask, next_agent: AgentSpec, validate: Any, max_repairs: int
        ) -> AgentRunResult:
            if agent.kind == "openhands-sdk":
                from pyforger.agent_diagnostics import sanitizer

                journal = getattr(session, "_forger_journal", None)
                if journal is not None:
                    journal.clean = sanitizer([], secrets=self._credential_values)
                session.switch_llm(llm_for(next_agent))
            _materialize(root, dict(task.files))
            try:
                result = asyncio.run(
                    run_validated_session(
                        session,
                        instruction=self._backend._instruction(root, task),
                        collect=lambda: _collect(root, task.outputs),
                        outputs=task.outputs,
                        validate=validate
                        if validate is not None
                        else lambda _: Measurement(passed=True),
                        max_repairs=max_repairs,
                        budget=self._budget,
                    )
                )
            except BaseException:
                if agent.kind == "openhands-sdk":
                    retained = self._backend._trajectory(session, next_agent)
                    current = self._diagnostics["current_trajectory"]
                    current.pop("steps", None)
                    current.update(retained, incomplete=True, phase="agent_turn")
                raise
            retained = (
                self._backend._trajectory(session, next_agent)
                if agent.kind == "openhands-sdk"
                else None
            )
            return AgentRunResult(
                status="completed",
                files=result.output,
                validation_attempts=result.attempts if validate is not None else (),
                trajectory=retained,
                diagnostics={
                    "iterations": self._budget.iterations,
                    "session_id": str(getattr(session, "id", "script")),
                },
            )

        return run


class SailSession(ReusableSession):
    def _initialize(self, task: WorkspaceTask, agent: AgentSpec) -> Any:
        import sail

        from pyforger.measurements import Measurement
        from pyforger.sail_sdk import remote_sdk_session
        from pyforger.sail_transport import SailTransport, egress_policy, workspace_path
        from pyforger.structured import generate_validated

        if any(
            key != "load_skills" or value is not False
            for key, value in agent.kwargs.items()
        ):
            raise ValueError(
                "Sail agents do not accept SDK overrides that could introduce host tools"
            )
        for name in (*task.files, *task.outputs):
            workspace_path(name)
        image = sail.Image.from_registry(self._backend.image)
        if agent.kind == "openhands-sdk":
            from pyforger.sail_sdk import SDK_REQUIREMENT, TOOLS_REQUIREMENT

            image = image.pip_install(SDK_REQUIREMENT, TOOLS_REQUIREMENT)
        if task.packages:
            image = image.pip_install(*task.packages)
        box = create_sailbox(
            sail.Sailbox.create,
            budget=self._budget,
            app=sail.App.find(self._backend.app, mint_if_missing=True),
            image=image,
            name="forger-" + uuid.uuid4().hex,
            egress_policy=egress_policy(task.network),
            max_lifetime_seconds=max(60, math.ceil(self._budget.remaining_s()) + 60),
        )
        self._diagnostics.update(sailbox_id=box.sailbox_id, cleanup="pending")

        def terminate() -> None:
            try:
                box.terminate()
                self._diagnostics["cleanup"] = "terminated"
            except BaseException:
                self._diagnostics["cleanup"] = "failed"
                raise

        self._stack.callback(terminate)
        transport = SailTransport(box, self._budget)
        setup = transport.run(
            [
                "bash",
                "-c",
                "mkdir -p /work && chown 65534:65534 /work && chmod 755 /work",
            ]
        )
        if setup.exit_code:
            raise RuntimeError("Cannot initialize Sail workspace")
        sdk_run = None
        if agent.kind == "openhands-sdk":
            sdk_run = self._stack.enter_context(
                remote_sdk_session(
                    self._backend, transport, task, agent, None, 0, self._diagnostics
                )
            )

        def run(
            task: WorkspaceTask, next_agent: AgentSpec, validate: Any, max_repairs: int
        ) -> AgentRunResult:
            for name in (*task.files, *task.outputs):
                workspace_path(name)
            transport.files(dict(task.files), write=True, replace=True)
            if sdk_run is not None:
                files, attempts = sdk_run(task, validate, max_repairs, next_agent)
            else:

                def generate(instruction: str) -> dict[str, bytes]:
                    self._budget.consume_step()
                    if agent.kind == "oracle":
                        result = transport.command(
                            agent.solution or "", feedback=instruction
                        )
                        if result.exit_code:
                            raise RuntimeError("Agent script failed")
                    return transport.files(
                        {name: b"" for name in task.outputs}, write=False
                    )

                def check(files: dict[str, bytes]) -> Any:
                    missing = [name for name in task.outputs if name not in files]
                    value = (
                        Measurement(
                            passed=False, reason=f"Missing declared outputs: {missing}"
                        )
                        if missing
                        else validate(files)
                    )
                    self._budget.remaining_s()
                    return value

                if validate is None:
                    files, attempts = generate(task.instruction), ()
                else:
                    result = generate_validated(
                        task.instruction,
                        generate=generate,
                        validate=check,
                        revise=lambda request, attempt: (
                            "Repair the outputs. Validation feedback:\n"
                            + attempt.measurement.model_dump_json()
                        ),
                        max_attempts=max_repairs + 1,
                    )
                    files, attempts = result.output, result.attempts
            self._diagnostics["iterations"] = self._budget.iterations
            return AgentRunResult(
                status="completed",
                files=files,
                validation_attempts=attempts,
                diagnostics=dict(self._diagnostics),
                trajectory=self._diagnostics.get("trajectory"),
            )

        return run

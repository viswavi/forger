"""Local backend: agents and retained programs run on this host.

The SDK agent runs in a temporary workspace with terminal and file tools;
that is host-trust execution, not a security boundary. Retained programs
replay through the restricted-Python worker without credentials, as before.
Deterministic agent kinds (``oracle``, ``nop``) run without any model.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from pyforger.agent_backend import (
    AgentRunResult,
    AgentSpec,
    WorkspaceTask,
    reward_passed,
)
from pyforger.agent_backends import require_learner_runtime
from pyforger.local_agent import _execute, agent_llm
from pyforger.measurements import Measurement


def _workspace_text(root: Path, text: str) -> str:
    return re.sub(r"(?<![\w/.:@-])/work(?=/|$|[\s\"'])", lambda _: str(root), text)


def _script_env(root: Path) -> dict[str, str]:
    return {
        "PATH": str(root / ".venv" / "bin") + ":/usr/local/bin:/usr/bin:/bin",
        "HOME": str(root),
    }


def _materialize(
    root: Path, files: dict[str, str | bytes], *, workspace: Path | None = None
) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        if path.suffix in {".py", ".sh"}:
            try:
                data = _workspace_text(workspace or root, data.decode()).encode()
            except UnicodeDecodeError:
                pass
        path.write_bytes(data)


def _collect(root: Path, outputs: tuple[str, ...]) -> dict[str, bytes]:
    files = {}
    for name in outputs:
        path = root / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix in {".py", ".sh"}:
            data = data.replace(str(root).encode(), b"/work")
        files[name] = data
    return files


def _run_script(
    root: Path, script: str, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
        ["bash", "-c", _workspace_text(root, script)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_script_env(root),
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )


class LocalBackend:
    """Execute agents in a host workspace and programs in the restricted worker."""

    name = "local"
    supports_science_rl_tools = True
    supports_inner_loop_validation = True
    supports_sdk_condensation = True

    def cache_key(self) -> object:
        """Return the local executor identity for cache keys."""
        return ("local-backend-v1", sys.version_info[:2])

    def execute_program(
        self, code: str, item: dict[str, Any], timeout_s: float
    ) -> list[str]:
        """Replay retained source without giving the worker host credentials."""
        return _execute(code, item, timeout_s)

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Materialize the workspace and run the agent in-process."""
        from pyforger.agent_session import SessionBudget

        require_learner_runtime(self, agent)
        budget = SessionBudget(agent.max_iterations, task.timeout_s)
        with tempfile.TemporaryDirectory(prefix="forger-local-agent-") as directory:
            root = Path(directory)
            _materialize(root, dict(task.files))
            if agent.kind == "oracle":
                _materialize(root, dict(task.solution_files))
            if task.packages:
                self._install(root, replace(task, timeout_s=budget.remaining_s()))
            status = "completed"
            diagnostics: dict[str, Any] = {}
            try:
                if agent.kind == "oracle":
                    result = _run_script(
                        root, agent.solution or "", budget.remaining_s()
                    )
                    diagnostics["stdout"] = result.stdout[-4000:]
                    diagnostics["exit_code"] = result.returncode
                    if result.returncode:
                        status = "failed"
                        diagnostics["stderr"] = result.stderr[-4000:]
                elif agent.kind == "openhands-sdk":
                    diagnostics.update(self._run_sdk(root, task, agent, budget))
            except subprocess.TimeoutExpired:
                status = "timed_out"
            passed = None
            failure_kind = None
            if task.verifier is not None and status == "completed":
                _materialize(
                    root / "verifier", dict(task.verifier_files), workspace=root
                )
                reward = root / "reward.txt"
                reward.unlink(missing_ok=True)
                result = _run_script(root, task.verifier, budget.remaining_s())
                diagnostics["verifier_exit_code"] = result.returncode
                diagnostics["verifier_stderr"] = result.stderr[-4000:]
                if result.returncode:
                    status = "failed"
                    failure_kind = "infrastructure"
                else:
                    passed = reward.is_file() and reward_passed(reward.read_bytes())
            return AgentRunResult(
                status=status,
                files=_collect(root, task.outputs),
                passed=passed,
                diagnostics=diagnostics,
                failure_kind=failure_kind,
                trajectory=diagnostics.get("trajectory"),
            )

    def run_agent_validated(
        self,
        task: WorkspaceTask,
        agent: AgentSpec,
        *,
        validate: Callable[[dict[str, bytes]], Measurement],
        max_repairs: int = 2,
    ) -> AgentRunResult:
        """Repair files in one SDK conversation and workspace with host validation."""
        from pyforger.agent_session import SessionBudget, run_validated_session

        if type(max_repairs) is not int or max_repairs < 0:
            raise ValueError("max_repairs must be a nonnegative integer")
        if task.verifier is not None:
            raise ValueError("Validated generation uses validate, not a task verifier")
        require_learner_runtime(self, agent)
        budget = SessionBudget(agent.max_iterations, task.timeout_s)
        with tempfile.TemporaryDirectory(prefix="forger-local-agent-") as directory:
            root = Path(directory)
            _materialize(root, dict(task.files))
            if agent.kind == "oracle":
                _materialize(root, dict(task.solution_files))
            if task.packages:
                self._install(root, replace(task, timeout_s=budget.remaining_s()))
            session = (
                self._sdk_session(root, task, agent, budget=budget)
                if agent.kind == "openhands-sdk"
                else _ScriptSession(root, agent, budget)
            )
            try:
                result = asyncio.run(
                    run_validated_session(
                        session,
                        instruction=self._instruction(root, task),
                        collect=lambda: _collect(root, task.outputs),
                        outputs=task.outputs,
                        validate=validate,
                        max_repairs=max_repairs,
                        budget=budget,
                    )
                )
                retained_trajectory = (
                    self._trajectory(session, agent)
                    if agent.kind == "openhands-sdk"
                    else None
                )
                return AgentRunResult(
                    status="completed",
                    files=result.output,
                    validation_attempts=result.attempts,
                    trajectory=retained_trajectory,
                    diagnostics={
                        "iterations": budget.iterations,
                        "session_id": str(getattr(session, "id", "script")),
                        **(
                            {"trajectory": retained_trajectory}
                            if agent.kind == "openhands-sdk"
                            else {}
                        ),
                    },
                )
            finally:
                session.close()

    @staticmethod
    def _trajectory(session: Any, agent: AgentSpec) -> dict[str, Any]:
        from pyforger.agent_diagnostics import trajectory

        return trajectory(session.state.events, agent)

    def _install(self, root: Path, task: WorkspaceTask) -> None:
        deadline = time.monotonic() + task.timeout_s
        venv = root / ".venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            check=True,
            capture_output=True,
            timeout=task.timeout_s,
        )
        subprocess.run(
            [str(venv / "bin" / "pip"), "install", "--no-cache-dir", *task.packages],
            check=True,
            capture_output=True,
            timeout=max(0.001, deadline - time.monotonic()),
        )

    @staticmethod
    def _instruction(root: Path, task: WorkspaceTask) -> str:
        python = root / ".venv" / "bin" / "python"
        prefix = (
            f"Use {python} for Python commands; task packages are installed there.\n"
            if task.packages
            else ""
        )
        return prefix + _workspace_text(root, task.instruction)

    def _sdk_session(
        self, root: Path, task: WorkspaceTask, agent: AgentSpec, *, budget: Any = None
    ) -> Any:
        from openhands.sdk import Agent, AgentContext, Conversation
        from openhands.sdk.tool import Tool
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool

        assert agent.model is not None
        agent_type = Agent
        if budget is not None:
            from pyforger.agent_session import BudgetedAgent

            agent_type = BudgetedAgent
        require_learner_runtime(self, agent)
        tools = [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]
        defaults = ["FinishTool"]
        kwargs = dict(agent.kwargs)
        if agent.runtime is not None and agent.runtime.tools == "science-rl":
            if kwargs.keys() & {"tools", "include_default_tools", "agent_context"}:
                raise ValueError("Agent kwargs conflict with the explicit tool profile")
            tools.append(Tool(name=TaskTrackerTool.name))
            defaults.append("ThinkTool")
            kwargs["agent_context"] = AgentContext(
                load_user_skills=False,
                load_public_skills=False,
                load_project_skills=False,
            )
        llm = agent_llm(agent.model)
        from pyforger._sdk_runtime import sdk_history_kwargs, sdk_iteration_limit

        kwargs.update(sdk_history_kwargs(agent, llm))
        sdk_agent: Any = agent_type(
            llm=llm,
            tools=tools,
            include_default_tools=defaults,
            **kwargs,
        )
        if budget is not None:
            sdk_agent.attach_budget(budget)
        return Conversation(
            agent=sdk_agent,
            workspace=root,
            visualizer=None,
            max_iteration_per_run=sdk_iteration_limit(agent.max_iterations),
        )

    def _run_sdk(
        self, root: Path, task: WorkspaceTask, agent: AgentSpec, budget: Any
    ) -> dict[str, Any]:
        session = self._sdk_session(root, task, agent, budget=budget)
        try:
            session.send_message(self._instruction(root, task))
            asyncio.run(asyncio.wait_for(session.arun(), budget.remaining_s()))
            status = session.state.execution_status
            if getattr(status, "value", status) != "finished":
                raise RuntimeError(f"Agent execution did not finish: {status}")
            from pyforger.agent_diagnostics import trajectory

            return {"trajectory": trajectory(session.state.events, agent)}
        finally:
            session.close()


class _ScriptSession:
    """Run deterministic oracle turns in the same workspace, without an LLM."""

    def __init__(self, root: Path, agent: AgentSpec, budget: Any) -> None:
        self.root, self.agent, self.budget = root, agent, budget
        self.state = self
        self.execution_status = "idle"
        self.feedback = ""

    def send_message(self, message: str) -> None:
        self.feedback = message
        self.execution_status = "idle"

    async def arun(self) -> None:
        self.budget.consume_step()
        if self.agent.kind == "oracle":
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                _workspace_text(self.root, self.agent.solution or ""),
                cwd=self.root,
                env={
                    **_script_env(self.root),
                    "FORGER_FEEDBACK": self.feedback,
                },
                start_new_session=True,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await process.communicate()
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.communicate()
            if process.returncode:
                raise RuntimeError(
                    f"Agent script failed: {stderr.decode(errors='replace')[-4000:]}"
                )
        self.execution_status = "finished"

    def close(self) -> None:
        pass

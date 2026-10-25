"""Sail backend: remote SDK conversation and isolated unprivileged commands.

Requires sail 0.14. PUBLIC uses Sail's public internet routing, which rejects
private destinations and has no IPv6 internet support. MODEL/OFFLINE use
no_network, including denial of the otherwise exempt Sail API destination.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from typing import Any

from pyforger.agent_backend import (
    AgentRunResult,
    AgentSpec,
    Network,
    ProgramValidationError,
    WorkspaceTask,
    reward_passed,
)
from pyforger.agent_backends import require_learner_runtime
from pyforger.agent_diagnostics import trajectory
from pyforger.measurements import Measurement
from pyforger.sail_sdk import SDK_VERSION


class _OracleExecutionFailed(Exception):
    """An executed oracle command returned a nonzero exit status."""


class SailBackend:
    """Run SDK supervisors and command workspaces in separate disposable VMs."""

    name = "sail"
    supports_inner_loop_validation = True
    supports_sdk_condensation = True

    def __init__(
        self, *, app: str = "forger", image: str = "docker.io/library/python:3.12-slim"
    ) -> None:
        """Select the Sail app and public registry base image."""
        self.app = app
        self.image = image

    def cache_key(self) -> object:
        """Identify functional execution configuration, excluding credentials."""
        return (
            "sail-backend-v4-native-terminal-sdk-" + SDK_VERSION,
            self.app,
            self.image,
        )

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Run one agent turn; verifier material is withheld until it completes."""
        return self._run(task, agent, None, 0)

    def run_agent_validated(
        self,
        task: WorkspaceTask,
        agent: AgentSpec,
        *,
        validate: Callable[[dict[str, bytes]], Measurement],
        max_repairs: int = 2,
    ) -> AgentRunResult:
        """Validate exact collected bytes on the host in one persistent session."""
        if type(max_repairs) is not int or max_repairs < 0:
            raise ValueError("max_repairs must be a nonnegative integer")
        return self._run(task, agent, validate, max_repairs)

    def _run(
        self, task: WorkspaceTask, agent: AgentSpec, validate: Any, max_repairs: int
    ) -> AgentRunResult:
        require_learner_runtime(self, agent)
        import sail

        from pyforger.sail_transport import SailTransport, egress_policy, workspace_path
        from pyforger.session_budget import SessionBudget
        from pyforger.structured import generate_validated

        if any(
            key != "load_skills" or value is not False
            for key, value in agent.kwargs.items()
        ):
            raise ValueError(
                "Sail agents do not accept SDK overrides "
                "that could introduce host tools"
            )
        for name in (
            *task.files,
            *task.outputs,
            *task.solution_files,
            *task.verifier_files,
        ):
            workspace_path(name)
        budget = SessionBudget(agent.max_iterations, task.timeout_s)
        image = sail.Image.from_registry(self.image)
        if agent.kind == "openhands-sdk":
            from pyforger.sail_sdk import SDK_REQUIREMENT, TOOLS_REQUIREMENT

            image = image.pip_install(SDK_REQUIREMENT, TOOLS_REQUIREMENT)
        if task.packages:
            image = image.pip_install(*task.packages)
        box = sail.Sailbox.create(
            app=sail.App.find(self.app, mint_if_missing=True),
            image=image,
            image_build_timeout=max(1, math.ceil(budget.remaining_s())),
            name="forger-" + uuid.uuid4().hex,
            egress_policy=egress_policy(task.network),
            timeout=budget.remaining_s(),
            max_lifetime_seconds=max(60, math.ceil(task.timeout_s) + 60),
        )
        diagnostics: dict[str, Any] = {
            "sailbox_id": box.sailbox_id,
            "cleanup": "pending",
        }
        failed_session = False
        try:
            transport = SailTransport(box, budget)
            setup = transport.run(
                [
                    "bash",
                    "-c",
                    "mkdir -p /work && chown 65534:65534 /work && chmod 755 /work",
                ]
            )
            if setup.exit_code:
                raise RuntimeError("Cannot initialize Sail workspace")
            transport.files(dict(task.files), write=True)
            if agent.kind == "oracle":
                transport.files(dict(task.solution_files), write=True)

            def collect() -> dict[str, bytes]:
                return transport.files(
                    {name: b"" for name in task.outputs}, write=False
                )

            attempts: Any = ()
            if agent.kind == "openhands-sdk":
                try:
                    files, attempts = self._sdk(
                        transport, task, agent, validate, max_repairs, diagnostics
                    )
                except Exception as exc:
                    if diagnostics.get("execution_status") in (None, "finished"):
                        raise
                    failed_session = True
                    diagnostics["iterations"] = budget.iterations
                    diagnostics["error"] = trajectory(
                        [{"type": type(exc).__name__, "message": str(exc)}], agent
                    )["steps"][0]
                    steps = diagnostics.get("trajectory", {}).get("steps", [])
                    errors = [
                        e for e in steps if e.get("kind") == "ConversationErrorEvent"
                    ]
                    limited = (
                        diagnostics.get("budget_exhausted", False)
                        or diagnostics.get("trajectory", {}).get("last_error_code")
                        == "MaxIterationsReached"
                        or bool(
                            errors and errors[-1].get("code") == "MaxIterationsReached"
                        )
                    )
                    return AgentRunResult(
                        status=(
                            "timed_out" if isinstance(exc, TimeoutError) else "failed"
                        ),
                        failure_kind=(
                            "budget"
                            if limited or isinstance(exc, TimeoutError)
                            else "execution"
                            if diagnostics.get("execution_status") == "stuck"
                            else "infrastructure"
                        ),
                        diagnostics=diagnostics,
                        trajectory=diagnostics.get("trajectory"),
                    )
            else:

                def generate(instruction: str) -> dict[str, bytes]:
                    budget.consume_step()
                    if agent.kind == "oracle":
                        result = transport.command(
                            agent.solution or "",
                            feedback=(
                                "" if instruction == task.instruction else instruction
                            ),
                        )
                        diagnostics["stdout"] = result.stdout[-4000:]
                        diagnostics["stderr"] = result.stderr[-4000:]
                        diagnostics["exit_code"] = result.exit_code
                        if result.exit_code:
                            raise _OracleExecutionFailed
                    return collect()

                if validate is None:
                    files = generate(task.instruction)
                else:

                    def check(files: dict[str, bytes]) -> Measurement:
                        missing = [name for name in task.outputs if name not in files]
                        measurement = (
                            Measurement(
                                passed=False,
                                reason=f"Missing declared outputs: {missing}",
                            )
                            if missing
                            else validate(files)
                        )
                        budget.remaining_s()
                        return measurement

                    validated = generate_validated(
                        task.instruction,
                        generate=generate,
                        validate=check,
                        revise=lambda request, attempt: (
                            "Write corrected outputs. Validation feedback:\n"
                            + attempt.measurement.reason
                        ),
                        max_attempts=max_repairs + 1,
                    )
                    files, attempts = validated.output, validated.attempts
            passed = None
            status = "completed"
            failure_kind = None
            if task.verifier is not None:
                # Terminate detached agent processes before revealing verifier material.
                stopped = transport.run(
                    [
                        "python3",
                        "-I",
                        "-c",
                        "import os, signal, time\n"
                        "from pathlib import Path\n"
                        "deadline=time.monotonic()+5\n"
                        "while True:\n"
                        " pids=[]\n"
                        " for p in Path('/proc').iterdir():\n"
                        "  try:\n"
                        "   if not p.name.isdigit(): continue\n"
                        "   if p.stat().st_uid != 65534: continue\n"
                        "   if (p/'stat').read_text().rsplit(') ', 1)[1][0] != 'Z':\n"
                        "    os.kill(int(p.name), signal.SIGSTOP)\n"
                        "    pids.append(int(p.name))\n"
                        "  except (ProcessLookupError, FileNotFoundError): pass\n"
                        " for pid in pids:\n"
                        "  try: os.kill(pid, signal.SIGKILL)\n"
                        "  except ProcessLookupError: pass\n"
                        " if not pids: break\n"
                        " if time.monotonic()>deadline:\n"
                        "  raise RuntimeError('Agent processes did not stop')\n"
                        " time.sleep(.05)",
                    ]
                )
                if stopped.exit_code:
                    raise RuntimeError(
                        "Cannot terminate agent processes before verification"
                    )
                # No generation/repair turn occurs after hidden material is staged.
                transport.files(
                    {
                        "verifier/" + name: data
                        for name, data in task.verifier_files.items()
                    },
                    write=True,
                )
                # Do not trust an agent-written reward from before verification.
                clean = transport.run(
                    [
                        "python3",
                        "-I",
                        "-c",
                        "import os; p='/work/reward.txt'; "
                        "os.unlink(p) if os.path.lexists(p) else None",
                    ]
                )
                if clean.exit_code:
                    raise RuntimeError("Cannot clear pre-verification reward")
                result = transport.command(task.verifier)
                diagnostics["verifier_exit_code"] = result.exit_code
                diagnostics["verifier_stderr"] = result.stderr[-4000:]
                reward = transport.files({"reward.txt": b""}, write=False)
                if result.exit_code:
                    status = "failed"
                    failure_kind = "infrastructure"
                else:
                    passed = reward_passed(reward.get("reward.txt", b""))
            if validate is None and task.verifier is not None:
                files = collect()
            diagnostics["iterations"] = budget.iterations
            return AgentRunResult(
                status=status,
                files=files,
                passed=passed,
                diagnostics=diagnostics,
                validation_attempts=attempts,
                failure_kind=failure_kind,
                trajectory=diagnostics.get("trajectory"),
            )
        except _OracleExecutionFailed:
            diagnostics["iterations"] = budget.iterations
            return AgentRunResult(
                status="failed",
                files=collect(),
                failure_kind="execution",
                diagnostics=diagnostics,
            )
        finally:
            try:
                box.terminate()
                diagnostics["cleanup"] = "terminated"
            except Exception as exc:
                diagnostics["cleanup"] = "failed"
                diagnostics["cleanup_error_type"] = type(exc).__name__
                if not failed_session:
                    raise

    def _sdk(
        self,
        transport: Any,
        task: WorkspaceTask,
        spec: AgentSpec,
        validate: Any,
        max_repairs: int,
        diagnostics: dict[str, Any],
    ) -> tuple[Any, Any]:
        from pyforger.sail_sdk import run_remote_sdk

        return run_remote_sdk(
            self, transport, task, spec, validate, max_repairs, diagnostics
        )

    def execute_program(
        self, code: str, item: dict[str, Any], timeout_s: float
    ) -> list[str]:
        """Replay retained code offline in a fresh microVM with no model client."""
        worker = r"""
import ast, contextlib, io, json
request = json.load(open('request.json'))
try:
    tree = ast.parse(request['code'])
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not functions: raise ValueError('Source must define an entry function')
    namespace = {'__name__': 'forger_dynamic'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, '<retained-source>', 'exec'), namespace)
        output = namespace[functions[-1].name](request['item'])
    if not isinstance(output, list) or any(not isinstance(v, str) for v in output):
        raise TypeError('The pipeline must return a list of strings')
    result = {'output': output}
except Exception as exc:
    result = {'error': str(exc), 'type': type(exc).__name__}
open('result.json', 'w').write(json.dumps(result))
"""
        result = self.run_agent(
            WorkspaceTask(
                instruction="Replay retained program",
                files={
                    "worker.py": worker,
                    "request.json": json.dumps({"code": code, "item": item}),
                },
                outputs=("result.json",),
                network=Network.OFFLINE,
                timeout_s=timeout_s,
            ),
            AgentSpec(kind="oracle", solution="python3 -I worker.py"),
        )
        value = json.loads(result.files["result.json"])
        if "error" in value:
            raise ProgramValidationError(value["error"])
        output = value["output"]
        if not isinstance(output, list) or any(not isinstance(v, str) for v in output):
            raise TypeError("The pipeline must return a list of strings")
        return output

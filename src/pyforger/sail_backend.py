"""Direct Sail backend: host SDK conversation, remote unprivileged commands.

Requires sail 0.14. PUBLIC uses Sail's public internet routing, which rejects
private destinations and has no IPv6 internet support. MODEL/OFFLINE use
no_network, including denial of the otherwise exempt Sail API destination.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
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
from pyforger.agent_diagnostics import trajectory
from pyforger.measurements import Measurement


class SailBackend:
    """Execute in disposable Sail microVMs, without Harbor or remote agent servers."""

    name = "sail"
    supports_inner_loop_validation = True

    def __init__(
        self, *, app: str = "forger", image: str = "docker.io/library/python:3.12-slim"
    ) -> None:
        """Select the Sail app and public registry base image."""
        self.app = app
        self.image = image

    def cache_key(self) -> object:
        """Identify functional execution configuration, excluding credentials."""
        return ("sail-backend-v1", self.app, self.image)

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
        import sail

        from pyforger.agent_session import SessionBudget
        from pyforger.sail_transport import SailTransport, egress_policy, workspace_path
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
                    limited = bool(
                        errors and errors[-1].get("code") == "MaxIterationsReached"
                    )
                    return AgentRunResult(
                        status=(
                            "timed_out" if isinstance(exc, TimeoutError) else "failed"
                        ),
                        failure_kind=(
                            "budget"
                            if limited or isinstance(exc, TimeoutError)
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
                        if result.exit_code:
                            raise RuntimeError(
                                f"Sail agent command failed ({result.exit_code}): "
                                f"{result.stderr[-2000:]}"
                            )
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
        from openhands.sdk import AgentContext
        from openhands.sdk.conversation.impl.local_conversation import LocalConversation
        from openhands.sdk.tool import FinishTool, Tool, register_tool, resolve_tool

        from pyforger.agent_session import BudgetedAgent, run_validated_session
        from pyforger.local_agent import agent_llm
        from pyforger.sail_transport import TRANSPORTS, SailCommandTool

        class SailAgent(BudgetedAgent):
            def _initialize(self, state: Any) -> None:
                if self._initialized:
                    return
                # Avoid SDK auto-discovery of host vision profiles and tools.
                tools = list(resolve_tool(self.tools[0], state))
                tools.extend(FinishTool.create(state))
                self._tools = {tool.name: tool for tool in tools}
                self._initialized = True

        class SealedConversation(LocalConversation):
            def _ensure_plugins_loaded(self) -> None:
                # SDK 1.55 discovers ambient user plugins unconditionally. These
                # can add host tools/hooks, so remote-only sessions opt out.
                self._plugins_loaded = True

        assert spec.model is not None
        token = uuid.uuid4().hex
        TRANSPORTS[token] = transport
        register_tool(SailCommandTool.name, SailCommandTool)
        try:
            with tempfile.TemporaryDirectory(prefix="forger-sail-host-") as directory:
                llm = agent_llm(spec.model)
                if spec.api_key_env:
                    from pydantic import SecretStr

                    llm = llm.model_copy(
                        update={"api_key": SecretStr(os.environ[spec.api_key_env])}
                    )
                agent = SailAgent(
                    llm=llm,
                    tools=[Tool(name=SailCommandTool.name, params={"token": token})],
                    include_default_tools=["FinishTool"],
                    agent_context=AgentContext(
                        load_user_skills=False,
                        load_public_skills=False,
                        load_project_skills=False,
                    ),
                )
                agent.attach_budget(transport.budget)
                session: Any = SealedConversation(
                    agent=agent,
                    workspace=directory,
                    visualizer=None,
                    max_iteration_per_run=spec.max_iterations,
                )
                try:
                    diagnostics["conversation_id"] = str(session.id)

                    async def collect() -> dict[str, bytes]:
                        diagnostics.setdefault("snapshots", []).append(
                            {
                                "conversation_id": str(session.id),
                                "sailbox_id": transport.box.sailbox_id,
                                "events": len(session.state.events),
                            }
                        )
                        return await asyncio.to_thread(
                            transport.files,
                            {name: b"" for name in task.outputs},
                            write=False,
                        )

                    async def run() -> tuple[Any, Any]:
                        if validate is not None:
                            result = await run_validated_session(
                                session,
                                instruction=task.instruction,
                                collect=collect,
                                outputs=task.outputs,
                                validate=validate,
                                max_repairs=max_repairs,
                                budget=transport.budget,
                            )
                            return result.output, result.attempts
                        session.send_message(task.instruction)
                        await asyncio.wait_for(
                            session.arun(), transport.budget.remaining_s()
                        )
                        status = session.state.execution_status
                        if getattr(status, "value", status) != "finished":
                            raise RuntimeError(
                                f"Agent execution did not finish: {status}"
                            )
                        return await collect(), ()

                    return asyncio.run(run())
                finally:
                    status = session.state.execution_status
                    diagnostics["execution_status"] = getattr(status, "value", status)
                    try:
                        diagnostics["trajectory"] = trajectory(
                            session.state.events, spec
                        )
                    finally:
                        try:
                            session.close()
                            diagnostics["session_cleanup"] = "closed"
                        except Exception as exc:
                            diagnostics["session_cleanup"] = "failed"
                            diagnostics["session_cleanup_error_type"] = type(
                                exc
                            ).__name__
                            if diagnostics["execution_status"] == "finished":
                                raise
        finally:
            TRANSPORTS.pop(token, None)

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

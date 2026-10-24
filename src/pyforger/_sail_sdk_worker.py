"""Private persistent SDK worker. Credentials arrive only through the in-memory wire."""

from __future__ import annotations

import faulthandler
import json
import os
import re
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast


def main(wire: Any) -> None:
    """Run the persistent SDK worker loop until the controller closes it."""
    # SDK/provider logs must not become protocol frames or leak credentials.
    # Redact before writing to disk, including provider and native fd output.
    log_path = os.environ.get("FORGER_WORKER_LOG", "/tmp/forger-worker.log")
    try:
        sink = os.fdopen(
            os.open(
                log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600
            ),
            "w",
            buffering=1,
        )
    except OSError:
        sink = open(os.devnull, "w")

    class RedactedLog:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.pending = ""

        def write(self, text: str) -> int:
            with self.lock:
                self.pending += text
                while "\n" in self.pending:
                    line, self.pending = self.pending.split("\n", 1)
                    for secret in wire.secrets:
                        line = line.replace(secret, "[REDACTED]")
                    line = re.sub(
                        r"(?i)(Bearer|Basic)\s+[^\s\"']+", r"\1 [REDACTED]", line
                    )
                    sink.write(line + "\n")
                # Avoid unbounded memory on a provider's unterminated log line;
                # never spill a potentially partial credential to disk.
                if len(self.pending) > 1024 * 1024:
                    self.pending = "[oversize log line omitted]"
            return len(text)

        def flush(self) -> None:
            sink.flush()

        def isatty(self) -> bool:
            return False

    safe_log = RedactedLog()
    reader, output = os.pipe()
    os.dup2(output, 1)
    os.dup2(output, 2)
    os.close(output)

    def capture() -> None:
        with os.fdopen(reader, "r", errors="replace") as stream:
            for line in stream:
                safe_log.write(line)
                safe_log.flush()

    threading.Thread(target=capture, name="sdk-logs", daemon=True).start()
    sys.stderr = safe_log
    sys.stdout = safe_log

    # One process-wide dump, before the host's 60s write timeout; never recurring.
    faulthandler.enable(file=sink)
    faulthandler.dump_traceback_later(45, repeat=False, file=sink)

    @contextmanager
    def phase(name: str) -> Any:
        started = time.monotonic()
        print(
            json.dumps({"phase": name, "state": "started", "at": started}),
            file=sink,
            flush=True,
        )
        try:
            yield
        finally:
            ended = time.monotonic()
            print(
                json.dumps(
                    {
                        "phase": name,
                        "state": "finished",
                        "at": ended,
                        "duration_s": ended - started,
                    }
                ),
                file=sink,
                flush=True,
            )

    def send(value: dict[str, Any]) -> None:
        wire.send(value)

    def receive() -> dict[str, Any]:
        with phase("receive"):
            value = cast(dict[str, Any], wire.receive())
            kind = value.get("kind", "request")
            kind = (
                kind
                if kind in {"request", "turn", "configure", "close", "observation"}
                else "unknown"
            )
            print(
                json.dumps(
                    {
                        "received_kind": kind,
                        "encoded_bytes": len(json.dumps(value).encode()),
                        "at": time.monotonic(),
                    }
                ),
                file=sink,
                flush=True,
            )
            return value

    request = receive()
    deadline = time.monotonic() + request["timeout_s"]
    from _sdk_runtime import sdk_history_kwargs, sdk_iteration_limit
    from agent_diagnostics import trajectory
    from openhands.sdk import LLM, Agent, AgentContext
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.llm import LLMCallContext
    from openhands.sdk.tool import (
        Action,
        FinishTool,
        Observation,
        Tool,
        ToolDefinition,
        ToolExecutor,
        register_tool,
        resolve_tool,
    )
    from pydantic import Field

    class CommandAction(Action):
        command: str = Field(description="Bash command in the isolated /work workspace")

    class CommandObservation(Observation):
        pass

    class CommandExecutor(ToolExecutor[CommandAction, CommandObservation]):
        def __call__(self, action: Any, conversation: Any = None) -> Any:
            command = {"kind": "command", "command": action.command}
            if len(json.dumps(command).encode()) + 1 > 256 * 1024:
                raise ValueError("SDK command exceeds 256 KiB protocol frame limit")
            send(command)
            reply = receive()
            if reply.get("kind") != "observation":
                raise RuntimeError("Invalid controller response")
            return CommandObservation.from_text(reply["text"])

    class WorkspaceCommandTool(ToolDefinition[CommandAction, CommandObservation]):
        @classmethod
        def create(cls, conv_state: Any = None, **params: Any) -> list[Any]:
            return [
                cls(
                    description=(
                        "Execute bash in the isolated workspace. "
                        "Use shell commands to read and edit files."
                    ),
                    action_type=CommandAction,
                    observation_type=CommandObservation,
                    executor=CommandExecutor(),
                )
            ]

    register_tool(WorkspaceCommandTool.name, WorkspaceCommandTool)
    redaction_spec = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(api_key=request["llm"]["api_key"])
        ),
        api_key_env=None,
    )
    redaction_specs = [redaction_spec]
    iterations = 0

    class WorkerLLM(LLM):
        def generate(self, *args: Any, **kwargs: Any) -> Any:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("SDK deadline exhausted")
            kwargs["timeout"] = min(self.timeout or remaining, remaining)
            kwargs["max_retries"] = 0
            if request["disable_prompt_cache_key"]:
                kwargs["call_context"] = replace(
                    kwargs.get("call_context") or LLMCallContext(),
                    prompt_cache_key=None,
                )
            with phase("model.generate"):
                return super().generate(*args, **kwargs)

    class WorkerAgent(Agent):
        def _initialize(self, state: Any) -> None:
            if self._initialized:
                return
            tools = list(resolve_tool(self.tools[0], state))
            tools.extend(FinishTool.create(state))
            self._tools = {tool.name: tool for tool in tools}
            self._initialized = True

        @property
        def prompt_dir(self) -> str:
            import inspect
            from pathlib import Path

            return str(Path(inspect.getfile(Agent)).parent / "prompts")

        def step(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            nonlocal iterations
            if time.monotonic() >= deadline:
                raise TimeoutError("SDK deadline exhausted")
            if (
                request["max_iterations"] is not None
                and iterations >= request["max_iterations"]
            ):
                raise RuntimeError("Shared iteration budget exhausted")
            iterations += 1
            native: Any = super()
            native.step(*args, **kwargs)

    with phase("model.initialize"):
        llm = WorkerLLM(**request["llm"], stream=False)
    if llm.custom_tokenizer and llm._chat_template_tokenizer is None:
        send({"kind": "error", "error_type": "TokenizerUnavailable", "iterations": 0})
        return
    spec = SimpleNamespace(
        runtime=SimpleNamespace(**request["runtime"]) if request["runtime"] else None,
        kwargs={},
    )
    agent = WorkerAgent(
        llm=llm,
        tools=[Tool(name=WorkspaceCommandTool.name)],
        include_default_tools=["FinishTool"],
        agent_context=AgentContext(
            load_user_skills=False, load_public_skills=False, load_project_skills=False
        ),
        **sdk_history_kwargs(spec, llm),
    )

    class SealedConversation(LocalConversation):
        def _ensure_plugins_loaded(self) -> None:
            self._plugins_loaded = True

    session: Any = SealedConversation(
        agent=agent,
        workspace="/tmp",
        visualizer=None,
        max_iteration_per_run=sdk_iteration_limit(request["max_iterations"]),
    )
    emitted = 0

    def report_error(exc: Exception, **details: Any) -> None:
        error = {
            "kind": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "iterations": iterations,
            **details,
        }
        for prior_spec in redaction_specs:
            error = trajectory([error], prior_spec)["steps"][0]
        print(error["traceback"], file=sink, flush=True)
        try:
            emit_events()
        except Exception:
            safe_log.write(traceback.format_exc())
            safe_log.flush()
        send(error)

    def emit_events() -> None:
        nonlocal emitted
        for event in list(session.state.events)[emitted:]:
            retained = trajectory([event], redaction_spec)
            for prior_spec in redaction_specs:
                retained = trajectory(retained["steps"], prior_spec)
            encoded = json.dumps(retained["steps"][0])
            # Chunk individual events too: scientific observations can be large.
            for offset in range(0, len(encoded), 8000):
                send({"kind": "event_chunk", "text": encoded[offset : offset + 8000]})
            send({"kind": "event_end"})
            emitted += 1

    try:
        send({"kind": "ready"})
        while True:
            turn = receive()
            if turn["kind"] == "close":
                return
            if turn["kind"] == "configure":
                try:
                    with phase("configure.model.initialize"):
                        replacement = WorkerLLM(**turn["llm"], stream=False)
                    if (
                        replacement.custom_tokenizer
                        and replacement._chat_template_tokenizer is None
                    ):
                        raise RuntimeError("TokenizerUnavailable")
                    with phase("configure.switch_llm"):
                        session.switch_llm(replacement)
                    request["disable_prompt_cache_key"] = turn[
                        "disable_prompt_cache_key"
                    ]
                    redaction_spec = SimpleNamespace(
                        model=SimpleNamespace(
                            config=SimpleNamespace(api_key=turn["llm"]["api_key"])
                        ),
                        api_key_env=None,
                    )
                    redaction_specs.append(redaction_spec)
                    send({"kind": "configured"})
                except Exception as exc:
                    report_error(exc)
                    return
                continue
            if turn["kind"] != "turn":
                raise RuntimeError("Invalid controller request")
            try:
                if "timeout_s" in turn:
                    deadline = time.monotonic() + turn["timeout_s"]
                with phase("turn.send_message"):
                    session.send_message(turn["instruction"])
                with phase("run"):
                    session.run()
                status = session.state.execution_status
                status = getattr(status, "value", status)
                emit_events()
                send(
                    {
                        "kind": "result",
                        "status": status,
                        "iterations": iterations,
                        "conversation_id": str(session.id),
                        "events": len(session.state.events),
                    }
                )
            except Exception as exc:
                report_error(
                    exc,
                    budget_exhausted=(
                        time.monotonic() >= deadline
                        or (
                            request["max_iterations"] is not None
                            and iterations >= request["max_iterations"]
                        )
                    ),
                )
                return
    finally:
        try:
            session.close()
        except Exception:
            safe_log.write(traceback.format_exc())
            safe_log.flush()

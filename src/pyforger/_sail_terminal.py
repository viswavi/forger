"""Native terminal executor in the credential-free task sandbox."""

import os
from types import SimpleNamespace


def main(mailbox):
    status = {}
    with open("/proc/self/status") as source:
        for line in source:
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
    if (os.getuid(), os.getgid()) != (65534, 65534) or os.getgroups():
        raise PermissionError(
            "Task terminal requires isolated uid/gid and empty groups"
        )
    if status.get("NoNewPrivs") != "1" or any(
        int(status.get(key, "1"), 16)
        for key in ("CapEff", "CapPrm", "CapInh", "CapAmb", "CapBnd")
    ):
        raise PermissionError(
            "Task terminal requires no privilege escalation or capabilities"
        )
    from openhands.sdk.conversation.secret_registry import SecretRegistry
    from openhands.tools.terminal.definition import TerminalAction
    from openhands.tools.terminal.impl import TerminalExecutor

    registry = SecretRegistry()
    conversation = SimpleNamespace(state=SimpleNamespace(secret_registry=registry))
    executor = TerminalExecutor(
        working_dir="/work", terminal_type="subprocess", no_change_timeout_seconds=5
    )
    try:
        initialize = mailbox.receive()
        if initialize.get("kind") != "initialize":
            raise ValueError("Invalid terminal initialization")
        mailbox.send({"kind": "ready"})
        while True:
            frame = mailbox.receive()
            if frame["kind"] == "close":
                executor.close()
                mailbox.send({"kind": "closed"})
                return
            if frame["kind"] != "terminal":
                raise ValueError("Invalid task terminal request")
            action = TerminalAction.model_validate(frame["action"])
            if action.timeout is None:
                action = action.model_copy(update={"timeout": 10})
            observation = executor(action, conversation)
            mailbox.send(
                {
                    "kind": "terminal_observation",
                    "observation": observation.model_dump(mode="json"),
                }
            )
    finally:
        executor.close()

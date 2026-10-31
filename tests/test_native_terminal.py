"""Real native terminal behavior; container isolation suite runs this entrypoint."""

import json
import queue
import threading
import time

from pyforger._sail_terminal import main


def exercise():
    incoming, outgoing = queue.Queue(), queue.Queue()

    class Mailbox:
        receive = staticmethod(incoming.get)
        send = staticmethod(outgoing.put)

    errors = []

    def run():
        try:
            main(Mailbox())
        except BaseException as exc:
            errors.append(exc)
            outgoing.put({"error": repr(exc)})

    worker = threading.Thread(target=run)
    worker.start()
    incoming.put({"kind": "initialize"})
    assert outgoing.get(timeout=30) == {"kind": "ready"}, errors

    def command(text, **kwargs):
        incoming.put({"kind": "terminal", "action": {"command": text, **kwargs}})
        reply = outgoing.get(timeout=30)
        assert reply["kind"] == "terminal_observation", reply
        return reply["observation"]

    def text(result):
        return json.dumps(result)

    try:
        result = command(
            'id -u; id -g; test -z "$SERVICE_CREDENTIAL" && echo clean-env; cat /private/key 2>/dev/null || echo private-denied; cat /proc/1/environ 2>/dev/null || echo proc-denied',
            timeout=5,
        )
        assert "clean-env" in text(result)
        assert "private-denied" in text(result) and "proc-denied" in text(result)
        assert "service-only-sentinel" not in text(result)
        command("mkdir -p sub; cd sub", timeout=5)
        assert "/work/sub" in text(command("pwd", timeout=5))
        command(
            "cat > data <<'END'\n"
            + "x" * 100
            + "\n"
            + ("x" * 100 + "\n") * 128
            + "x" * 100
            + "\nEND",
            timeout=5,
        )
        assert "13130" in text(command("wc -c < data", timeout=5))
        command("sleep 1; echo delayed", timeout=0.2)
        time.sleep(1)
        assert "delayed" in text(command("", is_input=True, timeout=2))
        command("sleep 30", timeout=0.2)
        command("C-c", is_input=True, timeout=2)
        assert "recovered" in text(command("echo recovered", timeout=2))
        command(
            "nohup sh -c 'sleep 1; echo background > job.log' >/dev/null 2>&1 & echo $! >job.pid",
            timeout=2,
        )
        time.sleep(1.5)
        assert "background" in text(command("cat job.log", timeout=2))
        command("", reset=True, timeout=2)
        assert "reset-ok" in text(command("echo reset-ok", timeout=2))
    finally:
        incoming.put({"kind": "close"})
        assert outgoing.get(timeout=10) == {"kind": "closed"}
        worker.join(10)
    assert not worker.is_alive() and not errors, errors
    # Exercise the SDK's supported grant API directly, not a Forger wire feature.
    from types import SimpleNamespace
    from openhands.sdk.conversation.secret_registry import SecretRegistry
    from openhands.tools.terminal.definition import TerminalAction
    from openhands.tools.terminal.impl import TerminalExecutor

    registry = SecretRegistry()
    registry.update_secrets({"TASK_TOKEN": "explicit-task-sentinel"})
    executor = TerminalExecutor(working_dir="/work", terminal_type="subprocess")
    try:
        result = executor(
            TerminalAction(command="printf '%s' \"$TASK_TOKEN\"", timeout=5),
            SimpleNamespace(state=SimpleNamespace(secret_registry=registry)),
        )
        assert "explicit-task-sentinel" not in result.text
        assert "<secret-hidden>" in result.text
    finally:
        executor.close()
    print("native terminal isolation and behavior checks passed")


if __name__ == "__main__":
    exercise()


def test_native_terminal_fails_closed_without_boundary():
    import pytest

    with pytest.raises(PermissionError):
        main(None)


def test_native_terminal_container():
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    import pytest

    image = "python:3.13.17-slim-bookworm"
    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True
        ).returncode
    ):
        pytest.skip("Native isolation test requires local Docker Python3.13 image")
    root = Path(__file__).resolve().parents[1]
    deps = Path(sys.executable).parent.parent / "lib/python3.13/site-packages"
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{deps}:/deps:ro",
            "-v",
            f"{root}:/code:ro",
            "-e",
            "SERVICE_CREDENTIAL=service-only-sentinel",
            image,
            "sh",
            "-c",
            "mkdir /work /private; chmod 700 /private; echo service-only-sentinel >/private/key; "
            "chown 65534:65534 /work; setpriv --reuid=65534 --regid=65534 --clear-groups "
            "--no-new-privs --bounding-set=-all --inh-caps=-all --ambient-caps=-all "
            "env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/work PYTHONPATH=/code/src:/deps "
            "LITELLM_LOCAL_MODEL_COST_MAP=True python /code/tests/test_native_terminal.py",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr

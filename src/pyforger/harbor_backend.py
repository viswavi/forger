"""Harbor backend: agents and retained programs run in Harbor environments.

The backend translates a backend-agnostic
:class:`~pyforger.agent_backend.WorkspaceTask` into a native Harbor task,
runs the specified agent there, and collects declared outputs. It contains
no agent functionality of its own: agent identity arrives as an
:class:`~pyforger.agent_backend.AgentSpec`, and retained programs replay
through Oracle without any model.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pyforger.agent_backend import (
    AgentRunResult,
    AgentSpec,
    Network,
    ProgramValidationError,
    WorkspaceTask,
)
from pyforger.agent_backends import require_learner_runtime
from pyforger.artifacts import ArtifactRef, read_text, write_input
from pyforger.local_agent import _record_key

_NETWORK = {
    Network.OFFLINE: "no-network",
    Network.MODEL: "no-network",
    Network.PUBLIC: "public",
}


def _package(
    task: WorkspaceTask,
    agent: AgentSpec,
    root: Path,
    outputs: list[str],
    *,
    trusted_verifier_files: dict[str, bytes] | None = None,
) -> ArtifactRef:
    if any(
        not name.startswith("/work/") or name in {"/work/reward.txt", "/work/"}
        for name in outputs
    ):
        raise ValueError(
            "Harbor outputs must be explicit workspace artifacts, not reward or workspace root"
        )
    root.mkdir()
    environment = root / "environment"
    files_dir = environment / "files"
    files_dir.mkdir(parents=True)
    for name, content in task.files.items():
        path = files_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_input(path, content)
    installs = (
        "RUN pip install --no-cache-dir "
        + " ".join(shlex.quote(package) for package in task.packages)
        + "\n"
        if task.packages
        else ""
    )
    tokenizer_build = ""
    if agent.model is not None:
        from pyforger.sdk_config import llm_options

        tokenizer = llm_options(agent.model).get("custom_tokenizer")
        if tokenizer:
            script = (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download({tokenizer!r}, local_dir='/opt/tokenizer', "
                "allow_patterns=['*token*','*vocab*','merges.txt','config.json',"
                "'*.model','*.tiktoken','*.jinja'], token=False)"
            )
            tokenizer_build = (
                "RUN /opt/openhands-sdk-venv/bin/python -c "
                + shlex.quote(script)
                + "\n"
            )
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        + installs
        + "RUN apt-get update && apt-get install -y --no-install-recommends "
        "curl coreutils "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m venv /opt/openhands-sdk-venv "
        "&& /opt/openhands-sdk-venv/bin/pip install --no-cache-dir "
        "openhands-sdk==1.58.0 openhands-tools==1.58.0 fastapi==0.135.2\n"
        + tokenizer_build
        + "RUN mkdir /work && chown 65534:65534 /work\n"
        "ENV HOME=/tmp\nWORKDIR /work\nCOPY --chown=65534:65534 files /work\nUSER root\n"
    )
    (root / "instruction.md").write_text(task.instruction)
    (root / "task.toml").write_text(
        'schema_version = "1.4"\n'
        f"artifacts = {json.dumps(outputs)}\n"
        f"[agent]\ntimeout_sec = {max(1, int(task.timeout_s))}\n"
        'user = "65534"\n'
        '[verifier]\nenvironment_mode = "separate"\nuser = "root"\n'
        f"timeout_sec = {max(1, int(task.timeout_s))}\n"
        'network_mode = "no-network"\n'
        f'[environment]\nnetwork_mode = "{_NETWORK[task.network]}"\n'
        "cpus = 2\nmemory_mb = 4096\n"
    )
    if agent.kind == "oracle":
        solution = root / "solution"
        files_root = solution / "files"
        files_root.mkdir(parents=True)
        for name, content in task.solution_files.items():
            path = files_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_input(path, content)
        (solution / "solve.sh").write_text(
            "#!/bin/sh\nset -eu\ncp -r /solution/files/. /work/ 2>/dev/null || true\n"
            "cd /work\n" + (agent.solution or "") + "\n"
        )
    tests = root / "tests"
    tests.mkdir()
    trusted = tests / "trusted"
    trusted.mkdir()
    for name, body in (trusted_verifier_files or {}).items():
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
            raise ValueError("Trusted verifier filenames must be simple basenames")
        (trusted / name).write_bytes(body)
    public = tests / "public"
    public.mkdir()
    for name, content in task.files.items():
        path = public / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_input(path, content)
    (tests / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        + installs
        + "RUN mkdir -p /work /tests /logs/verifier && chmod 700 /tests /logs/verifier\n"
        "COPY . /tests\nCOPY public /work\n"
        "RUN mkdir -p /opt/trusted && cp -r /tests/trusted/. /opt/trusted/ "
        "&& chmod 755 /opt/trusted && chmod -R go-w /opt/trusted "
        "&& chmod 700 /tests /logs/verifier\n"
        "WORKDIR /tests\nUSER root\n"
    )
    if task.verifier is not None:
        verifier_dir = tests / "verifier"
        verifier_dir.mkdir()
        for name, content in task.verifier_files.items():
            path = verifier_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_input(path, content)
        (tests / "test.sh").write_text(
            "#!/bin/sh\nset -eu\numask 077\nchmod 700 /tests /logs/verifier\n"
            "rm -f /work/reward.txt /logs/verifier/reward.txt /logs/verifier/reward.json\n"
            "touch /work/reward.txt && chmod 600 /work/reward.txt\n"
            "WORKSPACE=/work VERIFIER_DIR=/tests/verifier PYTHONSAFEPATH=1 sh -c "
            + shlex.quote("cd /work\n" + task.verifier)
            + "\nif [ ! -s /logs/verifier/reward.txt ]; then "
            "cp /work/reward.txt /logs/verifier/reward.txt 2>/dev/null "
            "|| echo 0 > /logs/verifier/reward.txt; fi\n"
        )
    else:
        checks = " && ".join(f"test -f {shlex.quote(o)}" for o in outputs) or "true"
        (tests / "test.sh").write_text(
            "#!/bin/sh\nmkdir -p /logs/verifier\n"
            f"if {checks}; then echo 1; else echo 0; fi "
            "> /logs/verifier/reward.txt\n"
        )
    original = tests / "verifier-command.sh"
    (tests / "test.sh").rename(original)
    (tests / "test.sh").write_text("#!/bin/sh\nexec python3 -I /tests/launch.py\n")
    (tests / "launch.py").write_text(
        "import os, pathlib, subprocess\n"
        "os.umask(0o077)\n"
        "for name in ('/tests', '/logs/verifier'):\n"
        "    pathlib.Path(name).chmod(0o700)\n"
        "subprocess.run(['sh', '/tests/verifier-command.sh'], check=True, cwd='/tests', "
        "env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/root','PYTHONSAFEPATH':'1'})\n"
    )
    return ArtifactRef.capture(root)


def _failure_kind(outcome: dict[str, Any]) -> str | None:
    exception = outcome.get("exception") or {}
    name = exception.get("exception_type", "") if isinstance(exception, dict) else ""
    if name in {
        "AgentTimeoutError",
        "TrialTimeoutError",
        "AgentExecutionTimeoutError",
        "AgentIterationLimitError",
    }:
        return "budget"
    if name in {"AgentExecutionError", "AgentExitError"} or outcome.get(
        "exit_code"
    ) not in (None, 0, "0"):
        return "execution"
    if exception:
        return "infrastructure"
    if outcome["status"] == "timed_out":
        return "budget"
    return None if outcome["status"] == "completed" else "infrastructure"


class HarborBackend:
    """Execute agents in a Harbor environment (Docker by default)."""

    name = "harbor"
    supports_inner_loop_validation = False
    supports_sdk_condensation = True

    def __init__(
        self,
        environment: Any = None,
        *,
        retain_trials: str | Path | None = None,
        trusted_verifier_files: dict[str, bytes] | None = None,
    ) -> None:
        """Snapshot backend configuration without importing the optional SDK.

        ``retain_trials`` keeps each native trial directory (trajectory,
        verifier output, result) under that root after the run; by default
        trial directories are transient and only normalized outcomes and
        collected files survive.
        """
        self._environment = copy.deepcopy(
            environment if environment is not None else {"type": "docker"}
        )
        self._retain_trials = Path(retain_trials) if retain_trials else None
        self._trusted_verifier_files = dict(trusted_verifier_files or {})
        backend_env = (
            self._environment.get("env")
            if isinstance(self._environment, dict)
            else getattr(self._environment, "env", None)
        )
        if backend_env:
            raise ValueError("Harbor backend environments must not contain env")

    def cache_key(self) -> object:
        """Return the environment/executor identity for cache keys."""
        from pyforger.harbor import _sdk, _secrets

        _, EnvironmentConfig, _, _, _ = _sdk()
        backend = (
            self._environment.model_copy(deep=True)
            if isinstance(self._environment, EnvironmentConfig)
            else EnvironmentConfig.model_validate(copy.deepcopy(self._environment))
        ).model_dump(mode="json")
        if _secrets(backend):
            raise ValueError("Harbor backend environments must not contain credentials")
        return (
            tuple(
                (name, hashlib.sha256(body).hexdigest())
                for name, body in sorted(self._trusted_verifier_files.items())
            ),
            "harbor-backend-v2-separate-root-verifier",
            json.dumps(backend, sort_keys=True, allow_nan=False),
        )

    def _agent_config(self, agent: AgentSpec) -> Any:
        require_learner_runtime(self, agent)
        if agent.kind == "oracle":
            return {"name": "oracle"}
        if agent.kind == "nop":
            return {"name": "nop"}
        from pyforger.harbor import harbor_agent_config

        assert agent.model is not None
        config = harbor_agent_config(
            agent.model,
            max_iterations=agent.max_iterations,
            api_key_env=agent.api_key_env,
            agent_kwargs=dict(agent.kwargs),
        )
        if agent.runtime is not None and agent.runtime.condenser_max_tokens is not None:
            config.env["FORGER_CONDENSER_MAX_TOKENS"] = str(
                agent.runtime.condenser_max_tokens
            )
        return config

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Package the task natively and run the agent in Harbor."""
        require_learner_runtime(self, agent)
        from pyforger.harbor import (
            HarborRequest,
            HarborTaskRunner,
            collect_trial_files,
            trial_directory,
            trial_outcome,
        )

        outputs = ["/work/" + name for name in task.outputs]
        with tempfile.TemporaryDirectory(prefix="forger-harbor-agent-") as directory:
            root = Path(directory)
            packaged = _package(
                task,
                agent,
                root / "task",
                outputs,
                trusted_verifier_files=self._trusted_verifier_files,
            )
            runner = HarborTaskRunner(
                agent=self._agent_config(agent), environment=self._environment
            )

            async def run() -> Any:
                async with asyncio.timeout(task.timeout_s):
                    return await runner.arun(
                        HarborRequest(trial_id="agent", task=packaged), root / "native"
                    )

            try:
                rollout = asyncio.run(run())
            except TimeoutError:
                native = root / "native"
                agent_started = any(
                    native.glob("trial-*/agent/trajectory.json")
                ) or any(native.glob("trial-*/agent/exit-code.txt"))
                diagnostics: dict[str, Any] = {
                    "wall_timeout_s": task.timeout_s,
                    "phase": (
                        "agent_or_verifier" if agent_started else "startup_unresolved"
                    ),
                }
                if self._retain_trials is not None and (root / "native").exists():
                    retained = self._retain_trials / ("timeout-" + root.name)
                    shutil.copytree(root / "native", retained)
                    diagnostics["trial_dir"] = str(retained)
                return AgentRunResult(
                    status="timed_out",
                    passed=False if agent_started else None,
                    failure_kind="budget" if agent_started else "infrastructure",
                    diagnostics=diagnostics,
                )
            outcome = trial_outcome(rollout)
            native_trajectory = trial_directory(rollout) / "agent/trajectory.json"
            normalized_trajectory = None
            if native_trajectory.is_file():
                from pyforger.agent_diagnostics import trajectory

                native = json.loads(native_trajectory.read_text())
                events = (
                    native.get("steps", native.get("events", []))
                    if isinstance(native, dict)
                    else native
                )
                if not isinstance(events, list):
                    raise ValueError("Harbor trajectory must contain a list of steps")
                normalized_trajectory = trajectory(events, agent)
            if self._retain_trials is not None:
                source = trial_directory(rollout)
                retained = self._retain_trials / source.name
                shutil.copytree(source, retained, dirs_exist_ok=True)
                outcome = {
                    **outcome,
                    "harbor_result": str(retained / "result.json"),
                    "trial_dir": str(retained),
                }
            collected: dict[str, bytes] = {}
            if task.outputs:
                (root / "collected").mkdir()
                destinations = {
                    name: f"artifact-{index}" for index, name in enumerate(task.outputs)
                }
                collect_trial_files(
                    rollout,
                    {"/work/" + name: target for name, target in destinations.items()},
                    root / "collected",
                )
                for name in task.outputs:
                    path = root / "collected" / destinations[name]
                    if path.is_file():
                        collected[name] = path.read_bytes()
            return AgentRunResult(
                status=str(outcome["status"]),
                trajectory=normalized_trajectory,
                failure_kind=_failure_kind(outcome),
                files=collected,
                passed=(
                    outcome.get("passed")
                    if isinstance(outcome.get("passed"), bool)
                    else None
                ),
                diagnostics={k: v for k, v in outcome.items() if k != "status"},
            )

    def execute_program(
        self, code: str, item: dict[str, Any], timeout_s: float
    ) -> list[str]:
        """Replay the exact retained source through Oracle, without an LLM."""
        from pyforger.harbor import execute_script

        _record_key(item)
        with tempfile.TemporaryDirectory(prefix="forger-harbor-exec-") as directory:
            root = Path(directory)
            task = WorkspaceTask(
                instruction="Execute the retained program.",
                files={
                    "evidence/index.json": json.dumps({"request": "request.json"}),
                    "evidence/request.json": json.dumps(
                        {
                            "code": code,
                            "item": item,
                            "timeout_s": timeout_s,
                            "validation_result": True,
                        }
                    ),
                },
                outputs=("result.json",),
                network=Network.OFFLINE,
                timeout_s=timeout_s + 30,
            )
            packaged = _package(
                task, AgentSpec(kind="nop"), root / "task", ["/work/result.json"]
            )
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "worker.py").write_bytes(
                Path(__file__).with_name("_harbor_dynamic_worker.py").read_bytes()
            )
            rollout = execute_script(
                packaged,
                ArtifactRef.capture(scripts),
                root / "replay",
                environment=self._environment,
                script_path="worker.py",
                timeout_s=timeout_s + 5,
            )
            from pyforger.harbor import collect_trial_files, trial_outcome

            outcome = trial_outcome(rollout)
            if outcome["status"] != "completed" or outcome["passed"] is not True:
                raise RuntimeError(f"Harbor program replay failed: {outcome}")
            collect_trial_files(rollout, {"/work/result.json": "result.txt"}, root)
            result: object = json.loads(read_text(root / "result.txt"))
            if isinstance(result, dict) and isinstance(
                result.get("validation_error"), str
            ):
                raise ProgramValidationError(result["validation_error"])
            if not isinstance(result, list) or any(
                not isinstance(v, str) for v in result
            ):
                raise TypeError("The pipeline must return a list of strings")
            return result

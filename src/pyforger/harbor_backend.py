"""Harbor backend: agents and retained programs run in Harbor environments.

The backend translates a backend-agnostic
:class:`~pyforger.agent_backend.WorkspaceTask` into a native Harbor task,
runs the specified agent there, and collects declared outputs. It contains
no agent functionality of its own: agent identity arrives as an
:class:`~pyforger.agent_backend.AgentSpec`, and retained programs replay
through Oracle without any model.
"""

from __future__ import annotations

import copy
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
    task: WorkspaceTask, agent: AgentSpec, root: Path, outputs: list[str]
) -> ArtifactRef:
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
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        + installs
        + "RUN apt-get update && apt-get install -y --no-install-recommends "
        "curl coreutils "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m venv /opt/openhands-sdk-venv "
        "&& /opt/openhands-sdk-venv/bin/pip install --no-cache-dir "
        "openhands-sdk==1.58.0 openhands-tools==1.58.0 fastapi==0.135.2\n"
        "RUN mkdir /work && chown 65534:65534 /work\n"
        "ENV HOME=/tmp\nWORKDIR /work\nCOPY files /work\nUSER 65534:65534\n"
    )
    (root / "instruction.md").write_text(task.instruction)
    (root / "task.toml").write_text(
        'schema_version = "1.3"\n'
        f"artifacts = {json.dumps(outputs)}\n"
        f"[agent]\ntimeout_sec = {max(1, int(task.timeout_s))}\n"
        'user = "65534"\n'
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
    if task.verifier is not None:
        verifier_dir = tests / "verifier"
        verifier_dir.mkdir()
        for name, content in task.verifier_files.items():
            path = verifier_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            write_input(path, content)
        (tests / "test.sh").write_text(
            "#!/bin/sh\nmkdir -p /logs/verifier\n"
            "WORKSPACE=/work VERIFIER_DIR=/tests/verifier sh -c "
            + shlex.quote("cd /work\n" + task.verifier)
            + "\ncp /work/reward.txt /logs/verifier/reward.txt 2>/dev/null "
            "|| echo 0 > /logs/verifier/reward.txt\n"
        )
    else:
        checks = " && ".join(f"test -f {shlex.quote(o)}" for o in outputs) or "true"
        (tests / "test.sh").write_text(
            "#!/bin/sh\nmkdir -p /logs/verifier\n"
            f"if {checks}; then echo 1; else echo 0; fi "
            "> /logs/verifier/reward.txt\n"
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

    def __init__(
        self, environment: Any = None, *, retain_trials: str | Path | None = None
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
            "harbor-backend-v1",
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
        return harbor_agent_config(
            agent.model,
            max_iterations=agent.max_iterations,
            api_key_env=agent.api_key_env,
            agent_kwargs=dict(agent.kwargs),
        )

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
            packaged = _package(task, agent, root / "task", outputs)
            rollout = HarborTaskRunner(
                agent=self._agent_config(agent), environment=self._environment
            ).run(HarborRequest(trial_id="agent", task=packaged), root / "native")
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
                collect_trial_files(
                    rollout,
                    {"/work/" + n: n.replace("/", "_") for n in task.outputs},
                    root / "collected",
                )
                for name in task.outputs:
                    path = root / "collected" / name.replace("/", "_")
                    if path.is_file():
                        collected[name] = path.read_bytes()
            return AgentRunResult(
                status=str(outcome["status"]),
                trajectory=normalized_trajectory,
                usage=outcome["usage"],
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

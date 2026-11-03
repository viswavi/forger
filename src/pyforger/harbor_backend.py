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
import hashlib
import json
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyforger._harbor_deadline import remaining
from pyforger._sdk_install import SDK_COMMIT, SDK_REQUIREMENT, TOOLS_REQUIREMENT
from pyforger.agent_backend import (
    AgentRunResult,
    AgentSpec,
    Network,
    ProgramValidationError,
    WorkspaceTask,
)
from pyforger.agent_backends import require_learner_runtime
from pyforger.agent_diagnostics import ArtifactSafetyError, credential_values
from pyforger.artifacts import (
    ArtifactRef,
    FileInput,
    OwnedFileInput,
    read_text,
    write_input,
)
from pyforger.evidence_journal import evidence_file_chunks
from pyforger.local_agent import _record_key

if TYPE_CHECKING:
    from pyforger.harbor import HarborResult

_NETWORK = {
    Network.OFFLINE: "no-network",
    Network.MODEL: "no-network",
    Network.PUBLIC: "public",
}


def _validate_packaged_inputs(task: WorkspaceTask, agent: AgentSpec) -> None:
    secrets = {value.encode("utf-8") for value in credential_values([agent]) if value}
    if not secrets:
        return
    overlap = max(map(len, secrets)) - 1
    inputs = [task.files, task.verifier_files]
    if agent.kind == "oracle":
        inputs.append(task.solution_files)
    for files in inputs:
        for content in files.values():
            chunks = (
                content.chunks()
                if isinstance(content, FileInput)
                else [content.encode("utf-8") if isinstance(content, str) else content]
            )
            tail = b""
            for chunk in chunks:
                block = tail + chunk
                if any(secret in block for secret in secrets):
                    raise ArtifactSafetyError(
                        "Credential detected in native task inputs; remove credential literals before execution"
                    )
                tail = block[-overlap:] if overlap else b""


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
    _validate_packaged_inputs(task, agent)
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
        "curl coreutils git "
        "&& rm -rf /var/lib/apt/lists/*\n"
        "RUN python -m venv /opt/openhands-sdk-venv "
        "&& /opt/openhands-sdk-venv/bin/pip install --no-cache-dir "
        + shlex.join((SDK_REQUIREMENT, TOOLS_REQUIREMENT, "fastapi==0.135.2"))
        + "\n"
        + tokenizer_build
        + "RUN mkdir /work && chown 65534:65534 /work\n"
        "ENV HOME=/tmp\nWORKDIR /work\n"
        + ("COPY --chown=65534:65534 files /work\n" if task.files else "")
        + "USER root\n"
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
        "COPY . /tests\n"
        + ("COPY public /work\n" if task.files else "")
        + "RUN mkdir -p /opt/trusted && cp -r /tests/trusted/. /opt/trusted/ "
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
            "cp /work/reward.txt /logs/verifier/reward.txt; fi\n"
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
    (tests / "launch.py").write_bytes(
        Path(__file__).with_name("_harbor_verifier.py").read_bytes()
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


def _collect_outputs(
    rollout: HarborResult, outputs: tuple[str, ...], output: Path
) -> dict[str, bytes | FileInput]:
    from pyforger.harbor import collect_trial_files

    collected: dict[str, bytes | FileInput] = {}
    if not outputs:
        return collected
    output.mkdir()
    destinations = {name: f"artifact-{index}" for index, name in enumerate(outputs)}
    collect_trial_files(
        rollout,
        {"/work/" + name: target for name, target in destinations.items()},
        output,
    )
    for name, destination in destinations.items():
        path = output / destination
        if path.is_file():
            collected[name] = OwnedFileInput.snapshot(path)
        elif path.is_dir():
            for entry in ArtifactRef.capture(path).files:
                collected[name + "/" + entry.path] = OwnedFileInput.snapshot(
                    path / entry.path
                )
    return collected


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
        trial directories are transient; scrubbed native evidence is exported
        as disk-backed references alongside normalized outcomes and collected files.
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
            "harbor-backend-v3-verifier-execution-record",
            SDK_COMMIT,
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
        if agent.runtime is not None and agent.runtime.condenser_max_events is not None:
            config.env["FORGER_CONDENSER_MAX_EVENTS"] = str(
                agent.runtime.condenser_max_events
            )
        return config

    def run_agent(self, task: WorkspaceTask, agent: AgentSpec) -> AgentRunResult:
        """Package the task natively and run the agent in Harbor."""
        deadline = time.monotonic() + task.timeout_s
        remaining(deadline)
        require_learner_runtime(self, agent)
        from pyforger.harbor import (
            HarborRequest,
            HarborTaskRunner,
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
            remaining(deadline)
            runner = HarborTaskRunner(
                agent=self._agent_config(agent),
                environment=self._environment,
                timeout_s=task.timeout_s,
                deadline=deadline,
            )

            rollout = runner.run(
                HarborRequest(trial_id="agent", task=packaged), root / "native"
            )
            remaining(deadline)
            outcome = trial_outcome(rollout)
            trial = trial_directory(rollout)
            native_evidence = {}
            for entry in ArtifactRef.capture(trial).files:
                native_evidence[entry.path] = evidence_file_chunks(
                    FileInput.from_artifact(
                        rollout.artifacts,
                        str((trial / entry.path).relative_to(rollout.artifacts.root)),
                    ).chunks()
                )[0]
            outcome["native_evidence"] = native_evidence
            native_trajectory = trial_directory(rollout) / "agent/trajectory.json"
            normalized_trajectory = None
            if native_trajectory.is_file():
                from pyforger._streaming_json import trajectory_events
                from pyforger.evidence_journal import journal_events

                normalized_trajectory = journal_events(
                    trajectory_events(native_trajectory), [agent]
                )
            if self._retain_trials is not None:
                source = trial_directory(rollout)
                retained = self._retain_trials / source.name
                shutil.copytree(source, retained, dirs_exist_ok=True)
                outcome = {
                    **outcome,
                    "harbor_result": str(retained / "result.json"),
                    "trial_dir": str(retained),
                }
            collected = _collect_outputs(rollout, task.outputs, root / "collected")
            remaining(deadline)
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

        deadline = time.monotonic() + timeout_s
        remaining(deadline)

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
                timeout_s=timeout_s,
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
                timeout_s=timeout_s,
                deadline=deadline,
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
            remaining(deadline)
            return result

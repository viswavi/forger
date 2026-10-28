"""Optional native Harbor execution with host-owned agents and environments."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pyforger._streaming_json import JSONReader
from pyforger.acquire.model import Model
from pyforger.artifacts import ArtifactRef, ArtifactStore, replace_file_secrets
from pyforger.usage import combine_usage, usage_record


class HarborRequest(BaseModel):
    """Native Harbor input; models and task trees are backend-specific."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    trial_id: str = Field(min_length=1)
    task: ArtifactRef
    model: str | None = None
    configuration: dict[str, JsonValue] = Field(default_factory=dict)

    def cache_key(self) -> tuple[str, str, str | None, str]:
        """Verify the native task before identifying this execution."""
        return (
            self.trial_id,
            self.task.cache_key(),
            self.model,
            json.dumps(self.configuration, sort_keys=True, allow_nan=False),
        )


class HarborResult(BaseModel):
    """Native execution diagnostics, independent of admission policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    status: Literal["completed", "failed", "timed_out"]
    artifacts: ArtifactRef
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    usage: dict[str, JsonValue] = Field(default_factory=dict)


_SENSITIVE = re.compile(
    r"(^|_)(api_key|token|password|secret|credential|private_key)($|_)", re.I
)
_PARAMETERS = {"max_iterations", "timeout_s"}

# Native execution infrastructure retries only failures that cannot have
# consumed agent work: the trial ended before an agent result exists and
# its diagnostics indicate a transient provider/transport problem (an
# HTTP 429/5xx surfaced by the environment service, or an explicitly
# transient error type). Agent, verifier, and task failures are returned
# untouched; whether a returned failure excludes a trial from grading
# remains caller policy.
_STARTUP_MAX_ATTEMPTS = 3
_STARTUP_BACKOFF_S = (2.0, 8.0)
_TRANSIENT_HTTP = re.compile(r"\bHTTP (?:429|5\d\d)\b")
_TRANSIENT_NAMES = re.compile(
    r"timeout|temporar|transient|connection|unavailable|rate.?limit|"
    r"overloaded|try again|api.?error",
    re.I,
)


def _transient_startup_failure(result: Any) -> bool:
    """Return True when a trial failed transiently before any agent result."""
    exception = getattr(result, "exception_info", None)
    if exception is None or getattr(result, "agent_result", None) is not None:
        return False
    evidence = " ".join(
        str(getattr(exception, field, "") or "")
        for field in ("exception_type", "exception_message", "exception_traceback")
    )
    return bool(_TRANSIENT_HTTP.search(evidence) or _TRANSIENT_NAMES.search(evidence))


def _sdk() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            TrialConfig,
        )
        from harbor.trial.trial import Trial
    except ImportError as exc:
        raise ImportError(
            "HarborTaskRunner requires the optional Harbor SDK and Python >=3.12"
        ) from exc
    return AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, Trial


def harbor_agent_config(
    model: Model,
    *,
    api_key_env: str | None = None,
    max_iterations: int | None = None,
    agent_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Convert a Model to native Harbor AgentConfig without retaining credentials.

    Credentials must exist in a host environment variable (explicitly selected
    with api_key_env, or matched by value). Arbitrary LiteLLM kwargs are rejected;
    common ModelSettings and disable_prompt_cache_key are supported. Non-model
    native SDK options may be supplied through agent_kwargs.
    """
    from pyforger._harbor_model import ForgerOpenHandsSDK, runner_source
    from pyforger.sdk_config import llm_options

    if model._kwargs.keys() - {"disable_prompt_cache_key"}:
        raise ValueError("Unsupported Harbor Model kwargs")
    kwargs = dict(agent_kwargs or {})
    if kwargs.keys() & {
        "temperature",
        "reasoning_effort",
        "collect_token_ids",
        "config",
    }:
        raise ValueError("agent_kwargs must not override model settings")
    options = llm_options(model)
    key = options.pop("api_key")
    base_url = options["base_url"]
    if options["model"].startswith("openai/") and base_url in {
        "https://api.openai.com",
        "https://api.openai.com/",
    }:
        base_url = options["base_url"] = None
    if api_key_env is not None:
        selected = os.environ.get(api_key_env)
        if not selected or (key is not None and selected != key):
            raise ValueError("api_key_env must identify the Model credential")
    else:
        api_key_env = next(
            (
                name
                for name, value in os.environ.items()
                if key and value == key and _SENSITIVE.search(name)
            ),
            None,
        )
    if api_key_env is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("Harbor requires a credential in a host api_key_env variable")
    fields = (
        "model",
        "custom_tokenizer",
        "base_url",
        "temperature",
        "max_output_tokens",
        "num_retries",
        "timeout",
        "reasoning_effort",
        "litellm_extra_body",
        "api_mode",
        "capability_overrides",
    )
    hosts: list[str] = []
    if base_url:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Model endpoint must be a credential-free http/https URL")
        hosts = [parsed.hostname]
    settings = {name: options[name] for name in fields}
    # Reject embedded secrets in otherwise public model configuration.
    if _secrets(settings) or (key and key in json.dumps(settings)):
        raise ValueError("Model settings must not contain embedded credentials")
    if max_iterations is not None:
        if "max_iterations" in kwargs:
            raise ValueError("max_iterations specified twice")
        kwargs["max_iterations"] = _positive(
            max_iterations, "max_iterations", integer=True
        )
    ForgerOpenHandsSDK.parse_options(kwargs)
    runner_source()  # Fail before execution if the installed runner is incompatible.
    AgentConfig, _, _, _, _ = _sdk()
    return AgentConfig(
        import_path="pyforger._harbor_model:ForgerOpenHandsSDK",
        model_name=model.config.model_id,
        kwargs=kwargs,
        extra_allowed_hosts=hosts,
        env={
            "LLM_API_KEY": "${" + api_key_env + "}",
            "FORGER_LLM_CONFIG": json.dumps(settings),
            **({"LLM_BASE_URL": base_url} if base_url else {}),
            "FORGER_DISABLE_PROMPT_CACHE": (
                "1" if model._kwargs.get("disable_prompt_cache_key") else "0"
            ),
        },
    )


def _positive(value: Any, name: str, *, integer: bool = False) -> Any:
    if value is not None and (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive {'integer' if integer else 'number'}"
        )
    return value


def _secrets(value: Any, key: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            found.update(_secrets(child, str(name)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_secrets(child, key))
    elif isinstance(value, str):
        if key.endswith("_env"):
            secret = os.environ.get(value)
            if secret:
                found.add(secret)
        elif _SENSITIVE.search(key) and value:
            found.add(value)
    return found


def _redact(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SENSITIVE.search(key) and not key.endswith("_env")
                else _redact(child, secrets)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact(child, secrets) for child in value]
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value
    return value


def _scrub(output: Path, secrets: set[str]) -> None:
    # Validate every retained entry before opening any file for rewriting.
    ref = ArtifactRef.capture(output)
    replacements = set()
    for secret in secrets:
        replacements.add(secret.encode())
        replacements.add(json.dumps(secret)[1:-1].encode())
    for entry in ref.files:
        path = output / entry.path
        replace_file_secrets(path, replacements)
        if path.suffix != ".json":
            continue
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".scrub-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                with path.open(encoding="utf-8") as source:
                    reader = JSONReader(source)
                    reader.rewrite(
                        target,
                        lambda value: _redact(value, secrets),
                        lambda key: (
                            bool(_SENSITIVE.search(key)) and not key.endswith("_env")
                        ),
                    )
                    trailing, token = reader.token(target)
                    if token:
                        raise ValueError("Trailing JSON content")
                    target.write(trailing)
            os.replace(temporary, path)
        except (ValueError, UnicodeDecodeError):
            # Non-JSON native logs still receive exact-byte secret redaction.
            pass
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _output_path(output: Path, task: ArtifactRef) -> Path:
    output = Path(os.path.abspath(output))
    for ancestor in (output, *output.parents):
        if ancestor.is_symlink():
            raise ValueError("Harbor output path must not contain symlinks")
    task_root = task.root.resolve()
    if (
        output == task_root
        or task_root in output.parents
        or output in task_root.parents
    ):
        raise ValueError("Harbor output must not overlap the task artifact")
    if os.path.lexists(output):
        raise FileExistsError(f"Harbor output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("Harbor output parent must already exist")
    return output


def _status(result: Any) -> Literal["completed", "timed_out", "failed"]:
    exception = result.exception_info
    if exception is None:
        return "completed"
    name = exception.exception_type
    if "Timeout" in name or name == "TimeoutError":
        return "timed_out"
    return "failed"


class HarborTaskRunner:
    """Execute trusted Harbor task packages using any native agent/backend.

    The task is an immutable Harbor-format directory (instruction.md, task.toml,
    environment/, and optional solution/ and tests/). Harbor owns public/private
    staging, verifier behavior, and environment lifecycle. Backend sandbox policy
    and custom import paths must be configured by a trusted caller, not a task.
    Credentials should be resolved by the SDK from host environment variables;
    prefer agent kwargs such as api_key_env over literal credential values.
    Native trial files are retained, with credential redaction before capture.
    """

    def __init__(
        self,
        *,
        agent: Any,
        environment: Any,
        max_iterations: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Snapshot trusted AgentConfig/EnvironmentConfig objects or dictionaries."""
        self._agent = copy.deepcopy(agent)
        self._environment = copy.deepcopy(environment)
        self._max_iterations = _positive(max_iterations, "max_iterations", integer=True)
        self._timeout_s = _positive(timeout_s, "timeout_s")

    def _configuration(self, request: HarborRequest, task: Path, output: Path) -> Any:
        AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, _ = _sdk()
        unexpected = request.configuration.keys() - _PARAMETERS
        if unexpected:
            raise ValueError(
                f"Unsupported Harbor request parameters: {sorted(unexpected)}"
            )
        agent = (
            self._agent.model_copy(deep=True)
            if isinstance(self._agent, AgentConfig)
            else AgentConfig.model_validate(copy.deepcopy(self._agent))
        )
        backend = (
            self._environment.model_copy(deep=True)
            if isinstance(self._environment, EnvironmentConfig)
            else EnvironmentConfig.model_validate(copy.deepcopy(self._environment))
        )
        iterations = _positive(
            request.configuration.get("max_iterations", self._max_iterations),
            "max_iterations",
            integer=True,
        )
        timeout = _positive(
            request.configuration.get("timeout_s", self._timeout_s), "timeout_s"
        )
        if request.model is not None:
            if (
                agent.import_path == "pyforger._harbor_model:ForgerOpenHandsSDK"
                and request.model != agent.model_name
            ):
                raise ValueError("Harbor request must not override Model settings")
            agent.model_name = request.model
        if iterations is not None:
            agent.kwargs["max_iterations"] = iterations
        if timeout is not None:
            agent.override_timeout_sec = timeout
        return TrialConfig(
            task=TaskConfig(path=task),
            trial_name="trial-" + uuid.uuid4().hex,
            trials_dir=output,
            agent=agent,
            environment=backend,
        )

    def run(
        self, request: HarborRequest, output: Path, *, environment: Any = None
    ) -> HarborResult:
        """Run a native trial synchronously; use arun inside an event loop."""
        if environment is not None:
            raise ValueError("HarborTaskRunner owns its native execution environment")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(request, output))
        raise RuntimeError(
            "Use await HarborTaskRunner.arun() inside an active event loop"
        )

    async def arun(
        self, request: HarborRequest, output: Path, *, environment: Any = None
    ) -> HarborResult:
        """Retain the entire native trial tree before returning verified evidence."""
        if environment is not None:
            raise ValueError("HarborTaskRunner owns its native execution environment")
        identity = request.cache_key()
        output = _output_path(output, request.task)
        output.mkdir(mode=0o700)
        staged = ArtifactStore(output / "tasks").put(request.task.root)
        if staged.digest != request.task.digest:
            raise ValueError("Harbor task changed during staging")
        config = self._configuration(request, staged.root, output)
        secrets = _secrets(config.model_dump(mode="python"))
        # Native serializers mask literal env values before we can collect them.
        secrets.update(_secrets(config.agent.env))
        secrets.update(_secrets(config.environment.env))
        secrets.update(
            value
            for key, value in os.environ.items()
            if value and _SENSITIVE.search(key)
        )
        _, _, _, _, Trial = _sdk()
        try:
            result = None
            attempt_usage = []
            for attempt in range(_STARTUP_MAX_ATTEMPTS):
                trial = await Trial.create(config)
                expected = output / config.trial_name
                if trial.paths.trial_dir != expected:
                    raise ValueError("Harbor returned an unexpected trial output path")
                result = await trial.run()
                attempt_usage.append(_trial_usage(result))
                if not _transient_startup_failure(result):
                    break
                if attempt + 1 < _STARTUP_MAX_ATTEMPTS:
                    await asyncio.sleep(_STARTUP_BACKOFF_S[attempt])
            assert result is not None
            payload = _redact(result.model_dump(mode="json"), secrets)
            usage = combine_usage(attempt_usage)
            status = _status(result)
        finally:
            _scrub(output, secrets)
        if request.cache_key() != identity:
            raise ValueError("Harbor changed the trial request or task artifact")
        return HarborResult(
            status=status,
            artifacts=ArtifactRef.capture(output),
            metadata={
                "harbor_result": payload,
                "harbor_result_path": f"{config.trial_name}/result.json",
                "trial_id": request.trial_id,
            },
            usage=usage,
        )


def execute_script(
    task: ArtifactRef,
    script: ArtifactRef,
    output: Path,
    *,
    environment: Any,
    script_path: str = "analysis.py",
    command: str = "python3",
    destination: str = "/work/analysis.py",
    model: str = "oracle",
    timeout_s: float | None = None,
) -> HarborResult:
    """Execute a retained program through Oracle on a fresh task copy."""
    task.verify()
    script.verify()
    source = script.path(script_path)
    output = _output_path(output, task)
    output.mkdir(mode=0o700)
    staged = output / "task"
    staged.mkdir()
    for entry in task.files:
        target = staged / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(task.path(entry.path).read_bytes())
    solution = staged / "solution"
    solution.mkdir(exist_ok=True)
    (solution / "program.py").write_bytes(source.read_bytes())
    solve = solution / "solve.sh"
    solve.write_text(
        "#!/bin/sh\nset -eu\ncp /solution/program.py "
        + shlex.quote(destination)
        + "\nexec "
        + shlex.quote(command)
        + " "
        + shlex.quote(destination)
        + "\n"
    )
    solve.chmod(0o755)
    return HarborTaskRunner(
        agent={"name": "oracle"}, environment=environment, timeout_s=timeout_s
    ).run(
        HarborRequest(
            trial_id="script-" + uuid.uuid4().hex,
            task=ArtifactRef.capture(staged),
            model=model,
        ),
        output / "native",
    )


def _trial_usage(result: Any) -> dict[str, Any]:
    n_input, n_cache, n_output, cost = result.compute_token_cost_totals()
    contexts = (
        [result.agent_result]
        if result.agent_result is not None
        else [
            step.agent_result
            for step in getattr(result, "step_results", None) or []
            if step.agent_result is not None
        ]
    )
    known = 0.0
    complete = bool(contexts)
    for context in contexts:
        models = getattr(context, "model_usage", None) or {}
        if models:
            known += sum(amount.cost_usd or 0.0 for amount in models.values())
            complete &= all(amount.cost_usd is not None for amount in models.values())
        else:
            known += context.cost_usd or 0.0
            complete &= context.cost_usd is not None
    return usage_record(
        {
            "prompt_tokens": n_input,
            "cached_tokens": n_cache,
            "completion_tokens": n_output,
            "cost_usd": known if complete else None,
            "known_cost_usd": known,
            "cost_complete": complete,
        }
    )


def trial_directory(rollout: HarborResult) -> Path:
    """Locate the verified native Harbor trial directory."""
    rollout.artifacts.verify()
    result_path = rollout.metadata["harbor_result_path"]
    if not isinstance(result_path, str):
        raise ValueError("Harbor rollout requires a result path")
    return rollout.artifacts.path(result_path).parent


def trial_outcome(rollout: HarborResult) -> dict[str, Any]:
    """Decode native rewards and errors without assigning failure attribution."""
    trial = trial_directory(rollout)
    native = rollout.metadata["harbor_result"]
    if not isinstance(native, dict):
        raise ValueError("Harbor rollout requires a native result object")
    exit_code = trial / "agent/exit-code.txt"
    result: dict[str, Any] = {
        "status": rollout.status,
        "passed": None,
        "harbor_result": str(trial / "result.json"),
        "usage": usage_record(dict(rollout.usage)),
    }
    exception = native.get("exception_info")
    execution_path = trial / "verifier/execution.json"
    if execution_path.is_file():
        try:
            execution = json.loads(execution_path.read_text())
            code = execution["exit_code"]
            if code is not None and type(code) is not int:
                raise ValueError("Invalid verifier exit code")
        except (ValueError, KeyError, TypeError):
            code = None
        result["verifier_execution"] = {
            "exit_code": code,
            "stdout_path": "verifier/stdout.txt",
            "stderr_path": "verifier/stderr.txt",
        }
        if code != 0:
            return {
                **result,
                "status": "failed",
                "exception": {
                    "exception_type": "VerifierExecutionError",
                    "exception_message": (
                        f"Verifier did not complete successfully (exit_code={code})"
                    ),
                },
                **({"native_exception": exception} if exception else {}),
            }
    if exception:
        if execution_path.is_file() and exception.get("exception_type") in {
            "RewardFileNotFoundError",
            "RewardFileEmptyError",
            "VerifierOutputParseError",
        }:
            return {
                **result,
                "status": "failed",
                "exception": {
                    "exception_type": "VerifierRewardProtocolError",
                    "exception_message": (
                        "Verifier completed without a valid binary reward"
                    ),
                },
                "native_exception": exception,
            }
        return {**result, "exception": exception}
    if exit_code.exists() and exit_code.read_text().strip() != "0":
        return {**result, "exit_code": exit_code.read_text().strip()}
    verifier_result = native.get("verifier_result")
    rewards = (
        verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    )
    if rollout.status == "completed":
        if (
            not isinstance(rewards, dict)
            or set(rewards) != {"reward"}
            or rewards["reward"] not in (0, 1)
        ):
            result["status"] = "failed"
            result["exception"] = {
                "exception_type": "VerifierRewardProtocolError",
                "exception_message": (
                    "Verifier completed without a valid binary reward"
                ),
            }
        else:
            result["passed"] = rewards["reward"] == 1

    return result


def collect_trial_files(
    rollout: HarborResult,
    sources: dict[str, str],
    output: Path,
    *,
    max_bytes: int | None = None,
) -> None:
    """Collect declared files from a verified native Harbor manifest."""
    from pyforger.artifacts import FileInput

    trial = trial_directory(rollout)
    manifest = json.loads((trial / "artifacts/manifest.json").read_text())
    output = Path(output)
    for source, name in sources.items():
        target = output / name
        if target.parent != output or target.is_symlink():
            raise ValueError("Output names must be plain basenames")
        entries = [
            row for row in manifest if row["source"] == source and row["status"] == "ok"
        ]
        if entries:
            relative = (
                trial.relative_to(rollout.artifacts.root) / entries[0]["destination"]
            )
            source_file = FileInput.capture(rollout.artifacts.path(relative.as_posix()))
            if max_bytes is not None and source_file.size > max_bytes:
                raise ValueError("Artifact exceeds read limit")
            if target.exists():
                target.unlink()
            source_file.copy_to(target)
    rollout.artifacts.verify()

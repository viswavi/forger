"""Sandbox-compatible Python execution over shared environments.

Only strict JSON crosses the boundary, never pickle or remote callables. Docker
runs unrestricted Python, not RestrictedPython. The bridge separates stdout and
results but is not an evaluator security boundary: code can tamper with its own
container evidence. Trusted images must already contain Python; no pulls occur.
"""
from __future__ import annotations

import json
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any, Protocol

from pyforger.execution import (
    DockerEnvironmentFactory,
    EnvironmentFactory,
    EnvironmentSpec,
)
from pyforger.sandbox import SandboxTimeout


class PythonExecutor(Protocol):
    """Execution surface shared by Sandbox and remote Python executors."""

    def execute(
        self, code: str, *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Any:
        """Execute source and call its last function."""
        ...

    def run_program(
        self, code: str, *, stdin: str = "", timeout_s: float | None = None
    ) -> str:
        """Run a script with text input and return stdout."""
        ...


class CodeExecutionError(RuntimeError):
    """Remote exception or environment failure with diagnostic details."""

    def __init__(
        self,
        message: str,
        *,
        remote_type: str | None = None,
        traceback: str = "",
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Retain remote exception type, traceback and bounded logs."""
        super().__init__(message)
        self.remote_type = remote_type
        self.traceback = traceback
        self.stdout = stdout
        self.stderr = stderr


def _validate_json(value: Any) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise TypeError(
        "strict JSON required: finite numbers, lists, string keys; no tuples or objects"
    )


_BRIDGE = r"""
import io
import json
import math
import sys
import traceback
from pathlib import Path


def validate(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            validate(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            validate(item)
        return
    raise TypeError("result must be strict JSON: finite numbers, lists, string keys")


request = json.loads(Path(sys.argv[1]).read_text())
result_path = Path(sys.argv[2])
namespace = {"__name__": "sandbox"}
try:
    bytecode = compile(request["code"], "<executor>", "exec")
    if request["mode"] == "program":
        namespace["__name__"] = "__main__"
        sys.stdin = io.TextIOWrapper(
            io.BytesIO(request["stdin"].encode()), encoding="utf-8"
        )
        try:
            exec(bytecode, namespace)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        value = None
    else:
        exec(bytecode, namespace)
        if request["mode"] == "method":
            cls = namespace.get(request["class_name"])
            if not isinstance(cls, type):
                raise ValueError("requested name is not a class")
            function = getattr(cls(), request["method_name"])
        else:
            function = next((obj for name, obj in reversed(list(namespace.items()))
                             if name != "__builtins__" and callable(obj)
                             and hasattr(obj, "__code__")), None)
            if function is None:
                raise ValueError("code did not define any function to call")
        value = function(*request["args"], **request["kwargs"])
    validate(value)
    response = {"ok": True, "value": value}
except BaseException as exc:
    response = {"ok": False, "type": type(exc).__name__, "message": str(exc)[:8192],
                "traceback": traceback.format_exc()[-16384:]}
result_path.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
"""


class ContainerPythonExecutor:
    """Execute every call in a fresh environment using strict JSON values.

    Tuples, objects, non-string keys and nonfinite floats are rejected. Remote
    compile_function/call is intentionally unsupported; no host callable is made.
    """

    def __init__(
        self,
        factory: EnvironmentFactory,
        image: str | EnvironmentSpec = "python:3.13-slim",
        *,
        spec: EnvironmentSpec | None = None,
    ) -> None:
        """Choose a shared factory and image or complete environment spec."""
        if spec is not None and image != "python:3.13-slim":
            raise ValueError("pass either image or spec, not both")
        self.factory = factory
        self.spec = spec or (
            image if isinstance(image, EnvironmentSpec) else EnvironmentSpec(image)
        )

    def cache_key(self) -> tuple:
        """Provide process-local identity for tracing, not memoization."""
        return (type(self).__name__, id(self))

    def execute(
        self, code: str, *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Any:
        """Execute source and call its last function with JSON arguments."""
        return self._invoke(code, "function", args, kwargs, timeout_s)

    def run_program(
        self, code: str, *, stdin: str = "", timeout_s: float | None = None
    ) -> str:
        """Run a fresh script and return bounded standard output."""
        if not isinstance(stdin, str):
            raise TypeError("stdin must be a string")
        return self._invoke(code, "program", (), {}, timeout_s, stdin=stdin)

    def call_method(
        self,
        code: str,
        class_name: str,
        method_name: str,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Instantiate a class fresh and call a public method."""
        if not isinstance(class_name, str) or not isinstance(method_name, str):
            raise TypeError("class and method names must be strings")
        if method_name.startswith("_"):
            raise ValueError("method name must not start with '_'")
        return self._invoke(
            code,
            "method",
            args,
            kwargs,
            timeout_s,
            class_name=class_name,
            method_name=method_name,
        )

    def _invoke(self, code, mode, args, kwargs, timeout_s, **extra):
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        request = {
            "code": code,
            "mode": mode,
            "args": list(args),
            "kwargs": kwargs,
            **extra,
        }
        _validate_json(request)
        payload = json.dumps(request, allow_nan=False).encode("utf-8")
        budget = self.spec.timeout_s if timeout_s is None else timeout_s
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        prefix = "bridge-" + uuid.uuid4().hex
        bridge, request_path, response_path = (
            prefix + suffix for suffix in (".py", ".json", "-result.json")
        )
        try:
            with self.factory.open(self.spec) as environment:
                environment.write_files({bridge: _BRIDGE.encode("utf-8")})
                environment.write_files({request_path: payload})
                result = environment.run(
                    ("python", "-I", bridge, request_path, response_path), budget
                )
                if result.status == "timed_out":
                    raise SandboxTimeout("container Python execution timed out")
                if result.status != "completed":
                    raise CodeExecutionError(
                        f"execution {result.status}: {result.stderr}",
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                with tempfile.TemporaryDirectory(prefix="forger-python-") as temporary:
                    artifacts = environment.collect(
                        (response_path,), Path(temporary) / "response"
                    )
                    response = json.loads(
                        artifacts.path(response_path).read_text(encoding="utf-8")
                    )
                    _validate_json(response)
                if (
                    not isinstance(response, dict)
                    or type(response.get("ok")) is not bool
                ):
                    raise ValueError("invalid bridge response")
                if not response["ok"]:
                    if not all(
                        isinstance(response.get(key), str)
                        for key in ("type", "message", "traceback")
                    ):
                        raise ValueError("invalid bridge error response")
                    raise CodeExecutionError(
                        f'{response["type"]}: {response["message"]}',
                        remote_type=response["type"],
                        traceback=response["traceback"],
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                if "value" not in response:
                    raise ValueError("missing bridge result")
                return result.stdout if mode == "program" else response["value"]
        except (SandboxTimeout, CodeExecutionError):
            raise
        except Exception as exc:
            raise CodeExecutionError(f"environment or bridge failure: {exc}") from exc


class DockerPythonExecutor(ContainerPythonExecutor):
    """Convenience executor using Docker unless a shared factory is supplied."""

    def __init__(
        self,
        factory: EnvironmentFactory | None = None,
        image: str | EnvironmentSpec = "python:3.13-slim",
        *,
        spec: EnvironmentSpec | None = None,
    ) -> None:
        """Use Docker by default while retaining the factory-neutral bridge."""
        super().__init__(
            factory if factory is not None else DockerEnvironmentFactory(),
            image,
            spec=spec,
        )

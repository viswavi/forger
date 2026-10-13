"""Credential-free subprocess entry point for dynamic Python steps."""

import contextlib
import importlib.metadata
import importlib.util
import io
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "forger_dynamic_sandbox", Path(__file__).resolve().parents[1] / "sandbox.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load Forger sandbox")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Sandbox = module.Sandbox


def main():
    """Execute one submitted program and write its JSON result to stdout."""
    version = importlib.metadata.version("RestrictedPython")
    if tuple(map(int, version.split(".")[:2])) < (8, 4):
        raise RuntimeError("dynamic_step requires RestrictedPython >=8.4")
    request = json.loads(sys.stdin.read())
    sandbox = Sandbox(allowed_imports=("json", "re", "math", "html"))
    with contextlib.redirect_stdout(io.StringIO()):
        result = sandbox.execute(
            request["code"], request["item"], timeout_s=request["timeout_s"]
        )
    if not isinstance(result, list) or any(
        not isinstance(value, str) for value in result
    ):
        raise TypeError("The pipeline must return a list of strings")
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()

"""Credential-free subprocess entry point for dynamic Python steps."""

import contextlib
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
    request = json.loads(sys.stdin.read())
    sandbox = Sandbox(allowed_imports=("json", "re", "math", "string", "html"))
    with contextlib.redirect_stdout(io.StringIO()):
        result = sandbox.execute(request["code"], request["item"])
    if not isinstance(result, str):
        raise TypeError("The pipeline must return a string")
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()

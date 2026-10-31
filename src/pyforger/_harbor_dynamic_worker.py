"""Credential-free program execution entry point, staged inside Harbor only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run retained source in a bounded child and publish only valid results."""
    evidence = Path("/work/evidence")
    index = json.loads((evidence / "index.json").read_text())
    request = json.loads((evidence / index["request"]).read_text())
    if len(sys.argv) > 1:
        request["code"] = Path(sys.argv[1]).read_text()
    child = """
import ast, contextlib, io, json, sys
request = json.loads(sys.stdin.read())
tree = ast.parse(request['code'])
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
if not functions:
    raise ValueError('Source must define an entry function')
namespace = {'__name__': 'forger_dynamic'}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(tree, '<retained-source>', 'exec'), namespace)
    result = namespace[functions[-1].name](request['item'])
if not isinstance(result, list) or any(not isinstance(v, str) for v in result):
    raise TypeError('The pipeline must return a list of strings')
sys.stdout.write(json.dumps(result, allow_nan=False))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", child],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=request["timeout_s"],
            env={"PATH": os.defpath, "HOME": "/tmp"},
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if not request.get("validation_result"):
            raise
        Path("/work/result.json").write_text(
            json.dumps({"validation_error": exc.stderr[-4000:]})
        )
        return
    output = json.loads(result.stdout)
    if not isinstance(output, list) or any(not isinstance(v, str) for v in output):
        raise TypeError("The pipeline must return a list of strings")
    Path("/work/result.json").write_text(json.dumps(output))


if __name__ == "__main__":
    main()

"""Isolated verifier launcher; copied into the root-only verifier image."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from pathlib import Path

os.umask(0o077)
logs = Path("/logs/verifier")
for directory in (Path("/tests"), logs):
    directory.chmod(0o700)
record = logs / "execution.json"
record.write_text(json.dumps({"exit_code": None}))
code = 1
with (
    (logs / "stdout.txt").open("wb") as stdout,
    (logs / "stderr.txt").open("wb") as stderr,
):
    try:
        process = subprocess.run(
            ["sh", "/tests/verifier-command.sh"],
            cwd="/tests",
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/root",
                "PYTHONSAFEPATH": "1",
            },
            check=False,
            stdout=stdout,
            stderr=stderr,
        )
        code = process.returncode
    except OSError:
        stderr.write(traceback.format_exc().encode())
    finally:
        if code:
            # A grader can write a provisional reward before crashing.
            for name in ("reward.txt", "reward.json"):
                (logs / name).unlink(missing_ok=True)
        record.write_text(json.dumps({"exit_code": code}))
raise SystemExit(code if code >= 0 else 128 - code)

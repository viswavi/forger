"""One disposable RestrictedPython execution, with no API credentials in its env."""
import contextlib
import json
import math
import resource
import sys
from pathlib import Path

# -I ignores PYTHONPATH and the current directory. Import only our sibling clone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rubric_sandbox import Sandbox  # noqa: E402 - isolated worker/module bootstrap


def main():
    """Run the command-line utility described in the module documentation."""
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    request = json.load(sys.stdin)
    sandbox = Sandbox(
        allowed_imports=(
            "json",
            "re",
            "math",
            "string",
            "collections",
            "itertools",
            "functools",
            "statistics",
        ),
        default_timeout_s=5,
    )

    class Discard:
        def write(self, value):
            return len(value)

        def flush(self):
            pass

    try:
        with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(
            Discard()
        ):
            # Select the named entry point, even if helper functions follow it.
            def execute():
                namespace = sandbox._exec_namespace(request["code"])
                return namespace["grade"](
                    request["prompt"], request["response"], request["ground_truth"]
                )

            value = sandbox.call(execute)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("score must be a finite number in [0,1]")
        print(json.dumps({"score": float(value)}, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}))


if __name__ == "__main__":
    main()

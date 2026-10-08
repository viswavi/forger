"""Exercise shared launcher shutdown without GPUs, sockets, or inference libraries."""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    """Regression checks for launcher."""

    def test_term_stops_server_and_client(self):
        """Verify term stops server and client."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            task = root / "task"
            task.mkdir()
            (task / "evaluate_model.py").write_text(
                (
                    "import os,sys,time\nfrom pathlib import Path\nif "
                    '"--validate-only" in sys.argv: sys.exit(0)\n'
                    'Path(os.environ["PROBE_DIR"],"client.pid").write_text(str('
                    "os.getpid()))\ntime.sleep(60)\n"
                )
            )
            (bin_dir / "python").write_text(
                (
                    '#!/usr/bin/env bash\nif [[ "${1:-}" == *eval_server.py ]]; '
                    'then exit 0; fi\nif [[ "${1:-}" == -c && "${2:-}" == '
                    '*importlib.util* ]]; then exit 0; fi\nexec "'
                )
                + sys.executable
                + '" "$@"\n'
            )
            (bin_dir / "vllm").write_text(
                "#!"
                + sys.executable
                + (
                    "\nimport os,time\nfrom pathlib import Path\n"
                    'Path(os.environ["PROBE_DIR"],"server.pid").write_text(str('
                    "os.getpid()))\ntime.sleep(60)\n"
                )
            )
            for p in bin_dir.iterdir():
                p.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(bin_dir) + ":" + os.environ["PATH"],
                "CUDA_VISIBLE_DEVICES": "0",
                "PROBE_DIR": str(root),
                "EVAL_STAGGER_SECONDS": "0",
            }
            with (root / "console.log").open("w") as log:
                proc = subprocess.Popen(
                    [
                        "bash",
                        str(ROOT / "coding/eval_launcher.sh"),
                        str(task),
                        "--split",
                        "val",
                        "--out-dir",
                        str(root / "out"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                targets = []
                try:
                    deadline = time.monotonic() + 15
                    while not all(
                        (root / name).exists() for name in ("server.pid", "client.pid")
                    ):
                        if proc.poll() is not None or time.monotonic() > deadline:
                            self.fail((root / "console.log").read_text())
                        time.sleep(0.05)
                    targets = [
                        int((root / name).read_text())
                        for name in ("server.pid", "client.pid")
                    ]
                    proc.send_signal(signal.SIGTERM)
                    self.assertEqual(proc.wait(timeout=8), 143)
                    for pid in targets:
                        stat = Path(f"/proc/{pid}/stat")
                        self.assertTrue(
                            not stat.exists() or stat.read_text().split()[2] == "Z",
                            f"process {pid} survived",
                        )
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    for pid in targets:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

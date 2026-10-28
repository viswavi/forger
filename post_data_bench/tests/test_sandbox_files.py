"""Every module the sandbox's training helpers import is copied into the sandbox."""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / "coding" / "rl"
# Copied into every sandbox apart from the allowlist (see launch_agent's binds).
ALWAYS_COPIED = {"token_budget.py", "token_units.py"}


def allowlisted_rl_files() -> set[str]:
    """The rl/*.py names in launch_agent.py's TASK_PUBLIC_FILES tuples."""
    source = (ROOT / "launch_agent.py").read_text()
    names = set()
    for block in re.findall(r'f"rl/\{name\}"\s+for name in \((.*?)\)', source, re.S):
        names.update(re.findall(r'"([\w.]+\.py)"', block))
    return names


def relative_imports(path: Path) -> set[str]:
    """Modules `path` imports from its own package (`from .x import y`)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            found.add(node.module.split(".")[0] + ".py")
    return found


class SandboxFilesTest(unittest.TestCase):
    """The sandbox's copy of rl/ is closed under its own imports."""

    def test_imports_of_copied_modules_are_copied(self):
        """A missing rl/lineage.py made both round-2 agents write their own."""
        copied = allowlisted_rl_files() | ALWAYS_COPIED
        self.assertIn("agent_training.py", copied)
        missing = {
            f"{name} imports {dep}"
            for name in sorted(copied)
            if (RL / name).is_file()
            for dep in relative_imports(RL / name)
            if dep not in copied
        }
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()

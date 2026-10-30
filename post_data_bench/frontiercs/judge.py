"""The C++ judge, shared with the RL reward in coding/rl/cpp_judge.py.

cpp_judge ports FrontierCS@12ab268 algorithmic/judge/src/judge_engine.js; its
docstring stays benchmark-neutral because rl/ is copied into agent sandboxes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coding.rl.cpp_judge import (  # noqa: E402 - standalone script path setup
    Case,
    Problem,
    compile_cpp,
    compiled_judge_program,
    default_cache_dir,
    judge,
    judge_case,
    judge_interactive_case,
    launcher,
    load_problem,
    to_bytes,
    to_seconds,
)

__all__ = [
    "Case",
    "Problem",
    "compile_cpp",
    "compiled_judge_program",
    "default_cache_dir",
    "judge",
    "judge_case",
    "judge_interactive_case",
    "launcher",
    "load_problem",
    "to_bytes",
    "to_seconds",
]

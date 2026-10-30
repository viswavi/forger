"""Shared task goal and prompt builder for split construction and evaluation."""
from __future__ import annotations

# The task's goal specification, shown to every candidate model as the system prompt.
# Verbatim from the benchmark's own solution generator (see make_splits.py): it
# defines the task, so it is kept unchanged. A thinking model's reasoning precedes
# this "code only" answer.
SYSTEM_PROMPT = (
    "You are a competitive programmer. You will be given a problem statement, please "
    "implement a solution in C++. The execution time and memory limit are also "
    "stated in the statement so be aware of the complexity of the program. Please "
    "wrap the code in ```cpp and ``` so that it is properly formatted. Your response "
    "should ONLY contain the C++ code, with no additional explanation or text."
)


def format_limits(time_limit_s: float, memory_mb: int) -> str:
    """The judging line after every statement, e.g. "... time limit 2 seconds ...".

    About half the statements give no limits and a dozen state other ones than the
    judge enforces, so the enforced values follow the (unedited) statement.
    """
    seconds = f"{time_limit_s:g} second" + ("" if time_limit_s == 1 else "s")
    return (
        "Judging limits (these apply even where the statement says otherwise): "
        f"time limit {seconds} of CPU time per test; memory limit {memory_mb} MB. "
        "The solution is compiled with g++ -O2 -std=gnu++17."
    )


def build_user_content(statement: str, time_limit_s: float, memory_mb: int) -> str:
    """Render the user-turn text: the benchmark generator's, plus the judging limits."""
    limits = format_limits(time_limit_s, memory_mb)
    return f"Problem:\n\n{statement}\n\n{limits}\n\nGenerate solution code:"


def build_candidate_prompt(user_content: str) -> list[dict[str, str]]:
    """Build the OpenAI-chat-format prompt for one row's task_text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

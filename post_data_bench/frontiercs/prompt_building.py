"""Shared task goal and prompt builder for split construction and evaluation."""
from __future__ import annotations

# The task's goal specification, shown to every candidate model as the system prompt.
# Verbatim from FrontierCS's own solution generator (algorithmic/scripts/
# generate_solutions.py, CPP_SYSTEM_PROMPT): it defines the task, so it is kept
# unchanged. A thinking model's reasoning precedes this "code only" answer.
SYSTEM_PROMPT = (
    "You are a competitive programmer. You will be given a problem statement, please "
    "implement a solution in C++. The execution time and memory limit are also "
    "stated in the statement so be aware of the complexity of the program. Please "
    "wrap the code in ```cpp and ``` so that it is properly formatted. Your response "
    "should ONLY contain the C++ code, with no additional explanation or text."
)


def build_user_content(statement: str) -> str:
    """Render the user-turn text for one problem, as FrontierCS's generator does."""
    return f"Problem:\n\n{statement}\n\nGenerate solution code:"


def build_candidate_prompt(user_content: str) -> list[dict[str, str]]:
    """Build the OpenAI-chat-format prompt for one row's task_text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

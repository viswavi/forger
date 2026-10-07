"""Build shared prompts for reference models and candidate evaluation.

Inline LCB’s question/starter formatting to avoid its Anthropic dependency
and import-time paths. Prepared statements are already harmonized; add the
same public-test block for every row.
"""
from __future__ import annotations

from checkers import CodeGenerationProblem

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)

FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin, solve the problem, and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads the "
    "inputs, runs the algorithm, and writes output to STDOUT."
)

ABSTENTION_MARKER = "This task is impossible to complete given the provided tests."

ABSTENTION_INSTRUCTION = (
    "\n\nSome tasks in this evaluation are unsolvable as specified -- for example, "
    "a stated example's output may be inconsistent with the problem's own "
    "described behavior. If, after carefully working through the problem, you "
    f"conclude this is such a case, respond with exactly the string "
    f'"{ABSTENTION_MARKER}" instead of code. Only do this when you are genuinely '
    "convinced the task cannot be solved as specified -- most tasks are solvable."
)


def _render_public_test_cases(problem: CodeGenerationProblem) -> str:
    lines = ["### Public Test Cases"]
    for i, test in enumerate(problem.public_test_cases, start=1):
        lines.append(f"Example {i}:")
        lines.append(f"Input: {test.input}")
        lines.append(f"Output: {test.output}")
    return "\n".join(lines)


def _question_body(problem: CodeGenerationProblem) -> str:
    parts = [f"### Question:\n{problem.question_content}\n"]
    if problem.starter_code:
        parts.append(
            f"### Format: {FORMATTING_WITH_STARTER_CODE}\n"
            f"```python\n{problem.starter_code}\n```\n"
        )
    else:
        parts.append(
            f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            "```python\n# YOUR CODE HERE\n```\n"
        )
    parts.append(_render_public_test_cases(problem))
    parts.append("\n### Answer: (use the provided format with backticks)\n")
    return "\n".join(parts)


def build_candidate_prompt(
    problem: CodeGenerationProblem, *, include_abstention_instruction: bool
) -> list[dict[str, str]]:
    """Build chat messages, omitting abstention instructions for gold generation."""
    system = SYSTEM_MESSAGE + (
        ABSTENTION_INSTRUCTION if include_abstention_instruction else ""
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _question_body(problem)},
    ]

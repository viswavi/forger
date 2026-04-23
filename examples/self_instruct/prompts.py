"""Prompt templates for Self-Instruct (Tables 5-8 from the paper).

Reference: https://arxiv.org/abs/2212.10560, Appendix A.4
"""
from __future__ import annotations

# Table 5: Instruction generation (adapted for chat models)
# The paper used GPT-3 davinci (completion model). Chat models need a
# system prompt to generate tasks instead of answering them.
INSTRUCTION_SYSTEM_PROMPT = (
    "You are a task generator. Given a list of example tasks, generate new "
    "diverse tasks in the same format. Output ONLY new tasks, one per line, "
    'in the format "Task N: <instruction>". Do not explain or answer the tasks.'
)


def build_instruction_messages(few_shot_tasks: list[dict]) -> list[dict[str, str]]:
    """Build Table 5 prompt as chat messages."""
    examples = ""
    for i, task in enumerate(few_shot_tasks, 1):
        examples += f"Task {i}: {task['instruction']}\n"
    n = len(few_shot_tasks) + 1
    return [
        {"role": "system", "content": INSTRUCTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here are some example tasks:\n\n{examples}\n"
                f"Generate 7 more tasks continuing from Task {n}:"
            ),
        },
    ]


# Table 6: Classification identification (verbatim from paper)
# 12 classification + 19 non-classification few-shot examples from seeds.
# The paper's Table 6 truncates with "..." — shown examples reproduced below.
CLASSIFICATION_PROMPT = """\
Can the following task be regarded as a classification task with finite output labels?

Task: Given my personality and the job, tell me if I would be suitable.
Is it a classification task? Yes

Task: Give me an example of a time when you had to use your sense of humor.
Is it a classification task? No

Task: Replace the placeholders in the given text with appropriate named entities.
Is it a classification task? No

Task: Fact checking - tell me if the statement is true, false, or unknown, \
based on your knowledge and common sense.
Is it a classification task? Yes

Task: Return the SSN number for the person.
Is it a classification task? No

Task: Detect if the Reddit thread contains hate speech.
Is it a classification task? Yes

Task: Analyze the sentences below to identify biases.
Is it a classification task? No

Task: Select the longest sentence in terms of the number of words in the \
paragraph, output the sentence index.
Is it a classification task? Yes

Task: Find out the toxic word or phrase in the sentence.
Is it a classification task? No

Task: Rank these countries by their population.
Is it a classification task? No

Task: You are provided with a news article, and you need to identify all \
the categories that this article belongs to. Possible categories include: \
Music, Sports, Politics, Tech, Finance, Basketball, Soccer, Tennis, \
Entertainment, Digital Game, World News. Output its categories one by one, \
seperated by comma.
Is it a classification task? Yes

Task: Given the name of an exercise, explain how to do it.
Is it a classification task? No

Task: Select the oldest person from the list.
Is it a classification task? Yes

Task: Find the four smallest perfect numbers.
Is it a classification task? No

Task: Does the information in the document supports the claim? You can \
answer "Support" or "Unsupport".
Is it a classification task? Yes

Task: Create a detailed budget for the given hypothetical trip.
Is it a classification task? No

Task: Given a sentence, detect if there is any potential stereotype in it. \
If so, you should explain the stereotype. Else, output no.
Is it a classification task? No

Task: To make the pairs have the same analogy, write the fourth word.
Is it a classification task? No

Task: Given a set of numbers, find all possible subsets that sum to a given number.
Is it a classification task? No

Task: {value}
Is it a classification task?"""


def build_input_first_prompt(
    target_instruction: str, few_shot_tasks: list[dict]
) -> str:
    """Build Table 7 prompt: input-first instance generation for non-clf tasks."""
    lines = [
        "Come up with examples for the following tasks. Try to generate "
        "multiple examples when possible. If the task doesn't require "
        "additional input, you can generate the output directly.\n"
    ]
    for task in few_shot_tasks:
        lines.append(f"Task: {task['instruction']}")
        if "instances" in task:
            for j, inst in enumerate(task["instances"], 1):
                if inst.get("input"):
                    lines.append(f"Example {j}")
                    lines.append(f"Input: {inst['input']}")
                    lines.append(f"Output: {inst['output']}")
                else:
                    lines.append(f"Output: {inst['output']}")
        lines.append("")
    lines.append(f"Task: {target_instruction}")
    return "\n".join(lines)


def build_output_first_prompt(
    target_instruction: str, few_shot_tasks: list[dict]
) -> str:
    """Build Table 8 prompt: output-first instance generation for clf tasks."""
    lines = [
        "Given the classification task definition and the class labels, "
        "generate an input that corresponds to each of the class labels. "
        "If the task doesn't require input, just generate the correct "
        "class label.\n"
    ]
    for task in few_shot_tasks:
        lines.append(f"Task: {task['instruction']}")
        if "instances" in task:
            for inst in task["instances"]:
                lines.append(f"Class label: {inst['output']}")
                if inst.get("input"):
                    lines.append(f"Input: {inst['input']}")
        lines.append("")
    lines.append(f"Task: {target_instruction}")
    return "\n".join(lines)

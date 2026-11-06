"""An RL dataset of math problems with integer answers, each with a code rubric.

Each step's docstring says what it does and why: the options we compared, how we
measured them, and what we chose. The numbers in those justifications are
illustrative, written for this example rather than recorded from a run.

The output is one JSON line per problem: the ``prompt`` and its ``rubric``, a list
of ``[criterion, weight]`` pairs whose one criterion is code that checks the final
answer (the format ``acquire.generate_custom_rubric`` produces).

Run with STRONG_MODEL_ID and WEAK_MODEL_ID replaced by gateway model ids:

    FORGER_LIMIT=3 python examples/rl_math/pipeline.py     # a smoke run, 3 rows a step
    python examples/rl_math/pipeline.py rl_math.jsonl      # the full run
"""

from __future__ import annotations

import functools
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from pyforger import acquire, alter, filter, inspect, pipeline, step
from pyforger.acquire import LocalModel, Model

SOURCE = "EleutherAI/hendrycks_math"
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
ASK = "\n\nReason step by step, and put your final answer in \\boxed{}."

# The rubric's one criterion: the last \boxed{} after the reasoning must hold the
# answer. __ANSWER__ is replaced per row. No imports or augmented assignment, so
# it runs in a restricted grading sandbox.
GRADE_CODE = r"""
def grade(prompt, response, ground_truth):
    answer = __ANSWER__
    if "<think>" in response and "</think>" not in response:
        return 0.0
    text = response.split("</think>")[-1]
    start = text.rfind("\\boxed{")
    if start < 0:
        return 0.0
    depth = 0
    got = None
    begin = start + len("\\boxed{")
    for j in range(begin, len(text)):
        if text[j] == "{":
            depth = depth + 1
        elif text[j] == "}":
            if depth == 0:
                got = text[begin:j]
                break
            depth = depth - 1
    if got is None:
        return 0.0
    for junk in (" ", ",", "$", "\\!", "\\,"):
        got = got.replace(junk, "")
    return 1.0 if got == answer else 0.0
""".strip()


def as_integer(answer: object) -> str | None:
    """``"1,024"``, ``"$-7$"`` and ``12`` to ``"1024"``, ``"-7"``, ``"12"``."""
    text = re.sub(r"[\s,$]|\\!|\\,", "", str(answer))
    return text if re.fullmatch(r"-?\d+", text) else None


def rubric_for(answer: str) -> list:
    """One code criterion, weight 1, checking the final answer."""
    return [[{"code": GRADE_CODE.replace("__ANSWER__", repr(answer))}, 1.0]]


@functools.lru_cache(maxsize=None)
def _grader(code: str) -> Callable[[str, str, str], float]:
    namespace: dict = {}
    exec(code, namespace)  # our own rubric code, not model output
    return namespace["grade"]


def reward(response: str, row: dict) -> float:
    """The row's rubric score for ``response``: the weighted mean of its criteria."""
    total = sum(weight for _, weight in row["rubric"])
    return (
        sum(
            weight * _grader(criterion["code"])(row["prompt"], response, "")
            for criterion, weight in row["rubric"]
        )
        / total
    )


def solved(response: str, row: dict) -> bool:
    """A full-score response, for ``filter.by_pass_at_k``."""
    return reward(response, row) == 1.0


def whole(completion: str) -> str:
    """Grade the whole completion: the rubric finds the answer itself."""
    return completion


def probes() -> dict[str, Callable[[dict], str]]:
    """Responses per row that only a broken rubric would pay for."""

    def largest_number(row: dict) -> str:
        numbers = re.findall(r"-?\d+", row["problem"]) or ["0"]
        return max(numbers, key=lambda s: abs(int(s)))

    return {
        "number_from_statement": lambda row: f"\\boxed{{{largest_number(row)}}}",
        "every_box": lambda row: " ".join(f"\\boxed{{{i}}}" for i in range(-10, 101)),
        "unclosed_thinking": lambda row: f"<think>so it is \\boxed{{{row['answer']}}}",
    }


EXTRACT = """Name the topic this competition problem tests, as a short phrase
specific enough to write a different problem from (e.g. "counting lattice paths
that avoid a blocked point", not "combinatorics").

Problem: {problem}"""


@step
def seed_topics(model: Model) -> dict[str, list[dict]]:
    """Retrieve competition problems to identify salient topics; hold out validation.

    Justification: we compared 3 candidates:
        1. Write a list of topics ourselves from memory, and generate problems.
        2. Use the validation examples directly as seeds.
        3. Retrieve existing problems, extract topics from those, and generate
           problems from the topics.

    We generated 50 problems from each and ran inspect.embedding_dispersion(...,
    reference=validation). #2 put 25% of its problems near a validation problem
    (near-reference share 0.25; about 0.10 is expected for unrelated problems
    drawn alike), which was too high. #1 covered 48% of the validation set
    (its uncovered examples were mostly geometry and precalculus) against 81%
    for #3, with Vendi 21 against 34. Therefore, we chose #3.
    """
    problems, validation = [], []
    for subject in SUBJECTS:
        problems += acquire.retrieve_from_hf(
            SOURCE, config=subject, split="train", n=80, seed=0
        )
        # n caps at the split's size: all of it.
        validation += acquire.retrieve_from_hf(
            SOURCE, config=subject, split="test", n=10**5
        )
    topics = alter.generate_from_context(
        problems, EXTRACT, model, output_schema={"topic": str}, merge=True
    )
    topics = [{"topic": t["topic"].strip(), "subject": t["type"]} for t in topics]
    topics = filter.by_embedding_similarity(topics, field="topic", threshold=0.9)
    return {"topics": topics, "validation": validation}


WRITE = """Write a new competition math problem on this topic: {topic}.

The answer must be a single integer. Then solve the problem step by step and end
the solution with the answer in \\boxed{{}}."""


@step
def write(
    topics: list[dict], validation: list[dict], writer: Model, verifier: Model
) -> list[dict]:
    """Three new problems per topic, each with the writer's solution and integer answer.

    Kept problems have an integer answer that is not printed in the statement,
    that the verifier reproduces on at least 3 of 4 tries, and no near-copy
    among the kept problems or in the validation set.

    Justification: we compared 3 writer models:
        #1: the strong model at no reasoning effort
        #2: the weak model at low effort
        #3: the weak model at no effort
    and two workflows:
        #4: joint problem-solution generation
        #5: independent problem and solution generation

    Tested on 10 problems from each, from the same topics, using the verifier's
    pass@4 (filter.by_pass_at_k with the strong model at low effort) as our
    quality metric. All three models scored about the same, so we compared cost
    per problem kept: #1 was 20% cheaper than #2 and 55% cheaper than #3, since
    the weak model's drafts more often lacked an integer answer or failed
    verification. Averaged across models, #4 led to a higher pass@4 than #5
    (80% vs 72%) at lower cost. Therefore, we use #1 and #4.

    The verification bar (3 of 4) and the duplicate threshold came from their
    histograms: filter.by_pass_at_k(..., histogram=True) on 300 drafts, and
    inspect.embedding_dispersion(..., duplicate_threshold=t) for t in 0.95,
    0.9 and 0.85. At 0.9 the groups were siblings with their numbers swapped;
    at 0.85 they joined distinct problems that opened alike.
    """
    drafts = alter.generate_from_context(
        topics,
        WRITE,
        writer,
        samples_per_item=3,
        output_schema={"problem": str, "solution": str, "answer": str},
        merge=True,
    )
    rows = []
    for d in drafts:
        answer = as_integer(d["answer"])
        rows.append(
            {
                "subject": d["subject"],
                "topic": d["topic"],
                "problem": d["problem"].strip(),
                "prompt": d["problem"].strip() + ASK,
                "solution": d["solution"],
                "answer": answer,
                "rubric": rubric_for(answer) if answer is not None else None,
            }
        )
    rows = filter.by_predicate(
        rows, lambda r: r["answer"] is not None, label="integer answer"
    )
    rows = filter.by_predicate(
        rows,
        lambda r: r["answer"] not in re.findall(r"-?\d+", r["problem"]),
        label="answer not in statement",
    )
    rows = filter.by_pass_at_k(
        rows,
        model=verifier,
        input_field="prompt",
        k=4,
        min_rate=0.75,
        max_rate=1.0,
        metric=solved,
        extract_answer=whole,
        min_yield=0.4,
    )
    return filter.by_embedding_similarity(
        rows, field="problem", threshold=0.9, reference=validation
    )


@step
def calibrate(rows: list[dict], student: LocalModel) -> list[dict]:
    """Filter out problems the student solves 0% or 100% of the time (bad for RL).

    Justification: RL samples 8 responses per prompt, and a prompt whose 8 all
    score the same gives no gradient. We measured the student with RL's settings
    (8 samples at temperature 1.0, 16,384 tokens) on 892 problems:
    filter.by_pass_at_k(..., min_rate=0, max_rate=1, histogram=True) reported
    384 rows with every sample graded the same (125 all 0, 259 all 1). We
    compared keeping [1/8, 7/8] (508 rows) with [1/8, 6/8] (419 rows); nothing
    we measured favored the narrower band, so we kept [1/8, 7/8].
    """
    return filter.by_pass_at_k(
        rows,
        model=student,
        input_field="prompt",
        k=8,
        min_rate=1 / 8,
        max_rate=7 / 8,
        metric=solved,
        extract_answer=whole,
    )


@step
def check_rubrics(rows: list[dict], student: LocalModel, baseline: Model) -> list[dict]:
    r"""Score solutions, a weak baseline, probes and student samples with the rubrics.

    Justification: we compared two rubric criteria:
        1. Any \boxed{} in the response holds the answer.
        2. The last \boxed{} after the reasoning holds the answer.

    inspect.rubric_scores with the probes below paid 0.71 on average to
    every_box and 1.00 to unclosed_thinking under #1, against 0.00 under #2,
    while solutions scored 1.00 under both. number_from_statement scored 0.06
    under #1, which is why write drops answers printed in the statement.
    Therefore, we use #2. Under it, the baseline (the weak model at no effort)
    scored below the solution on 88 of 200 rows and tied on 112, rows both got
    right.
    """
    rows = alter.generate_from_context(
        rows, "{prompt}", baseline, merge=True, output_field="baseline"
    )
    rows = [{**r, "baseline_solution": r["baseline"]["text"]} for r in rows]
    inspect.rubric_scores(
        rows,
        reward,
        n=200,
        response_fields=["solution", "baseline_solution"],
        probes=probes(),
        model=student,
        k=8,
    )
    return rows


@pipeline
def main(
    writer: Model, verifier: Model, student: LocalModel, baseline: Model
) -> list[dict]:
    """Topics, then verified problems, a difficulty band, and checked rubrics."""
    seeds = seed_topics(baseline)
    rows = write(seeds["topics"], seeds["validation"], writer, verifier)
    rows = calibrate(rows, student)
    return check_rubrics(rows, student, baseline)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "rl_math.jsonl")
    rows = main(
        writer=Model("STRONG_MODEL_ID", reasoning_effort="none"),
        verifier=Model("STRONG_MODEL_ID", reasoning_effort="low"),
        # RL's sampling settings, so calibrate predicts RL's groups.
        student=LocalModel("Qwen/Qwen3-8B", temperature=1.0, max_tokens=16384),
        baseline=Model("WEAK_MODEL_ID", reasoning_effort="none"),
    )
    # What RL trains on: each prompt with the rubric that scores its responses.
    out.write_text(
        "".join(
            json.dumps({"prompt": r["prompt"], "rubric": r["rubric"]}) + "\n"
            for r in rows
        )
    )
    print(f"wrote {len(rows)} prompts and rubrics to {out}")

"""Build an RL dataset of math problems with integer answers, checked by exact match.

This file is where the decisions ended up; DECISIONS.md beside it is how they were
made: the options compared at each step and the measurements that chose between
them. Read that first. The pipeline:

    seeds        competition problems from a public training split, integer answers
    write        a strong model writes new problems that test the same idea
    verify       an independent model reproduces each stated answer on 3 of 4 tries
    dedupe       drop near-copies of each other and of the held-out test split
    calibrate    keep problems the student solves on 1 to 7 of 8 samples
    check_reward the reward on references, a weak baseline, probes, and the student
    finish       coverage against held-out training problems; write the rows

Run (needs LITELLM_API_KEY for the writer and verifier, TINKER_API_KEY for the
student):

    FORGER_LIMIT=3 python examples/rl_math/pipeline.py    # a smoke run, 3 rows a step
    python examples/rl_math/pipeline.py rl_math.jsonl     # the full run
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from pyforger import acquire, alter, filter, inspect, pipeline, step
from pyforger.acquire import Model, TinkerModel
from pyforger.filter.model_difficulty import answer_segment

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
SEEDS_PER_SUBJECT = 80
VARIANTS_PER_SEED = 3
# One-line math problems embed closer than long ones: one changed number scores
# about 0.93, so 0.95 misses re-parameterized copies (DECISIONS.md, D3).
DUPLICATE_THRESHOLD = 0.9
ASK = "\n\nReason step by step, and put your final answer in \\boxed{}."


# --- The reward -----------------------------------------------------------------


def boxed(text: str | None) -> str | None:
    r"""The content of the last ``\boxed{...}`` in ``text``, braces balanced."""
    if not text:
        return None
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed{")
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[i:j]
            depth -= 1
    return None


def as_integer(answer: str | None) -> str | None:
    """``"1,024"``, ``"$-7$"`` and ``" 12 "`` to ``"1024"``, ``"-7"``, ``"12"``."""
    if answer is None:
        return None
    text = re.sub(r"[\s,$]|\\!|\\,", "", answer)
    return text if re.fullmatch(r"-?\d+", text) else None


def final_answer(completion: str) -> str | None:
    """The last boxed integer after the reasoning; None if the thinking never closed.

    Only the last box counts, and only after ``</think>``: the probes in
    DECISIONS.md (D5) showed a grader taking any box rewarded a response that
    boxes every integer, and one reading inside the thinking rewarded
    truncated responses.
    """
    segment = answer_segment(completion)
    return as_integer(boxed(segment)) if segment else None


def correct(prediction: str | None, row: dict) -> bool | None:
    """The verifier ``by_pass_at_k`` calls on ``final_answer``'s output."""
    return None if prediction is None else prediction == row["answer"]


def reward(response: str, row: dict) -> float:
    """The RL reward: 1 for the right final integer, else 0."""
    return float(final_answer(response) == row["answer"])


# Responses the reward must not pay for (DECISIONS.md, D5).
def _largest_number(row: dict) -> str:
    numbers = re.findall(r"-?\d+", row["problem"]) or ["0"]
    return max(numbers, key=lambda s: abs(int(s)))


def probes() -> dict[str, Callable[[dict], str]]:
    """Responses per row that only a broken reward would pay for."""
    return {
        # Is the answer given away? The statement's largest number, boxed.
        "number_from_statement": lambda row: f"\\boxed{{{_largest_number(row)}}}",
        # Does the grader take any box? Every integer from -10 to 100, boxed.
        "every_box": lambda row: " ".join(f"\\boxed{{{i}}}" for i in range(-10, 101)),
        # Does a truncated response earn reward? The right answer, thinking open.
        "unclosed_thinking": lambda row: f"<think>so it is \\boxed{{{row['answer']}}}",
    }


# --- Steps ----------------------------------------------------------------------


@step
def sources() -> dict[str, list[dict]]:
    """Seeds and held-out problems from the training split, and the test split.

    Each subject's training problems are shuffled once and split in two: seeds
    to write from, and a reference the seeds never touch, for coverage (D1).
    The test split is only for decontamination.
    """
    seed_rows, reference, test = [], [], []
    for subject in SUBJECTS:
        train = acquire.retrieve_from_hf(
            SOURCE, config=subject, split="train", n=2 * SEEDS_PER_SUBJECT, seed=0
        )
        seed_rows += train[:SEEDS_PER_SUBJECT]
        reference += train[SEEDS_PER_SUBJECT:]
        # n caps at the split's size: every test problem.
        test += acquire.retrieve_from_hf(
            SOURCE, config=subject, split="test", n=10**5
        )
    seed_rows = [
        {**r, "seed_answer": as_integer(boxed(r["solution"]))} for r in seed_rows
    ]
    seed_rows = filter.by_predicate(
        seed_rows, lambda r: r["seed_answer"] is not None, label="integer answer"
    )
    return {"seeds": seed_rows, "reference": reference, "test": test}


WRITE = """Here is a competition math problem and its solution.

Problem: {problem}

Solution: {solution}

Write a new problem that tests the same idea in a different setting. Change the
objects, the numbers and what is asked, so that no step of this solution can be
copied into the new one. The answer must be a single integer. Then solve your
problem carefully."""


@step
def write(seed_rows: list[dict], writer: Model) -> list[dict]:
    """Three new problems per seed, each with the writer's integer answer."""
    drafts = alter.generate_from_context(
        seed_rows,
        WRITE,
        writer,
        samples_per_item=VARIANTS_PER_SEED,
        output_schema={"new_problem": str, "new_answer": str},
        merge=True,
    )
    rows = [
        {
            "subject": d["type"],
            "level": d["level"],
            "seed_problem": d["problem"],
            "problem": d["new_problem"].strip(),
            "prompt": d["new_problem"].strip() + ASK,
            "answer": as_integer(str(d["new_answer"])),
        }
        for d in drafts
    ]
    rows = filter.by_predicate(
        rows, lambda r: r["answer"] is not None, label="integer answer"
    )
    # The number_from_statement probe found answers printed in the problem (D5).
    return filter.by_predicate(
        rows,
        lambda r: r["answer"] not in re.findall(r"-?\d+", r["problem"]),
        label="answer not in statement",
    )


@step
def verify(rows: list[dict], verifier: Model) -> list[dict]:
    """Keep problems whose answer another model reproduces on 3 or 4 of 4 tries."""
    rows = filter.by_pass_at_k(
        rows,
        model=verifier,
        input_field="prompt",
        k=4,
        min_rate=0.75,
        max_rate=1.0,
        metric=correct,
        extract_answer=final_answer,
        completions_field="verifier_samples",
        outcomes_field="verifier_grades",
        min_yield=0.4,
    )
    # A passing verifier solution is the reference response for check_reward.
    return [
        {
            **r,
            "reference_solution": next(
                c
                for c, ok in zip(
                    r["verifier_samples"]["model"], r["verifier_grades"]["model"]
                )
                if ok
            ),
        }
        for r in rows
    ]


@step
def dedupe(rows: list[dict], test: list[dict]) -> list[dict]:
    """Drop near-copies of a kept problem or of a test problem."""
    return filter.by_embedding_similarity(
        rows,
        field="problem",
        threshold=DUPLICATE_THRESHOLD,
        reference=test,
        reference_field="problem",
    )


@step
def calibrate(rows: list[dict], student: TinkerModel) -> list[dict]:
    """Keep problems the student solves on 1 to 7 of 8 samples, RL's group size."""
    return filter.by_pass_at_k(
        rows,
        model=student,
        input_field="prompt",
        k=8,
        min_rate=1 / 8,
        max_rate=7 / 8,
        metric=correct,
        extract_answer=final_answer,
        outcomes_field="student_grades",
    )


@step
def check_reward(
    rows: list[dict], student: TinkerModel, baseline: TinkerModel
) -> list[dict]:
    """Grade references, a weak baseline, probes and student samples with the reward."""
    rows = alter.generate_from_context(
        rows, "{prompt}", baseline, merge=True, output_field="baseline"
    )
    rows = [{**r, "baseline_solution": r["baseline"]["text"]} for r in rows]
    inspect.rubric_scores(
        rows,
        reward,
        n=200,
        response_fields=["reference_solution", "baseline_solution"],
        probes=probes(),
        model=student,
        k=8,
    )
    return rows


@step
def finish(rows: list[dict], reference: list[dict]) -> list[dict]:
    """Measure the final rows against held-out training problems; keep RL fields."""
    inspect.embedding_dispersion(
        rows,
        "problem",
        reference=reference,
        duplicate_threshold=DUPLICATE_THRESHOLD,
    )
    return [
        {"prompt": r["prompt"], "answer": r["answer"], "subject": r["subject"]}
        for r in rows
    ]


@pipeline
def main(
    writer: Model, verifier: Model, student: TinkerModel, baseline: TinkerModel
) -> list[dict]:
    """Seed, write, verify, dedupe, calibrate, check the reward, and measure."""
    source = sources()
    rows = write(source["seeds"], writer)
    rows = verify(rows, verifier)
    rows = dedupe(rows, source["test"])
    rows = calibrate(rows, student)
    rows = check_reward(rows, student, baseline)
    return finish(rows, source["reference"])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "rl_math.jsonl")
    rows = main(
        # Writer and verifier differ, so a shared mistake is less likely (D2).
        writer=Model("litellm_proxy/azure_ai/FW-GLM-5.3"),
        verifier=Model("litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"),
        # RL's sampling settings, so calibrate predicts RL's groups (D4).
        student=TinkerModel("Qwen/Qwen3-8B", temperature=1.0, max_tokens=16384),
        # The student without thinking: weaker on the same problems (D5).
        baseline=TinkerModel("Qwen/Qwen3-8B", enable_thinking=False, max_tokens=4096),
    )
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {out}")

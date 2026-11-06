"""An RL dataset of math problems with integer answers, each with a code rubric.

Each step's docstring says what it does and why: the options we compared, how we
measured them, and what we chose. The numbers in those justifications are
illustrative, written for this example rather than recorded from a run.

The output is one JSON line per problem: the ``prompt`` and its ``rubric``, a list
of ``[criterion, weight]`` pairs whose one criterion is code that checks the final
answer (the format ``acquire.generate_custom_rubric`` produces).

Run with STRONG_MODEL_ID and WEAK_MODEL_ID replaced by gateway model ids, and
BRAVE_API_KEY set for web search:

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

# Web searches for competition-style problems; topics come from their results.
QUERIES = [
    "AMC 12 counting and probability problems with solutions",
    "AIME number theory problems modular arithmetic",
    "AIME combinatorics problems with answers",
    "HMMT February algebra round problems",
    "HMMT combinatorics round problems and solutions",
    "ARML team round problems",
    "AMC 10 geometry problems area and similar triangles",
    "AIME geometry problems circles and tangents",
    "olympiad polynomial problems roots and coefficients",
    "competition math problems on sequences and recursions",
    "competition math functional equations integer answer",
    "Mathcounts national problems with solutions",
    "competition math problems on digits and base representation",
    "competition probability problems expected value",
    "competition math problems complex numbers roots of unity",
    "competition trigonometry problems with integer answers",
    "competition math inequalities AM-GM problems",
    "combinatorial geometry competition problems lattice points",
]
# The alternative we compared in seed_topics (#3), kept for reference: problems
# from a public dataset's subjects, retrieve_from_hf(SUBJECTS_DATASET, config=s).
SUBJECTS_DATASET = "EleutherAI/hendrycks_math"
SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
# Held out: problems like the ones we want the student to improve on.
VALIDATION = "HuggingFaceH4/MATH-500"
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


EXTRACT = """Below is text from a web page found by searching for "{query}".

List the competition math topics its problems test, at most five, each as a short
phrase specific enough to write a different problem from (e.g. "counting lattice
paths that avoid a blocked point", not "combinatorics"). List nothing if the page
has no competition problems.

{text}"""


def extract_prompt(row: dict) -> str:
    """The extraction prompt, with the page cut to its first 8,000 characters."""
    return EXTRACT.format(query=row["query"], text=row["text"][:8000])


@step
def seed_topics(model: Model) -> dict[str, list[dict]]:
    """Search the web for competition problems to identify salient topics.

    Justification: we compared 4 candidates:
        1. Write a list of topics ourselves from memory, and generate problems.
        2. Use the validation examples directly as seeds.
        3. Retrieve problems from a public dataset by subject (SUBJECTS) and
           rephrase them into new problems.
        4. Search the web (QUERIES), extract topics from the results, and
           generate problems.

    We generated 50 problems from each and used inspect.embedding_dispersion(...,
    reference=validation) to analyze their embeddings and overlap with the validation
    set. #3 saw 25% of its problems near a validation problem, which is too high -- the
    public dataset shares its source with our validation set, and rephrasing does not
    help enough. #2 came second (19%), because the extracted topics were too specific,
    leading to problems too close to the val set. #1 mostly avoided contamination and
    covered 48% of the validation set; it failed to cover a few important topics (e.g.
    geometry and precalculus). #4 achieved a coverage of 79% for #4, with Vendi 21
    against 36, and it's near-reference overlap was only 8%.
    Therefore, we picked #4: extract topics via search.
    """
    pages = acquire.retrieve_from_web(QUERIES, n=5)
    extracted = alter.generate_from_context(
        pages, extract_prompt, model, output_schema={"topics": list}, merge=True
    )
    topics = [
        {"topic": topic.strip(), "query": page["query"], "url": page["url"]}
        for page in extracted
        if isinstance(page["topics"], list)
        for topic in page["topics"]
        if isinstance(topic, str) and topic.strip()
    ]
    topics = filter.by_embedding_similarity(topics, field="topic", threshold=0.9)
    validation = acquire.retrieve_from_hf(VALIDATION, split="test", n=10**5)
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
    quality metric. All three models scored about the same, so we chose #3
    because it is cheaper (20% cheaper than #2 and 55% cheaper than #1).
    Averaged across models, #4 led to a higher pass@4 than #5 (80% vs 72%) at
    lower cost. Therefore, we use #3 and #4.

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
                "topic": d["topic"],
                "source_url": d["url"],
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

    Justification: we considered two pass@8 ranges, [2/8, 6/8] and [1/8, 7/8].
    On 892 problems, the former kept 361 and the latter 508, so we opted for
    the latter.
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
def check_rubrics(rows: list[dict], student: LocalModel, skyline: Model) -> list[dict]:
    r"""Score solutions, a skyline, probes and student samples with the rubrics.

    The skyline is a model stronger than the student (here the writer's model)
    solving each problem afresh: what a capable response earns.

    Justification: we compared two rubric criteria:
        1. Any \boxed{} in the response holds the answer.
        2. The last \boxed{} after the reasoning holds the answer.

    inspect.rubric_scores with the probes below paid 0.71 on average to
    every_box and 1.00 to unclosed_thinking under #1, against 0.00 under #2,
    while solutions scored 1.00 under both. number_from_statement scored 0.06
    under #1, which is why write drops answers printed in the statement.
    Therefore, we use #2. Under it, the skyline tied the solution on 182 of 200
    rows and scored below it on 18 (mean 0.91), while the student's samples
    averaged 0.52: room between the student and a stronger model.
    """
    rows = alter.generate_from_context(
        rows, "{prompt}", skyline, merge=True, output_field="skyline"
    )
    rows = [{**r, "skyline_solution": r["skyline"]["text"]} for r in rows]
    inspect.rubric_scores(
        rows,
        reward,
        n=200,
        response_fields=["solution", "skyline_solution"],
        probes=probes(),
        model=student,
        k=8,
    )
    return rows


@pipeline
def main(writer: Model, verifier: Model, student: LocalModel) -> list[dict]:
    """Topics, then verified problems, a difficulty band, and checked rubrics."""
    seeds = seed_topics(writer)
    rows = write(seeds["topics"], seeds["validation"], writer, verifier)
    rows = calibrate(rows, student)
    return check_rubrics(rows, student, skyline=writer)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "rl_math.jsonl")
    rows = main(
        writer=Model("WEAK_MODEL_ID", reasoning_effort="none"),
        verifier=Model("STRONG_MODEL_ID", reasoning_effort="low"),
        # RL's sampling settings, so calibrate predicts RL's groups.
        student=LocalModel("Qwen/Qwen3-8B", temperature=1.0, max_tokens=16384),
    )
    # What RL trains on: each prompt with the rubric that scores its responses.
    out.write_text(
        "".join(
            json.dumps({"prompt": r["prompt"], "rubric": r["rubric"]}) + "\n"
            for r in rows
        )
    )
    print(f"wrote {len(rows)} prompts and rubrics to {out}")

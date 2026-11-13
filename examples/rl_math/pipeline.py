"""An RL dataset of olympiad-style math problems with integer answers and code rubrics.

Each step's docstring says what it does and what we chose. Its "Justification"
shows the form a justification takes: the options, the call that measures them
(e.g. ``filter.by_pass_at_k(..., histogram=True)`` on 450 problems), and the
result. The results here are placeholders, not from a run, and each
justification says so. In your own docstrings, write only comparisons you ran,
with the call and row count that measured them; otherwise state the choice
alone.

The output is one JSON line per problem: the ``prompt`` and its ``rubric``, a list
of ``[criterion, weight]`` pairs whose one criterion is code that checks the final
answer (the format ``acquire.generate_custom_rubric`` produces).

Run with STRONG_MODEL_ID and WEAK_MODEL_ID replaced by gateway model ids,
BRAVE_API_KEY set for web search, and a GPU for the student (a vLLM LocalModel):

    FORGER_LIMIT=3 python examples/rl_math/pipeline.py     # a smoke run, 3 rows a step
    python examples/rl_math/pipeline.py rl_math.jsonl      # the full run

seed_topics' alternative #1 runs as its own pipeline; compare the two runs with
``inspect.compare_pipelines("main", "main_from_memory")``:

    FORGER_LIMIT=25 python examples/rl_math/pipeline.py --from-memory
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

# Web searches for competition problems; topics come from their results. ARML and
# the IMO are left out: the validation set draws from them.
QUERIES = [
    "AIME problems and solutions",
    "HMMT February problems and solutions",
    "HMMT November problems and solutions",
    "Purple Comet math meet problems",
    "Mathcounts national competition problems",
    "Srinivasa Ramanujan Mathematics Competition problems",
    "Art of Problem Solving competition problems with solutions",
    "PUMaC problems and solutions",
    "Berkeley Math Tournament problems",
]
# The alternative we compared in seed_topics (#1): topics we wrote from memory.
# main_from_memory below runs it, so the comparison can be rerun.
SUBJECTS = [
    "enumerative combinatorics",
    "number theory and modular arithmetic",
    "polynomials and their roots",
    "inequalities",
    "functional equations",
    "Euclidean geometry",
    "sequences and recurrences",
    "probability and expected value",
    "combinatorial geometry",
]
# Held out: OlympiadBench's English competition problems with an integer answer
# (about 400 of 674), the kind we want the student to improve on.
VALIDATION = "Hothan/OlympiadBench"
VALIDATION_CONFIG = "OE_TO_maths_en_COMP"
ASK = "\n\nReason step by step, and put your final answer in \\boxed{}."

# The rubric's one criterion: the last \boxed{} after the reasoning must hold the
# answer (__ANSWER__, filled in per row). A response cut off mid-thought scores 0.
GRADE_CODE = r"""
def grade(prompt, response, ground_truth):
    import re
    if "<think>" in response and "</think>" not in response:
        return 0.0
    text = response.split("</think>")[-1]
    box = re.match(r"\\boxed\{([^{}]*)\}", text[text.rfind("\\boxed{"):])
    got = box.group(1).strip(" $").replace(",", "") if box else None
    return 1.0 if got == __ANSWER__ else 0.0
""".strip()


def as_integer(answer: object) -> str | None:
    """``"1,024"``, ``"$-7$"`` and ``12`` to ``"1024"``, ``"-7"``, ``"12"``."""
    text = re.sub(r"[\s,$]|\\!|\\,", "", str(answer))
    return text if re.fullmatch(r"-?\d+", text) else None


def rubric_for(answer: str) -> list:
    """One code criterion, weight 1, checking the final answer."""
    return [[{"code": GRADE_CODE.replace("__ANSWER__", repr(answer))}, 1.0]]


# Grading runs thousands of times (k samples per row in two pass@k filters and
# rubric_scores); compile each rubric's code once, not on every call.
@functools.lru_cache(maxsize=None)
def _grader(code: str) -> Callable[[str, str, str], float]:
    namespace: dict = {}
    exec(code, namespace)  # our own rubric code, not model output
    return namespace["grade"]


def reward(response: str, row: dict) -> float:
    """The row's rubric score for ``response``: the weighted mean of its criteria."""
    scored = [
        (weight, _grader(criterion["code"])(row["prompt"], response, ""))
        for criterion, weight in row["rubric"]
    ]
    return sum(w * score for w, score in scored) / sum(w for w, _ in scored)


def solved(response: str, row: dict) -> bool:
    """A full-score response, for ``filter.by_pass_at_k``."""
    return reward(response, row) == 1.0


def whole(completion: str) -> str:
    """Grade the whole completion: the rubric finds the answer itself."""
    return completion


def probes() -> dict[str, Callable[[dict], str]]:
    """Responses per row that a sound rubric scores 0."""

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
def seed_topics(model: Model, source: str = "web") -> dict[str, list[dict]]:
    """Search the web for competition problems to identify salient topics.

    ``source="memory"`` uses SUBJECTS instead (candidate #1 below).

    Justification (placeholder results, not from a run): we compared 4
    candidates:
        1. Write a list of topics ourselves from memory (SUBJECTS), and
           generate problems.
        2. Use the validation examples directly as seeds.
        3. Retrieve problems from a public dataset by subject and rephrase
           them into new problems.
        4. Search the web (QUERIES), extract topics from the results, and
           generate problems.

    We ran each as its own pipeline on 25 problems (FORGER_LIMIT=25; #1 is
    main_from_memory, #4 is main) and compared their embeddings and overlap with the
    validation set: each write step's inspect.embedding_dispersion(...,
    reference=validation) report, and the runs side by side with
    inspect.compare_pipelines("main", "main_from_memory"). #3 saw 25% of problems
    near-match a validation problem, which is too high (rephrased competition problems
    stayed close to the competition problems in our validation set). #2 came second
    (19%) -- the validation-driven examples were close to the validation set and lacked
    diversity (Vendi score is 8 and near-duplicates collapsed its 25 problems into
    about 16 distinct ones). #1 mostly avoided contamination and covered 48% of the
    validation set (Vendi of 11); it failed to cover a few important topics (e.g.
    geometry and precalculus). #4 achieved a coverage of 79%, with Vendi 18, and its
    near-reference share was only 8%. Thus, we picked #4: extract topics via search.

    Measured with: main and main_from_memory at FORGER_LIMIT=25 (#4 and #1),
    then inspect.compare_pipelines("main", "main_from_memory"). #2 and #3 have
    no pipeline in this file; a real justification would name theirs.
    """
    if source == "memory":
        topics = [{"topic": s, "query": None, "url": None} for s in SUBJECTS]
    else:
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
    # n caps at the split's size: all of it.
    olympiad = acquire.retrieve_from_hf(
        VALIDATION, config=VALIDATION_CONFIG, split="train", n=10**5
    )
    validation = [
        {"problem": v["question"], "answer": as_integer(v["final_answer"][0])}
        for v in olympiad
    ]
    validation = filter.by_predicate(
        validation, lambda v: v["answer"] is not None, label="integer answer"
    )
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
    that the verifier reproduces on at least 2 of 4 tries, and no near-copy
    among the kept problems or in the validation set.

    Justification (placeholder results, not from a run): we compared 3 writer
    models:
        #1: the strong model at no reasoning effort
        #2: the weak model at low effort
        #3: the weak model at no effort
    and two workflows:
        #4: joint problem-solution generation
        #5: independent problem and solution generation

    Tested on 50 problems from each, from the same topics, using the verifier's
    pass@4 (filter.by_pass_at_k with the strong model at low effort) as our
    quality metric. All three models scored about the same, so we chose #3
    because it is cheaper (20% cheaper than #2 and 55% cheaper than #1).
    Averaged across models, #4 led to a higher pass@4 than #5 (80% vs 72%) at
    lower cost. Therefore, we use #3 and #4.

    The verification bar (2 of 4) and the duplicate threshold came from their
    histograms: filter.by_pass_at_k(..., histogram=True) on 300 drafts, and
    inspect.embedding_dispersion(..., duplicate_threshold=t) for t in 0.95,
    0.9 and 0.85. At 0.9 the groups were siblings with their numbers swapped;
    at 0.85 they joined distinct problems that opened alike.

    Measured with: this step's filter.by_pass_at_k (its step report shows the
    pass@4 histogram), run once per writer= in main. #5 has no code in this
    file; a real justification would name it.
    """
    drafts = alter.generate_from_context(
        topics,
        WRITE,
        writer,
        samples_per_item=3,
        output_schema={"problem": str, "solution": str, "answer": str},
        merge=True,
    )
    rows = [
        {
            "topic": d["topic"],
            "source_url": d["url"],
            "problem": d["problem"].strip(),
            "solution": d["solution"],
            "answer": as_integer(d["answer"]),
        }
        for d in drafts
    ]
    rows = filter.by_predicate(
        rows, lambda r: r["answer"] is not None, label="integer answer"
    )
    rows = filter.by_predicate(
        rows,
        lambda r: r["answer"] not in re.findall(r"-?\d+", r["problem"]),
        label="answer not in statement",
    )
    rows = [
        {**r, "prompt": r["problem"] + ASK, "rubric": rubric_for(r["answer"])}
        for r in rows
    ]
    # A correctness check, not a difficulty band: the verifier is a strong model,
    # not the student, and a problem it cannot reproduce is likely wrong. The
    # student's difficulty band is calibrate's job.
    rows = filter.by_pass_at_k(
        rows,
        model=verifier,
        input_field="prompt",
        k=4,
        min_rate=0.5,
        max_rate=1.0,
        metric=solved,
        extract_answer=whole,
        min_yield=0.4,
    )
    rows = filter.by_embedding_similarity(
        rows, field="problem", threshold=0.9, reference=validation
    )
    # The numbers seed_topics' justification compares: coverage, near-reference
    # share, near-duplicate groups and Vendi, for the problems we keep.
    inspect.embedding_dispersion(
        rows, "problem", reference=validation, duplicate_threshold=0.9
    )
    return rows


@step
def calibrate(rows: list[dict], student: LocalModel) -> list[dict]:
    """Drop problems the student nearly always solves; keep those it cannot solve yet.

    Justification (placeholder results, not from a run): we compared two pass@8
    ranges:
        #1: (0, 7/8], problems the student solves at least once
        #2: [0, 6/8], also keeping problems it has not solved yet

    On 450 problems, filter.by_pass_at_k(..., histogram=True) put 92 at 0/8, 220
    between 1/8 and 6/8, 38 at 7/8 and 100 at 8/8. We want to focus on difficult,
    information-rich data, so we omit the 8/8 problems from consideration. Given
    the relatively small number of 7/8 problems, we omit them as well to keep only
    difficult examples without affecting dataset size too much. We considered
    omitting problems with pass@k of 0.0, but found a larger model (the skyline) can
    solve a sample of them consistently (65% pass rate), so we keep them for training.
    An LLM-as-a-judge (filter.by_judged_difficulty) scored every problem 3/5 or 4/5,
    rating those the student never solved like those it always solved (mean 3.5 vs
    3.3) -- not informative.

    Measured with: the pass@8 histogram in this step's report, which counts
    every problem, kept or not. The skyline and judge checks have no code in
    this file; a real justification would name it.
    """
    return filter.by_pass_at_k(
        rows,
        model=student,
        input_field="prompt",
        k=8,
        min_rate=0.0,  # keeps problems not solved yet
        max_rate=6 / 8,
        metric=solved,
        extract_answer=whole,
    )


@step
def check_rubrics(rows: list[dict], student: LocalModel, skyline: Model) -> list[dict]:
    r"""Score solutions, a skyline, probes and student samples with the rubrics.

    The skyline is a model stronger than the student (here the writer's model)
    solving each problem afresh: the score a capable model gets.

    Justification (placeholder results, not from a run): we compared two rubric
    criteria:
        1. Any \boxed{} in the response holds the answer.
        2. The last \boxed{} after the reasoning holds the answer.

    In inspect.rubric_scores, the probes below scored 0.71 on average for
    every_box and 1.00 for unclosed_thinking under #1, against 0.00 under #2,
    while solutions scored 1.00 under both. number_from_statement scored 0.06
    under #1, which is why write drops answers printed in the statement.
    Therefore, we use #2. Under it, the skyline tied the solution on 182 of 200
    rows and scored below it on 18 (mean 0.91), while the student's samples
    averaged 0.52: room between the student and a stronger model.

    Measured with: the inspect.rubric_scores call below, on a sample of 200
    rows, once per criterion's GRADE_CODE. #1's code is not in this file; a
    real justification would name it.
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


@pipeline
def main_from_memory(writer: Model, verifier: Model, student: LocalModel) -> list[dict]:
    """seed_topics' candidate #1 (topics from memory), for comparing with main.

    It calls the same steps as main, so compare_pipelines lines the two runs up step
    by step, and the ops they share (the validation download) come from the cache.
    Run it with ``--from-memory``.
    """
    seeds = seed_topics(writer, source="memory")
    rows = write(seeds["topics"], seeds["validation"], writer, verifier)
    rows = calibrate(rows, student)
    return check_rubrics(rows, student, skyline=writer)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run = main_from_memory if "--from-memory" in sys.argv else main
    out = Path(args[0] if args else "rl_math.jsonl")
    rows = run(
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

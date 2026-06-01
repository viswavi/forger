"""basic_math_4.py - FINAL pipeline: challenging-but-basic math word problems.

Produces a GSM8K-style dataset of {"question", "answer"} pairs where each answer
is a multi-step solution with inline ``<<expr=result>>`` calculator annotations
and a final ``#### <number>`` line.

Pipeline (each acquire/judge stage is an @op, so reruns are cached & cheap):
  1. acquire : generate problems over the full (concept x scenario) grid
  2. filter  : local arithmetic verification + clean-numbers (free, strong)
  3. filter  : ROUGE-L dedup on the question text (diversity)
  4. filter  : independent LLM judge (stronger model) for well-posedness /
               correct modeling -- catches errors the verifier cannot
Output: examples/basic_math/basic_math_dataset.jsonl

Calibration (basic_math_2/3) measured: ~90% pass local verification, ~0% lost to
dedup (high diversity), ~92% kept by the judge. The generation prompt here adds
two fixes learned from inspecting judge rejections: ask for exactly ONE quantity,
and forbid mid-solution rounding/"adjusting" of numbers.

Run:
    export LITELLM_API_KEY=sk-...
    export LITELLM_BASE_URL="https://cmu.litellm.ai"
    export FORGER_CACHE=.forger_cache
    python examples/basic_math/basic_math_4.py
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

from dotenv import load_dotenv

import pyforger
from pyforger import filter as ffilter
from pyforger import op
from pyforger.acquire import Model
from pyforger.alter.generate import generate_from_context

from verify_utils import check_answer, check_question, has_clean_numbers

load_dotenv()

GEN_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
JUDGE_MODEL_ID = "litellm_proxy/gemini/gemini-3.5-flash"
JUDGE_THRESHOLD = 4

CONCEPTS = [
    "relative speed / catch-up (one mover chasing another)",
    "sequential fractions of a remaining quantity",
    "percentages, discounts, and sales tax",
    "ages of people across past/future years",
    "ratios and proportions",
    "combined work / rate problems (e.g. two pumps or workers)",
    "unit price, total cost, and change received",
    "averages and the mean of several quantities",
    "saving or spending money over several weeks",
    "counting and combining items in a collection",
    "distance = speed x time (single mover)",
    "area and perimeter of rectangles",
    "sharing or dividing items, with remainders",
    "comparisons using 'times as many' / multiples",
    "mixtures combining two or more quantities",
    "percentage increase or decrease across stages",
    "buying and selling for profit or loss",
    "unit conversion (minutes/hours, cm/m, etc.)",
    "totals across multiple people or groups",
    "repeated addition and simple arithmetic sequences",
]

SCENARIOS = [
    "a grocery shopping trip", "a baker in a bakery", "animals on a farm",
    "planting a vegetable garden", "a school classroom", "a sports team",
    "a birthday party with balloons", "books in a library", "a visit to the zoo",
    "a family road trip", "filling a swimming pool", "a lemonade stand",
    "a bag of candy", "saving coins in a piggy bank", "a toy factory",
    "following a cooking recipe", "a fishing trip", "marbles in jars",
    "harvesting fruit in an orchard", "buying movie tickets", "sharing a pizza",
    "commuting by bus or train", "a knitting project", "a charity fundraiser",
    "collecting trading cards", "birds at a feeder", "fish in an aquarium",
    "a bookstore sale", "stacking boxes in a warehouse", "a marathon race",
]

SYSTEM = (
    "You are an expert math problem author creating grade-school-style word "
    "problems to train a math reasoning assistant. You write clean, unambiguous "
    "problems and fully worked, arithmetically correct solutions."
)

INSTRUCTIONS = """Write ONE challenging but basic math word problem and its full solution.

Mathematical concept to use: {concept}
Real-world scenario: {scenario}

Strict requirements:
- The problem must be self-contained: every number needed is stated in the problem.
- Ask for exactly ONE numeric quantity at the end. Do NOT ask for two different \
things in the same problem.
- Solving it must require MULTIPLE steps (at least 3 distinct arithmetic operations).
- Use only basic arithmetic: addition, subtraction, multiplication, division \
(simple fractions, decimals, or percentages are fine). Do not require algebra.
- Choose the numbers so the arithmetic stays CLEAN: every intermediate result and the \
final result must come out to a whole number or a short decimal. NEVER round, \
truncate, or 'adjust' a number in the middle of the solution -- design the problem \
so this is never needed.
- Write a clear step-by-step solution. After EVERY arithmetic calculation, annotate it \
inline using the exact format <<expression=result>>, where `expression` contains ONLY \
digits and the operators + - * / and parentheses (no words, no % sign, no units).
  Example annotation: "she has 3*12 = <<3*12=36>> eggs".
- The VERY LAST calculation in the solution must produce the final answer, and the \
solution must end with a line "#### <final_number>" containing exactly that number.
- Make the arithmetic genuinely correct. Re-check every <<...>> calculation.

Respond with a JSON object with exactly two keys:
  "question": the problem statement (a string),
  "answer": the full step-by-step solution ending with the #### line (a string).
Output only the JSON object, no markdown fences, no extra text."""

JUDGE_SYSTEM = (
    "You are a meticulous math teacher grading auto-generated word problems that "
    "will be used to train a student model. You reward problems that are clear, "
    "unambiguous, and correctly solved, and you penalize anything flawed."
)

JUDGE_USER = """Evaluate the following math word problem and its solution.

PROBLEM:
{question}

SOLUTION:
{answer}

Score it from 1 to 5 on ALL of these criteria together:
- The problem is self-contained, unambiguous, and solvable from the numbers given.
- The problem asks for exactly one quantity.
- The solution's reasoning correctly models the problem (right operations, right setup).
- The final number after '####' is actually correct for the question asked.
Give 5 only if the problem is clean AND the solution is fully correct. Give 1-2 if the
problem is ambiguous/under-specified, asks for two things, or the final answer is wrong.

Respond with ONLY a raw JSON object, no markdown, no code fences, no extra text.
Format: {{"score": <integer 1 to 5>}}"""


def build_prompt(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTIONS.format_map(item)},
    ]


def build_seeds(seed: int = 0) -> list[dict]:
    """Full concept x scenario grid, shuffled for balanced batching."""
    combos = [{"concept": c, "scenario": s} for c in CONCEPTS for s in SCENARIOS]
    random.Random(seed).shuffle(combos)
    return combos


@op
def generate_problems(seeds: list[dict], model: Model) -> list[dict]:
    raw = generate_from_context(
        seeds,
        build_prompt,
        model,
        samples_per_item=1,
        output_schema={"question": str, "answer": str},
        temperature=1.0,
        batch_size=25,
    )
    return [r for r in raw if r is not None]


def verify_filter(problems: list[dict]) -> list[dict]:
    """Free local filters: arithmetic correctness + clean numbers + sane question."""
    kept = []
    for p in problems:
        q, a = p.get("question", ""), p.get("answer", "")
        if not check_question(q)[0]:
            continue
        if not check_answer(a)[0]:
            continue
        if not has_clean_numbers(a):
            continue
        kept.append({"question": q, "answer": a})
    return kept


@op
def judge_scores(problems: list[dict], judge_model: Model) -> list[int | None]:
    """One integer score (or None) per problem from an independent stronger judge."""
    def judge_prompt(item: dict) -> list[dict]:
        return [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    question=item["question"], answer=item["answer"]
                ),
            },
        ]

    results = generate_from_context(
        problems,
        judge_prompt,
        judge_model,
        samples_per_item=1,
        output_schema={"score": int},
        temperature=0.0,
        batch_size=25,
    )
    scores: list[int | None] = []
    for r in results:
        if r is None:
            scores.append(None)
            continue
        try:
            scores.append(int(r["score"]))
        except (KeyError, ValueError, TypeError):
            scores.append(None)
    return scores


def main() -> None:
    if "LITELLM_API_KEY" not in os.environ:
        raise SystemExit("Please set LITELLM_API_KEY.")

    gen_model = Model(
        GEN_MODEL_ID,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ.get("LITELLM_BASE_URL"),
        temperature=1.0,
    )
    judge_model = Model(
        JUDGE_MODEL_ID,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ.get("LITELLM_BASE_URL"),
        temperature=0.0,
    )

    seeds = build_seeds()
    print(f"[0] seeds (concept x scenario grid): {len(seeds)}")

    raw = pyforger.run(generate_problems, seeds, gen_model)
    print(f"[1] generated (parsed JSON): {len(raw)}")

    verified = verify_filter(raw)
    print(f"[2] passed local verification + clean-numbers: {len(verified)}")

    deduped = ffilter.by_rouge_l(verified, field="question", threshold=0.6)
    print(f"[3] after ROUGE-L question dedup: {len(deduped)}")

    scores = pyforger.run(judge_scores, deduped, judge_model)
    print(f"[4] judge score distribution: {dict(sorted(Counter(scores).items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    final = [
        p for p, s in zip(deduped, scores)
        if s is not None and s >= JUDGE_THRESHOLD
    ]
    print(f"[5] FINAL kept (judge score >= {JUDGE_THRESHOLD}): {len(final)}")

    out_path = os.path.join(os.path.dirname(__file__), "basic_math_dataset.jsonl")
    with open(out_path, "w") as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(final)} problems to {out_path}")


if __name__ == "__main__":
    main()

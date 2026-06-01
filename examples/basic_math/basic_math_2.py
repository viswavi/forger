"""basic_math_2.py - calibration run for the math word-problem pipeline.

This iteration exercises the full pipeline at medium scale to measure the
keep-rate of each stage before committing to a large final run:

  1. acquire  : generate problems from (concept x scenario) seeds
  2. filter   : local arithmetic verification (free, strong)
  3. filter   : clean-number check (drop ugly repeating decimals)
  4. filter   : ROUGE-L dedup on the question text (diversity)
  5. filter   : LLM-as-judge for well-posedness / correct modeling

The local verifier guarantees the arithmetic *inside* each solution is
self-consistent, but it cannot tell whether the solution actually models the
question correctly. The LLM judge (a stronger model) closes that gap.

Run:
    export LITELLM_API_KEY=sk-...
    export LITELLM_BASE_URL="https://cmu.litellm.ai"
    python examples/basic_math/basic_math_2.py
"""
from __future__ import annotations

import json
import os
import random

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

# Diverse basic-math concepts (the "method" axis).
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

# Everyday scenarios (the "surface form" axis), for diversity of phrasing.
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
- Solving it must require MULTIPLE steps (at least 3 distinct arithmetic operations).
- Use only basic arithmetic: addition, subtraction, multiplication, division \
(simple fractions, decimals, or percentages are fine). Do not require algebra.
- Choose the numbers so the arithmetic stays CLEAN: intermediate and final results \
should be whole numbers or short decimals. Avoid long or repeating decimals.
- The answer must be a single number.
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

JUDGE_CRITERIA = (
    "The problem must be well-posed, self-contained, and unambiguous, and the "
    "step-by-step solution must correctly model and solve it, arriving at the "
    "correct final number after the '####'. Give a low score if the problem is "
    "ambiguous or under-specified, if the solution's setup does not actually "
    "answer the question asked, or if the final answer is wrong."
)


def build_prompt(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTIONS.format_map(item)},
    ]


def build_seeds(n: int, seed: int = 0) -> list[dict]:
    """Build n distinct (concept, scenario) seeds, spread across both axes."""
    combos = [{"concept": c, "scenario": s} for c in CONCEPTS for s in SCENARIOS]
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:n]


@op
def generate_problems(seeds: list[dict], model: Model) -> list[dict]:
    raw = generate_from_context(
        seeds,
        build_prompt,
        model,
        samples_per_item=1,
        output_schema={"question": str, "answer": str},
        temperature=1.0,
        batch_size=20,
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

    seeds = build_seeds(n=80)
    raw = pyforger.run(generate_problems, seeds, gen_model)
    print(f"[1] generated (parsed JSON): {len(raw)} / {len(seeds)} seeds")

    verified = verify_filter(raw)
    print(f"[2] passed local verification + clean-numbers: {len(verified)}")

    deduped = ffilter.by_rouge_l(verified, field="question", threshold=0.6)
    print(f"[3] after ROUGE-L question dedup: {len(deduped)}")

    judged = ffilter.by_llm_score(
        deduped,
        model=judge_model,
        input_field="question",
        output_field="answer",
        score_threshold=4,
        max_score=5,
        num_judges=1,
        agreement_threshold=1.0,
        criteria=JUDGE_CRITERIA,
        temperature=0.0,
    )
    print(f"[4] after LLM-judge (score>=4/5): {len(judged)}")

    out_path = os.path.join(os.path.dirname(__file__), "calibration_sample.jsonl")
    with open(out_path, "w") as f:
        for p in judged:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(judged)} problems to {out_path}")

    print("\n=== Sample of final problems ===")
    for p in judged[:4]:
        print("Q:", p["question"])
        print("A:", p["answer"])
        print()


if __name__ == "__main__":
    main()

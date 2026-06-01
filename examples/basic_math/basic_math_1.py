"""basic_math_1.py - small smoke test for the math word-problem pipeline.

Goal of this iteration: confirm connectivity, confirm the JSON output format
parses, and confirm the local arithmetic verifier accepts model outputs at a
reasonable rate. Generates only a handful of problems.

Run:
    export LITELLM_API_KEY=sk-...
    export LITELLM_BASE_URL="https://cmu.litellm.ai"
    python examples/basic_math/basic_math_1.py
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv

import pyforger
from pyforger import op
from pyforger.acquire import Model
from pyforger.alter.generate import generate_from_context

from verify_utils import check_answer, check_question

load_dotenv()

MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"

# A few (concept, scenario) seeds just to test the format end to end.
SEEDS = [
    {"concept": "relative speed / catch-up", "scenario": "a cyclist chasing a bus"},
    {"concept": "fractions of a quantity", "scenario": "a baker using bags of flour"},
    {"concept": "percentages and discounts", "scenario": "shopping for shoes on sale"},
    {"concept": "ages of people", "scenario": "a family's ages over several years"},
    {"concept": "ratios and proportions", "scenario": "mixing paint colors"},
    {"concept": "work / rate problems", "scenario": "two pumps filling a tank"},
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
(fractions/decimals/percentages are fine). No algebra variables in the final answer.
- The answer must be a single number.
- Write a clear step-by-step solution. After EVERY arithmetic calculation, annotate it \
inline using the exact format <<expression=result>>, where `expression` contains ONLY \
digits and the operators + - * / and parentheses (no words, no % sign, no units).
  Example annotation: "she has 3*12 = <<3*12=36>> eggs".
- The VERY LAST calculation in the solution must produce the final answer, and the \
solution must end with a line "#### <final_number>" containing exactly that number.
- Make the arithmetic genuinely correct. Double-check every <<...>> calculation.

Respond with a JSON object with exactly two keys:
  "question": the problem statement (a string),
  "answer": the full step-by-step solution ending with the #### line (a string).
Output only the JSON object, no markdown fences, no extra text."""


def build_prompt(item: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTIONS.format_map(item)},
    ]


@op
def generate_problems(seeds: list[dict], model: Model) -> list[dict]:
    schema = {"question": str, "answer": str}
    raw = generate_from_context(
        seeds,
        build_prompt,
        model,
        samples_per_item=1,
        output_schema=schema,
        temperature=1.0,
        batch_size=len(seeds),
    )
    return [r for r in raw if r is not None]


def main() -> None:
    if "LITELLM_API_KEY" not in os.environ:
        raise SystemExit("Please set LITELLM_API_KEY.")
    model = Model(
        MODEL_ID,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ.get("LITELLM_BASE_URL"),
        temperature=1.0,
    )

    problems = pyforger.run(generate_problems, SEEDS, model)
    print(f"Generated {len(problems)} parsed problems from {len(SEEDS)} seeds.\n")

    n_valid = 0
    for i, p in enumerate(problems):
        q_ok, q_reason = check_question(p.get("question", ""))
        a_ok, a_reason = check_answer(p.get("answer", ""))
        valid = q_ok and a_ok
        n_valid += valid
        print(f"--- Problem {i} | valid={valid} (q:{q_reason}, a:{a_reason}) ---")
        print("Q:", p.get("question", ""))
        print("A:", p.get("answer", ""))
        print()

    print(f"VALID: {n_valid}/{len(problems)}")
    if model.last_usage:
        print("Last batch usage:", model.last_usage)


if __name__ == "__main__":
    main()

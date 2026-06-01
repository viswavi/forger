"""Instruction-following dataset generation — iteration 1 (small validation run).

Goal: generate challenging instruction-following (prompt, answer) pairs across a
broad array of domains (programming, math, home/DIY, personal/business advice,
logic puzzles, writing-with-constraints, etc.) for training an expert assistant.

Constraints discovered about the CMU LiteLLM proxy:
  * Model id must be ``litellm_proxy/gemini/gemini-3.1-flash-lite`` (and the
    ``gemini-3.5-flash`` variant for harder answering).
  * Gemini rejects ``n > 1`` ("Multiple candidates is not enabled"), so every
    generation uses ``samples_per_item=1`` via ``generate_from_context`` (which
    batches DISTINCT prompts at n=1) and judges use ``num_judges=1``.
  * Gemini requires ``temperature=1.0``.

Two-stage design:
  Stage 1 (cheap, creative model): pose one challenging question per (domain,
    angle) seed.
  Stage 2 (strong model): produce a correct, fully instruction-following answer.
  Then dedup (ROUGE-L) and quality-filter (LLM judge).

This file only runs a tiny smoke test and prints results + cost.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-aVC1-6i1uXBnNxvZf8C45w")
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), ".forger_cache"),
)

from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402

API_KEY = os.environ["LITELLM_API_KEY"]
API_BASE = "https://cmu.litellm.ai"
GEN_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
STRONG_MODEL_ID = "litellm_proxy/gemini/gemini-3.5-flash"


def make_model(model_id: str) -> Model:
    return Model(model_id, api_key=API_KEY, api_base=API_BASE, temperature=1.0)


# A few (domain, angle) seeds just to validate the end-to-end flow.
TEST_SEEDS = [
    {"domain": "competitive programming in C++", "angle": "a short problem with a tricky edge case, including expected I/O"},
    {"domain": "personal finance advice", "angle": "a constraint-puzzle (e.g. reach an exact amount under specific rules)"},
    {"domain": "creative writing with hard format constraints", "angle": "an exact word count plus a letter/structure rule"},
]


def question_prompt(item: dict) -> str:
    return (
        "You are designing a CHALLENGING instruction-following task to train an "
        "expert AI assistant.\n"
        f"Domain: {item['domain']}.\n"
        f"Style: {item['angle']}.\n\n"
        "Write ONE self-contained user request that (a) is genuinely difficult, "
        "(b) contains clear, checkable instructions or constraints the assistant "
        "must follow exactly, and (c) has a single well-defined correct response. "
        "Make it specific and realistic, not generic. Do not include the answer.\n\n"
        'Respond with a JSON object only, no markdown fences. Required key: '
        '["prompt"]. Example: {"prompt": "<the user request>"}'
    )


def answer_prompt(item: dict) -> str:
    return (
        "You are an expert assistant. Respond to the following user request. "
        "Follow EVERY instruction and constraint exactly. Be correct, complete, "
        "and concise. If code is requested, give correct, runnable code. If a "
        "format/length/structure is specified, satisfy it precisely.\n\n"
        f"USER REQUEST:\n{item['prompt']}\n\n"
        'Respond with a JSON object only, no markdown fences. Required key: '
        '["answer"]. Put your full response (markdown allowed inside the string) '
        'as the value. Example: {"answer": "<your full response>"}'
    )


def main() -> None:
    gen = make_model(GEN_MODEL_ID)
    strong = make_model(STRONG_MODEL_ID)

    questions = generate_from_context(
        TEST_SEEDS, question_prompt, gen,
        samples_per_item=1, output_schema={"prompt": str},
        temperature=1.0, max_retries=2, batch_size=8,
    )
    q_cost = gen.last_usage["total_cost"] if gen.last_usage else 0.0
    questions = [q for q in questions if q is not None]
    print(f"Generated {len(questions)} questions (cost ~${q_cost:.4f}):")
    for q in questions:
        print("  Q:", q["prompt"][:160])

    answers = generate_from_context(
        questions, answer_prompt, strong,
        samples_per_item=1, output_schema={"answer": str},
        temperature=1.0, max_retries=2, batch_size=8,
    )
    a_cost = strong.last_usage["total_cost"] if strong.last_usage else 0.0

    pairs = []
    for q, a in zip(questions, answers):
        if a is None:
            continue
        pairs.append({"prompt": q["prompt"], "answer": a["answer"]})

    print(f"\nProduced {len(pairs)} pairs (answer cost ~${a_cost:.4f}):\n")
    for p in pairs:
        print("=" * 80)
        print("PROMPT:", p["prompt"])
        print("-" * 40)
        print("ANSWER:", p["answer"][:600])
    print("\nTOTAL approx cost: $%.4f" % (q_cost + a_cost))


if __name__ == "__main__":
    main()

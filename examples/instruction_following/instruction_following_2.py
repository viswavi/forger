"""Instruction-following dataset generation — iteration 2 (full pipeline, TEST toggle).

Pipeline (all generation uses ``generate_from_context`` with samples_per_item=1
because Gemini on the CMU proxy rejects n>1):

  1. acquire/pose: cheap creative model writes ONE challenging, constraint-rich
     user request per (domain, angle) seed.
  2. dedup: drop None + near-duplicate prompts (ROUGE-L).
  3. answer: STRONG model produces a correct, concise, fully instruction-following
     answer (capped max_tokens to control cost and match the target answer style).
  4. judge: LLM-as-judge (strong model, strict) keeps only high-quality pairs.

Set TEST=True for a small validation run that prints cost; TEST=False for the
full run that writes JSONL.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-aVC1-6i1uXBnNxvZf8C45w")
os.environ.setdefault(
    "FORGER_CACHE", os.path.join(os.path.dirname(__file__), ".forger_cache")
)

import pyforger  # noqa: E402
from pyforger import filter as ffilter  # noqa: E402
from pyforger import op  # noqa: E402
from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402

API_KEY = os.environ["LITELLM_API_KEY"]
API_BASE = "https://cmu.litellm.ai"
GEN_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
STRONG_MODEL_ID = "litellm_proxy/gemini/gemini-3.5-flash"

TEST = True


def make_model(model_id: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_id, api_key=API_KEY, api_base=API_BASE,
        temperature=1.0, max_tokens=max_tokens, num_retries=2,
    )


# ---------------------------------------------------------------------------
# Seed grid: broad array of domains x constraint-rich angles.
# ---------------------------------------------------------------------------
DOMAINS = [
    "competitive programming in C++ (with input/output spec)",
    "a Python coding task (functions, edge cases)",
    "a SQL query problem over a described schema",
    "a bash / shell one-liner or script task",
    "a regular-expression construction task",
    "a JavaScript / front-end logic task",
    "a data-structures & algorithms problem (explain + code)",
    "a debugging task: find and fix the bug in given code",
    "a pandas / data-wrangling task",
    "algebra and equation solving",
    "combinatorics and counting",
    "probability and expected value",
    "number theory (divisibility, modular arithmetic, primes)",
    "geometry (with a numeric answer)",
    "a multi-step word problem",
    "a logic / constraint-satisfaction puzzle",
    "a lateral-thinking riddle with a unique answer",
    "a 'reach an exact value under rules' puzzle (like making change)",
    "home repair / DIY troubleshooting",
    "cooking or baking with quantity/substitution constraints",
    "travel itinerary planning under constraints",
    "scheduling / time-zone arithmetic",
    "personal advice with a concrete dilemma",
    "career advice for a specific situation",
    "business strategy or pricing decision",
    "marketing copy under strict constraints",
    "negotiation or communication scripting",
    "a medical-advisor triage scenario (general first aid / when to seek care)",
    "personal-finance planning with numbers",
    "creative writing with an exact word count",
    "an acrostic / constrained-letter writing task",
    "writing that must avoid a specific letter or follow a structure",
    "produce valid JSON/YAML matching a precise spec",
    "design a regex AND explain what it matches",
    "explain a science concept at a specified level and length",
    "a grammar / language-usage correction task",
    "a translation or wordplay puzzle with constraints",
    "convert / reformat data into a required structure (table, CSV, etc.)",
    "an estimation / Fermi problem with a method",
    "a step-by-step how-to with a strict numbered format",
]

ANGLES = [
    "Embed at least two explicit, checkable constraints (format, length, value, or structure).",
    "Specify the exact output format the assistant must produce.",
    "Include a subtle edge case that a careless solver would miss.",
    "Require a single, unambiguous correct answer.",
    "Add a numeric or counting constraint that must be satisfied exactly.",
    "Demand a specific length, ordering, or structural rule in the response.",
]


def build_seeds(target: int) -> list[dict]:
    seeds: list[dict] = []
    i = 0
    while len(seeds) < target:
        domain = DOMAINS[i % len(DOMAINS)]
        angle = ANGLES[(i // len(DOMAINS)) % len(ANGLES)]
        seeds.append({"domain": domain, "angle": angle, "nonce": i})
        i += 1
    return seeds


def question_prompt(item: dict) -> str:
    return (
        "You are designing a CHALLENGING instruction-following task to train an "
        "expert AI assistant.\n"
        f"Domain: {item['domain']}.\n"
        f"Requirement: {item['angle']}\n\n"
        "Write ONE self-contained user request that (a) is genuinely difficult, "
        "(b) contains clear, checkable instructions/constraints the assistant must "
        "follow exactly, and (c) has a single well-defined correct response. Make it "
        "specific, realistic and concrete (include any data, schema, or numbers "
        "needed to solve it). Do NOT include the answer or solution.\n\n"
        'Respond with a JSON object only, no markdown fences. Required key: '
        '["prompt"]. Example: {"prompt": "<the user request>"}'
    )


def answer_prompt(item: dict) -> str:
    return (
        "You are an expert assistant. Respond to the user request below. Follow "
        "EVERY instruction and constraint exactly. Be correct and complete, but "
        "concise: no filler, no restating the question. If code is requested, give "
        "correct, runnable code. If a format/length/value/structure is specified, "
        "satisfy it precisely and double-check it.\n\n"
        f"USER REQUEST:\n{item['prompt']}\n\n"
        'Respond with a JSON object only, no markdown fences. Required key: '
        '["answer"]. Put your full response (markdown allowed inside the string) as '
        'the value. Example: {"answer": "<your full response>"}'
    )


JUDGE_CRITERIA = (
    "Score 5 only if: the answer is CORRECT, it FOLLOWS EVERY explicit constraint "
    "in the request exactly (format, length, counts, structure), and the request "
    "itself is a clear, challenging, well-posed instruction-following task. Score 1-2 "
    "if the answer is wrong, violates a stated constraint, the task is trivial/vague, "
    "or the answer is incomplete/rambling."
)


@op(name="ifds.pose_questions")
def pose_questions(seeds, model):
    raw = generate_from_context(
        seeds, question_prompt, model,
        samples_per_item=1, output_schema={"prompt": str},
        temperature=1.0, max_retries=2, batch_size=16,
    )
    return [q for q in raw if q is not None and q.get("prompt", "").strip()]


@op(name="ifds.answer_questions")
def answer_questions(questions, model):
    answers = generate_from_context(
        questions, answer_prompt, model,
        samples_per_item=1, output_schema={"answer": str},
        temperature=1.0, max_retries=2, batch_size=16,
    )
    pairs = []
    for q, a in zip(questions, answers):
        if a is None or not a.get("answer", "").strip():
            continue
        pairs.append({"prompt": q["prompt"], "answer": a["answer"]})
    return pairs


def build_dataset(seeds, gen_model, strong_model):
    questions = pose_questions(seeds, gen_model)
    questions = ffilter.by_rouge_l(questions, field="prompt", threshold=0.55)
    pairs = answer_questions(questions, strong_model)
    pairs = ffilter.by_llm_score(
        pairs, model=strong_model, input_field="prompt", output_field="answer",
        score_threshold=4, max_score=5, num_judges=1, temperature=0.0,
        criteria=JUDGE_CRITERIA,
    )
    return questions, pairs


def main() -> None:
    gen = make_model(GEN_MODEL_ID, max_tokens=900)
    strong = make_model(STRONG_MODEL_ID, max_tokens=1600)

    target = 8 if TEST else 340
    seeds = build_seeds(target)

    questions, pairs = pyforger.run(build_dataset, seeds, gen, strong)

    gen_cost = gen.last_usage["total_cost"] if gen.last_usage else 0.0
    strong_cost = strong.last_usage["total_cost"] if strong.last_usage else 0.0

    print(f"\nseeds={len(seeds)}  questions(after dedup)={len(questions)}  "
          f"final_pairs={len(pairs)}")
    print(f"(last_usage cost — gen step: ${gen_cost:.4f}, strong step: ${strong_cost:.4f})")

    if TEST:
        for p in pairs:
            print("=" * 80)
            print("PROMPT:", p["prompt"][:300])
            print("-" * 40)
            print("ANSWER:", p["answer"][:500])
    else:
        out = os.path.join(os.path.dirname(__file__), "instruction_following.jsonl")
        with open(out, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()

"""Instruction-following dataset generation — iteration 3 (robust, TEST toggle).

Changes from iteration 2 (which failed because long code-bearing answers broke
strict JSON parsing, and the framework judge tripped on Gemini's markdown fences):

  * Questions and answers are generated as RAW TEXT (output_schema=None), so an
    answer containing code/quotes/newlines never has to survive JSON escaping or
    truncation.  ``generate_from_context`` stores raw output under ``{"text": ...}``.
  * A custom ``quality_filter`` @op grades each pair with the strong model using a
    JSON ``{"score": int}`` schema (n=1, temperature=1.0 as Gemini requires) and a
    strict rubric — more robust than the built-in judge against fenced output, and
    it sidesteps the n>1 / temperature constraints.

Pipeline: pose (cheap) -> ROUGE-L dedup -> answer (strong) -> quality filter (strong).
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

# ---------------------------------------------------------------------------
# Seed grid: broad array of domains x constraint-rich angles.
# ---------------------------------------------------------------------------
DOMAINS = [
    "competitive programming in C++ (give the full input/output spec)",
    "a Python coding task (functions, edge cases)",
    "a SQL query problem over a described schema",
    "a bash / shell one-liner or script task",
    "a regular-expression construction task",
    "a JavaScript / front-end logic task",
    "a data-structures & algorithms problem (explain + code)",
    "a debugging task: include buggy code and ask for the fix",
    "a pandas / data-wrangling task",
    "algebra and equation solving",
    "combinatorics and counting",
    "probability and expected value",
    "number theory (divisibility, modular arithmetic, primes)",
    "geometry with a numeric answer",
    "a multi-step quantitative word problem",
    "a logic / constraint-satisfaction puzzle",
    "a lateral-thinking riddle with a unique answer",
    "a 'reach an exact value under rules' puzzle (like making change with coins)",
    "home repair / DIY troubleshooting",
    "cooking or baking with quantity/substitution constraints",
    "travel itinerary planning under constraints",
    "scheduling / time-zone arithmetic",
    "personal advice with a concrete dilemma",
    "career advice for a specific situation",
    "business strategy or pricing decision",
    "marketing copy under strict length/keyword constraints",
    "negotiation or difficult-conversation scripting",
    "a medical-advisor triage scenario (general first aid / when to seek care)",
    "personal-finance planning with concrete numbers",
    "creative writing with an exact word count",
    "an acrostic or constrained-letter writing task",
    "writing that must avoid a specific letter or obey a structural rule",
    "produce valid JSON or YAML matching a precise spec",
    "design a regex AND explain exactly what it matches",
    "explain a science concept at a specified level and length",
    "a grammar / language-usage correction task",
    "a translation or wordplay puzzle with constraints",
    "convert or reformat data into a required structure (table, CSV, etc.)",
    "an estimation / Fermi problem requiring a stated method",
    "a step-by-step how-to in a strict numbered format",
]

ANGLES = [
    "Embed at least two explicit, checkable constraints (format, length, value, or structure).",
    "Specify the exact output format the assistant must produce.",
    "Include a subtle edge case that a careless solver would miss.",
    "Require a single, unambiguous correct answer.",
    "Add a numeric or counting constraint that must be satisfied exactly.",
    "Demand a specific length, ordering, or structural rule in the response.",
]


def make_model(model_id: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_id, api_key=API_KEY, api_base=API_BASE,
        temperature=1.0, max_tokens=max_tokens, num_retries=2,
    )


def build_seeds(target: int) -> list[dict]:
    seeds: list[dict] = []
    i = 0
    while len(seeds) < target:
        seeds.append({
            "domain": DOMAINS[i % len(DOMAINS)],
            "angle": ANGLES[(i // len(DOMAINS)) % len(ANGLES)],
            "nonce": i,
        })
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
        "specific, realistic and concrete: include any data, schema, code, or numbers "
        "needed to solve it. Do NOT include the answer or solution.\n\n"
        "Output ONLY the user request text, with no preamble, no quotes, and no "
        "labels like 'Prompt:'."
    )


def answer_prompt(item: dict) -> str:
    return (
        "You are an expert assistant. Respond to the user request below. Follow "
        "EVERY instruction and constraint exactly. Be correct and complete, but "
        "concise: no filler, no restating the question. If code is requested, give "
        "correct, runnable code. If a format/length/value/structure is specified, "
        "satisfy it precisely and verify it before answering.\n\n"
        f"USER REQUEST:\n{item['text']}\n\n"
        "Output ONLY your final response (markdown is fine). Do not include any "
        "meta-commentary about following the instructions."
    )


def judge_prompt(item: dict) -> str:
    return (
        "You are a strict grader of (request, answer) pairs used to train an expert "
        "assistant. Grade the answer from 1 to 5.\n"
        "Give 5 only if ALL hold: the answer is factually/logically CORRECT; it obeys "
        "EVERY explicit constraint in the request exactly (format, length, counts, "
        "structure); and the request is a clear, non-trivial instruction-following "
        "task. Give 1-2 if the answer is wrong, violates any stated constraint, the "
        "task is trivial or ill-posed, or the answer rambles/omits required parts. "
        "Be harsh about constraint violations (e.g. wrong word count).\n\n"
        f"REQUEST:\n{item['prompt']}\n\n"
        f"ANSWER:\n{item['answer']}\n\n"
        'Respond with a JSON object only, no markdown fences. Required key: '
        '["score"]. Example: {"score": 4}'
    )


@op(name="ifds.pose_questions")
def pose_questions(seeds, model):
    raw = generate_from_context(
        seeds, question_prompt, model,
        samples_per_item=1, output_schema=None,
        temperature=1.0, max_retries=1, batch_size=16,
    )
    out = []
    for q in raw:
        if q is None:
            continue
        text = q.get("text", "").strip()
        if len(text) > 40:  # drop empties / degenerate
            out.append({"prompt": text})
    return out


@op(name="ifds.answer_questions")
def answer_questions(questions, model):
    # answer_prompt reads item['text']; questions store the request under 'prompt'.
    items = [{"text": q["prompt"]} for q in questions]
    answers = generate_from_context(
        items, answer_prompt, model,
        samples_per_item=1, output_schema=None,
        temperature=1.0, max_retries=1, batch_size=16,
    )
    pairs = []
    for q, a in zip(questions, answers):
        if a is None or not a.get("text", "").strip():
            continue
        pairs.append({"prompt": q["prompt"], "answer": a["text"].strip()})
    return pairs


@op(name="ifds.quality_filter")
def quality_filter(pairs, model, threshold=4):
    scored = generate_from_context(
        pairs, judge_prompt, model,
        samples_per_item=1, output_schema={"score": int},
        temperature=1.0, max_retries=1, batch_size=16,
    )
    kept = []
    for pair, s in zip(pairs, scored):
        if s is None:
            continue
        try:
            score = int(s["score"])
        except (TypeError, ValueError, KeyError):
            continue
        if score >= threshold:
            kept.append(pair)
    return kept


def build_dataset(seeds, gen_model, strong_model):
    questions = pose_questions(seeds, gen_model)
    questions = ffilter.by_rouge_l(questions, field="prompt", threshold=0.55)
    pairs = answer_questions(questions, strong_model)
    kept = quality_filter(pairs, strong_model, threshold=4)
    return questions, pairs, kept


def main() -> None:
    gen = make_model(GEN_MODEL_ID, max_tokens=700)
    strong = make_model(STRONG_MODEL_ID, max_tokens=1600)

    target = 10 if TEST else 340
    seeds = build_seeds(target)

    questions, pairs, kept = pyforger.run(build_dataset, seeds, gen, strong)

    gen_cost = gen.last_usage["total_cost"] if gen.last_usage else 0.0
    strong_cost = strong.last_usage["total_cost"] if strong.last_usage else 0.0
    print(f"\nseeds={len(seeds)}  questions(dedup)={len(questions)}  "
          f"answered={len(pairs)}  kept={len(kept)}")
    print(f"(last_usage — gen step ${gen_cost:.4f}, strong step ${strong_cost:.4f})")

    if TEST:
        for p in kept:
            print("=" * 80)
            print("PROMPT:", p["prompt"][:400])
            print("-" * 40)
            print("ANSWER:", p["answer"][:600])
    else:
        out = os.path.join(os.path.dirname(__file__), "instruction_following.jsonl")
        with open(out, "w") as f:
            for p in kept:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(kept)} pairs to {out}")


if __name__ == "__main__":
    main()

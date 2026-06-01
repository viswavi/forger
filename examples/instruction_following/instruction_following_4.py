"""Instruction-following dataset generation — iteration 4 (final pipeline).

Generates challenging instruction-following (prompt, answer) pairs across a broad
array of domains (programming in many languages, math, logic puzzles, home/DIY,
personal/business advice, medical triage, constrained writing, data formatting,
...) for training an expert assistant.

Why the design looks like this (constraints of the CMU LiteLLM Gemini proxy):
  * Model ids are ``litellm_proxy/gemini/gemini-3.1-flash-lite`` (cheap poser) and
    ``litellm_proxy/gemini/gemini-3.5-flash`` (strong answerer/judge).
  * Gemini rejects n>1 ("Multiple candidates is not enabled"), so ALL generation
    runs through ``generate_from_context`` with ``samples_per_item=1`` (it batches
    DISTINCT prompts at n=1) and the judge uses a single grader.
  * Gemini requires ``temperature=1.0``.
  * Answers are RAW TEXT (output_schema=None) so code/quotes/newlines never have to
    survive JSON escaping; only the short judge step uses a JSON schema.

Pipeline (topic-based generation + overgenerate-and-filter):
  pose questions (cheap)  ->  ROUGE-L dedup  ->  answer (strong)  ->  LLM-judge
  quality filter (strong).

Scale via env vars (so the same file runs a smoke test or the full job):
  IFDS_TARGET   number of seed tasks to generate (default 450)
  IFDS_ANS_TOK  max answer tokens (default 2800)
  IFDS_OUT      output JSONL path
Run a quick check first:  IFDS_TARGET=16 python instruction_following_4.py
Then the full run:        python instruction_following_4.py
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

# ---------------------------------------------------------------------------
# Seed grid: broad array of domains x constraint-rich angles.
# ---------------------------------------------------------------------------
DOMAINS = [
    "competitive programming in C++ (give the full input/output spec and an example)",
    "a Python coding task (functions, tricky edge cases)",
    "a SQL query problem over a described schema",
    "a bash / shell scripting task",
    "a regular-expression construction task",
    "a JavaScript / TypeScript logic task",
    "a data-structures & algorithms problem (explain the approach + give code)",
    "a debugging task: include short buggy code and ask for the corrected version",
    "a pandas / data-wrangling task with a small described dataframe",
    "a Rust or Go coding task",
    "algebra and equation solving",
    "combinatorics and counting",
    "probability and expected value",
    "number theory (divisibility, modular arithmetic, primes)",
    "geometry with a numeric answer",
    "calculus (derivative/integral/limit) with a clean answer",
    "a multi-step quantitative word problem",
    "a logic / constraint-satisfaction puzzle",
    "a lateral-thinking riddle with a unique answer",
    "a 'reach an exact value under rules' puzzle (e.g. making change with coins)",
    "home repair / DIY troubleshooting",
    "cooking or baking with quantity/substitution constraints",
    "travel itinerary planning under explicit constraints",
    "scheduling or time-zone arithmetic",
    "personal advice with a concrete dilemma",
    "career advice for a specific situation",
    "business strategy or pricing decision with numbers",
    "marketing copy under strict length/keyword constraints",
    "negotiation or difficult-conversation scripting",
    "a medical-advisor triage scenario (general first aid / when to seek care)",
    "personal-finance planning with concrete numbers",
    "creative writing with an exact word count",
    "an acrostic or constrained-letter writing task",
    "writing that must avoid a specific letter or obey a structural rule",
    "produce valid JSON or YAML matching a precise spec",
    "explain a science concept at a specified level and length",
    "a grammar / language-usage correction task",
    "a translation or wordplay puzzle with constraints",
    "convert or reformat data into a required structure (markdown table, CSV, etc.)",
    "an estimation / Fermi problem requiring a stated method and a final number",
    "a step-by-step how-to in a strict numbered format",
    "a unit-conversion or dimensional-analysis problem",
    "a chemistry or physics calculation with units",
    "a spreadsheet formula construction task (e.g. Excel/Sheets)",
    "an ethics or trade-off scenario requiring a reasoned recommendation",
]

ANGLES = [
    "Embed at least two explicit, checkable constraints (format, length, value, or structure).",
    "Specify the exact output format the assistant must produce.",
    "Include a subtle edge case that a careless solver would miss.",
    "Require a single, unambiguous correct answer.",
    "Add a numeric or counting constraint that must be satisfied exactly.",
    "Demand a specific length, ordering, or structural rule in the response.",
    "Make it a realistic scenario someone would actually ask, but non-trivial.",
    "Require combining two skills or steps to reach the answer.",
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
        "Write ONE self-contained user request that (a) is genuinely challenging but "
        "CLEANLY and UNAMBIGUOUSLY solvable by an expert in a focused response (about "
        "60-350 words, or a single code block), (b) contains clear, checkable "
        "instructions/constraints the assistant must follow exactly, and (c) has a "
        "single well-defined correct response. Make it specific, realistic and "
        "concrete: include any data, schema, code, or numbers needed to solve it. "
        "Avoid tasks that require a huge derivation, brute-force search, or pages of "
        "work to answer. Do NOT include the answer or solution.\n\n"
        "Output ONLY the user request text, with no preamble, no surrounding quotes, "
        "and no labels like 'Prompt:'."
    )


def answer_prompt(item: dict) -> str:
    return (
        "You are an expert assistant. Respond to the user request below. Follow "
        "EVERY instruction and constraint exactly. Be correct and complete, but "
        "concise: no filler, no restating the question, and keep it under ~400 words. "
        "If code is requested, give correct, runnable code. If the request imposes a "
        "strict output format, length, count, or structure, satisfy it EXACTLY — that "
        "matters more than showing your work. Verify any such constraint before "
        "answering.\n\n"
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
        "task. Give 1-2 if the answer is wrong, violates any stated constraint, looks "
        "truncated/incomplete, the task is trivial or ill-posed, or the answer "
        "rambles. Be harsh about constraint violations (e.g. wrong word count).\n\n"
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
    target = int(os.environ.get("IFDS_TARGET", "450"))
    ans_tok = int(os.environ.get("IFDS_ANS_TOK", "2800"))
    out_path = os.environ.get(
        "IFDS_OUT",
        os.path.join(os.path.dirname(__file__), "instruction_following.jsonl"),
    )

    gen = make_model(GEN_MODEL_ID, max_tokens=700)
    strong = make_model(STRONG_MODEL_ID, max_tokens=ans_tok)

    seeds = build_seeds(target)
    questions, pairs, kept = pyforger.run(build_dataset, seeds, gen, strong)

    gen_cost = gen.last_usage["total_cost"] if gen.last_usage else 0.0
    strong_cost = strong.last_usage["total_cost"] if strong.last_usage else 0.0
    print(f"\nseeds={len(seeds)}  questions(dedup)={len(questions)}  "
          f"answered={len(pairs)}  kept={len(kept)}")
    print(f"(last_usage — gen step ${gen_cost:.4f}, strong step ${strong_cost:.4f})")

    with open(out_path, "w") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} pairs to {out_path}")


if __name__ == "__main__":
    main()

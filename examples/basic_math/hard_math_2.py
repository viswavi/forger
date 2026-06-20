"""hard_math_2.py -- topic-conditioned generation + adaptive flash verification.

Pipeline:
  1. Generate AIME-style problems with a cheap writer (gemini-3.1-flash-lite),
     conditioned on a randomly sampled (topic, subtopic).  Writer emits only
     {question, answer} for JSON robustness.
  2. Early diversity dedup on the question text (by_rouge_l).
  3. Verify each problem by solving it independently with a STRONG model
     (gemini-3.5-flash): 2 solves; accept on 2/2 agreement, else a 3rd solve and
     accept on 2/3.  The agreed integer becomes the verified gold answer and the
     matching full solution is kept as the reference solution.

This iteration runs a small batch so we can inspect yield, correctness and cost.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("LITELLM_API_KEY", "sk-i1GxNCIGJnywAvUfy5iw1Q")
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "hard_math", ".forger_cache"),
)

from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402
from pyforger import filter as F  # noqa: E402

from hard_math_1 import extract_final_int  # noqa: E402

WRITER_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
SOLVER_ID = "litellm_proxy/gemini/gemini-3.5-flash"


def make_model(model_id: str) -> Model:
    return Model(
        model_id,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ["LITELLM_BASE_URL"],
        temperature=1.0,
        num_retries=3,
    )


# A broad topic taxonomy so we cover the competition-math distribution.
TOPICS: dict[str, list[str]] = {
    "number theory": [
        "modular arithmetic and remainders", "divisibility and digit sums",
        "Diophantine equations", "base representations", "gcd/lcm and factorization",
        "order and primitive roots",
    ],
    "algebra": [
        "polynomial roots and Vieta's", "functional equations", "systems with logarithms",
        "sequences and recursions", "inequalities yielding equality cases",
        "telescoping sums and products",
    ],
    "combinatorics": [
        "counting with casework", "Burnside / symmetry counting",
        "paths on grids", "stars and bars with constraints",
        "recursive counting", "pigeonhole extremal counts",
    ],
    "geometry": [
        "triangle lengths and areas", "coordinate geometry loci",
        "circles and power of a point", "similar triangles and ratios",
        "3D solids and cross-sections",
    ],
    "probability": [
        "expected value", "conditional probability and counting",
        "geometric probability", "states and Markov-style recursion",
    ],
    "trigonometry & complex": [
        "roots of unity sums", "trig identities and sums",
        "complex numbers in geometry",
    ],
}


def sample_seeds(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    seeds = []
    flat = [(t, s) for t, subs in TOPICS.items() for s in subs]
    for i in range(n):
        topic, sub = rng.choice(flat)
        seeds.append({"topic": topic, "subtopic": sub, "idx": i})
    return seeds


def writer_prompt(item: dict):
    return [
        {
            "role": "system",
            "content": (
                "You are an experienced olympiad problem author writing ORIGINAL "
                "competition mathematics problems at the level of the AIME (hard) and "
                "USAMO (easy). Each problem must be self-contained, unambiguous, and "
                "have a UNIQUE final answer that is a non-negative integer below 1000 "
                "(follow the AIME convention: ask for m+n, a remainder, a count, a "
                "floor, etc.). Avoid cliché textbook problems; be creative but keep "
                "the problem fully well-posed and solvable by hand."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write one original AIME-style problem in {item['topic']} "
                f"(focus: {item['subtopic']}). Make it genuinely challenging "
                "(roughly AIME #10-15 difficulty). Respond ONLY with a JSON object "
                "(no markdown fences) with keys \"question\" (the problem statement, "
                "LaTeX allowed) and \"answer\" (the final integer as a string)."
            ),
        },
    ]


def solver_prompt(item: dict):
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous competition mathematician. Solve the problem "
                "rigorously, showing your key steps. The final answer is a "
                "non-negative integer below 1000. End with a line exactly of the form "
                "'ANSWER: <integer>'."
            ),
        },
        {"role": "user", "content": f"Solve this problem:\n\n{item['question']}"},
    ]


def solve_pass(problems: list[dict], model: Model, tag: str) -> list[int | None]:
    """One independent solve per problem; returns extracted ints (parallel batch)."""
    items = [{"question": p["question"], "_tag": tag, "pid": p["pid"]} for p in problems]
    out = generate_from_context(
        items, solver_prompt, model, samples_per_item=1, temperature=1.0, batch_size=24,
    )
    results = []
    for o in out:
        results.append(extract_final_int(o.get("text", "")) if o else None)
    # stash full solution text alongside for the reference solution
    for p, o in zip(problems, out):
        p.setdefault("_solutions", []).append(o.get("text", "") if o else "")
    return results


def main(n_seeds: int = 18, seed: int = 0):
    writer = make_model(WRITER_ID)
    solver = make_model(SOLVER_ID)

    # Stage 1: generate
    seeds = sample_seeds(n_seeds, seed=seed)
    gen = generate_from_context(
        seeds, writer_prompt, writer, samples_per_item=1,
        output_schema={"question": str, "answer": str},
        temperature=1.0, batch_size=12,
    )
    problems = []
    for s, g in zip(seeds, gen):
        if g is None or not g.get("question"):
            continue
        problems.append({
            "pid": len(problems), "topic": s["topic"], "subtopic": s["subtopic"],
            "question": g["question"], "writer_answer": extract_final_int(g.get("answer", "")),
        })
    print(f"[gen] {len(problems)}/{n_seeds} parsed")

    # Stage 2: early dedup
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i
    print(f"[dedup] {len(problems)} after rouge-l")

    # Stage 3: adaptive verification with strong solver
    v1 = solve_pass(problems, solver, "v1")
    v2 = solve_pass(problems, solver, "v2")
    accepted, need3 = [], []
    for p, a1, a2 in zip(problems, v1, v2):
        p["_votes"] = [x for x in (a1, a2) if x is not None]
        if a1 is not None and a1 == a2:
            p["gold"] = a1
            accepted.append(p)
        else:
            need3.append(p)
    if need3:
        v3 = solve_pass(need3, solver, "v3")
        for p, a3 in zip(need3, v3):
            votes = p["_votes"] + ([a3] if a3 is not None else [])
            c = Counter(votes)
            if c:
                modal, cnt = c.most_common(1)[0]
                if cnt >= 2:
                    p["gold"] = modal
                    accepted.append(p)

    print(f"[verify] accepted {len(accepted)}/{len(problems)} "
          f"({100*len(accepted)//max(1,len(problems))}%)")

    cost = writer.last_usage["total_cost"] + solver.last_usage["total_cost"]
    print(f"[cost] this run (uncached portion): ${cost:.4f}")

    # Dump for inspection
    out_path = os.path.join(os.path.dirname(__file__), "..", "hard_math", "inspect_v2.jsonl")
    with open(out_path, "w") as f:
        for p in accepted:
            # pick reference solution whose extracted answer == gold
            ref = ""
            for sol in p.get("_solutions", []):
                if extract_final_int(sol) == p["gold"]:
                    ref = sol
                    break
            f.write(json.dumps({
                "topic": p["topic"], "subtopic": p["subtopic"],
                "question": p["question"], "gold": p["gold"],
                "writer_answer": p["writer_answer"], "votes": p["_votes"],
                "reference_solution": ref,
            }) + "\n")
    print(f"[write] {len(accepted)} -> {out_path}")


if __name__ == "__main__":
    main()

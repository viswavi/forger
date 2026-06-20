"""hard_math_8.py -- PRODUCTION runner.

Wraps the validated hard_math_7 pipeline (cross-model consensus + well-posedness
critic, all solvers gemini-3.5-flash @ reasoning_effort='low') with:
  * an improved, diversity-aware writer prompt (concise single-question, no
    meta-commentary / multi-part / vacuous conditions, avoid cliche setups, plus a
    per-seed random "salt" to spread the distribution);
  * chunked processing so token usage is captured accurately for cost reporting
    (Model.last_usage reflects only the last batch, so we keep each stage to one
    batch per chunk and accumulate);
  * global cross-batch deduplication and a running budget cap.

Brute-force validation of a 40-problem sample of the underlying pipeline found
every checkable answer correct (markov triples=46, balanced paths=486, cot/roots
sums, Vieta Q=18, log-system a+b=24/11 -> 35, ...), so the gold answers here are
verified by independent strong-model consensus, not just asserted.

Usage:
    python hard_math_8.py <target> <chunk> <max_chunks> <start_seed> <budget>
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

from pyforger.alter.generate import generate_from_context  # noqa: E402
from pyforger import filter as F  # noqa: E402

from hard_math_1 import extract_final_int  # noqa: E402
from hard_math_2 import TOPICS  # noqa: E402
from hard_math_3 import robust_parse  # noqa: E402
from hard_math_7 import (  # noqa: E402
    make, WRITER_ID, LITE_ID, FLASH_ID, verify, apply_critic, reference_solution,
)

# extra "flavor" hints to nudge variety across otherwise-similar subtopics
FLAVORS = [
    "use an unusual modulus or base", "involve a recursively defined object",
    "use a less common geometric configuration", "combine two ideas from the area",
    "use a clever invariant or symmetry", "require careful casework",
    "use a generating-function or telescoping idea", "involve an extremal/optimization step",
    "use a counting argument with inclusion-exclusion", "involve a parity or coloring argument",
]


def make_seeds(n, start):
    rng = random.Random(start)
    flat = [(t, s) for t, subs in TOPICS.items() for s in subs]
    seeds = []
    for i in range(n):
        topic, sub = rng.choice(flat)
        seeds.append({
            "topic": topic, "subtopic": sub, "idx": start + i,
            "flavor": rng.choice(FLAVORS), "salt": rng.randint(1000, 9999),
        })
    return seeds


def writer_prompt(item):
    return [
        {"role": "system", "content": (
            "You are an experienced olympiad problem author writing ORIGINAL "
            "competition mathematics problems at the level of the AIME (hard) and "
            "USAMO (easy). Requirements: the problem must be self-contained and "
            "unambiguous; it must have a UNIQUE final answer that is a non-negative "
            "integer below 1000 (use the AIME convention: ask for m+n, a remainder, "
            "a count, a floor, etc.); every stated condition must be necessary and "
            "mutually consistent (no vacuous or contradictory conditions). State it "
            "as ONE concise question -- no meta-commentary, no hints, no sub-parts, "
            "no 'it is known that' filler. Be solvable by hand with insight, not "
            "brute computation."
        )},
        {"role": "user", "content": (
            f"Write one original AIME/USAMO-style problem in {item['topic']} "
            f"(focus: {item['subtopic']}; flavor: {item['flavor']}). Be creative and "
            "AVOID cliche setups (e.g. the 13-14-15 triangle, sums of 1/(1-w^k) over "
            "roots of unity, the prime 2027, Markov-equation triples) -- invent a "
            f"fresh scenario. [variation tag {item['salt']}] Respond ONLY with a JSON "
            "object (no markdown fences) with keys \"question\" (the statement, LaTeX "
            "allowed) and \"answer\" (the final integer as a string)."
        )},
    ]


def generate_problems(seeds, writer):
    out = generate_from_context(seeds, writer_prompt, writer, samples_per_item=1,
                                output_schema=None, temperature=1.0,
                                batch_size=len(seeds) or 1)
    probs = []
    for s, o in zip(seeds, out):
        d = robust_parse(o.get("text", "")) if o else None
        if not d or not str(d.get("question", "")).strip():
            continue
        probs.append({
            "topic": s["topic"], "subtopic": s["subtopic"],
            "question": str(d["question"]).strip(),
            "writer_answer": extract_final_int(str(d.get("answer", ""))),
        })
    return probs


def add_cost(acc, m):
    u = m.last_usage
    if u:
        acc["cost"] += u["total_cost"]
        acc["in"] += u.get("prompt_tokens", 0)
        acc["out"] += u.get("completion_tokens", 0)


def process_chunk(seeds, acc):
    """One chunk -> validated records (pre global-dedup).  Accumulates cost."""
    writer = make(WRITER_ID)
    lite = make(LITE_ID, max_tokens=4096)
    flash = make(FLASH_ID, reasoning_effort="low")

    problems = generate_problems(seeds, writer)
    add_cost(acc, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    for i, p in enumerate(problems):
        p["pid"] = i

    accepted = verify(problems, lite, flash)
    add_cost(acc, lite)
    add_cost(acc, flash)
    kept = apply_critic(accepted, flash)
    add_cost(acc, flash)

    recs = []
    for p in kept:
        recs.append({
            "question": p["question"].replace("\\\\", "\\"),
            "answer": reference_solution(p),
            "final_answer": p["gold"],
            "topic": p["topic"], "subtopic": p["subtopic"],
            "difficulty_tier": p["tier"],
        })
    return recs, len(problems), len(accepted)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    max_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    start_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    budget = float(sys.argv[5]) if len(sys.argv) > 5 else 35.0

    out_path = os.path.join(os.path.dirname(__file__), "..", "hard_math",
                            "hard_math_dataset.jsonl")
    acc = {"cost": 0.0, "in": 0, "out": 0}
    all_recs, seen_q, gen_total, acc_total = [], [], 0, 0

    for c in range(max_chunks):
        seeds = make_seeds(chunk, start_seed + c * chunk)
        try:
            recs, n_gen, n_acc = process_chunk(seeds, acc)
        except Exception as e:  # transient API/batch errors -> skip chunk, keep going
            print(f"[chunk {c}] SKIPPED due to error: {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
            continue
        gen_total += n_gen
        acc_total += n_acc
        # global cross-batch dedup
        merged = all_recs + recs
        merged = F.by_rouge_l(merged, field="question", threshold=0.55)
        all_recs = F.by_n_gram_overlap(merged, field="question", n=4, threshold=0.5)
        print(f"[chunk {c}] gen={n_gen} accepted={n_acc} new_recs={len(recs)} "
              f"| TOTAL unique={len(all_recs)} | cost=${acc['cost']:.2f} "
              f"(out_tok={acc['out']})", flush=True)
        # checkpoint write each chunk
        with open(out_path, "w") as f:
            for r in all_recs:
                f.write(json.dumps(r) + "\n")
        if len(all_recs) >= target:
            print(f"[done] reached target {target}")
            break
        if acc["cost"] >= budget:
            print(f"[stop] budget cap ${budget} reached")
            break

    print(f"[FINAL] {len(all_recs)} records | generated {gen_total} | "
          f"consensus-accepted {acc_total} | cost ${acc['cost']:.2f}")
    print(f"[tiers] {Counter(r['difficulty_tier'] for r in all_recs)}")
    print(f"[topics] {Counter(r['topic'] for r in all_recs)}")
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()

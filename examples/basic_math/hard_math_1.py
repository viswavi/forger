"""hard_math_1.py -- smoke test of the core generate -> solve -> consensus loop.

Goal of this iteration: confirm that (a) topic-conditioned problem generation works,
(b) independent solver passes return parseable integer answers, and (c) a simple
consensus over the solvers + writer gives us a verified gold answer.

Run:
    export LITELLM_API_KEY=sk-...
    export LITELLM_BASE_URL=https://cmu.litellm.ai
    python examples/basic_math/hard_math_1.py
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

# Make src importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

os.environ.setdefault("LITELLM_API_KEY", "sk-i1GxNCIGJnywAvUfy5iw1Q")
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")
# Keep a project-local cache so iterations reuse work.
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "hard_math", ".forger_cache"),
)

from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402

GEN_MODEL_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"


def make_model(model_id: str = GEN_MODEL_ID, temperature: float = 1.0) -> Model:
    return Model(
        model_id,
        api_key=os.environ["LITELLM_API_KEY"],
        api_base=os.environ["LITELLM_BASE_URL"],
        temperature=temperature,
        num_retries=3,
    )


# --- answer extraction ------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{\s*0*(-?\d+)\s*\}")
_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*0*(-?\d+)", re.I)
_HASH = re.compile(r"####\s*0*(-?\d+)")


def extract_final_int(text: str) -> int | None:
    """Pull a final integer answer out of a free-form solution."""
    if not text:
        return None
    for pat in (_HASH, _BOXED, _ANSWER_LINE):
        m = list(pat.finditer(text))
        if m:
            try:
                return int(m[-1].group(1))
            except ValueError:
                continue
    # Fallback: last integer in the text.
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


# --- prompts ----------------------------------------------------------------
def writer_prompt(item: dict):
    return [
        {
            "role": "system",
            "content": (
                "You are an experienced olympiad problem author who writes original "
                "competition mathematics problems at the level of the AIME (hard) and "
                "USAMO (easy). Every problem must be self-contained, unambiguous, and "
                "have a UNIQUE final answer that is an integer between 0 and 999 "
                "(use the standard AIME convention of asking for m+n, a remainder, a "
                "count, etc. so the answer is a non-negative integer < 1000)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write one original AIME-style problem in the area of {item['topic']} "
                f"(specifically {item['subtopic']}). Make it genuinely challenging. "
                "Provide a complete, correct, rigorous solution and the final integer "
                "answer. Respond ONLY with a JSON object (no markdown fences) with keys "
                '"question" (string, the problem statement using LaTeX), "solution" '
                '(string, full worked solution), and "answer" (string, the final '
                "integer)."
            ),
        },
    ]


def solver_prompt(item: dict):
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous competition mathematician. Solve the problem "
                "carefully and show your reasoning. The answer is a non-negative "
                "integer below 1000. End your response with a line of the form "
                "'ANSWER: <integer>'."
            ),
        },
        {"role": "user", "content": f"Solve this problem:\n\n{item['question']}"},
    ]


def main():
    model = make_model()

    topics = [
        {"topic": "number theory", "subtopic": "divisibility and digit sums"},
        {"topic": "algebra", "subtopic": "systems with logarithms"},
        {"topic": "combinatorics", "subtopic": "counting with symmetry"},
    ]

    # Stage 1: generate problems (one batched call, n=1 per item).
    gen = generate_from_context(
        topics,
        writer_prompt,
        model,
        samples_per_item=1,
        output_schema={"question": str, "solution": str, "answer": str},
        temperature=1.0,
        batch_size=8,
    )
    problems = [g for g in gen if g is not None]
    print(f"Generated {len(problems)}/{len(topics)} problems\n")

    # Stage 2: solve each problem K times independently.
    K = 3
    solve_items = []
    for pid, p in enumerate(problems):
        for k in range(K):
            solve_items.append({"pid": pid, "k": k, "question": p["question"]})

    sols = generate_from_context(
        solve_items,
        solver_prompt,
        model,
        samples_per_item=1,
        temperature=1.0,
        batch_size=16,
    )

    # Stage 3: consensus per problem.
    by_pid: dict[int, list[int]] = {}
    for item, sol in zip(solve_items, sols):
        if sol is None:
            continue
        ans = extract_final_int(sol.get("text", ""))
        if ans is not None:
            by_pid.setdefault(item["pid"], []).append(ans)

    for pid, p in enumerate(problems):
        writer_ans = extract_final_int(p.get("answer", ""))
        votes = by_pid.get(pid, [])
        all_votes = votes + ([writer_ans] if writer_ans is not None else [])
        c = Counter(all_votes)
        modal, count = (c.most_common(1)[0] if c else (None, 0))
        print("=" * 70)
        print(f"PID {pid} | topic={topics[pid]['topic']}")
        print(f"Q: {p['question'][:200]}")
        print(f"writer_ans={writer_ans}  solver_votes={votes}")
        print(f"  -> consensus={modal} with {count}/{len(all_votes)} votes")
    print(f"\nlast_usage: {model.last_usage}")


if __name__ == "__main__":
    main()

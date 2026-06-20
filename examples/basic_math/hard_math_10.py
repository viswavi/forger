"""hard_math_10.py -- add gemini's internal reasoning ("thinking") to each record.

Adds a 3rd field so each record cleanly separates reasoning from the final answer:

    {question, thinking, answer, final_answer, topic, subtopic, difficulty_tier}

  * thinking : gemini-3.5-flash's *internal* chain-of-thought (reasoning_content /
               thinking_blocks), which Model.generate_batch normally discards.
  * answer   : UNCHANGED -- the previously verified worked solution ending "#### n".

The cache only stored visible content, so the thinking must be re-generated: we
re-solve each problem once at reasoning_effort="medium", capturing both the visible
solution and the hidden reasoning.  The thinking is attached ONLY when the fresh
solve's final answer still equals the verified gold, so every trace corresponds to a
correct solution.  Cost is tracked via a litellm callback with a hard cap, and a
per-(question, effort) disk cache makes re-runs (e.g. to try a higher effort) cheap.

Usage:
    LITELLM_API_KEY=<key> python hard_math_10.py [effort] [cap_usd]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")

import litellm  # noqa: E402

from hard_math_1 import extract_final_int  # noqa: E402

HM = os.path.join(os.path.dirname(__file__), "..", "hard_math")
DATASET = os.path.join(HM, "hard_math_dataset.jsonl")
CACHE = os.path.join(HM, "thinking_cache.jsonl")
MODEL = "litellm_proxy/gemini/gemini-3.5-flash"
BATCH = 20

COST = {"usd": 0.0, "calls": 0}


def _cb(k, r, s, e):
    try:
        c = k.get("response_cost")
        if c is None:
            c = litellm.completion_cost(completion_response=r)
        COST["usd"] += float(c or 0.0)
        COST["calls"] += 1
    except Exception:
        pass


litellm.success_callback = [_cb]


def solver_messages(q):
    return [
        {"role": "system", "content": (
            "You are a meticulous competition mathematician. Solve the problem "
            "rigorously, showing your reasoning. The final answer is a non-negative "
            "integer below 1000. End with a line exactly of the form "
            "'ANSWER: <integer>'.")},
        {"role": "user", "content": f"Solve this problem:\n\n{q}"},
    ]


def extract_thinking(msg) -> str:
    """Pull the model's internal reasoning out of a litellm message object."""
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        return rc.strip()
    tb = getattr(msg, "thinking_blocks", None)
    if tb:
        parts = [b.get("thinking", "") for b in tb if isinstance(b, dict)]
        if any(parts):
            return "\n".join(p for p in parts if p).strip()
    psf = getattr(msg, "provider_specific_fields", None)
    if isinstance(psf, dict):
        rc = psf.get("reasoning_content")
        if rc:
            return str(rc).strip()
        tb = psf.get("thinking_blocks")
        if tb:
            parts = [b.get("thinking", "") for b in tb if isinstance(b, dict)]
            if any(parts):
                return "\n".join(p for p in parts if p).strip()
    return ""


def _key(q, effort):
    return hashlib.sha256(f"{effort}||{q}".encode()).hexdigest()


def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try:
                d = json.loads(line)
                cache[d["key"]] = d
            except Exception:
                pass
    return cache


def main():
    effort = sys.argv[1] if len(sys.argv) > 1 else "medium"
    cap = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_BASE_URL"]

    rows = [json.loads(l) for l in open(DATASET)]
    cache = load_cache()
    print(f"[load] {len(rows)} records; {len(cache)} cached solves; "
          f"effort={effort} cap=${cap}", flush=True)

    # (re)solve anything not yet successfully captured: missing, errored (ans=None),
    # empty thinking, or whose fresh answer disagreed with the verified gold.
    def needs_solve(r):
        c = cache.get(_key(r["question"], effort))
        if c is None:
            return True
        return (c.get("ans") is None or not c.get("thinking")
                or c.get("ans") != r["final_answer"])
    todo = [r for r in rows if needs_solve(r)]
    print(f"[plan] {len(rows) - len(todo)} already captured, {len(todo)} to (re)solve",
          flush=True)

    cf = open(CACHE, "a")
    for start in range(0, len(todo), BATCH):
        if COST["usd"] >= cap:
            print(f"[stop] cost cap ${cap} reached (true ${COST['usd']:.2f})", flush=True)
            break
        batch = todo[start:start + BATCH]
        msgs = [solver_messages(r["question"]) for r in batch]
        try:
            resps = litellm.batch_completion(
                model=MODEL, messages=msgs, api_key=api_key, api_base=api_base,
                temperature=1.0, reasoning_effort=effort, num_retries=3)
        except Exception as ex:
            print(f"[batch {start}] error: {type(ex).__name__}: {str(ex)[:90]}",
                  flush=True)
            continue
        for r, resp in zip(batch, resps):
            try:
                m = resp.choices[0].message
                content = m.content or ""
                thinking = extract_thinking(m)
            except Exception:
                content, thinking = "", ""
            rec = {"key": _key(r["question"], effort),
                   "ans": extract_final_int(content), "thinking": thinking,
                   "content_tail": content[-120:]}
            cache[rec["key"]] = rec
            cf.write(json.dumps(rec) + "\n")
        cf.flush()
        done = min(start + BATCH, len(todo))
        print(f"[solve] {done}/{len(todo)} | true_spend=${COST['usd']:.2f} "
              f"({COST['calls']} calls)", flush=True)
    cf.close()

    # attach thinking where the fresh solve agrees with the verified gold
    attached, mismatched, missing = 0, 0, 0
    out = []
    for r in rows:
        rec = cache.get(_key(r["question"], effort))
        thinking = ""
        if rec is None:
            missing += 1
        elif rec.get("ans") == r["final_answer"] and rec.get("thinking"):
            thinking = rec["thinking"]
            attached += 1
        else:
            mismatched += 1
        out.append({
            "question": r["question"],
            "thinking": thinking,
            "answer": r["answer"],
            "final_answer": r["final_answer"],
            "topic": r["topic"], "subtopic": r["subtopic"],
            "difficulty_tier": r["difficulty_tier"],
        })

    with open(DATASET, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    tl = sorted(len(r["thinking"]) for r in out if r["thinking"])
    med = tl[len(tl) // 2] if tl else 0
    print(f"[attach] thinking on {attached}/{len(rows)} "
          f"(mismatch {mismatched}, not-yet-solved {missing})")
    print(f"[thinking len] min {tl[0] if tl else 0} median {med} max {tl[-1] if tl else 0}")
    print(f"[FINAL] true_spend=${COST['usd']:.2f}  -> {DATASET}")


if __name__ == "__main__":
    main()

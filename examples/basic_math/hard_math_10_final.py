"""hard_math_10_final.py -- self-contained, end-to-end hard-math dataset pipeline.

    {question, thinking, answer, final_answer, topic, subtopic, difficulty_tier}

Stages (all in this file; uses the Forger framework + raw litellm):

  1. GENERATE   A cheap writer (gemini-3.1-flash-lite) writes original AIME/USAMO-style
                problems conditioned on a sampled (topic, subtopic, flavor), with a
                diversity nudge.  Robust JSON-ish parsing -> {question, answer}.
  2. VERIFY     Cross-model consensus that gives trustworthy integer answers:
                solve lite x2 (self-consistency gate) then ONE strong solve
                (gemini-3.5-flash, reasoning_effort="low"); accept iff all three
                agree (gold = the strong model's value).  This rejects the
                "weak model confidently wrong" class.
  3. CRITIC     A strict well-posedness referee (strong model) drops problems whose
                conditions are inconsistent or whose claimed answer form is invalid.
  4. ASSEMBLE   The strong solve's worked solution becomes `answer`, rewritten to end
                with a GSM8K-style "#### n".  Cross-chunk dedup (rouge-L + n-gram).
  5. THINKING   Re-solve each accepted problem once at reasoning_effort="medium",
                capturing gemini's INTERNAL chain-of-thought
                (message.reasoning_content / thinking_blocks, which the framework's
                Model normally discards).  Attach as `thinking` ONLY when the fresh
                answer still equals the verified gold, so every trace is for a
                correct solution.

Cost note: gemini-3.5-flash is a *thinking* model; an uncapped solve can cost ~$0.14
and a hard max_tokens cap truncates the answer, so solvers use `reasoning_effort`
instead.  Verification runs ~$0.10/accepted record; medium thinking ~$0.15-0.29/solve.
True spend is tracked via a litellm success callback (Model.last_usage only reports
the last batch and undercounts).  Reproducing the full dataset costs roughly
~$35 (verify) + ~$50 (medium thinking) end to end; use the cap args to bound it.

Usage:
    export LITELLM_API_KEY=<key>
    export LITELLM_BASE_URL=https://cmu.litellm.ai
    python hard_math_10_final.py \
        [target=300] [chunk=20] [verify_cap_usd=35] [think_effort=medium] [total_cap_usd=85]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter

# --- Forger framework on the import path (this references the framework, not any
#     sibling hard_math_* script) ------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
os.environ.setdefault("LITELLM_BASE_URL", "https://cmu.litellm.ai")
os.environ.setdefault(
    "FORGER_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "hard_math", ".forger_cache_final"),
)

import litellm  # noqa: E402
from pyforger.acquire import Model  # noqa: E402
from pyforger.alter.generate import generate_from_context  # noqa: E402
from pyforger import filter as F  # noqa: E402

WRITER_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
LITE_ID = "litellm_proxy/gemini/gemini-3.1-flash-lite"
FLASH_ID = "litellm_proxy/gemini/gemini-3.5-flash"
OUT = os.path.join(os.path.dirname(__file__), "..", "hard_math",
                   "hard_math_dataset.jsonl")

# --- accurate global cost accumulator (fires for every litellm completion, incl.
#     the framework's Model calls) -------------------------------------------------
COST = {"usd": 0.0, "calls": 0}


def _cost_cb(kwargs, response_obj, start, end):
    try:
        c = kwargs.get("response_cost")
        if c is None:
            c = litellm.completion_cost(completion_response=response_obj)
        COST["usd"] += float(c or 0.0)
        COST["calls"] += 1
    except Exception:
        pass


litellm.success_callback = [_cost_cb]


def make(model_id, **kw):
    return Model(model_id, api_key=os.environ["LITELLM_API_KEY"],
                 api_base=os.environ["LITELLM_BASE_URL"], temperature=1.0,
                 num_retries=3, **kw)


# ---------------------------------------------------------------------------
# answer / json parsing helpers
# ---------------------------------------------------------------------------
_BOXED = re.compile(r"\\boxed\{\s*0*(-?\d+)\s*\}")
_ANSWER_LINE = re.compile(r"(?:final answer|answer)\s*[:=]\s*0*(-?\d+)", re.I)
_HASH = re.compile(r"####\s*0*(-?\d+)")


def extract_final_int(text):
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
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def robust_parse(text):
    """Recover {question, answer} from a possibly fenced / noisy LLM response."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t.strip())
    candidates = [t]
    first, last = t.find("{"), t.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(t[first:last + 1])
    for c in candidates:
        try:
            d = json.loads(c)
            if isinstance(d, dict) and "question" in d and "answer" in d:
                return d
        except (json.JSONDecodeError, TypeError):
            continue
    q = re.search(r'"question"\s*:\s*"(.*?)"\s*,\s*"answer"', t, re.S)
    a = re.search(r'"answer"\s*:\s*"?([^"\n}]+)"?', t, re.S)
    if q and a:
        return {"question": q.group(1), "answer": a.group(1)}
    return None


# ---------------------------------------------------------------------------
# topic taxonomy + diversity-aware seeds
# ---------------------------------------------------------------------------
TOPICS = {
    "number theory": [
        "modular arithmetic and remainders", "divisibility and digit sums",
        "Diophantine equations", "base representations", "gcd/lcm and factorization",
        "order and primitive roots"],
    "algebra": [
        "polynomial roots and Vieta's", "functional equations",
        "systems with logarithms", "sequences and recursions",
        "inequalities yielding equality cases", "telescoping sums and products"],
    "combinatorics": [
        "counting with casework", "Burnside / symmetry counting", "paths on grids",
        "stars and bars with constraints", "recursive counting",
        "pigeonhole extremal counts"],
    "geometry": [
        "triangle lengths and areas", "coordinate geometry loci",
        "circles and power of a point", "similar triangles and ratios",
        "3D solids and cross-sections"],
    "probability": [
        "expected value", "conditional probability and counting",
        "geometric probability", "states and Markov-style recursion"],
    "trigonometry & complex": [
        "roots of unity sums", "trig identities and sums",
        "complex numbers in geometry"],
}

FLAVORS = [
    "use an unusual modulus or base", "involve a recursively defined object",
    "use a less common geometric configuration", "combine two ideas from the area",
    "use a clever invariant or symmetry", "require careful casework",
    "use a generating-function or telescoping idea",
    "involve an extremal/optimization step",
    "use a counting argument with inclusion-exclusion",
    "involve a parity or coloring argument",
]


def make_seeds(n, start):
    rng = random.Random(start)
    flat = [(t, s) for t, subs in TOPICS.items() for s in subs]
    seeds = []
    for i in range(n):
        topic, sub = rng.choice(flat)
        seeds.append({"topic": topic, "subtopic": sub, "idx": start + i,
                      "flavor": rng.choice(FLAVORS), "salt": rng.randint(1000, 9999)})
    return seeds


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
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
            "brute computation.")},
        {"role": "user", "content": (
            f"Write one original AIME/USAMO-style problem in {item['topic']} "
            f"(focus: {item['subtopic']}; flavor: {item['flavor']}). Be creative and "
            "AVOID cliche setups (e.g. the 13-14-15 triangle, sums of 1/(1-w^k) over "
            "roots of unity, the prime 2027, Markov-equation triples) -- invent a "
            f"fresh scenario. [variation tag {item['salt']}] Respond ONLY with a JSON "
            "object (no markdown fences) with keys \"question\" (the statement, LaTeX "
            "allowed) and \"answer\" (the final integer as a string).")},
    ]


def solver_prompt(item):
    return [
        {"role": "system", "content": (
            "You are a meticulous competition mathematician. Solve the problem "
            "rigorously, showing your reasoning. The final answer is a non-negative "
            "integer below 1000. End with a line exactly of the form "
            "'ANSWER: <integer>'.")},
        {"role": "user", "content": f"Solve this problem:\n\n{item['question']}"},
    ]


CRITIC_SYS = (
    "You are a strict competition-math referee checking a problem for "
    "WELL-POSEDNESS. Check explicitly and reason: (1) Are all stated conditions "
    "mutually CONSISTENT -- does a configuration satisfying every condition "
    "simultaneously actually exist? (2) Is every claimed property TRUE -- if it "
    "says a quantity equals m/n with integers, is that quantity actually real and "
    "rational? verify any 'it is known that' assertion. (3) Is the requested answer "
    "UNIQUELY determined and a non-negative integer below 1000? Only output INVALID "
    "if you find a DEFINITE flaw; otherwise VALID. End with exactly "
    "'VERDICT: VALID' or 'VERDICT: INVALID'."
)


def critic_prompt(item):
    return [{"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content": "Problem:\n\n" + item["question"]}]


# ---------------------------------------------------------------------------
# generation + verification (Forger generate_from_context, @op-cached)
# ---------------------------------------------------------------------------
def solve_pass(problems, model, tag):
    """One independent solve per problem; stores full text under _sol_<tag>."""
    items = [{"question": p["question"], "_tag": tag} for p in problems]
    out = generate_from_context(items, solver_prompt, model, samples_per_item=1,
                                temperature=1.0, batch_size=24)
    res = []
    for p, o in zip(problems, out):
        txt = o.get("text", "") if o else ""
        p["_sol_" + tag] = txt
        res.append(extract_final_int(txt))
    return res


def generate_problems(seeds, writer):
    out = generate_from_context(seeds, writer_prompt, writer, samples_per_item=1,
                                output_schema=None, temperature=1.0,
                                batch_size=len(seeds) or 1)
    probs = []
    for s, o in zip(seeds, out):
        d = robust_parse(o.get("text", "")) if o else None
        if not d or not str(d.get("question", "")).strip():
            continue
        probs.append({"topic": s["topic"], "subtopic": s["subtopic"],
                      "question": str(d["question"]).strip(),
                      "writer_answer": extract_final_int(str(d.get("answer", "")))})
    return probs


def verify_cheap(problems, lite, flash):
    """lite x2 self-consistency gate -> single strong confirm. gold = strong value."""
    l1 = solve_pass(problems, lite, "lite1")
    l2 = solve_pass(problems, lite, "lite2")
    survivors = []
    for p, a, b in zip(problems, l1, l2):
        if a is not None and a == b:
            p["_L"] = a
            survivors.append(p)
    accepted = []
    if survivors:
        f1 = solve_pass(survivors, flash, "flash1")
        for p, v in zip(survivors, f1):
            if v is not None and v == p["_L"]:
                p["gold"], p["tier"] = v, "core"
                accepted.append(p)
    return accepted


def apply_critic(accepted, flash):
    if not accepted:
        return []
    out = generate_from_context(accepted, critic_prompt, flash, samples_per_item=1,
                                temperature=1.0, batch_size=24)
    kept = []
    for p, o in zip(accepted, out):
        txt = o.get("text", "") if o else ""
        verdicts = re.findall(r"VERDICT:\s*(VALID|INVALID)", txt, re.I)
        if verdicts and verdicts[-1].upper() == "VALID":
            kept.append(p)
    return kept


def reference_solution(p):
    """Strong solve's worked solution, tail rewritten to '#### <gold>'."""
    gold = p["gold"]
    for tag in ("flash1", "lite1", "lite2"):
        s = p.get("_sol_" + tag, "")
        if s and extract_final_int(s) == gold:
            body = s.split("ANSWER:")[0].rstrip()
            return f"{body}\n\n#### {gold}"
    return f"#### {gold}"


def process_chunk(seeds, writer, lite, flash):
    problems = generate_problems(seeds, writer)
    problems = F.by_rouge_l(problems, field="question", threshold=0.6)
    accepted = verify_cheap(problems, lite, flash)
    kept = apply_critic(accepted, flash)
    recs = []
    for p in kept:
        recs.append({"question": p["question"].replace("\\\\", "\\"),
                     "answer": reference_solution(p), "final_answer": p["gold"],
                     "topic": p["topic"], "subtopic": p["subtopic"],
                     "difficulty_tier": p["tier"]})
    return recs, len(problems), len(accepted)


# ---------------------------------------------------------------------------
# thinking capture (raw litellm -- the framework's Model discards reasoning_content)
# ---------------------------------------------------------------------------
def think_messages(q):
    return [
        {"role": "system", "content": (
            "You are a meticulous competition mathematician. Solve the problem "
            "rigorously, showing your reasoning. The final answer is a non-negative "
            "integer below 1000. End with a line exactly of the form "
            "'ANSWER: <integer>'.")},
        {"role": "user", "content": f"Solve this problem:\n\n{q}"},
    ]


def extract_thinking(msg):
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
        for fld in ("reasoning_content", "thinking_blocks"):
            v = psf.get(fld)
            if isinstance(v, str) and v:
                return v.strip()
            if isinstance(v, list):
                parts = [b.get("thinking", "") for b in v if isinstance(b, dict)]
                if any(parts):
                    return "\n".join(p for p in parts if p).strip()
    return ""


def attach_thinking(records, effort, total_cap, batch=20):
    """Re-solve each record at `effort`, attach internal CoT where answer matches."""
    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_BASE_URL"]
    for r in records:
        r.setdefault("thinking", "")
    todo = [r for r in records if not r["thinking"]]
    for start in range(0, len(todo), batch):
        if COST["usd"] >= total_cap:
            print(f"[think] total cap ${total_cap} reached (${COST['usd']:.2f})",
                  flush=True)
            break
        chunk = todo[start:start + batch]
        try:
            resps = litellm.batch_completion(
                model=FLASH_ID, messages=[think_messages(r["question"]) for r in chunk],
                api_key=api_key, api_base=api_base, temperature=1.0,
                reasoning_effort=effort, num_retries=3)
        except Exception as ex:
            print(f"[think batch {start}] {type(ex).__name__}: {str(ex)[:80]}",
                  flush=True)
            continue
        for r, resp in zip(chunk, resps):
            try:
                m = resp.choices[0].message
                content, thinking = m.content or "", extract_thinking(m)
            except Exception:
                content, thinking = "", ""
            if thinking and extract_final_int(content) == r["final_answer"]:
                r["thinking"] = thinking
        print(f"[think] {min(start + batch, len(todo))}/{len(todo)} | "
              f"spend=${COST['usd']:.2f}", flush=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    verify_cap = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
    effort = sys.argv[4] if len(sys.argv) > 4 else "medium"
    total_cap = float(sys.argv[5]) if len(sys.argv) > 5 else 85.0

    writer = make(WRITER_ID)
    lite = make(LITE_ID, max_tokens=4096)
    flash = make(FLASH_ID, reasoning_effort="low")

    # ---- stages 1-4: generate -> verify -> critic -> assemble (until target) ----
    all_recs = []
    c = 0
    while len(all_recs) < target and COST["usd"] < verify_cap:
        seeds = make_seeds(chunk, 1_000_000 + c * chunk)
        try:
            recs, n_gen, n_acc = process_chunk(seeds, writer, lite, flash)
        except Exception as ex:
            print(f"[chunk {c}] skipped: {type(ex).__name__}: {str(ex)[:80]}",
                  flush=True)
            c += 1
            continue
        merged = all_recs + recs
        merged = F.by_rouge_l(merged, field="question", threshold=0.55)
        all_recs = F.by_n_gram_overlap(merged, field="question", n=4, threshold=0.5)
        with open(OUT, "w") as f:                       # checkpoint
            for r in all_recs:
                f.write(json.dumps({**r, "thinking": r.get("thinking", "")}) + "\n")
        print(f"[chunk {c}] gen={n_gen} accepted={n_acc} +{len(recs)} | "
              f"TOTAL={len(all_recs)} | spend=${COST['usd']:.2f}", flush=True)
        c += 1

    print(f"[verify done] {len(all_recs)} records | spend=${COST['usd']:.2f}", flush=True)

    # ---- stage 5: attach internal thinking traces ----
    attach_thinking(all_recs, effort, total_cap)

    # final write with the 3-field schema (thinking between question and answer)
    out = []
    for r in all_recs:
        out.append({"question": r["question"], "thinking": r.get("thinking", ""),
                    "answer": r["answer"], "final_answer": r["final_answer"],
                    "topic": r["topic"], "subtopic": r["subtopic"],
                    "difficulty_tier": r["difficulty_tier"]})
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    have = sum(1 for r in out if r["thinking"])
    tl = sorted(len(r["thinking"]) for r in out if r["thinking"])
    print(f"[FINAL] {len(out)} records | thinking on {have} | "
          f"thinking median {tl[len(tl)//2] if tl else 0} chars | "
          f"total spend ${COST['usd']:.2f}")
    print(f"[topics] {dict(Counter(r['topic'] for r in out))}")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()

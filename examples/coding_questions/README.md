# Difficult programming-questions dataset

HumanEval-style Python coding problems for training an expert programming agent,
generated with the Forger pipeline.

## Files

| File | Contents |
|---|---|
| `coding_questions.jsonl` | **Final dataset** (319 examples). |
| `coding_questions_raw.jsonl` | All 426 execution-verified problems before LLM quality scoring (provenance). |
| `coding_questions_scored.jsonl` | The 425 problems with their judge `_score` (1–5), for re-thresholding. |
| `validate.py` | Standalone validator: re-runs every example's doctests + `check()` in a sandboxed subprocess. |

## Schema (each JSONL line)

Matches the HumanEval format:

- `prompt` — function signature + docstring (with `>>>` doctests).
- `canonical_solution` — the function **body** (append directly after `prompt` to form a complete function).
- `test` — defines `check(candidate)` with ≥ 6 asserts (plus a `METADATA` dict).
- `entry_point` — the function name (extra convenience field; pass it as `candidate`).

Reconstruct + run a problem:

```python
prog = ex["prompt"] + "\n" + ex["canonical_solution"] + "\n" + ex["test"]
ns = {}; exec(prog, ns); ns["check"](ns[ex["entry_point"]])
```

## How it was built

Pipelines live in `examples/instruction_following/coding_questions_{1..5}.py`.

1. **Generate** (`gemini-3.1-flash-lite`, temperature 1.0): one hard problem per
   (algorithmic topic × difficulty twist) seed, emitted in delimiter tags
   (robust for code) rather than JSON.
2. **Execution verify** (the key quality gate): run `prompt + canonical_solution`,
   then run the docstring doctests **and** all `check()` asserts in a sandboxed
   subprocess with an 8 s timeout. Keep only self-consistent problems.
3. **Self-repair**: failures get one repair round with the traceback fed back.
4. **Dedup**: drop near-duplicate prompts (ROUGE-L ≥ 0.7).
5. **Quality filter**: drop solutions that ignore their inputs (hardcoded stubs),
   then an LLM rubric judge scores each 1–5; keep score ≥ 4.

Every example in the final set is execution self-consistent (its solution passes
its own doctests and tests) and judged a correct, general algorithm.

Total LLM spend: ≈ $3 (budget was $20).

## Re-validate

```bash
python examples/coding_questions/validate.py
```

# Basic math word problems

A GSM8K-style dataset of challenging-but-basic multi-step math word problems,
built with Forger. Each example is `{"question", "answer"}`, where the answer is
a step-by-step solution with inline `<<expr=result>>` calculator annotations and
a final `#### <number>` line.

## Dataset

- **`basic_math_dataset.jsonl`** — 484 problems (target was ≥200).
- Every row is machine-verified: all `<<expr=result>>` annotations evaluate
  correctly, the solution is multi-step (median 4, up to 14 steps), and the
  `####` answer matches the final calculation.
- 484 unique questions across 20 math concepts × 30 everyday scenarios.

## Pipeline

The final pipeline is **`basic_math_4.py`**. Stages:

1. **acquire** — `alter.generate_from_context` over a (concept × scenario) grid,
   author model `gemini-3.1-flash-lite`, temperature 1.0, JSON output schema.
2. **filter (local, free)** — `verify_utils.check_answer/check_question/has_clean_numbers`:
   re-evaluates every calculator annotation, requires ≥2 steps, requires the
   `####` answer to equal the last annotated calculation, and drops ugly
   repeating decimals. This is the strongest, cheapest quality lever.
3. **filter (diversity)** — `filter.by_rouge_l` on the question text (threshold 0.6).
4. **filter (judge)** — an independent stronger model (`gemini-3.5-flash`) scores
   each problem 1–5 for well-posedness and correct modeling; keep score ≥ 4.
   This catches errors the local verifier cannot (e.g. a solution that is
   internally consistent but silently changes a number, or an ambiguous /
   two-part question).

Run yield (600 seeds): 600 generated → 554 verified → 542 deduped →
484 kept by judge (462 scored 5/5).

### Why a custom judge instead of `filter.by_llm_score`

`gemini-3.5-flash` wraps its JSON in ```` ```json ```` fences, which the
library's strict `json.loads` rejects (every response came back "unparseable").
The judge here uses `generate_from_context` with a prompt that explicitly forbids
code fences, which the stronger model obeys.

### Model-specific constraint

This proxy's Gemini models reject `n > 1` ("Multiple candidates is not enabled"),
so `acquire.sample_from_LLM` (which batches via `n`) is unusable. We instead use
`generate_from_context` with `samples_per_item=1` over many distinct prompts,
which fans out via `litellm.batch_completion` (one completion each).

## Iteration history

- `basic_math_1.py` — smoke test: connectivity, JSON format, verifier (6/6 valid).
- `basic_math_2.py` — calibration (80 seeds); surfaced the broken library judge.
- `basic_math_3.py` — calibration with the fixed custom judge; inspected rejects.
- `basic_math_4.py` — final full-grid run producing `basic_math_dataset.jsonl`.

## Reproduce

```bash
export LITELLM_API_KEY=sk-...
export LITELLM_BASE_URL="https://cmu.litellm.ai"
export FORGER_CACHE=.forger_cache   # @op disk cache; reruns are nearly free
python examples/basic_math/basic_math_4.py
```

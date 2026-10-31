# AGENTS.md: Forger pipeline reference

Forger builds and runs pipelines that generate synthetic data or RL environments.

## Execution model

A pipeline is a `@pyforger.pipeline` function calling `@pyforger.step` functions. A step is a named, cached group of
ops (typically acquire or alter, then filter) from `list[dict]` rows to rows.

- Each step prints a report, and the pipeline a summary table. Read them after every run.
- `WARNING:` lines are data problems (e.g. "20 rows dropped: 34 of 120 completions ... hit max_tokens=16384"). Fix
  each, or suppress it with a reason using the flag in brackets at the end of the line, e.g.
  `@step(suppress_truncation_warning="<why these rows are still fine>")`. Dim `INFO:` lines are about the code.
- Run each new pipeline version with `FORGER_LIMIT=3` first: every op on three rows, in about a minute.
- A batch of model requests that would not fit a cap or quota stops after about 5% of it (`SpendProjectionError`).
- `pyforger.inspect.runs()` lists saved runs with rows, cost, step yields and warnings ("Saved pipeline runs" in
  [`inspect.md`](agents/inspect.md)).

More, including limited runs and spend projection: [`pipelines.md`](agents/pipelines.md).

## Basic strategy for using Forger

Forger is both a framework for building synthetic data pipelines and an ideology for how that process should be
structured. Writing data generation pipelines is often seen as a trial-and-error process. Forger aims to make it a
deliberate, systematic process. At each step of the pipeline, you should consider multiple options (e.g. compare
sources for your starting task descriptions before choosing one, consider multiple ways to refine them if necessary,
compare various techniques for generating ground truth for each task before committing to one, and compare the
effect of various filters based on how they shape the data distribution). Each of these decisions should be
deliberate, and they should use functions in pyforger.inspect (or similar, hand-rolled data inspection routines) and
the data samples and cost profiles produced for each Forger step to inform your choices. Finally, each @step should
include a filter step that serves as a contract to verify that the data meets the expected properties and quality,
using `min_yield=` and `max_yield=`, e.g. `filter.by_llm_score(rows, ..., min_yield=0.2, max_yield=0.8)` to enforce
the contract. You can call `pyforger.inspect.inspect_step("step_name")` to see more sample rows when a step's yield
or samples look off. [`pipelines.md`](agents/pipelines.md) has a complete example.

## Operation reference

Ops (all but `filter.by_predicate`) and retrieval are cached on disk. Pass `_no_cache=True` for a fresh result
(also for RANDOM mode without a seed), or `_cache_only=True` to read only the cache (a miss raises `FileNotFoundError`).

Every op's signature and defaults are in [OPS.md](OPS.md), generated from the code (anchors like
[`OPS.md#filter-by_llm_score`](OPS.md#filter-by_llm_score)).

## Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.op` | `src/pyforger/op.py` | Decorator for caching operations. |
| `pyforger.step` | `src/pyforger/steps.py` | Decorator for a named, cached, reported group of ops. |
| `pyforger.pipeline` | `src/pyforger/steps.py` | Decorator for a function that calls steps. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing any. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.inspect` | `src/pyforger/inspect/` | Report on rows without modifying them. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`, `canonical`. |

## Reference files

Read the file for the ops you use:

- [`docs/OPS.md`](OPS.md): every op's signature, generated from the code.
- [`acquire.md`](agents/acquire.md): `sample_from_LLM`, student sampling (`LocalModel`, `TinkerModel`), retrieval, rubrics.
- [`alter.md`](agents/alter.md): `generate_from_context`, translation, code execution, running Python or C++
  programs on tests (partial scores, checker programs), STaR.
- [`filter.md`](agents/filter.md): LLM judge, `by_predicate`, `by_tests`, difficulty filters,
  `by_embedding_similarity`, `by_topic_frequency`.
- [`inspect.md`](agents/inspect.md): `topic_based_clustering`, `eval_tree`, `rubric_scores`, `embedding_dispersion`,
  `compare_to_reference`, `inspect_step`, `spend`, `runs`, `student_log_likelihood`.
- [`models.md`](agents/models.md): `Model`, gateway credentials, concurrency, `n` fan-out, reasoning effort.
- [`caching.md`](agents/caching.md): `@op` caching, the request cache, tracing.
- [`pipelines.md`](agents/pipelines.md): `@step` / `@pipeline`, reports, spend caps, composing pipelines, patterns.
- [`dynamic_step.md`](agents/dynamic_step.md): input-specific sub-pipelines with `dynamic_step`.

## Credentials and Cost

### Credentials

- Read keys from the environment (`LITELLM_API_KEY`, else provider variables such as `OPENAI_API_KEY`), e.g.
  `os.environ.get("LITELLM_API_KEY")`. Never look at, print, or commit a key.
- Use the model id the user gives (e.g. "base/org_name/gpt-5.6-sol") as the LiteLLM model name. If none is given,
  stop and ask. Other models in the same account are fair to try.
- Default Azure models: "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731" (fast, cheap, good for most things) and
  "litellm_proxy/azure_ai/FW-GLM-5.3" (accurate, strong reasoning, expensive).

### Cost

Benchmark quotas count token units: DeepSeek-V4-Flash costs 1 unit per input token and 3 per output token; GLM-5.3
costs 4 and 12.5.

- Cached input tokens count at full weight. Completion tokens already include reasoning; do not add it again.
- Other models report `NaN` units, and budgeted calls to them stop. Local GPU inference is free and exempt.
- Check spend with `model.total_token_units`. Cached `@op` results cost nothing.
- `FORGER_TOKEN_USAGE_FILE` (a JSON ledger) and `FORGER_TOKEN_UNIT_LIMIT` share one quota across models and
  processes; `reset_history()` never resets it. Each request holds its worst case (prompt plus `max_tokens` per
  completion, 65,536 without it) until its response replaces it with actual usage; new calls stop once used plus
  held units reach the quota. A request that times out, loses its connection, or whose process is killed stays
  charged at its hold, since the gateway may still bill it (`unreturned_calls`; a killed process's holds move there
  after two hours).
- The benchmark launcher sets a generation quota and a separate judge quota, both given in the task instructions.
- Direct API calls must use the launcher's `rl.token_budget` helpers to be counted; they make one attempt per call
  (no `num_retries`), so retry in your own code.
- `TinkerModel` is charged in dollars to a separate ledger (see [`acquire.md`](agents/acquire.md)).
- Cap a run in units with `@pipeline(max_token_units=...)` or `FORGER_MAX_TOKEN_UNITS`.

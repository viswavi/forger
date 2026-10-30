# AGENTS.md: Forger pipeline reference

Forger builds and runs pipelines that generate synthetic data or RL environments.

## Basic strategy for using Forger

Iterate: build a pipeline, run it, inspect outputs, adjust. Start simple (e.g. one `retrieval` or
`sample_from_LLM` step), then add `alter` and `filter` steps to improve quality. The cache skips unchanged expensive
steps, so iterating is cheap.

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
  processes. In-flight calls can overshoot it; `reset_history()` never resets it.
- The benchmark launcher sets a generation quota and a separate judge quota, both given in the task instructions.
- Direct API calls must use the launcher's `rl.token_budget` helpers to be counted.
- `TinkerModel` is charged in dollars to a separate ledger (see [`acquire.md`](agents/acquire.md)).
- Cap a run in units with `@pipeline(max_token_units=...)` or `FORGER_MAX_TOKEN_UNITS`.

### Execution model

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

### Operation reference

Ops (all but `filter.by_predicate`) and retrieval are cached on disk. Pass `_no_cache=True` for a fresh result
(also for RANDOM mode without a seed), or `_cache_only=True` to read only the cache (a miss raises `FileNotFoundError`).

Every op's signature and defaults are in [OPS.md](OPS.md), generated from the code (anchors like
[`OPS.md#filter-by_llm_score`](OPS.md#filter-by_llm_score)).

### Public API summary

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

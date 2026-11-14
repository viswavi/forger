# AGENTS.md: Forger pipeline reference

Forger is a tool for describing and executing pipelines for generating synthetic data or RL environments. This
reference describes the execution model, the public API, and the conventions the test suite expects.

## Contents

- [Basic strategy for using Forger](#basic-strategy-for-using-forger)
- [Credentials and Cost](#credentials-and-cost)
- [Execution model](#execution-model)
- [Operation reference](#operation-reference)
- [Public API summary](#public-api-summary)
- [Reference files](#reference-files)

## Basic strategy for using Forger

Data generation is usually an iterative process of designing a pipeline, running it, inspecting the outputs, and
adjusting the pipeline. Forger supports this workflow with flexible primitives and caching for quick iteration. A
common pattern is to start with a simple pipeline (e.g. a single `retrieval` or `sample_from_LLM` step) and then add
`alter` and `filter` steps to improve quality. Forger's caching capabilities avoid repeating expensive steps while
tweaking parameters.

For a complete worked example, read [`examples/rl_math/pipeline.py`](../examples/rl_math/pipeline.py): an RL dataset
of math problems with code rubrics, from web search to a difficulty band and rubric probes. Each step's docstring
states what was chosen and shows the form of a justification: the options, the call that measures them, and the
result. Its results are placeholders, not from a run; write only comparisons you ran, with the call that measured them.

## Credentials and Cost

### Credentials

Many of the operations in Forger involve calls to LLMs or external APIs, which may incur costs. These include LLM
APIs (e.g. accessing Anthropic's models via the LiteLLM library) or retrieval APIs (e.g. Brave Search, arXiv). These
APIs typically require credentials (e.g. API keys) that should be kept secure and not committed to version control.
They can either be stored on disk (e.g. in a `.env` file loaded by your shell or other external tooling) or managed
through provider-specific environment variables.
LLMs will almost always be accessed via LiteLLM-backed model wrappers. Generally the user should also provide a model
identifier (e.g. "base/org_name/XXX") in the prompt, which you should set as the model name in LiteLLM. If not given,
pause and ask the user to provide one. This identifier should include account/user information (e.g. "base/org_name")
a model name (e.g. "gpt-5.6-sol"). Within this structure, you can potentially try other models within the same account
if the user-provided model is undesirable or unavailable. Within Azure specifically, the following models are available
by default:
"litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731" (fast, cheap, surprisingly good for most things),
"litellm_proxy/azure_ai/FW-GLM-5.3" (accurate, strong reasoning, expensive).

Typically, the user should set LITELLM_API_KEY environment variable. If none is given, then LiteLLM will fall back to
provider-specific environment variables (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). When instantiating a `Model` with
such variables, do not look at or print the key; simply pass it to the client via `os.environ.get("LITELLM_API_KEY")`
(or equivalent).  Additional `**kwargs` are forwarded to `litellm.completion` (`api_base`, `organization`).

You can check whether the LITELLM_API_KEY environment variable is provided in Python with:

```python
from dotenv import load_dotenv
import os
load_dotenv()

if "LITELLM_API_KEY" not in os.environ:
    print("Please provide a LITELLM_API_KEY environment variable.")
```

### Cost

Token-unit quotas use fixed weights, independent of provider prices:
DeepSeek-V4-Flash costs 1 unit per input token and 3 per output token; GLM-5.3 costs
4 per input token and 12.5 per output token. The judge's DeepSeek-V4.1-Flash costs
1 per input token and 3 per output token, on the judge quota only.
Cached input tokens count at full weight. Completion tokens already include reasoning;
do not add reasoning tokens again. Other model families have no defined unit weight.

`Model` exposes per-call `result.usage["token_units"]` and `result.usage["cost_usd"]`,
plus cumulative `model.total_token_units`, `model.total_cost`, and raw `model.total_usage`.
Unknown model weights or missing usage produce `NaN` token units, including in the
cumulative total. Local GPU inference has zero API cost and is exempt from quotas;
models outside the fixed list still report `NaN` units. `TinkerModel` reports Tinker
dollars in `cost_usd` and is charged to a separate dollar ledger, not to token units
(see `acquire.md`). Step and pipeline reports and the spend ledger show units next to dollars, and
`@pipeline(max_token_units=...)` / `FORGER_MAX_TOKEN_UNITS` cap a run or process in units
(see [`docs/agents/pipelines.md`](agents/pipelines.md)).
INFO logs show both totals after the first call and every million units (raw tokens
when weights are unknown). Dollar costs use provider metadata or LiteLLM pricing;
unavailable prices warn and contribute zero to `total_cost`.
`reset_history()` resets the model's view, but never resets a configured shared quota.
Cached `@op` results do not make API calls and do not consume the quota again.

Set `FORGER_TOKEN_USAGE_FILE` to a persistent JSON ledger and
`FORGER_TOKEN_UNIT_LIMIT` to share a quota across model instances and processes.
Ledgers retain `cost_usd` and count missing prices in `unpriced_calls`.
Each request holds its worst case (prompt plus `max_tokens` per completion, 65,536 without it) while in flight,
and new calls stop once used plus held units reach the quota. A response replaces its hold with actual usage;
a request that times out or loses its connection keeps its hold as spent, because the gateway may still bill it
(`unreturned_calls` in the ledger). A process that dies or is interrupted mid-request releases its holds
(`released_calls`): the next ledger read frees them, or after two hours when the process's death cannot be seen.
Unknown model weights or missing token counts stop budgeted calls instead of counting as free.

### Execution model

A pipeline is a Python function. It calls operations from `pyforger.acquire`, `pyforger.alter`, `pyforger.filter`,
and `pyforger.inspect`. Each operation takes a `list[dict]` (or generation parameters). Acquire, alter, and filter
operations generally return a `list[dict]`; rubric generation returns one rubric per input.
`acquire.generate_custom_rubric_criteria` maps
`list[str]` to `list[list[tuple[str, float]]]`. Inspect operations return a report about the rows without modifying
rows.
Control flow (`for`, `if`, comprehensions, helper functions) in plain Python can be used

Structure a pipeline as a `@pyforger.pipeline` function that calls `@pyforger.step` functions. A step is a named
function from rows to rows built from Forger ops (typically acquire or alter, then filter); it may also take several
lists of rows and return a dict of named row lists. Each step is cached like an `@op` and, when it returns, prints a
report; the pipeline prints a summary table of its steps at the end. Read these reports after every run.
`WARNING:` lines are data problems: each says what happened to your rows (e.g. "20 rows dropped: 34 of 120
completions ... hit max_tokens=16384"), and the summary ends with all of them. Fix each one; a few that may not matter
for your data (e.g. truncation in a draft step) can be suppressed with a reason, e.g.
`@step(suppress_truncation_warning="<why these rows are still fine>")`; a line that can be suppressed ends with its
flag in brackets. Dim `INFO:` lines are about how the code is written. Run each new pipeline version with
`FORGER_LIMIT=3` first: it runs every op on three rows in about a minute (see "Limited runs" in `pipelines.md`).
A batch of model requests that would not fit a cap or quota stops early, after about 5% of it
(`SpendProjectionError`; see "Spend projection").
Every pipeline run is saved, and `pyforger.inspect.runs()` lists them with their rows, cost, step yields, and
warnings (see "Saved pipeline runs" in `inspect.md`). `inspect.compare_pipelines("main_a", "main_b")` shows which
step two runs first differ at and why (see "Comparing pipeline runs" in `inspect.md`).
See [`docs/agents/pipelines.md`](agents/pipelines.md) for the reports, warnings, suppression flags, spend ledger, and
spending caps.

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (`sample_from_LLM`,
`generate_custom_rubric_criteria`,
every `alter.*` op, every `filter.*` op but the cheap `by_predicate`, every `inspect.*` op) are decorated internally
with `@op`, so their results are cached on disk and traced. The retrieval functions (`retrieve`, `retrieve_from_*`)
are also internally cached and traced by default. Pass `_no_cache=True` to skip cache reads and writes for an
invocation, or `_cache_only=True` to read an existing result without fetching (a miss raises `FileNotFoundError`).
This includes RANDOM mode, even without a seed: use `_no_cache=True` for a fresh sample.
Direct calls to source `.fetch()` methods remain uncached.

For the full call signature of every op (parameter names, defaults, field parameters, caching), see
[OPS.md](OPS.md), which is generated from the code. It starts with a table of contents, and each op can be linked as
`OPS.md#<module>-<op>` in lowercase, e.g. [`OPS.md#filter-by_llm_score`](OPS.md#filter-by_llm_score).

### Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.op` | `src/pyforger/op.py` | Decorator for caching operations. |
| `pyforger.step` | `src/pyforger/steps.py` | Decorator for a named, cached, reported group of ops (rows to rows). |
| `pyforger.pipeline` | `src/pyforger/steps.py` | Decorator for a function that calls steps; prints a summary table. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing items. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.inspect` | `src/pyforger/inspect/` | Examine rows without modifying them; returns a report. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`, `canonical`. |

Forger's loggers print to stderr themselves and also propagate to the root logger. Once the root logger has a handler
that prints to the terminal (e.g. after `logging.basicConfig()`), Forger lines are printed by that handler, in its
format, and not a second time by Forger's own.

## Reference files

The detailed reference is split by topic under `docs/agents/`; read the file for the operations you use:

- [`docs/OPS.md`](OPS.md): Every op's signature and docstring summary, generated from the code.
- [`docs/agents/acquire.md`](agents/acquire.md): Generation with `sample_from_LLM`, `TinkerModel` student sampling,
  retrieval, rubric generation.
- [`docs/agents/alter.md`](agents/alter.md): Row transforms: `generate_from_context`, translation, code execution,
  running Python or C++ programs against tests (partial scores, checker programs), STaR.
- [`docs/agents/filter.md`](agents/filter.md): Quality and diversity filters, including the LLM judge, `by_predicate`,
  `by_tests`, `by_rubric_score` (every row's grade score), the difficulty filters, `by_embedding_similarity`, and
  `by_topic_frequency`.
- [`docs/agents/inspect.md`](agents/inspect.md): Reports on rows: `topic_based_clustering`, `eval_tree`,
  `rubric_scores`, `embedding_dispersion`, `compare_to_reference`, `inspect_step`, `spend`, `runs` and
  `compare_pipelines` (saved pipeline runs), and `student_log_likelihood`.
- [`docs/agents/models.md`](agents/models.md): The `Model` class, gateway credentials, concurrency, `n` fan-out,
  reasoning effort, prompt types.
- [`docs/agents/caching.md`](agents/caching.md): How `@op` caching, the request cache, and tracing work.
- [`docs/agents/pipelines.md`](agents/pipelines.md): `@step` / `@pipeline`, step reports, spend ledger and caps,
  composing pipelines, and common data-generation patterns.
- [`docs/agents/dynamic_step.md`](agents/dynamic_step.md): Input-specific sub-pipelines with `dynamic_step`.

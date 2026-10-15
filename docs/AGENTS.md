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
(or equivalent).  Additional `**kwargs` are forwarded to `litellm.completion` (`api_base`, `organization`, `timeout`).

You can check whether the LITELLM_API_KEY environment variable is provided in Python with:

```python
from dotenv import load_dotenv
import os
load_dotenv()

if "LITELLM_API_KEY" not in os.environ:
    print("Please provide a LITELLM_API_KEY environment variable.")
```

### Cost

Benchmark LLM quotas use fixed token units, independent of provider prices:
DeepSeek-V4-Flash costs 1 unit per input token and 3 per output token; GLM-5.3 costs
4 per input token and 12.5 per output token.
Cached input tokens count at full weight. Completion tokens already include reasoning;
do not add reasoning tokens again. Other model families have no defined unit weight.

`Model` exposes per-call `result.usage["token_units"]` and `result.usage["cost_usd"]`,
plus cumulative `model.total_token_units`, `model.total_cost`, and raw `model.total_usage`.
Unknown model weights or missing usage produce `NaN` token units, including in the
cumulative total. Local GPU inference has zero API cost and is exempt from quotas;
models outside the fixed list still report `NaN` units. `TinkerModel` reports Tinker
dollars in `cost_usd` and is charged to a separate dollar ledger, not to token units
(see `acquire.md`).
INFO logs show both totals after the first call and every million units (raw tokens
when weights are unknown). Dollar costs use provider metadata or LiteLLM pricing;
unavailable prices warn and contribute zero to `total_cost`.
`reset_history()` resets the model's view, but never resets a configured shared quota.
Cached `@op` results do not make API calls and do not consume the quota again.

Set `FORGER_TOKEN_USAGE_FILE` to a persistent JSON ledger and
`FORGER_TOKEN_UNIT_LIMIT` to share a quota across model instances and processes.
Ledgers retain `cost_usd` and count missing prices in `unpriced_calls`.
New calls stop once the recorded total reaches the quota; in-flight calls can overshoot it.
Unknown model weights or missing token counts stop budgeted calls instead of counting as free.
The benchmark launcher sets a 230M-unit generation quota and a separate judge quota,
given in the task instructions.
Direct API calls must use the launcher's `rl.token_budget` helpers to participate in accounting.

### Execution model

A pipeline is a Python function. It calls operations from `pyforger.acquire`, `pyforger.alter`, `pyforger.filter`,
and `pyforger.inspect`. Each operation takes a `list[dict]` (or generation parameters). Acquire, alter, and filter
operations generally return a `list[dict]`; rubric generation returns one rubric per input.
`acquire.generate_custom_rubric_criteria` maps
`list[str]` to `list[list[tuple[str, float]]]`. Inspect operations return a report about the rows without modifying rows.
Control flow (`for`, `if`, comprehensions, helper functions) in plain Python can be used

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (`sample_from_LLM`,
`generate_custom_rubric_criteria`,
every `alter.*` op, every `filter.*` op, every `inspect.*` op) are decorated internally with `@op`, so their results
are cached on disk and traced. The retrieval functions (`retrieve`, `retrieve_from_*`) are also internally cached
and traced by default. Pass `_no_cache=True` to skip cache reads and writes for an invocation, or
`_cache_only=True` to read an existing result without fetching (a miss raises `FileNotFoundError`).
This includes RANDOM mode, even without a seed: use `_no_cache=True` for a fresh sample.
Direct calls to source `.fetch()` methods remain uncached.

### Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.op` | `src/pyforger/op.py` | Decorator for caching operations. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing items. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.inspect` | `src/pyforger/inspect/` | Examine rows without modifying them; returns a report. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`, `canonical`. |

## Reference files

The detailed reference is split by topic under `docs/agents/`; read the file for the operations you use:

- [`docs/agents/acquire.md`](agents/acquire.md): Generation with `sample_from_LLM`, `TinkerModel` student sampling,
  retrieval, rubric generation.
- [`docs/agents/alter.md`](agents/alter.md): Row transforms: `generate_from_context`, translation, code execution, STaR.
- [`docs/agents/filter.md`](agents/filter.md): Quality and diversity filters, including the LLM judge.
- [`docs/agents/inspect.md`](agents/inspect.md): Reports on rows: `topic_based_clustering` and `eval_tree`.
- [`docs/agents/models.md`](agents/models.md): The `Model` class, gateway credentials, reasoning effort, prompt types.
- [`docs/agents/caching.md`](agents/caching.md): How `@op` caching and tracing work.
- [`docs/agents/pipelines.md`](agents/pipelines.md): Composing pipelines and common data-generation patterns.
- [`docs/agents/dynamic_step.md`](agents/dynamic_step.md): Input-specific sub-pipelines with `dynamic_step`.

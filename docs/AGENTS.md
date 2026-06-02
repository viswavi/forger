# AGENTS.md: Forger pipeline reference

Forger is a tool for describing and executing pipelines for generating synthetic data or RL environments. This
reference describes the execution model, the public API, and the conventions the test suite expects.

## Contents

- [Basic Strategy for Using Forger](#basic-strategy-for-using-forger)
- [Credentials and Resource Management](#credentials-and-cost)
- [Execution model](#execution-model)
- [Public API summary](#public-api-summary)
- [Operation reference](#operation-reference)
  - [`pyforger.acquire`](#pyforgeracquire)
  - [`pyforger.alter`](#pyforgeralter)
  - [`pyforger.filter`](#pyforgerfilter)
- [Models, prompts, and credentials](#models-prompts-and-credentials)
- [Caching and tracing with `@op`](#caching-and-tracing-with-op)
- [Composing a pipeline](#composing-a-pipeline)

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
They can either be stored on disk (e.g. in a `.env` file) or managed through environment variables (e.g. `export
LITELLM_API_KEY="your_api_key"`).

LLMs will almost always be accessed via LiteLLM -- if the user does not provide a LITELLM_API_KEY environment
variable or in the prompt, you should pause and ask the user to provide one. Generally the user should also provide
a model identifier (e.g. "litellm_proxy/lab_name/XXX") in the prompt, which you should set as the model name in
LiteLLM. If not given, similarly pause and ask the user to provide one. This model identifier will typically include
both account/user information (e.g. "litellm_proxy/azure/") and a model name (e.g. "gpt-5-mini"). Within this
structure, you can potentially try other models within the same account if the user-provided model is undesirable or
unavailable. Within Azure specifically, the following models are available by default:
"gemini/gemini-3.1-flash-lite" (fast, cheap), "gemini/gemini-3.5-flash" (accurate, expensive),
"azure_ai/DeepSeek-R1-0528", "azure_ai/DeepSeek-V4-Flash", "anthropic/claude-haiku-4-5" (fast),
"anthropic/claude-sonnet-4-6" (medium speed/cost), "anthropic/claude-opus-4-7" (expensive), "azure_ai/gpt-5.1-codex",
"azure_ai/gpt-5.4-mini", "azure_ai/gpt-5.4-nano", "azure_ai/gpt-5.4-pro", "azure_ai/gpt-5.5" (slowest/most accurate),
"azure_ai/gpt-oss-120b", "azure_ai/Kimi-K2.6", "azure_ai/Kimi-K2-Thinking", "azure_ai/Llama-3.3-70B-Instruct",
"azure_ai/Llama-4-Maverick-17B-128E-Instruct-FP8", "azure_ai/text-embedding-3-large" (text embeddings),
"azure_ai/text-embedding-3-small" (text embeddings), "azure_ai/FLUX.2-pro" (image generation)

You can check whether the LITELLM_API_KEY environment variable is provided in Python with:

```python
from dotenv import load_dotenv
import os
load_dotenv()

if "LITELLM_API_KEY" not in os.environ:
    print("Please provide a LITELLM_API_KEY environment variable.")
```

### Cost

LLM calls and external API calls may incur costs. For LLM calls, the cost is based on the number of input/output
tokens and the model used. For example, gpt-5.4-nano is generally much cheaper than gpt-5.4-pro or gpt-5.5.

### Execution model

A pipeline is a Python function. It calls operations from `pyforger.acquire`, `pyforger.alter`, and `pyforger.filter`.
Each operation takes a `list[dict]` (or generation parameters) and returns a `list[dict]`. Control flow (`for`, `if`,
comprehensions, helper functions) in plain Python can be used

Caching, tracing, and pipeline-level optimization are opt-in through two modules: `pyforger.op` (`@op` decorator for
per-call disk cache and tracing) and `pyforger.pipeline` (`run` and `optimize` hooks).

### Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.run(fn, *a, **kw)` | `pyforger.pipeline.run` | Calls `fn(*a, **kw)`. Hook for future trace and telemetry. |
| `pyforger.optimize(fn, [transforms])` | `pyforger.pipeline.optimize` | Applies `Callable[[Fn], Fn]` rewriters in
order. |
| `pyforger.op` | `pyforger.op.op` | Decorator: disk cache and trace. Trace helpers are `from pyforger.op import
trace, clear_trace`. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing items. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`. |

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (`sample_from_LLM`
every `alter.*` op, every `filter.*` op) are decorated internally with `@op`, so their results are cached on disk and
traced. The retrieval functions (`retrieve`, `retrieve_from_*`) are not internally cached; wrap them with `@op` to
memoize their results.

#### `pyforger.acquire`

Generate or retrieve new rows.

| Export | Cached | Description |
|---|---|---|
| `Model` |   | LiteLLM-backed wrapper. `generate(prompt, *, n=1, ...) -> list[str]` and `generate_batch(prompts, *,
n=1, ...) -> list[list[str]]`. Exposes`model.last_usage` (prompt tokens, completion tokens, total cost from the most
recent call). |
| `sample_from_LLM` | yes | First-class generation source. Supports JSON-schema validation with retry. |
| `retrieve` | no | Generic retrieval against a `RetrievalSource`. |
| `retrieve_from_web` | no | Brave Search backend (QUERY only). |
| `retrieve_from_wiki` | no | Wikipedia backend (QUERY or RANDOM). |
| `retrieve_from_hf` | no | Hugging Face datasets backend. |
| `retrieve_from_arxiv` | no | arXiv backend. |
| `retrieve_from_s2` | no | Semantic Scholar backend. |
| `RetrievalSource`, `RetrieveMode` |   | Abstract base class and mode enum (`QUERY`, `RANDOM`). |
| `WebRetrievalSource`, `WikiRetrievalSource`, `HFRetrievalSource`, `ArxivRetrievalSource`, `S2RetrievalSource` |   |
Per-backend source classes. |

#### `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |

`generate_from_context` is defined in `pyforger.alter.generate` but is not re-exported from `pyforger.alter`. Import
explicitly:

```python
from pyforger.alter.generate import generate_from_context
```

It performs `samples_per_item` LLM calls per input row and returns parsed dicts, with retry against an optional JSON
schema.

#### `pyforger.filter`

Remove rows by criterion. If a filter empties the list, downstream operations receive `[]` and return `[]`.

| Export | Description |
|---|---|
| `by_exact_agreement` | Keep rows where two fields agree (or disagree, with `disagree=True`). |
| `by_rouge_l_agreement` | ROUGE-L similarity gate between two fields. |
| `by_exact_match` | Keep up to `max_count` rows per unique field value. |
| `by_rouge_l` | Pairwise ROUGE-L diversity filter. |
| `by_n_gram_overlap` | N-gram overlap diversity filter. |
| `by_length` | Filter by word count within a given field. |
| `by_invalid_response` | Drop empty, identical, or near-identical responses to the instruction. |
| `by_keyword_blocklist` | Drop responses containing blocklisted keywords; default blocklist targets common refusal
phrases. |
| `by_llm_score` | LLM-as-judge with optional multi-judge consensus. |
| `by_syntax` | Validate JSON, YAML, Python, or XML, optionally against required keys or a JSON schema. |

These are important for assuring quality and diversity in generated data, and one should experiment with their
parameters and thresholds to find the right balance for a given use case. For example, `by_rouge_l` with a very high
threshold produces more diverse outputs but may allow low-quality ones through; `by_llm_score` with multiple judges
and a high consensus threshold produces higher-quality outputs but at the cost of more LLM calls.

### Models, prompts, and credentials

#### The `Model` class

```python
from pyforger.acquire import Model

model = Model(
    "openai/claude-4.5-haiku",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=None,
    num_retries=2,
)
```

As described at the beginning of this guide, Identifiers follow LiteLLM conventions (`openai/gpt-4o`, `anthropic
claude-sonnet-4-20250514`, etc.), but typically the user should provide one in the prompt. Unless the user explicitly
provides otherwise, set `api_key` as `LITELLM_API_KEY` (from the environment variables). Additional `**kwargs` are
forwarded to `litellm.completion` (`api_base`, `timeout`, `organization`).

#### Prompt types

Two type aliases are exported from `pyforger.acquire.model`:

```python
Message = dict[str, str]                 # {"role": "...", "content": "..."}
Prompt  = Union[str, list[Message]]      # string or full chat message list
```

A bare string is wrapped as a single user message. Pass a `list[Message]` for a system prompt or multi-turn structure.

For operations that prompt per row (e.g. `generate_from_context`), the template type is:

```python
PromptTemplate = Union[str, Callable[[dict], Prompt]]
```

A string template is filled with `str.format_map(item)` (use `{field_name}` placeholders). A callable receives the row
dict and must return a `Prompt`.

### Caching and tracing with `@op`

```python
from pyforger import op

@op
def expensive_step(rows, *, model, temperature=0.7):
    ...

@op(name="custom_op_name", cache=False)
def debug_step(rows):
    ...
```

**Cache key.** SHA-256 of: the op name, a canonical representation of the function's compiled body (bytecode,
constants, names, varnames, freevars, argcount, applied recursively to nested code objects), the captured closure
values, and the call arguments. Editing the decorated function's body invalidates that function's cache; editing an
unrelated helper does not.

**Cache location.** `$FORGER_CACHE` if set, otherwise `~/.cache/forger`. The variable is read at call time, so tests
can redirect it.

**Per-call bypass.** Pass `_no_cache=True` at the call site to skip the cache for a single invocation.

**Custom hash keys for objects.** If an argument defines `cache_key() -> Any`, the return value is hashed in place of
the object's full canonical form. `MockModel` defines one. The base `Model` class does not: its canonical form falls
through to `__repr__`, which includes only `model_id`. As a result, two `Model("openai/gpt-4o", temperature=0.7)` and
`Model("openai/gpt-4o", temperature=0.0)` produce the same cache key. Subclass `Model` and implement `cache_key()`
when configuration differences must be distinguished by the cache:

```python
class ConfiguredModel(Model):
    def cache_key(self):
        return (self._model_id, self._temperature, self._max_tokens)
```

**Trace.** Import the trace helpers from the module:

```python
from pyforger.op import trace, clear_trace
```

`trace()` returns a list of `CallRecord(op_name, input_hash, output_hash, cache_hit, duration_s)` for every `@op` call
in the current process. `clear_trace()` resets it.

### Composing a pipeline

A single-pass pipeline:

```python
import pyforger
from pyforger import acquire, alter, filter, op

@op
def build_dataset(seeds, model):
    rows = alter.translate.to_language(
        seeds, target_language="Spanish", fields=["text"], model=model,
    )
    rows = filter.by_length(rows, instruction_field="text", min_len=3, max_len=200)
    rows = filter.by_rouge_l(rows, field="text", threshold=0.7)
    return rows

def main(seeds, model):
    return pyforger.run(build_dataset, seeds, model)
```

An iterative pipeline keeps the outer loop in Python and decorates each round's helper with `@op` so that cached
rounds are reused across runs:

```python
import random

@op
def one_round(rows, model):
    in_context_examples = random.sample(rows, min(len(rows), 5))
    prompt = # construct a prompt using the in-context examples
    new = acquire.sample_from_LLM(prompt=prompt, model=model, n=64)
    return rows + filter.by_rouge_l(new, field="text", threshold=0.7)

def run_pipeline(seeds, model, rounds=5):
    rows = seeds
    for _ in range(rounds):
        rows = one_round(rows, model)
    return rows
```

Module import time should not execute work. LLM calls, file I/O, and any `@op`-decorated call belong inside a function body.

### Data Generation Patterns

We often see some common patterns in data generation pipelines. Here is an incomplete list of examples:

- **Topic-based generation**: First sample from a list of topics (e.g. "biology", "chemistry", "mathematics") and then
condition generation on the chosen topic. This can improve diversity across topics and allow for topic-specific
prompting or filtering.
- **Retrieval-augmented generation:** First retrieve relevant information (e.g. from Wikipedia or a custom document
collection) and then condition generation on that retrieved information.
- **Self-refinement:** For a complicated step (e.g. generating a program to pair with a given natural language query),
first generate a "draft" output. Then use `pyforger.alter.generate_from_context` with a prompt that includes the query
and draft and asks for a critique of this draft. Then, use another `generate_from_context` call with a prompt that
includes the query, draft, and critique and asks for a refined output.
- **Back-translation:** First retrieve or generate outputs (e.g. math proofs) and then "back-translate" a suitable
input that could have produced that output (e.g. an English description of a math problem). This can be done with two
separate `@op`-decorated functions, one for the forward direction and one for the backward direction.
- **Overgenerate and filter:** Generate more candidates than needed (especially at the early stages of a pipeline) and
then apply `pyforger.filter` operations to assure quality and diversity. For example, generate 1000 candidate
queries,  filter with `by_llm_score` to keep the 250 best according to a consensus of 5 judge calls, and then filter
with `by_rouge_l` to keep a diverse subset of 100.
- **Rubric-based filtering:** For complex tasks where quality is hard to assure with simple heuristics, generate a
rubric (using `pyforger.alter.generate_from_context`) and then use multiple `by_llm_score` calls with each rubric item
as a separate judge to filter for high-quality outputs that meet multiple criteria.
- **Ensemble of experts:** For a given step, generate multiple outputs using different models or different prompts
(e.g. a "creative" prompt and a "precise" prompt) and then use `pyforger.filter.by_rouge_l_agreement" to only keep
rows where these different experts have matching outputs. This can improve quality (at the cost of diversity and
computational cost).
- **Expert-and-nonexpert prompting:** For a given input, generate one output with a "non-expert" model (e.g. "Llama-3
3-70B-Instruct") and one with multiple "expert" models (e.g. multiple calls of "gpt-5.5" with slightly different
prompts or another strong model). Then choose only the rows where the expert models agree with each other (answer is
likely correct) but disagree with the non-expert model (input is not too easy).
- **Iterative in-context learning:** Start with a simple pipeline that generates some initial outputs. Then, keep an
outer loop in Python and at each round sample in-context examples from the outputs of the previous round to construct
a prompt for the next round. Decorate the inner loop function with `@op` so that improvements to the prompt or
sampling parameters are cached and reused across rounds. This pattern can improve diversity and quality over multiple
rounds, as the pool of in-context examples evolves.

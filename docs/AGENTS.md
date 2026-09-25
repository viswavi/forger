# AGENTS.md: Forger pipeline reference

Forger is a tool for describing and executing pipelines for generating synthetic data or RL environments. This
reference describes the execution model, the public API, and the conventions the test suite expects.

## Contents

- [Basic Strategy for Using Forger](#basic-strategy-for-using-forger)
- [Credentials and Cost](#credentials-and-cost)
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
They can either be stored on disk (e.g. in a `.env` file loaded by your shell or other external tooling) or managed
through provider-specific environment variables.
LLMs will almost always be accessed via LiteLLM-backed model wrappers. Generally the user should also provide a model
identifier (e.g. "base/org_name/XXX") in the prompt, which you should set as the model name in LiteLLM. If not given,
pause and ask the user to provide one. This identifier should include account/user information (e.g. "base/org_name")
a model name (e.g. "gpt-5.6-sol"). Within this structure, you can potentially try other models within the same account
if the user-provided model is undesirable or unavailable. Within Azure specifically, the following models are available
by default:
"gemini/gemini-3.1-flash-lite" (fast, cheap), "gemini/gemini-3.5-flash" (accurate, expensive),
"azure_ai/DeepSeek-R1-0528", "azure_ai/DeepSeek-V4-Flash", "azure_ai/gpt-5.6-terra" (fast),
"azure_ai/gpt-5.6-luna" (medium speed/cost), "azure_ai/gpt-5.6-sol" (expensive/most accurate).

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

LLM calls and external API calls may incur costs. For LLM calls, the cost is based on the number of input/output
tokens and the model used. For example, gpt-5.4-nano is generally much cheaper than gpt-5.4-pro or gpt-5.5.

### Execution model

A pipeline is a Python function that consists of a sequence of operations. Many useful implementations of operations
are given in `pyforger.{acquire/alter/filter}`. You can also implement your own operations. Each operation takes an
optional input list (`list[dict]`) and returns an output list (`list[dict]`). Caching and tracing are enabled by adding
the `@op` decorator.

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (including almost
everything in `pyforger.{acquire/alter/filter}`) should be decorated internally with `@op`, so their results are cached on
disk and traced. The retrieval functions (`retrieve`, `retrieve_from_*`) are not internally cached; wrap them with `@op`
to cache their results.

### Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.op` | `src/pyforger/op.py` | Decorator for caching operations. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing items. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`. |

#### `pyforger.acquire`

Generate or retrieve new rows.

| Export | Cached | Description |
|---|---|---|
| `sample_from_LLM` | yes | Sample data from an LLM with a prompt. Supports schema-based validation and retries. |
| `retrieve` | no | Generic retrieval against a `RetrievalSource`. |
| `retrieve_from_web` | no | Brave Search backend (QUERY only). |
| `retrieve_from_wiki` | no | Wikipedia backend (QUERY or RANDOM). |
| `retrieve_from_hf` | no | Hugging Face datasets backend. |
| `retrieve_from_arxiv` | no | arXiv backend. |
| `retrieve_from_s2` | no | Semantic Scholar backend. |
| `RetrievalSource`, `RetrieveMode` |   | Abstract base class and mode enum (`QUERY`, `RANDOM`). |
| `WebRetrievalSource`, `WikiRetrievalSource`, `HFRetrievalSource`, ... |   | Per-backend source classes. |
| `ArxivRetrievalSource`, `S2RetrievalSource` |   |Per-backend source classes. |
| `Model` |   | LLM wrapper (initialized by user). Otherwise, LLM generation should be called with `sample_from_LLM`. |

`sample_from_LLM` performs `n` LLM calls per input and returns parsed dicts. It does under-the-hood batching and
validation, making this a superior choice to calling `Model` directly.

#### `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Execute Python code in a sandboxed environment and return the result. |

`generate_from_context` is defined in `pyforger.alter.generate` but is not re-exported from `pyforger.alter`. Import
explicitly:

```python
from pyforger.alter.generate import generate_from_context
```

It performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an optional JSON schema.
If the validated LLM call fails after `num_retries`, we return `None` for the failing slots.

`self_taught_reasoner` (re-exported from `pyforger.alter`, also in `pyforger.alter.generate`) implements the
**Self-Taught Reasoner (STaR)** approach ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) -- closely
related to **On-Policy Distillation** (Agarwal et al., 2023), since both keep SFT traces drawn from the student's own
distribution rather than a separate teacher: for each input row it samples `n` responses from a student `model` and
keeps only those whose final answer matches the row's reference answer (`reference_field`, default `"final_answer"`).
Each kept response becomes one
output row carrying the student's raw response plus, when `split_thinking=True`, its reasoning and answer in
`thinking`/`answer` fields (reasoning comes from the provider's separate reasoning field when returned, otherwise from
the inline `<think>` block); the matched student answer becomes the new ground truth. A row with zero correct
samples contributes nothing (so 8 samples with 4 correct yield 4 rows). Acceptance defaults to extracting an answer
with `default_extract_answer` (reads `ANSWER:`/`\boxed{}`/`####` markers) and comparing by normalized string equality;
pass a custom `extract_answer`, `match_fn`, or a `filter_fn` (any `list[dict] -> list[dict]`, e.g. a `pyforger.filter`
op bound with `functools.partial`) to override. Pair it with `LocalModel` to generate student traces on-GPU for SFT.

```python
from pyforger.alter import self_taught_reasoner
from pyforger.acquire import LocalModel

student = LocalModel("Qwen/Qwen3-4B", max_model_len=40960)
rows = self_taught_reasoner(problems, solver_prompt, student, n=8, reference_field="final_answer")
```

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
| `by_keyword_blocklist` | Drop responses with certain keywords; default blocklist targets common refusal phrases. |
| `by_llm_score` | LLM-as-judge with optional multi-judge consensus. |
| `by_syntax` | Validate JSON, YAML, Python, or XML, optionally against required keys or a JSON schema. |

These are important for assuring quality and diversity in generated data, and one should experiment with their parameters
and thresholds to find the right balance for a given use case. For example, `by_rouge_l` with a very low threshold
produces more diverse outputs but may filter out many rows; `by_llm_score` with multiple judges and a high consensus
threshold produces higher-quality outputs but at the cost of more LLM calls.

### Models, prompts, and credentials

#### The `Model` class

```python
import os
from pyforger.acquire import Model

model = Model(
    os.environ.get("MODEL_ID"),
    api_key=os.environ.get("LITELLM_API_KEY"),
    temperature=0.7,
    max_tokens=None,
    num_retries=2,
)
```

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

**Custom hash keys for objects.** If an argument defines `cache_key() -> Any`, this is used in place of the argument's
canonical form. This is useful for objects that are expensive to serialize or contain ever-changing state. Here is
an example:

```python
class ExecutionSandbox:
    def __init__(self, allowed_imports):
        self._allowed_imports = frozenset(allowed_imports)
        self._compiled_fns = {}  # grows as code runs; not part of the sandbox's config

    def cache_key(self):
        return (type(self).__name__, tuple(sorted(self._allowed_imports)))
```

`_compiled_fns` accumulates entries across calls and isn't hashable. `cache_key()` here omits this field to avoid
cache misses. Repeated calls with the same sandbox stay cache hits.

**Trace.** Import the trace helpers from the module:

```python
from pyforger.op import trace, clear_trace
```

`trace()` returns a list of `CallRecord(op_name, input_hash, output_hash, cache_hit, duration_s)` for every `@op` call
in the current process up to a certain maximum (default 10000). `clear_trace()` resets it.

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
    prompt = ...  # construct a prompt using the in-context examples
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
then apply `pyforger.filter` operations to assure quality and diversity. For example, generate 1000 candidate queries,
filter with `by_llm_score` to keep the best according to a consensus of 5 judge calls, and then filter with
`by_rouge_l` to keep a subset with sufficient lexical diversity.
- **Rubric-based filtering:** For complex tasks where quality is hard to assure with simple heuristics, generate a
rubric (using `pyforger.alter.generate_from_context`) and then use multiple `by_llm_score` calls with each rubric item
as a separate judge to filter for high-quality outputs that meet multiple criteria.
- **Strong-vs-weak gap:** For a given input, generate one output with a "weak" model (e.g. "gemini-3.1") and one with
multiple "expert" models (e.g. multiple calls of "gemini-3.5" with slightly different
prompts or another strong model). Then choose only the rows where the expert models agree with each other (answer is
likely correct) but disagree with the non-expert model (input is not too easy).
- **Iterative in-context learning:** Start with a simple pipeline that generates some initial outputs. Then, keep an
outer loop in Python and at each round sample in-context examples from the outputs of the previous round to construct
a prompt for the next round. Decorate the inner loop function with `@op` so that improvements to the prompt or
sampling parameters are cached and reused across rounds. This pattern can improve diversity and quality over multiple
rounds, as the pool of in-context examples evolves.
- **Self-Taught Reasoner / STaR (extending thinking for reasoning problems):** This is the Self-Taught Reasoner (STaR)
approach ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)), closely related to On-Policy Distillation
(Agarwal et al., 2023). When you have prompts paired with *verified*
answers (e.g. competition-math problems with checked integer answers) but want long chain-of-thought traces for SFT,
do not copy reasoning from a stronger teacher -- generate it from the *student* you intend to finetune. This is
generally necessary for training reasoning-style models on tasks that require extended reasoning, such as difficult
math, programming, or embodied reasoning. Sample `n` solutions per prompt from the student (use `acquire.LocalModel` to
run an open-weights reasoning model on-GPU via vLLM with a generous `max_tokens`/`max_model_len` so traces are not
truncated), keep only the solutions whose final answer matches the verified answer, and use each kept trace as a
training target. `alter.self_taught_reasoner` implements exactly this: it returns one row per correct sample (a prompt
with 0 correct samples drops out), with the student's `<think>` trace and answer split into separate fields. Because the
traces are in the student's own distribution, this distills the skill of *extending its own thinking to reach the right
answer* far better than off-policy teacher traces, and answer-verification keeps only correct reasoning. Filtering on
the final answer alone admits the occasional "right answer, wrong reasoning" trace; add a `by_llm_score` judge over the
reasoning (passed as `filter_fn`, or as a downstream filter) only if inspection shows this is actually frequent --
prefer the cheap answer check first.

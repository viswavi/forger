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
  - [`pyforger.inspect`](#pyforgerinspect)
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

A pipeline is a Python function. It calls operations from `pyforger.acquire`, `pyforger.alter`, `pyforger.filter`,
and `pyforger.inspect`. Each operation takes a `list[dict]` (or generation parameters). Acquire, alter, and filter
operations return a `list[dict]`; inspect operations instead return a report about the rows without modifying rows.
Control flow (`for`, `if`, comprehensions, helper functions) in plain Python can be used

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (`sample_from_LLM`
every `alter.*` op, every `filter.*` op, every `inspect.*` op) are decorated internally with `@op`, so their results
are cached on disk and traced. The retrieval functions (`retrieve`, `retrieve_from_*`) are not internally cached; wrap
them with `@op` to memoize their results.

### Public API summary

| Name | Location | Description |
|---|---|---|
| `pyforger.op` | `src/pyforger/op.py` | Decorator for caching operations. |
| `pyforger.acquire` | `src/pyforger/acquire/` | Generate or retrieve rows. |
| `pyforger.alter` | `src/pyforger/alter/` | Transform rows without removing items. |
| `pyforger.filter` | `src/pyforger/filter/` | Remove rows by criterion. |
| `pyforger.inspect` | `src/pyforger/inspect/` | Examine rows without modifying them; returns a report. |
| `pyforger.utils` | `src/pyforger/utils.py` | `get_field`, `get_formatted_logger`, `canonical`. |

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
| `LocalModel` |   | On-GPU vLLM model with `Model`'s interface (needs the `vllm` extra). Zero cost, `<think>` inline. |
| `split_reasoning` |   | `str -> (thinking, answer)`: splits a reasoning-model completion on `</think>`. |

`sample_from_LLM` performs `n` LLM calls per input and returns parsed dicts. It does under-the-hood batching and
validation, making this a superior choice to calling `Model` directly. Any reasoning traces contained in a completion are
contained in the output dict under a `thinking_trace` key. Inline traces are not extracted: `LocalModel` leaves `<think>`
blocks in the completion text and supplies `None` for that metadata; use `split_reasoning` to separate them out.

#### `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Execute Python code in a sandboxed environment and return the result. |
| `dynamic_step` | Run dynamic input-specific sub-pipelines that return a string list per input. |
| `dynamic_step_source` | Generate or inspect sub-pipeline generated for a single input. |

`generate_from_context` performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an
optional JSON schema. If the validated LLM call fails after `num_retries`, we return `None` for the failing slots.
As with `sample_from_LLM`, reasoning traces are included under a `thinking_trace` key. Inline traces are not extracted:
`LocalModel` leaves `<think>` blocks in the completion text. Use `split_reasoning` to separate them out.

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
# LocalModel needs vLLM: pip install "pyforger[vllm]"
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
| `by_group_agreement` | Drop groups of rows (by `group_by_input_field`) whose `match_output_field` values differ. |
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

#### `pyforger.inspect`

Examine rows without modifying them: inspect ops return a *report* about the rows (e.g. `list[str]`), not rows. At any
point, inspection can be performed to generate a report about the current data, without altering the original rows or
producing new rows in the pipeline.

| Export | Cached | Description |
|---|---|---|
| `topic_based_clustering` | yes | Interpretable clustering of contents of one field from the input rows. |

Topic-based clustering uses the ClusterFusion clustering algorithm ([arXiv:2512.04350](https://arxiv.org/abs/2512.04350)),
a hybrid method in which embeddings guide how rows are partitioned and an LLM summarizes the partitions into
interpretable topics, then assigns each row to one of them.

```python
from pyforger.inspect import topic_based_clustering

clusters = topic_based_clustering(
    rows,
    field="problem",
    model=model,
    context="competition mathematics problems. Topics should describe the "
            "mathematical subject and technique, NOT the numeric answer.",
    num_clusters=20,
)
# clusters[i] is the topic name of rows[i]
```

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

Use `Model(..., reasoning_effort="low")` where supported by the model/provider.
`dynamic_step` forwards `reasoning_effort` and `extra_body` to OpenHands.

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
Pass `_cache_only=True` to retrieve an existing result without executing the function;
a miss raises `FileNotFoundError`.

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
with 0 correct samples drops out), with the student's reasoning trace and answer split into separate fields. Because the
traces are in the student's own distribution, this distills the skill of *extending its own thinking to reach the right
answer* far better than off-policy teacher traces, and answer-verification keeps only correct reasoning. Filtering on
the final answer alone admits the occasional "right answer, wrong reasoning" trace; add a `by_llm_score` judge over the
reasoning (passed as `filter_fn`, or as a downstream filter) only if inspection shows this is actually frequent --
prefer the cheap answer check first.

### Dynamic input-specific pipelines

`alter.dynamic_step(inputs, goal, agent, backend=None, max_parallel=10)`
returns a `list[list[str]]` in input order. Each inner list contains the strings
produced for one input and may be empty. The default `pyforger` install includes
OpenHands SDK and its dynamic-step dependencies; Python 3.12+ and Unix signal
timers are required for execution.
Inputs are JSON-compatible records (`list[dict]`). Each uncached unique input
gets its own agent run on the selected backend. By default, `LocalBackend` runs
an OpenHands SDK conversation in a temporary host workspace with terminal, file
editor, and finish tools. The agent writes `program.py`; Forger replays that
source and feeds validation failures back for another attempt. The last defined
function accepts one dict and returns a list of strings.

```python
from pyforger.agent_backend import AgentSpec

agent = AgentSpec(model, max_iterations=20)
outputs = alter.dynamic_step(
    [{"text": "Alice <alice@example.org>"}, {"text": "Bob <bob@example.net>"}],
    "Return a list containing the email address's domain",
    agent, max_parallel=2,
)
# Example output: [["example.org"], ["example.net"]]
```

`dynamic_step_source(item, goal, agent, backend=None)` stores generated source through standard
`@op` caching under `$FORGER_CACHE` (otherwise `~/.cache/forger`). Use `trace()`
to inspect hits and `_cache_only=True` to inspect source without launching an
agent on a miss. Reordered or overlapping batches reuse source, and cached code
executes again on every call. Duplicate inputs share generation and execution
within a batch. `@op` writes atomically, but simultaneous calls for the same
uncached input in separate batches may each start an agent. Failures raise;
successful source from other inputs is retained.

`examples/show_dynamic_step_source.py` accepts a goal and JSON object inputs, then prints
previously generated programs and their `@op` cache-hit status without opening
cache files directly. As with any
`@op`, changing the decorated function body invalidates its earlier cache entries.
The default source version was advanced for the record input contract, so
previously cached string-input programs are regenerated on first use.
Pass a new `source_version` to either operation to generate and cache a fresh
program for an input when its earlier source is stale.

`max_parallel` bounds active sessions, `AgentSpec.max_iterations` bounds agent steps,
and `timeout_s` bounds each generated-code replay. `dynamic_step` works best with fast models
suited to agent workflows (e.g. DeepSeek-v4-flash or gpt-5.6-luna) to avoid timeouts.
All model information (LiteLLM model ID, base URL, and API key, as well as hyperparameters) can come from the Model class.
Credentials and base URL fall back to the LiteLLM environment, which reads `.env` from the current working directory without
changing the process environment. Credentials are passed to the SDK as secrets and excluded from worker environments.
Only `json`, `re`, `math`, and `html` imports are available to generated
code. A URL-only record such as `{"url": "https://example.org/page"}` is
expanded before calling that code. For HTTPS HTML and PDF URLs, Forger retrieves
and extracts text. For HTTPS `.git` URLs on GitHub or GitLab, it shallow-clones
the repository into a temporary directory and passes a bounded snapshot of text
files. The generated function receives a dict with `kind`, `url`, and either
`content` (HTML/PDF/plain text) or `files` (Git). Records with other fields,
including a retrieved row with `text` and `url`, pass through unchanged and
do not trigger another fetch.
The prepared input is supplied to the agent in `request.json`. The Git snapshot
includes at most 200 files,
64 KiB per file, and 1 MiB total text. HTTP downloads have a 10 MiB body cap;
Git clones have a 60-second timeout, a 32 MiB tunnel transfer cap, and a
64 MiB monitored temporary disk cap. Source remains
cached by `@op`, while the URL is fetched or cloned again whenever the generated
step executes. The generated Python worker runs without credentials or direct
network, shell, and filesystem tools. OpenHands runs on the host; SDK-discovered
plugins and their hooks may still be loaded, so use an isolated account for
untrusted inputs.
HTTP and Git connections use validated public IP addresses while preserving
TLS hostname checks; Git tunnels only GitHub and GitLab. PDF parsing runs
before the generated Python timeout starts.
PDF extraction uses `pypdf.PdfReader` on up to 100 pages; image-only scans need
OCR, which this operation does not provide.
RestrictedPython provides accidental-damage guardrails, not isolation against
adversarial code. Cache directories must be trusted.

### Callable sampling

`alter.sample(rows, generate, sample_ids=["0", "1"])` attaches samples produced by
an ordinary callable accepting a row and a sample ID. Values may be text or JSON;
the callable may use an LLM, an agent, or code. `concurrency` bounds simultaneous
calls across rows and sample IDs while preserving input order.
Every alter operation uses `@op`; identical identified samples can be reused,
while distinct IDs or `no_cache` request fresh work.

### Local agent execution

Agent functionality and execution backends are separate layers.
`pyforger.agents` provides backend-agnostic functions (`generate_source`,
`generate_files`). Generation functions take an `AgentSpec`
(what the agent is) and an optional `AgentBackend` (where it runs; defaults to
`LocalBackend`). Backends replay retained code through `execute_program`,
which does not take an agent or call a model. Additional backends implement the same protocol.
Backends implement only `run_agent(task, agent)` and
`execute_program(code, item, timeout_s)`; they never receive agent functionality and
agent functions never know which backend executes them. There is no fallback
path: an incompatible composition fails instead of silently switching
backends. `LocalBackend` runs SDK agents in a host workspace and replays
retained programs through the restricted-Python worker (an
accidental-damage guardrail, not hostile-code isolation). Source caching
remains in the decorated alter operation.

Generated files can be validated on the host and repaired in the same
session. `agents.generate_files(..., validate=..., max_repairs=...)` runs the
caller's `Measurement` callback against each produced snapshot: `True`
accepts, `False` feeds the exact feedback back into the same conversation and
workspace for repair, and `None` stops inconclusively. Backends that cannot
run host validation inside a persistent session reject the request before
launch. `agents.generate_object` layers caller-owned parsing on top: it
returns the parsed object, retains rejected raw files as `previous-N/` inputs
on fresh attempts, and preserves exact parse feedback. Session iteration and
wall-time budgets are shared across generation and every repair turn.

### Independent evaluation and success-rate filtering

`inspect.measure(rows, evaluate)` reports `Measurement` values for existing data.
`filter.by_evaluation(rows, evaluate)` selects explicit passes using the same
measurements. Evaluators may be deterministic, LLM-based, or agent-based; an
exception propagates instead of being recorded as an incorrect answer.

`inspect.success_rates(rows)` summarizes an `evaluations` mapping from group labels
to existing boolean or missing grades. `filter.by_success_rate(rows)` selects rows
whose rate in every group falls within inclusive bounds (default 0.2–0.8).
Neither operation generates or evaluates samples. Missing grades are excluded;
all grades must be present unless `min_valid` explicitly permits exclusions.
Inspect retains reports for rejected rows. Domain-specific attribution, audit,
replay, and admission policy belong to the caller.

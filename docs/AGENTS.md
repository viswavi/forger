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
"litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731" (fast, cheap, surprisingly good for most things), "litellm_proxy/azure_ai/FW-GLM-5.3" (accurate, strong reasoning, expensive).

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
DeepSeek-V4-Flash input/output tokens weigh 1/3 units; GLM-5.3 tokens weigh 4/12.5.
Cached input tokens count at full weight. Completion tokens already include reasoning;
do not add reasoning tokens again. Other model families have no defined unit weight.

`Model` exposes per-call `result.usage["token_units"]` and `result.usage["cost_usd"]`,
plus cumulative `model.total_token_units`, `model.total_cost`, and raw `model.total_usage`.
Unknown model weights or missing usage produce `NaN` token units, including in the
cumulative total. Local GPU inference has zero API cost and is exempt from quotas;
models outside the fixed list still report `NaN` units.
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
are cached on disk and traced. The generic `retrieve` is never internally cached; wrap it with `@op` yourself. Each `retrieve_from_*`
wrapper instead takes a `cache: bool = False` kwarg -- pass `cache=True` to memoize its results via `@op`.

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
| `generate_custom_rubric` | per stage | Recommended: generate mixed code/prompt rubrics ready for the RL executor. |
| `convert_rubric_criteria` | yes | Convert existing textual rubrics, preserving order and weights. |
| `generate_custom_rubric_criteria` | yes | Generate weighted textual rubrics with optional context and candidates. |
| `retrieve` | no | Generic retrieval against a `RetrievalSource`. |
| `retrieve_from_web` | opt-in | Brave Search backend (QUERY only). |
| `retrieve_from_wiki` | opt-in | Wikipedia backend (QUERY or RANDOM). |
| `retrieve_from_hf` | opt-in | Hugging Face datasets backend. |
| `retrieve_from_arxiv` | opt-in | arXiv backend. |
| `retrieve_from_s2` | opt-in | Semantic Scholar backend. |
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

`generate_custom_rubric` is the recommended entry point for humans and agents.
It first generates textual criteria, then converts deterministic requirements
(such as exact counts or JSON structure) to code while retaining semantic and
subjective requirements as prompted judges. For example, Arabic-language
identification stays prompted, while an explicit Unicode range requirement uses
code; sharing a script does not establish language. No mode flag is needed:

```python
from pyforger.acquire import generate_custom_rubric

rubrics = generate_custom_rubric(
    ["Write a 400–600-word story about a shy dragon for children aged 6–8."],
    model=model,
)
# rubrics[i] is [[{"code": "def grade(prompt, response, ground_truth): ..."}, 100.0],
#                [{"prompt": "Is the story suitable for ages 6–8?"}, 75.0], ...]
```

The result uses JSON lists and single-key `code`/`prompt` objects accepted by
`rl.rubric_data.validate_rubric`. Each code criterion defines a synchronous
`grade(prompt, response, ground_truth)` returning a numeric score in `[0, 1]`.
Weights and criterion order are preserved; prompted criteria keep their exact
text. Ambiguous requirements and combined objective/subjective criteria stay
prompted rather than being approximated by heuristics. The main entry point
rejects empty or zero-total-weight rubrics; an empty input list returns `[]`.

To inspect or reuse stages independently, keep using the textual generator below
and then call `convert_rubric_criteria(text_rubrics, model, inputs=task_strings)`.
The optional aligned `inputs` and shared `meta_description` provide context.
Conversion preserves empty rubrics, retries malformed output or invalid code,
and raises with `(rubric, criterion)` indices on exhaustion. Provider errors
propagate. Conversion statically checks the signature, allowlisted imports (at top level
or inside the grader), and
RestrictedPython compilation; it does **not** execute generated code or establish
semantic correctness. Test generated graders with known passing/failing cases
before using them for training. Default word counts use `len(response.split())`.

Both stages are cached independently. `_no_cache=True` on the main entry point
refreshes both; `_cache_only=True` requires both cached stages. The result is a
rubric, not a full training row: callers still supply `prompt` and `ground_truth`.
The RL validator requires the latter to contain reasoning and a final answer.

`generate_custom_rubric_criteria` supports open-ended evaluation (e.g. personal advice, creative writing,
or deep research). Pass a caller-configured `model` and a list of strings containing task descriptions,
prompts, serialized conversations, or documents. `meta_description` provides optional shared evaluation
context. `candidate_responses`, when provided, must contain one list of strings per input; use `[]` for
inputs without candidates. Candidates help identify failure modes and are not treated as ground truth.

```python
from pyforger.acquire import generate_custom_rubric_criteria

rubrics = generate_custom_rubric_criteria(
    ["Write a story about starting a new job.", "Explain how rainbows form."],
    model=model,
    meta_description="Evaluate writing for a non-native English speaking audience",
    candidate_responses=[["On my first day ...", "I began work ..."], []],
)
# rubrics[i] is [(criterion_text, weight), ...] for input i.
```

Weights are independent finite floats in `[0, 100]`, not percentages summing to 100. An empty input list
returns `[]`; a model may return an empty rubric when no meaningful criteria can be grounded in the context.
`batch_size` (default 10) limits prompts per batch, `temperature` defaults to 0.7, and `max_retries`
(default 3) gives additional attempts for malformed outputs, including invalid nested pairs or weights.
Exhaustion raises `ValueError` identifying zero-based input indices; failed inputs are never silently dropped.
Results are cached with `@op`; `_no_cache=True` forces fresh generation.

The output is an intermediate textual rubric. For `rl.agent_training validate`, a later dataset adapter
must wrap each criterion as `{"prompt": criterion_text}` (or convert it to `{"code": source}`),
ensure a nonempty rubric with positive total weight, and supply the required `prompt` and `ground_truth`
fields. The current RL validator requires `ground_truth` to include nonempty `<think>...</think>` reasoning
and a final answer. This building block does not generate those fields or execute criteria.

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
the inline `<think>` block); the matched student answer becomes the new ground truth.

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
| `eval_tree` | yes | Hierarchical capability tree and statistical weakness profile for scored outputs. |

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

EvalTree ([paper](https://arxiv.org/abs/2503.08893)) annotates each problem with a required capability,
embeds those annotations, recursively groups them using K-means and cosine silhouette scores, and describes
each capability subtree. For binary correctness scores, each node reports accuracy and an exact binomial
confidence interval; subtrees whose interval is entirely below `threshold` form a weakness profile. The
implementation is adapted from the [reference repository](https://github.com/zhiyuan-zeng/EvalTree).

```python
import json
from pyforger.inspect import eval_tree

with open("post_data_bench/coding/model_outputs/Qwen__Qwen3-4B_v3/val/predictions.jsonl") as f:
    rows = [json.loads(line) for line in f]

report = eval_tree(rows, input_field="question", score_field="score", model=model,
                   threshold=0.5)
# report["tree"] has descriptions, row indices, accuracy, and child nodes.
# report["weaknesses"] lists the extracted capability groups.
```

The `question` alone is used to infer the skill, avoiding leakage from an incorrect model output. If verified
solutions are available, pass `reference_field` to include them in the capability annotation prompt. The
default embedding backend is the same as `topic_based_clustering` (`pyforger[clustering]`); `embed_fn` can
provide another backend. `threshold`, `alpha`, `min_ci_size`, `min_child_size`, `max_children`, and `seed`
control weakness extraction and tree construction. The result contains only ordinary Python dicts and lists
and is JSON serializable.

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


- **Agent-as-a-step:** Treat an agent as a single step in a larger pipeline. The agent can perform complex reasoning or multi-turn interactions internally, customized to an individual input in the pipeline. This can be slower but allows for more sophisticated processing within a single pipeline step.
- **Retrieval-augmented generation:** First retrieve relevant information (e.g. from the web) and then condition
data generation on this (either directly, for simple/consistently-structured content, or using the "dynamic step"
building block to perform case-by-case extraction of usable details from the retrieved links). It is almost always
better to retrieve knowledge before generating content, rather than relying solely on the model's parametric knowledge.
- **Rubric-based filtering:** For complex tasks where quality is hard to assure with simple heuristics, generate a
rubric (using `pyforger.acquire.generate_custom_rubric_criteria`) and then use multiple `by_llm_score` calls
with each rubric item as a separate judge to filter for high-quality outputs that meet multiple criteria.
- **Overgenerate and filter:** Generate more candidates than needed (especially early in a pipeline) and then apply
`pyforger.filter` operations to assure quality and diversity. Forger offers tons of filters to make this worthwhile.
- **Strong-vs-weak gap:** For a given input, generate one output with a "weak" model (e.g. "gemini-3.1") and one with
multiple "expert" models (e.g. multiple calls of "gemini-3.5" with slightly different
prompts or another strong model). Then choose only the rows where the expert models agree with each other (answer is
likely correct) but disagree with the non-expert model (input is not too easy).
- **Back-translation:** First retrieve or generate outputs (e.g. math proofs) and then "back-translate" a suitable
input that could have produced that output (e.g. an English description of a math problem). This can be done with two
separate `@op`-decorated functions, one for the forward direction and one for the backward direction.
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
with 0 correct samples drops out), with the student's reasoning trace and answer split into separate fields. 

### Dynamic input-specific pipelines

`alter.dynamic_step(inputs, goal, model, max_parallel=4, max_iterations=20)`
returns a `list[list[str]]` in input order. Each inner list contains the strings
produced for one input and may be empty. The default `pyforger` install includes
OpenHands SDK and its dynamic-step dependencies; Python 3.12+ and Unix signal
timers are required for execution.
Inputs are JSON-compatible records (`list[dict]`). Each uncached unique input
gets its own bare-metal OpenHands conversation with
restricted Python, resource inspection, Git inspection, and finish tools. Agents revise code
against execution feedback and submit a function accepting one dict and
returning a list of strings. The last defined function is the entry point.

```python
outputs = alter.dynamic_step(
    [{"text": "Alice <alice@example.org>"}, {"text": "Bob <bob@example.net>"}],
    "Return a list containing the email address's domain", model, max_parallel=2,
)
# Example output: [["example.org"], ["example.net"]]
```

`dynamic_step_source(item, goal, model)` stores generated source through standard
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

`max_parallel` bounds active sessions, `max_iterations` bounds agent steps,
and `execution_timeout_s` bounds each generated-code execution in a subprocess. `dynamic_step` works best with fast models
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
OpenHands has read-only resource and Git list/read tools to inspect the same
snapshot during construction. The Git snapshot includes at most 200 files,
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

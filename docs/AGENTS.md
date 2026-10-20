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
  - [Several inputs and output streams](#several-inputs-and-output-streams)

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

LLM calls and external API calls may incur costs. For LLM calls, the cost is based on the number of input/output
tokens and the model used. For example, gpt-5.4-nano is generally much cheaper than gpt-5.4-pro or gpt-5.5.

### Execution model

A pipeline is a Python function. It calls operations from `pyforger.acquire`, `pyforger.alter`, `pyforger.filter`,
and `pyforger.inspect`. Each operation takes a `list[dict]` (or generation parameters). Acquire, alter, and filter
operations return a `list[dict]`; inspect operations instead return a report about the rows without modifying rows.
Control flow (`for`, `if`, comprehensions, helper functions) in plain Python can be used

Structure a pipeline as a `@pyforger.pipeline` function that calls `@pyforger.step` functions. A step is a named
function from rows to rows built from Forger ops (typically acquire or alter, then filter); it may also take several
lists of rows and return a dict of named row lists (see [Several inputs and output
streams](#several-inputs-and-output-streams)). Each step is cached like
an `@op` and, when it returns, prints a report: the fields of its output rows; phrases and names repeated across
them; every op it ran with rows in/out, yield, time, tokens, and cost; a histogram of each filter's scores with what
other thresholds would have kept; warnings; and one sample output row. The pipeline prints a summary table of its
steps at the end. Read these reports after every run. A cached step prints one line,
`step translate · 42 → 40 rows · cached · originally 0.5s · $0.0034`, with the number of its warnings if it has any;
`inspect_step(name)` prints its full report, which lists the ops of the run that cached it, marked `cached`, with that
run's warnings and score histograms (`filter.by_llm_score (cached run) · …`).
A filter whose own result came from the op cache inside a step that ran also shows the scores of the run that cached
it, marked the same way (`×5 (2 cached)` when some chunks were cached). Results cached before scores were stored with
them, and score lists of over 100,000 rows, have none: rerun with `_no_cache=True` to see them.

- Fields line: `fields: seed_id int · topic str · score int|None · twist str?  (+ score, twist; − draft)`. Each field
  of the output rows with the types of its values (from all rows, or 5,000 evenly spaced rows of a longer list); `?`
  marks a field that some rows lack; `+` / `−` are fields added or removed relative to the input rows, when the step's
  rows arguments are lists of dicts.
- Repeated line: `repeated: rewrite "i need" 78/79 (99%) · context "is writing to" 64/79 (81%) · context "Maya" 46/79
  (58%)`. For string fields the step added or changed, the phrases (2–4 words) and capitalized names in a large share
  of the output rows: templating and name collapse that exact-match and ROUGE-L filters pass, since every row is
  distinct. A phrase needs 30% of the rows for values of up to 100 words, more for longer ones (60% at 200 words, at
  most 90%), since long values share phrases by chance; a name needs 30%. Each field gives its most repeated phrase,
  the most repeated name gets a place, and the line has at most 3 items, or is absent. Left out as template text:
  markup, headings and the section under a heading most rows share (an `## Input` section's "The first line
  contains"), lines most rows share (an instruction prefix), code (`import sys`), categorical and identifier fields,
  stopword-only and "the problem"-style phrases, and phrases the input rows had nearly as often (the seed or the
  template put them there). It reads at most 200 rows (evenly spaced) of 1,000 characters each, from steps with 10 or
  more output rows, and is skipped when reports are off. For the full lists, call `inspect.phrase_frequency`.
- Chunks of one op in a step share one row, e.g. `alter.generate_from_context ×5`, with summed rows, time, tokens,
  and cost, and the yield of the sums. Each call is still checked against its own `min_yield` / `max_yield`
  (`LOW 2/5 · min 0.60` means two of five calls missed, `RAISED 1/5` that one raised). Their filter scores pool into
  one histogram when the label, keep rule, and thresholds match, and get one histogram per rule otherwise. A call
  whose input rows are an earlier call's output rows (the same row objects, also through other ops) gets its own
  row: two `by_llm_score` rubric judges in a row read `10 → 8` and `8 → 4`, not one summed row. Rows rebuilt in
  Python (`{**r, ...}`) between two calls hide that, so two more things keep calls apart: different score labels
  (`by_predicate(..., label="runs")`, then `label=">=3 outputs"`), and input rows that are all new to the step after
  a call that took known rows. A loop over batches with one label still makes one row.
- `(python)` rows: when rows change between two ops with no op to account for it (e.g. `[r for r in rows if ...]`
  between `generate_from_context` and `by_exact_match`), the op table shows a dim row such as
  `(python)  356 → 330  0.93` between them, or before the first op / after the last one. They are inferred: by row
  identity when the next op got some of the same dicts, else from row counts (marked `inferred`) when Python rebuilt
  them all (`{**r, ...}`) and each op row's calls are contiguous. Merged, split, or ambiguous rows (several list
  arguments, interleaved chunks, rebuilt rows sent to two ops) show nothing. The pipeline table shows one between two
  steps when the second got some of the first's rows, by identity only, and per stream when the first returned a dict
  (`(python) train  5  3  0.60`). They are information, never warnings. To see why rows went, write the check as
  `filter.by_predicate`: it gets its own row and a kept/dropped bar.

- Hard errors (`StepContractError` / `PipelineContractError`): a step returns neither a list nor a dict of row lists,
  a step calls no Forger ops (keep pure-Python helpers as plain functions), a step is called inside another step, a
  pipeline calls no steps, or rows do not match a `list[X]` type hint where `X` is a pydantic model (rows stay dicts).
- Warnings: an op misses its `min_yield` / `max_yield`; a step's type hints cannot be resolved; `show=` names a field
  no output row has; a pipeline has only one step; no quality filter, no diversity filter, or no inspect op ran
  anywhere; an op was called in the pipeline outside a step.
- These checks use each op's category, not its name. Ops under `pyforger.acquire/alter/filter/inspect` have their
  package's category; your own `@op` is `"other"` (even `@op(name="filter.x")`) unless it declares one, e.g.
  `@op(category="filter")` for a custom quality filter.
- Filter notes: the pipeline summary lists the steps that ran no filter and the filters that removed nothing (op, step,
  and rows seen). These are notes, not warnings, and `strict=True` does not raise on them: a generate-only step
  followed by a filter step is fine, and a filter that removes nothing may be a check that passed or one that tests
  nothing. Do not add a filter only to quiet them.
- `strict=True` on `@step` or `@pipeline` turns warnings into errors, also on cached runs; a step that fails strict
  mode is not cached. `prod=True` or the environment variable `FORGER_STEP_REPORT=0` silences the reports.
  `pyforger.inspect.inspect_step(name)` re-prints the full report of the latest run of a step, with up to 10 sample
  rows.
- Sample rows: `@step(sample=3)` shows three output rows (of each stream) instead of one. `@step(show=["statement"])`
  prints those fields of each sample row in full, line breaks kept, as an indented block under the row (other values
  as JSON); the other fields stay in the table, cut or summarized.
- Long sample values are summarized by shape rather than cut, e.g. `[5 items] a.py, b.py, c.py, …` or
  `def solve(n): … (312 lines, 9.8k chars, python)`. By default these structural summaries are all you get; they are
  free. For a one-sentence LLM summary of each long value as well, call `pyforger.steps.set_report_model(model)` or
  set `FORGER_REPORT_MODEL=<litellm model id>` (key from `LITELLM_API_KEY`). The sentence is an extra `LLM:` line
  under the value's own text and says what the value says (its task, method, numbers), not whether it is good; check
  the value itself (`inspect_step`) before trusting it. That costs up to 8 LLM calls per printed
  report, cached on disk and left out of the step and pipeline totals (the spend ledger and `FORGER_MAX_COST_USD`
  still count them); silenced reports make no calls. A cheap model without reasoning is enough. Without a report
  model, the first report with a long value logs (at INFO) how to set one; `set_report_model(None)` turns summaries
  off and silences that.
- Threads: ops called from `ThreadPoolExecutor` workers are not itemized in the step report (the step gets a warning
  instead of an error) unless submitted as `pool.submit(contextvars.copy_context().run, fn, arg)`. Steps called from
  workers still join the pipeline, and their spend counts toward it (and its cap) as it arrives. Each op, step, and
  pipeline counts only its own LLM calls, also when several run at the same time. An op that raised is listed as
  `RAISED` with what it spent. LLM calls from workers that did not copy the context and run no step belong to no
  step or pipeline: while exactly one pipeline runs, its report lists them as one `unattributed` line (it may include
  other threads in the process), kept out of its totals and its `max_cost_usd`.
- Concurrent steps: `my_step.submit(*args, **kwargs)` starts a step in a background thread and returns a
  `concurrent.futures.Future`; `.result()` returns its rows or re-raises its error. Use it for independent branches:

  ```python
  impossible = make_impossible.submit(verified[:n], writer, judge)
  hard = student_difficulty(verified[n:], student)        # runs meanwhile
  rows = assemble(impossible.result(), hard)
  ```

  The step runs in a copy of the caller's context, so it joins the pipeline, counts toward its caps, and keeps its own
  report and usage. At most `pyforger.steps.SUBMIT_WORKERS` (8) submitted steps run at once; the rest queue.
  `.submit` inside a step raises like a nested call. The summary lists steps in the order they were started or
  submitted. When the pipeline function returns, it waits for submitted steps still running and raises the error of
  any whose `.result()` was never called; when it raises (or on Ctrl-C), queued steps are cancelled and running ones
  are not waited for.
- Cost: a cached step costs nothing this run; its line shows the time and cost of the run that cached it (`cached ·
  originally 37.9s · $0.50`), and the pipeline summary separates what was spent this run from
  `cached (originally $X)`.
- Spend ledger: every usage record (`Model`, `LocalModel`, `dynamic_step` agents) is appended and flushed as it is
  recorded to `$FORGER_SPEND_LEDGER`, else `$FORGER_CACHE/spend.jsonl` (one JSON line: timestamp, pid, run id,
  pipeline, step, op, model id, tokens, cost), so killed runs keep their record. A `Model` records each request when
  it completes for a single prompt or with `show_progress=True`; with the default `show_progress=False`, a
  multi-prompt batch is recorded once it returns. `FORGER_SPEND_LEDGER=0` turns the ledger off.
  `pyforger.inspect.spend(since="2026-10-16")` prints and returns the totals by run, pipeline, and step.
- Spending caps: `@pipeline(max_cost_usd=20)` caps one pipeline run, and `FORGER_MAX_COST_USD=40` caps all spend in
  the process (unattributed calls included). Both must be numbers >= 0. Once spend reaches a cap, the next model
  request batch raises `pyforger.BudgetExceededError`, as does `dynamic_step` before starting an agent or during its
  run. Requests already sent finish and are recorded, so spend can pass a cap by up to one batch (or one agent step)
  per thread sending them.

### Operation reference

All operations are plain functions. Operations that perform LLM calls or expensive in-process work (`sample_from_LLM`
every `alter.*` op, every `filter.*` op but the cheap `by_predicate`, every `inspect.*` op) are decorated internally
with `@op`, so their results are cached on disk and traced. The retrieval functions (`retrieve`, `retrieve_from_*`)
are not internally cached; wrap them with `@op` to memoize their results. Inside a `@step` they are listed as
`acquire.retrieve`.

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

The tables below give one line per op; [OPS.md](OPS.md) lists each op's signature and docstring summary.

Forger's loggers print to stderr themselves and also propagate to the root logger. Once the root logger has a handler
that prints to the terminal (e.g. after `logging.basicConfig()`), Forger lines are printed by that handler, in its
format, and not a second time by Forger's own.

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
| `retrieve_from_s2` | no | Semantic Scholar backend; `max_workers` runs queries concurrently. |
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
`max_retries` counts rounds whose replies all fail to parse. A request that still fails with a transient provider
error after the `Model`'s own `num_retries` is not sent again: its samples are dropped, so the op returns fewer than `n`
samples with a warning and is not cached. A row error (a content-filter block or an over-long prompt; see
`generate_from_context` below) is not sent again either, but when every request hit one and no sample arrived it
raises, since the one prompt is the template itself. Other errors raise.

`output_schema` (here and in `alter.generate_from_context`) is a dict of keys, e.g. `{"question": str, "answer": str}`,
and the prompt asks for a JSON object with those keys. Only the keys are checked:

- A response must contain a JSON object with every schema key; one without them fails to parse and is retried.
- Keys outside the schema are dropped, so a key the model invents (`"crafted_by"`) never reaches your rows.
- Value types are shown to the model as a format example but not checked: `{"answer": 42}` passes `{"answer": str}`.
  Check types or values yourself, e.g. `[r for r in rows if isinstance(r["answer"], str)]`.
- A ```` ```json ```` fence, prose around the object, or a leading `<think>...</think>` block is tolerated. The first
  object with every schema key wins; JSON that is not an object (a list, a number) fails.

`n` is per query, so three queries with `n=5` return up to 15 rows. A Semantic Scholar query asks the API for only `n`
results. `retrieve_from_s2` runs one query at a time by default, because an API key is usually limited to 1 request
per second and keyless requests share a throttled pool; a rate-limited request is retried after 5 to 60 seconds, so
workers beyond the limit are slower, not faster. With a higher limit, e.g.
`retrieve_from_s2(queries=queries, n=4, max_workers=3)` runs three queries at a time and keeps rows in query order.

#### `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Call the last function in each row's code with one argument, in-process (RestrictedPython). |
| `run_programs` | Run each row's stdin/stdout program on its tests in subprocesses; record per-test results. |
| `dynamic_step` | Run dynamic input-specific sub-pipelines that return a string list per input. |
| `dynamic_step_source` | Generate or inspect sub-pipeline generated for a single input. |

`generate_from_context` performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an
optional JSON schema. A prompt whose output fails to parse (a reply arrived but did not parse) is asked again up to
`max_retries` times; after that its slots are `None` and the other prompts' results are kept. A request that still
fails with a transient provider error (timeout, lost connection, 429, or 5xx) after the `Model`'s own `num_retries` is
not asked again, so with `num_retries=2` a request makes at most 3 attempts in all; its slots are `None`.
A row error belongs to one prompt, not to the key, model, or account, so it is never retried and also leaves that
prompt's slots `None`: the provider's content filter blocked the prompt or its reply
(`litellm.ContentPolicyViolationError`, or a 400 whose body says `content_filter` or `content_policy`), or the prompt
is longer than the context window (`litellm.ContextWindowExceededError`, or `pyforger.PromptTooLongError` from a
backend that checks lengths before sending). If every prompt of the call (at least 2) hits a row error, the op raises
the first one instead, since that points at the template or the model (e.g. a system prompt that trips the filter).
Other provider errors (a bad API key, an unknown model, any other 400, an exhausted quota) raise at once, since
retrying cannot fix them. When any slot failed at the provider, the op logs one warning that counts the failures by
type or reason (`3 of 64 prompts failed after the model's retries; their slots are None: 2 prompts blocked by the
provider's content filter, InternalServerError ×1`), shows it in the step report, and is not cached (nor is its
step). Successful requests are cached one by one in the request cache, so a rerun re-sends only the failed prompts.
Parse-failure `None`s are cached. `inspect.topic_based_clustering` does not ask a failed request
again either (see `filter.by_llm_score` below).
As with `sample_from_LLM`, reasoning traces are included under a `thinking_trace` key. Inline traces are not extracted:
`LocalModel` leaves `<think>` blocks in the completion text. Use `split_reasoning` to separate them out.

By default the op returns a flat list of `len(items) * samples_per_item` outputs (or `None`). `merge=True` returns the
input rows with the output attached instead, one row per successful sample (failed slots dropped), with `sample_idx`
when `samples_per_item > 1`. The output's keys win over the input's (a warning names the clashing keys);
`output_field="name"` nests the output under one key instead (a warning also names an input key it, or `sample_idx`,
replaces):

```python
rows = alter.generate_from_context(problems, "Solve: {problem}", model, output_schema={"answer": str}, merge=True)
# [{"problem": "...", "answer": "4"}, ...]
rows = alter.generate_from_context(problems, "Solve: {problem}", model, merge=True, output_field="solution")
# [{"problem": "...", "solution": {"text": "..."}}, ...]
```

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

##### Running programs against tests

`execute_code` runs a function, not a program: it calls the last function defined in `code_field` with the value of
`arg_field`, in-process. For programs that read stdin and write stdout, use `alter.run_programs`, which runs each
row's program on each of its tests and records what happened. A test is a dict: `{"input": "3\n1 2 3\n", "output":
"6"}`. `"output"` is optional: without it the run is recorded but not graded, e.g. to collect a brute-force
solution's answers as expected outputs (pass `max_chars=None` to keep whole outputs). `"args"` gives command-line
arguments (e.g. `["7", "small"]` for a test generator) and `"timeout_s"` overrides the op's timeout for that test.

Each row gets `test_results` (`result_field`), one result per test: `{"status", "passed", "exit_code", "timed_out",
"stdout", "stderr", "seconds"}`. `status` is `"ok"`, `"wrong_answer"`, `"runtime_error"` (non-zero exit or a signal;
running out of memory usually shows up here as a `MemoryError`), `"timeout"`, or `"output_limit"` (see below).
`passed` is `True`, `False`, or `None` for an ungraded test. A test whose `"input"` is missing or `None` gets empty
stdin. Output is compared with
`outputs_match`, equality up to whitespace (`"1 2\n3"` matches `"1 2 3"`); pass `checker=(stdout, expected) -> bool`
to change it. A row whose code is missing or empty gets `None`. Programs run in Python (this interpreter, with `-I`)
unless `command` names another interpreter, e.g. `command=["node", "{file}"], suffix=".js"`.

```python
from pyforger import alter, filter
from pyforger.alter.programs import last_code_block

rows = [{"statement": "Read two integers and print their sum.",
         "solution": "a, b = map(int, input().split())\nprint(a + b)",
         "tests": [{"input": "1 2\n", "output": "3"}, {"input": "5 5\n", "output": "10"}]}]
rows = alter.run_programs(rows, code_field="solution")
# rows[0]["test_results"][0] == {"status": "ok", "passed": True, "exit_code": 0, "timed_out": False,
#                                "stdout": "3\n", "stderr": "", "seconds": 0.04}

# Keep reference solutions that pass their statement's examples.
verified = filter.by_tests(rows, code_field="solution", tests_field="tests")

# Grade a student's sampled programs with the same tests.
hard = filter.by_pass_at_k(verified, model=student, input_field="statement",
                           metric=filter.program_verifier("tests"), completions_field="samples")
```

`filter.by_tests` keeps rows whose program passes at least `min_pass_rate` (default 1.0) of its graded tests; the runs
are cached apart from `min_pass_rate`, so a new bound runs nothing, and the step report shows a few dropped rows
with their first failing test (`test 2 of 3: wrong_answer, got '4', expected '5'`). `filter.program_verifier` takes the
last fenced
Python block from each sampled answer (`last_code_block`), runs it on the row's tests until one fails, and grades it
`True` only if all pass. An answer with no code block (or one `extract_code` maps to `None` or blank code) grades
`False` without running anything; only a row with no graded tests (none with an `"output"`) grades `None`. The
verifier never sees a `None` answer: the difficulty filter grades an empty answer `None` and, with `by_pass_at_k`'s
default `truncated="fail"`, an unclosed `<think>` block `False`, before calling it. `extract_code=last_code_block`
does the same for `run_programs` and `by_tests` when a field holds a model's answer.

Every run is its own process, so the ops are safe to call from threads and run up to `max_workers` (default
`min(8, cpu count)`) programs at a time. `timeout_s` is wall-clock time, so too many workers on a loaded machine
turn slow passing runs into `"timeout"`; `max_workers` is part of the cache key, so lowering it and rerunning runs
the programs again. Each process starts with an empty environment except `PATH` and the locale, so API keys are not
visible to it, in a fresh temporary directory that is removed afterwards. A run that exceeds `timeout_s` is killed
with its process group, which holds every process it started unless one left the group (`setsid`, a daemon); such a
process keeps running. `max_output_mb` (default 64, 0 for none) caps the size of every file a run writes (stdout,
stderr, and scratch files), and a write past it fails. The status is `"output_limit"` when stdout reached the cap,
or when the run failed after stderr or a file in its working directory reached it. `memory_mb` (default 2048) caps
the address space with `RLIMIT_AS` on Linux; macOS has no working `RLIMIT_AS`, so there the limit is skipped and a
warning says so once. Interpreters that reserve a large address space at startup (e.g. `node`, `java`) may need
`memory_mb=None` on Linux.

Neither `run_programs` nor `execute_code` is a security sandbox. `execute_code`'s RestrictedPython is a guardrail
against accidental damage from machine-generated code, not a boundary against adversarial code, and `run_programs`
runs each program as your user, with your files and network. Run untrusted code at scale in a container or VM.

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
| `by_llm_score` | LLM judge scoring on a rubric, feedback before score; multi-judge consensus, `group_by` candidates. |
| `by_judged_difficulty` | LLM judge rates each task's difficulty from 1 to 5; keeps rows within a range. |
| `by_syntax` | Validate JSON, YAML, Python, or XML, optionally against required keys or a JSON schema. |
| `by_predicate` | Keep rows where `fn(row)` is truthy: a plain Python check that shows in the step report. Not cached. |
| `by_tests` | Run each row's stdin/stdout program on its tests; keep rows passing at least `min_pass_rate`. |
| `by_comparative_difficulty` | Keep rows a strong model answers correctly and a weak model does not. |
| `by_pass_at_k` | Sample `k` answers per row; keep rows whose pass rate is within `[min_rate, max_rate]`. |
| `program_verifier` | Not an op: a `metric` for the difficulty filters that runs sampled programs on the row's tests. |
| `by_topic_frequency` | Cluster rows into topics; downsample over-represented topics and drop rare ones. |

These are important for assuring quality and diversity in generated data, and one should experiment with their parameters
and thresholds to find the right balance for a given use case. For example, `by_rouge_l` with a very low threshold
produces more diverse outputs but may filter out many rows; `by_llm_score` with multiple judges and a high consensus
threshold produces higher-quality outputs but at the cost of more LLM calls.

`by_rouge_l` and `by_rouge_l_agreement` give the same scores as `rouge_score`'s `RougeScorer(["rougeL"])` without
stemming: text is lowercased and split at every character outside `[a-z0-9]` (so accented and non-Latin letters are
separators, not words), and the score is the F1 of the longest common subsequence of tokens. `by_rouge_l` compares
each row with every kept row, so its time grows with rows times kept rows; it scores a row against the whole kept pool
in one bit-parallel pass, e.g. 157 messages of 50–600 words take about 0.2s (2m16s with `rouge_score`).

Every op that takes input rows (alter, filter, inspect) accepts `min_yield=` and `max_yield=`: the expected range of
output rows over input rows, e.g. `filter.by_llm_score(rows, ..., min_yield=0.3, max_yield=0.8)`. A violation, or an
empty input with `min_yield`, is a warning in the step report (`strict=True` raises); neither keyword is part of the
cache key. `acquire.*` ops reject them. Filters also record a score per row, so the step report shows a histogram and
the fraction of rows each nearby threshold would keep; use it to choose thresholds. `by_rouge_l` and
`by_n_gram_overlap` are the exception: each score compares a row with the rows kept so far, so another threshold would
change the scores, and the report shows only the current threshold. `by_llm_score` and `by_tests` also note why each
row scored as it did (the judge's feedback, the first failing test), and under the histogram the report shows up to
three dropped rows with their note. With `num_judges > 1`, `by_llm_score` records the
fraction of judges scoring at least `score_threshold`, which is what `agreement_threshold` is compared with.
`by_llm_score` caches the judge scores separately from its thresholds, so re-running it with a different
`score_threshold` or `agreement_threshold` makes no new judge calls (`_no_cache=True` re-asks the judges;
`_cache_only=True` finds only thresholds that were run before).

Write a Python check on rows as `by_predicate` rather than a list comprehension, so the step report lists it with
rows in/out and a kept/dropped bar (`label` names it there; default: the function's name, or `predicate` for a
lambda). An exception from `fn` propagates with the row index in its message. It counts as neither a quality nor a
diversity filter.

```python
rows = filter.by_predicate(rows, lambda r: len(r["papers"]) >= 3, label="has ≥3 papers", min_yield=0.5)
```

##### LLM judges

`by_llm_score` rates each row on 1 to `max_score`. Absolute judges tend to score nearly everything at the top of the
scale. Two things in the default prompt counter that, following Prometheus
([Kim et al., 2024](https://arxiv.org/abs/2310.08491)): a rubric that describes every score level, not just the
endpoints, and feedback written before the score, like chain of thought. The judge checks the output against each part
of `criteria` and lists every mistake, quoting the part it concerns and naming the highest score the rubric allows with
it (or, if it finds none, says what it checked); the score is at most the lowest of those caps, as
`{"feedback": "...", "score": 3}`. The prompt fences the input and output in `<input>` / `<output>` tags and says that
length or detail the criteria do not ask for counts for nothing. The feedback costs more output tokens per judge call.
Still check the step report's histogram; if it is piled up at the top, the threshold is not separating anything
(tighten `criteria` or the rubric). The criteria matter as much as the prompt: a criterion most outputs meet easily
("the scenario is realistic") still scores nearly everything 5, and one worded too literally ("same voice" for a
rewrite that must drop sarcasm) marks down good outputs, so read a few rows' feedback before trusting a threshold.
The step report helps: under the histogram it shows up to three dropped rows (their `output_field`, cut to 200
characters) with the judge's feedback, the lowest-scoring judge's when there are several.
With several samples per input, `group_by` separates candidates best.

A strong reasoning judge can cost more than the generation it judges: in one trial GLM-5.3 at
`reasoning_effort="low"` wrote about 7K output tokens per judge call (about 92K token units), more than writing the
problems did. Judge a few rows first and check the units column of the step report before judging the whole set.

The default rubric, `pyforger.filter.judge.DEFAULT_RUBRIC`, describes the five levels of the default `max_score=5` by
how serious the output's mistakes are (1: fails the task; 3: a mistake makes part of it wrong, misleading, or
incomplete; 4: nothing wrong or misleading, only minor shortcomings; 5: nothing a careful expert would change). The
default `score_threshold=4` keeps levels 4 and 5; 3 also keeps outputs a reviewer would have to fix. `rubric={1: "...",
..., N: "..."}` replaces it and must describe exactly the levels 1 to `max_score`; any other `max_score` needs one (and
a `score_threshold` when it is below 4).

```python
kept = filter.by_llm_score(
    rows,
    model=judge,
    input_field="question",
    output_field="answer",
    criteria="the final answer is correct and every step is justified",
    rubric={
        1: "Wrong final answer.",
        2: "Right answer, but a step is wrong or missing.",
        3: "Right answer, every step correct, but some steps are asserted without justification.",
        4: "Right answer, every step correct and justified.",
    },
    max_score=4,
    score_threshold=4,
    explanation_field="judge_feedback",
)
```

`group_by="question"` judges the rows that share a value together, e.g. `n=4` samples per prompt: one prompt shows the
input once and the candidates in `<candidate label="A">` tags, and the judge gives feedback and a score per candidate
on the same rubric (`{"candidates": [{"label": "A", "feedback": "...", "score": 4}, ...]}`). Seeing the alternatives
helps the judge notice what a candidate gets wrong; each candidate's feedback is asked to stand on its own, without
naming the others, since labels are shuffled. Each candidate is still kept by `score_threshold` and
`agreement_threshold`, so a group can keep all its rows or none; the result is the kept rows in input order, as
without `group_by`. Candidates are shuffled per group and judge `j` sees that order rotated by `j`, so each candidate
takes each position equally often. With the default prompt a group of one row gets the single-output prompt; with a
`prompt_template` it gets the group prompt with one candidate. A candidate the judge leaves out or scores unparseably
is no vote for that candidate only.

All judge calls (prompts × `num_judges`) go out in one `model.generate_batch` call, one `n=1` request each, so they
run concurrently; `num_judges > 1` needs `temperature > 0`. Judge answers are read leniently: only the text after a
closed `</think>` block (an unclosed `<think>` is unparseable), then the last JSON object with the expected key, also
inside code fences or prose (raw newlines in strings allowed), else the last `"score": <number>` pattern (per entry
for a group, in any key order). Scores keep
fractions (`4.9` stays `4.9`, so it fails `score_threshold=5`), and a string score gives its leading number (`"4/5"`
is 4). Unparseable answers count as no vote, and so does a request that still fails after the model's retries with a
transient error (a 504, a timeout): one warning in the step report names the count and the last error, and the result
is not cached, so a rerun asks again. Another warning counts the judge calls that gave no score by cause (failed,
unparseable, or stopped mid-thought in an unclosed `<think>`) and the rows they left unscored, e.g. with `group_by`
`2 of 10 judge calls gave no score (...) → 6 rows unscored`. A row error (a content-filter block or an over-long
prompt; see `generate_from_context`) counts as a failed request too, and the warning names its reason, e.g.
`1 prompt longer than the model's context window`; a row error on every prompt (at least 2) raises. Other errors a
retry cannot fix (a bad key, another 400, a spending cap or token-unit quota) still raise. The same holds for every
LLM filter: `by_judged_difficulty` leaves a row whose
request failed unrated, `by_pass_at_k` and `by_comparative_difficulty` grade a failed sample `None` (excluded), and
`inspect.topic_based_clustering` does not ask a failed request again, though it asks again after an unparseable answer:
a failed shard summary is skipped, a failed label request gives its document the fallback topic, and a failed codebook
request keeps the last codebook, or raises `RuntimeError` when there is none yet.

`prompt_template` replaces the prompt: a format string or a callable taking a dict. The placeholders are the row's
fields (for a group, its first row's) plus `{input}`, `{output}`, `{criteria}`, `{max_score}`, and `{rubric}` (the
rendered levels); with `group_by` every prompt, one-row groups included, also gets `{candidates}` (the tagged outputs,
also as `{output}`), `{labels}` (`"A, B, C"`), and `{num_candidates}`. Write literal braces as `{{ }}`. If the prompt
does not contain the JSON key in double quotes (`"score"`, or `"candidates"` for a group), the feedback-then-score
answer instruction is appended for you; it mentions the rubric only if the prompt shows `{rubric}`. With `group_by`, a
template whose own answer format has `"score"` but not `"candidates"` raises `ValueError`. The feedback is read from
`"feedback"`, or `"explanation"` for older templates.
`score_field="judge_score"` adds the judges' mean score to kept rows and `explanation_field="judge_feedback"` the list
of their feedback. Without these fields rows are unchanged.

##### Difficulty filters

`filter.by_comparative_difficulty` and `filter.by_pass_at_k` ask models to solve the task in `input_field` and grade
each answer against `ground_truth_field` or with a verifier. `by_comparative_difficulty` is the medium-cost filter in
both cost and reliability: one prediction per row from `weak_model` and from `strong_model`, keeping rows only the
strong model gets right. `by_pass_at_k` is the most expensive but most reliable difficulty filter: `k` samples per
row in one `n=k` call, keeping rows whose pass rate (passed / valid grades) is in `[min_rate, max_rate]` (default
`[0.1, 0.9]`, inclusive).
`by_judged_difficulty` (least expensive, least reliable) only asks a judge how hard a task looks.

No answer format is imposed. The prompt is the task text itself (`prompt_template` replaces it, e.g. to ask for a
final `ANSWER: ...` line), and each completion is graded on what `extract_answer` takes from it. The default,
`pyforger.filter.model_difficulty.answer_segment`, is the text after the completion's closed `<think>` block, stripped
of surrounding whitespace. `metric` is `"exact_match"` (normalized equality with the reference), `"rouge_l"` (F1 at
least `rouge_l_threshold`), or a verifier `(prediction, row) -> bool | None`, e.g. `filter.program_verifier("tests")`,
which runs generated programs on the row's tests (see [Running programs against tests](#running-programs-against-tests)).
`ground_truth_field` is needed only for `exact_match` and `rouge_l`; a verifier reads the row itself. Verifiers run
`verifier_workers` (default 8) calls at a time, so one must be safe to call from several threads. It is part of the
cache key, since a verifier with wall-clock timeouts (e.g. `program_verifier`) can grade differently under load.

By default the whole answer segment is compared, so for `exact_match` the task or `prompt_template` should ask for a
bare answer, or pass your own `extract_answer(completion) -> str | None`. It replaces the default and sees the raw
completion, thinking included; e.g. `lambda t: t.rsplit("ANSWER:", 1)[-1].strip()` keeps what follows the last
`ANSWER:`. An empty answer segment grades `None`, as does `extract_answer` or a verifier returning `None`: excluded,
not failed. A truncated completion (an opening `<think>` and no `</think>`: the model hit `max_tokens`) grades as
`truncated` says. `by_pass_at_k` defaults to `truncated="fail"`: such a sample grades `False` without calling
`extract_answer` or the verifier, since a task the model thinks itself to death on is hard, not ungradable. (Before,
it graded `None`; `truncated="exclude"` restores that.) `by_comparative_difficulty` defaults to `"exclude"`: a weak
model that ran out of tokens has not shown the row is hard. The step report counts truncated samples under the
filter's histogram, e.g. `5 of 12 samples truncated (<think> never closed), graded failed`; if that is most of them,
raise `max_tokens`. A sample whose request failed has no text, so it grades `None` (excluded), never truncated; with a
`Model` (any model whose `generate_batch` takes `return_exceptions`) this includes fan-out copies that failed, so a
prompt that came back short is padded with `None`. Otherwise, a model that returns a number of completions other
than `n` for a prompt raises `ValueError`: a sample missing for any reason other than a failed request is never
padded.
`by_pass_at_k` needs all `k` grades valid unless `min_valid` allows fewer, and a row needs both comparative grades.
With `outcomes_field` set, kept rows carry their grades as `{group: [True | False | None, ...]}`, one grade per
sample: `{"model": [...]}` for `by_pass_at_k`, `{"weak": [...], "strong": [...]}` for `by_comparative_difficulty`.
With `completions_field` set, they carry the raw completions in the same shape and order (`None` for an empty one),
e.g. to keep a student's passing samples as STaR targets without sampling again. Predictions are cached apart from
grades, so changing the rate bounds or adding `completions_field` makes no model calls, and changing the metric
regrades the cached predictions.

```python
hard = filter.by_pass_at_k(
    rows,
    model=student,
    input_field="problem",
    ground_truth_field="final_answer",
    k=8,
    min_rate=0.125,
    max_rate=0.5,
    outcomes_field="evaluations",  # kept rows get e.g. {"model": [True, False, ...]}
    completions_field="samples",  # and {"model": ["<think>...</think>42", ...]}, in the same order
)
targets = [
    {**row, "response": text}
    for row in hard
    for text, ok in zip(row["samples"]["model"], row["evaluations"]["model"])
    if ok
]
```

`filter.by_judged_difficulty` asks an LLM judge to rate how hard each task in `task_field` is on a 5-point scale and
keeps rows rated in `[min_difficulty, max_difficulty]` (default `[3, 5]`, inclusive). It is the least expensive but
least reliable of the difficulty filters: one judge call per row, but the judge only guesses how hard the task is.
`by_comparative_difficulty` (medium cost and reliability) and `by_pass_at_k` (most expensive, most reliable) are the
next rungs up. The judge's fixed system prompt is anchored by three editable rubric descriptions, for levels 1, 3, and
5 (levels 2 and 4 sit between them); their defaults are module constants in `pyforger.filter.difficulty`
(`DEFAULT_MINIMUM_DIFFICULTY_CRITERION`, `DEFAULT_MEDIUM_DIFFICULTY_CRITERION`,
`DEFAULT_MAXIMUM_DIFFICULTY_CRITERION`). With `num_judges > 1` (which needs `temperature > 0`) a row's difficulty is
the mean of its judges' ratings. A row with no parseable rating is dropped. As with `by_llm_score`, ratings are cached
apart from the bounds, so changing `min_difficulty` or `max_difficulty` makes no new judge calls.

```python
hard = filter.by_judged_difficulty(
    rows,
    model=model,
    task_field="problem",
    min_difficulty=4,
    num_judges=3,
    temperature=0.7,
    maximum_difficulty_criterion="A problem that most olympiad medalists could not solve in an hour.",
)
```

`by_topic_frequency` balances topics. It clusters `field` into `num_clusters` topics with
[`inspect.topic_based_clustering`](#pyforgerinspect). A topic whose share of the input rows is above `max_frequency`
keeps a random subset of `max_frequency` of the input (seeded by `sample_seed`); a topic below `min_frequency` is
dropped (bounds inclusive). `1 / num_clusters` is the uniform share, and `max_frequency` defaults to twice that.
`topic_field="topic"` adds each kept row's topic name. The clustering is
the expensive part (one LLM call per unique value) and is cached as its own op, so trying other bounds makes no new
model or embedding calls; use the report's histogram of topic shares to choose them.

```python
balanced = filter.by_topic_frequency(
    rows,
    field="problem",
    model=model,
    context="competition mathematics problems. Topics should describe the technique, NOT the numeric answer.",
    num_clusters=20,
    max_frequency=0.1,  # no topic keeps more than 10% of the rows
)
```

#### `pyforger.inspect`

Examine rows without modifying them: inspect ops return a *report* about the rows (e.g. `list[str]`), not rows. At any
point, inspection can be performed to generate a report about the current data, without altering the original rows or
producing new rows in the pipeline.

| Export | Cached | Description |
|---|---|---|
| `topic_based_clustering` | yes | Interpretable clustering of contents of one field from the input rows. |
| `inspect_step` | no | Print the full report of the latest run of a `@step` and return its `StepRecord`. |
| `spend` | no | Print and return the spend ledger's totals by run, pipeline, and step. |
| `phrase_frequency` | no | Print and return, per string field, the phrases and names the most rows share. No LLM. |

`phrase_frequency(rows, fields=None, *, ngram_range=(2, 4), top_k=10, min_share=0.05, max_rows=2000,
max_chars=2000)` counts document frequency: the share of rows whose value contains each word n-gram (lowercased) or
capitalized name, e.g. `rewrite  phrases: "i need" 78/79 (99%) · "i need you to" 46/79 (58%)` and `context  names:
"Maya" 46/79 (58%)`. It leaves out the same template text as the step report's `repeated:` line, lists a phrase only
when no longer phrase containing it is in nearly as many rows, and returns a `PhraseReport` (one `FieldPhrases` per
field, with `skipped` set for categorical, identifier, and code fields). Topic clustering can look balanced while
every row says "I need"; run both. Rows are not modified.

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

`filter.by_topic_frequency` runs this clustering (cached) to downsample over-represented topics.

With fewer unique `field` values than `num_clusters` (e.g. 6 for `num_clusters=20`), the op uses one cluster per
unique value and logs a warning, so there is no need to shrink `num_clusters` for small inputs yourself. With a single
unique value, every row gets that value (cut to 80 characters) as its topic, without calling the LLM.

Embeddings only decide which rows the LLM summarizes together. By default they come from the sentence-transformers
model `embedding_model` (`pip install "pyforger[clustering]"`, GPU when available). `embedding_model="tfidf"` uses a
built-in TF-IDF + SVD embedder on the CPU (scikit-learn only); if sentence-transformers cannot be imported, the op
falls back to it with a warning. To use another backend, pass `embed_fn`: a function that takes the field values as a
`list[str]` and returns an array-like of shape `(n_rows, dim)`, one vector per string. `embed_fn` overrides
`embedding_model`. The fallback is resolved before the cache lookup, so its result is cached as `"tfidf"` and is not
reused once sentence-transformers is installed.

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
    max_concurrency=16,
)
```

`max_concurrency` caps how many requests this `Model` has in flight at once, across every op, `generate_batch` call,
and thread that uses it (each `n` fan-out copy counts). `generate_from_context`, `sample_from_LLM`,
`self_taught_reasoner`, `translate.*`, and `topic_based_clustering` send all their prompts in one `generate_batch` call,
so set it to what your provider's rate limit allows; you should never need to write your own thread pool around ops.
`None` (default) allows up to 100 requests per `generate_batch` call. It is not part of the cache key, so changing it
keeps cached results. Ctrl-C (or any error in the calling thread) cancels the requests not yet sent; those in flight
finish. Each request's tokens and cost reach the step report as soon as it finishes, also when the call later raises
or is interrupted; `model.history` gets the merged results only when the call returns.

`generate_from_context`, `sample_from_LLM`, and `self_taught_reasoner` take `batch_size=None` (default): all prompts
(or, for `sample_from_LLM`, all `n` samples) in one `generate_batch` call. An int splits them into calls of that size,
sent one after another.

Progress is on by default for a `generate_batch` call that sends more than one request: a tqdm bar when stderr is a
terminal, otherwise a log line at most every 30 seconds (`LLM calls (<model>): 120/500 done, 31s`). Turn it off with
`Model(..., show_progress=False)` (or per call) or the environment variable `FORGER_PROGRESS=0`; an explicit
`show_progress` wins over the variable. `LocalModel` keeps vLLM's own bar, off unless `show_progress=True`.
The log line keeps coming every 30 seconds while requests are in flight, also when none finished since the last one.

Each request attempt times out after `Model(timeout=...)` seconds. The default, `"auto"`, is 600 seconds (10 minutes),
or `max_tokens / 25` when that is longer (`max_tokens=32768` gets 1311 s), so a long reasoning call at a slow 25
tokens/s still finishes. Pass a number to change it (`Model(..., timeout=1800)`), or `timeout=None` for LiteLLM's
own default (6000 s). The timeout is not part of the cache key, so changing it keeps cached results.

`Model` retries each request itself, up to `num_retries` times (default 2), and only on transient errors: a timeout,
a lost connection, a 408/409/425/429, or a 5xx. It waits about 1 s, then 2 s, 4 s, ... between attempts (a 429's
`Retry-After` header instead, up to 60 s), and logs each retry at INFO. A 400 (e.g. an over-long prompt or a
content-filter block, which ops treat as row errors; see `generate_from_context`), 401, 403,
404, or an exhausted quota, credit balance, or budget (OpenAI's `insufficient_quota`, LiteLLM's or Forger's budget
errors) raises at once; a per-minute rate limit is retried, also Gemini's "You exceeded your current quota" 429.
LiteLLM is sent `num_retries=0`, so neither it nor the openai client retries on top: a request makes at most
`num_retries + 1` attempts on every provider, and one that hangs on every attempt waits at most
`timeout × (num_retries + 1)` (30 minutes with the defaults). Lower `num_retries` or `timeout` to bound it. Ops
never re-send a request that failed after these retries (their `max_retries` count parse failures only), so the bound
holds through `generate_from_context`, `sample_from_LLM`, and the LLM filters. Passing `max_retries=` to `Model` turns
the openai client's own retries back on: that many more attempts inside each of Forger's.

When a request has been in flight much longer than the batch's finished ones (over 2 minutes and over 5 times their
median), Forger logs one warning for it, in every progress mode:
`LLM calls (<model>): 1 request in flight for 9m 12s (median of completed requests 41s); each attempt times out after
10m (Model(timeout=...)) and is retried`. It means the provider is slow on that request or the request is stuck; the
batch waits for it until it finishes or times out. If such warnings are common, lower `timeout`, or `max_tokens` if
the long requests are runaway generations.

`model.generate_batch(prompts, return_exceptions=True)` keeps finished results when a request still fails after
`num_retries` (e.g. a gateway 504): the list has one entry per prompt, a prompt whose requests all failed holds the
exception, and a prompt where only some fan-out copies failed holds the completions that arrived (fewer than `n`).
Usage of finished requests is recorded. The default (`False`) raises the first failure. On `LocalModel` an engine
error fills every slot, since vLLM runs the batch as one call.

Use `Model(..., reasoning_effort="low")` where supported by the model/provider.
`dynamic_step` forwards `reasoning_effort` and `extra_body` to OpenHands.

`model.generate_batch(prompts, n=k)` returns exactly `k` completions per prompt. Claude, Grok, Gemini, o1/o3/o4, and
gpt-5 models (also behind `litellm_proxy/`, `openrouter/`, or provider prefixes), and models whose LiteLLM supported
params lack `n`, get `k` concurrent `n=1` requests per prompt, merged into one `GenerateResult` with summed usage and
cost (`raw` is then the list of responses). Other models get one request with `n=k`, or for `k > 16` concurrent
requests of at most 16, merged the same way (`n=40` sends 14, 13, 13). If such a model returns fewer completions
(e.g. a gateway alias that ignores `n`), it logs one warning naming the model, sends only the missing completions as
concurrent `n=1` requests, and fans out that model id for the rest of the process. It never pads: if the `n=1`
requests come back short too, or a response has more than `n`, it raises a `ValueError` (add the model to
`_NO_N_BRANDS` / `_NO_N_PREFIXES` in `model.py`).

Importing `pyforger.acquire` sets `litellm.suppress_debug_info = True`, so provider errors no longer print LiteLLM's
"Give Feedback / Get Help" banner; the raised error is unchanged. Set it back to `False` to see the banner.

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

**Cache key.** SHA-256 of: the op name, the function's qualified name and compiled body (bytecode, constants, names,
varnames, freevars, argcount, applied recursively to nested code objects), the captured closure values, the module
globals it reads that are plain data or define `cache_key()`, and the call arguments. Plain data is `None`, `bool`,
numbers, `str`, `bytes`, and tuples, lists, dicts, and sets of those. A global counts when the body loads it by name,
including in lambdas, comprehensions, and inner functions (`r.split()` does not read a global `split`). Editing a
`PROMPT = "..."` constant or a config dict, or pointing `MODEL = Model(...)` at another model, therefore invalidates
the steps that read it. Globals are read at each call; closure values are fixed when the function is decorated. The
module name is not part of the key, so a step run as `python pipeline.py` (`__main__`) and the same step imported from
`pipeline` share cache entries. Docstrings (of the function and of functions nested in it) are not part of the key
either, so editing a step's description keeps its cache; a function that reads its own `__doc__` as a prompt
therefore gets stale results after a docstring edit (put the prompt in a constant instead).

Callable arguments are keyed by value too: a `functools.partial` by its function, args, and keywords, and an object
with `__call__` by `cache_key()` if it has one, else by its type, `__call__` code, and attributes (pickled, with a
warning, when they are not plain data).

Not in the key: the bodies of helper functions and classes the op calls (only their names), files read from disk,
environment variables, other objects held in globals (e.g. a client, a retrieval source, or an object mutated at
runtime), and closure values changed after decoration. `@op` logs a warning at decoration for such a global (loggers
and typing objects excepted) or for `os.environ`. When one of these changes, pass `_no_cache=True`. Upgrading to this
key layout makes existing cache entries miss once.

**Cache location.** `$FORGER_CACHE` if set, otherwise `~/.cache/forger`. The variable is read at call time, so tests
can redirect it.

**Large values are stored once.** A cached result's `str` or `bytes` values of 16K or more, and lists or tuples of that
total size (e.g. a row's tests), are written once to `$FORGER_CACHE/blobs/<sha256>`, and the entry keeps a reference.
A 47K-char thinking trace carried through six steps is then stored once, not in every step's entry. Cache keys hash
each `str` or `bytes` of 4K or more by its SHA-256, computed once per string object. Deleting `$FORGER_CACHE` still
clears everything; an entry whose blob is gone is a cache miss and is recomputed.

**Per-call bypass.** Pass `_no_cache=True` at the call site to skip the cache for a single invocation, including the
ops it calls. Pass `_cache_only=True` to retrieve an existing result without executing the function;
a miss raises `FileNotFoundError`.

**Skipping the write.** Code running inside an `@op` can call `pyforger.op.skip_cache_write()` to keep the result
out of the cache, e.g. when it holds `None`s for failed requests. It applies to every enclosing op and step too; the
result is still returned, and the next call runs again. An op in a worker thread started with plain `pool.submit` /
`pool.map` cannot see which op or step started the thread, so every op and step running in the process at that moment
is left uncached instead, and a step that is skipped this way gets the warning `not cached: an op in a worker thread
kept its result out of the cache`. Submit with `contextvars.copy_context().run` to limit it to the enclosing calls.
`my_step.submit(...)` already runs the step in a copy of the context, so a failure in it or elsewhere in the
pipeline leaves the other submitted steps cached.

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

**Request cache.** Below `@op`, `Model.generate_batch` stores every completion it receives under
`$FORGER_CACHE/requests/`, so a rerun only pays for requests that never completed: if one row of 260 changes, one
request 504s, or the run is killed, the finished requests are reused (each is written as it finishes; errors, empty
text, and tool calls are never cached). The op cache is checked first; the request cache makes the op's partial
reruns cheap. A completion is keyed on three things, never on the prompt text alone:

- **The op call:** the op's name, its body, and every argument except its input rows, e.g. the template, the model's
  id and output-changing params (temperature, seed, `api_base`, ..., not the API key, timeouts, or retries), and the
  output schema. Each op has its own scope, so two ops never share samples. In `A(D(C(x)))` then `A(B(C(x)))`, C's
  requests are reused, and B never gets D's samples, even with the same template. The sample-count argument (`n`,
  `samples_per_item`, `num_judges`) is left out, so raising it requests only the extra samples; as a consequence,
  `sample_from_LLM` pools that differ only in `n` overlap (pass another `name` for an independent pool).
- **The row** each prompt was built from, so changing one row re-sends only that row's requests, and
  `generate_from_context(rows, SOLVE, model)` run again on rows that now carry an `expert1` field draws fresh
  `expert2` samples. A prompt with no row of its own is keyed on all of the op's rows.
- **The messages and a slot index.** Slots are numbered within one op call, so a request asked again in it (a
  duplicate row, a retry, a later `batch_size` chunk) gets new samples, and a rerun of the same op with the same rows
  and arguments replays them in order, as the op cache would.

Your own `@op` that builds one prompt per row can pass `cache_rows=rows` (aligned with the prompts) to
`generate_batch`, and `@op(samples_arg="n")` names its sample-count argument. Outside an `@op` / `@step` (or from a
thread that did not copy the context) the request cache is off: direct `model.generate` calls always send their
requests. To draw fresh samples, pass `_no_cache=True` to the op (skips reading, still writes),
change an argument such as the seed, or turn the cache off with `Model(..., request_cache=False)` or
`FORGER_REQUEST_CACHE=0`. A reused completion adds no tokens or cost; it stays in `model.history` with zero usage and
`usage["cached_completions"]`, and each op call with hits logs
`request cache: 57 of 64 completions reused in alter.generate_from_context`. `LocalModel` is not request-cached (local
inference is free), nor are requests that ask for tools or logprobs, since a replay has only the text.

**Arguments left out of the key.** `@op(unkeyed=("show_progress",))` leaves the named parameters out of the cache
key. Use it only for knobs that cannot change what the op returns, e.g. a progress bar: `score_all(rows,
show_progress=False)` is then a cache hit after `score_all(rows)`. A worker count is not such a knob when the op has
wall-clock timeouts (more workers, more contention, more timeouts), so Forger's own worker counts are keyed.

**Trace.** Import the trace helpers from the module:

```python
from pyforger.op import trace, clear_trace
```

`trace()` returns a list of `CallRecord(op_name, input_hash, output_hash, cache_hit, duration_s)` for every `@op` call
in the current process up to a certain maximum (default 10000). `clear_trace()` resets it.

### Composing a pipeline

A single-pass pipeline with two steps:

```python
from pydantic import BaseModel
from pyforger import acquire, alter, filter, pipeline, step

class Translated(BaseModel):
    text: str
    spanish: str

@step
def translate(seeds, model) -> list[Translated]:
    """Translate each seed into Spanish and drop empty or very long results."""
    rows = alter.translate.to_language(
        seeds, target_language="Spanish", fields=["text"], target_fields=["spanish"], model=model
    )
    return filter.by_length(rows, instruction_field="spanish", min_len=3, max_len=200)

@step
def dedupe(rows: list[Translated]) -> list[Translated]:
    """Drop near-duplicate translations."""
    return filter.by_rouge_l(rows, field="spanish", threshold=0.7, min_yield=0.5)

@pipeline
def main(seeds, model):
    return dedupe(translate(seeds, model))
```

Each step prints a report when it returns, and the pipeline prints a summary at the end (the `dedupe` report is left
out here):

```text
step translate ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  Translate each seed into Spanish and drop empty or very long results.
  42 → 40 rows (yield 0.95) · 0.5s · 5,040 in / 630 out tokens · $0.0034
  fields: text str · spanish str  (+ spanish)
  rows validated: any → list[Translated]

  op                            in   out   yield   time   tokens in   tokens out   cost
  ────────────────────────────────────────────────────────────────────────────────────────
  alter.translate.to_language   42    42       -   0.5s       5,040          630   $0.0034
  filter.by_length              42    40    0.95   0.0s           -            -   -

  filter.by_length · words · keeps 3 to 200: 40 of 42 (0.95)
      words  rows                                kept
          0     1  ██▌                           no
          1     1  ██▌                           no
          5     3  ███████▋                      yes
          6     6  ███████████████▎              yes
          7     4  ██████████▏                   yes
          8    11  ████████████████████████████  yes
          9    10  █████████████████████████▌    yes
         10     3  ███████▋                      yes
         11     3  ███████▋                      yes

  sample rows (1 of 40)
  #   text                                             spanish
  ────────────────────────────────────────────────────────────────────────────────────────────────
  0   Explain the capital of France in one sentence.   Explica la capital de Francia en una frase.

pipeline main ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  2 steps · 42 → 27 rows (yield 0.64) · 0.5s · 5,040 in / 630 out tokens · $0.0034

  step        in   out   yield   time      cost
  ────────────────────────────────────────────────
  translate   42    40    0.95   0.5s   $0.0034
  dedupe      40    27    0.68   0.0s         -

  pipeline warnings
  ⚠ no quality filter ran (e.g. filter.by_llm_score)
  ⚠ no inspect op ran
```

Run again and both steps come from the cache, one line each, before the same summary:

```text
step translate · 42 → 40 rows · cached · originally 0.5s · $0.0034 ─────────────────────────────────────────────────────

step dedupe · 40 → 27 rows · cached · originally 0.0s ──────────────────────────────────────────────────────────────────
```

`pyforger.inspect.inspect_step("translate")` prints the full report of the latest run, with up to 10 sample rows.
When a filter removes nothing, or a step runs no filter, the summary adds a `filters` section:

```text
  filters
  steps without a filter: add_difficulty
  filters that removed nothing (a check that passed, or one that tests nothing):
    filter.by_exact_match in seed_topics · 10 rows
```

#### Several inputs and output streams

A step may take several lists of rows and return a dict of named row lists (streams); a later step takes one stream.
The rows arguments are the first argument when it is a list, and any other argument that is a non-empty list of dicts
or is annotated `list[dict]` / `list[X]`. Models, numbers, strings, and other lists (e.g. topic names after the first
argument) are not counted. The pipeline may return anything; its summary counts a returned list or dict of lists and
otherwise shows only the input (`10 rows in`).

```python
class Assembled(TypedDict):  # optional; dict[str, list[X]] checks every stream against X
    train: list[TrainRow]
    refusals: list[RefusalRow]

@step
def assemble(solvable, refusals) -> Assembled:
    """Label solved problems for training and keep refusals as their own set."""
    train = label_kind(solvable, "solve")
    refused = label_kind(refusals, "refuse")
    return {
        "train": filter.by_length(train, instruction_field="solution", min_len=2),
        "refusals": filter.by_exact_match(refused, field="problem"),
    }

@pipeline
def coding(writer, solver, judge_model):
    tried = attempt(write_problems(TOPICS, writer), solver)  # {"solvable": [...], "refusals": [...]}
    out = assemble(judge(tried["solvable"], judge_model), tried["refusals"])
    train = [r for r in out["train"] if "heap" not in r["problem"]]
    return {"train": format_train(train), "refusals": out["refusals"]}
```

The step report counts each input and stream, with a fields line and sample rows per stream (`sample=` and `show=`
apply to each); the yield uses the sums. The pipeline table keeps the sums in a step's row, with its inputs (`+`) and
streams (`→`) on dim rows below:

```text
step assemble ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Label solved problems for training and keep refusals as their own set.
  solvable 4 + refusals 3 → train 4 · refusals 3 (yield 1.00) · 0.0s
  train fields: problem str · solution str · refusal str · kind str  (+ kind)
  refusals fields: problem str · solution str · refusal str · kind str  (+ kind)
  rows validated: any → train list[TrainRow] · refusals list[RefusalRow]

pipeline coding ────────────────────────────────────────────────────────────────────────────────────────────────────────
  5 steps · 10 → train 3 · refusals 3 (yield 0.60) · 0.0s · 12,100 in / 4,284 out tokens · $0.052

  step             in   out   yield   time      cost
  ─────────────────────────────────────────────────────
  write_problems   10    10    1.00   0.0s    $0.018
  attempt          10    10    1.00   0.0s    $0.031
    → solvable            7
    → refusals            3
  judge             7     4    0.57   0.0s   $0.0028
  assemble          7     7    1.00   0.0s         -
    + solvable      4
    + refusals      3
    → train               4
    → refusals            3
  (python) train    4     3    0.75
  format_train      3     3    1.00   0.0s         -
```

A cached step reads `step assemble · solvable 4 + refusals 3 → train 4 · refusals 3 · cached · originally 0.0s`.
With a `TypedDict` return hint each stream is validated like `list[X]` and must be one the `TypedDict` names; errors
name the stream and row (`output stream 'refusals' row 1 does not match RefusalRow: ...`). A `list[X]` hint on any
rows parameter validates that input (`input refusals row 0 ...`). A dict with a value that is not a list is an error.

An iterative pipeline keeps the outer loop in Python and makes each round a `@step`, so cached rounds are reused
across runs:

```python
import random

@step
def one_round(rows, model):
    """Sample new rows from in-context examples and keep the diverse ones."""
    in_context_examples = random.sample(rows, min(len(rows), 5))
    prompt = ...  # construct a prompt using the in-context examples
    new = acquire.sample_from_LLM(prompt=prompt, model=model, n=64)
    return rows + filter.by_rouge_l(new, field="text", threshold=0.7)

@pipeline
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

`alter.dynamic_step(inputs, goal, model, max_parallel=4, max_iterations=20)`
returns a `list[list[str]]` in input order. Each inner list contains the strings
produced for one input and may be empty. Called directly in a `@pipeline`, it
counts as a step: it prints a step report, and `inspect_step("dynamic_step")`
shows it. The default `pyforger` install includes OpenHands SDK and its
dynamic-step dependencies; Python 3.12+ and Unix signal timers are required for
execution.
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

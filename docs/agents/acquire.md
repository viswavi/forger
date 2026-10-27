# `pyforger.acquire`

Generate or retrieve new rows.

| Export | Cached | Description |
|---|---|---|
| `sample_from_LLM` | yes | Sample data from an LLM with a prompt. Supports schema-based validation and retries. |
| `generate_custom_rubric` | per stage | Recommended: generate mixed code/prompt rubrics ready for the RL executor. |
| `convert_rubric_criteria` | yes | Convert existing textual rubrics, preserving order and weights. |
| `generate_custom_rubric_criteria` | yes | Generate weighted textual rubrics with optional context and candidates. |
| `retrieve` | yes | Generic retrieval against a `RetrievalSource`. |
| `retrieve_from_web` | yes | Brave Search backend (QUERY only). |
| `retrieve_from_wiki` | yes | Wikipedia backend (QUERY or RANDOM). |
| `retrieve_from_hf` | yes | Hugging Face datasets backend. |
| `retrieve_from_arxiv` | yes | arXiv backend. |
| `retrieve_from_s2` | yes | Semantic Scholar backend; `max_workers` runs queries concurrently. |
| `RetrievalSource`, `RetrieveMode` |   | Abstract base class and mode enum (`QUERY`, `RANDOM`). |
| `WebRetrievalSource`, `WikiRetrievalSource`, `HFRetrievalSource`, ... |   | Per-backend source classes. |
| `ArxivRetrievalSource`, `S2RetrievalSource` |   |Per-backend source classes. |
| `Model` |   | LLM wrapper (initialized by user). Otherwise, LLM generation should be called with `sample_from_LLM`. |
| `LocalModel` |   | On-GPU vLLM model with `Model`'s interface (needs the `vllm` extra). Zero cost, `<think>` inline. |
| `TinkerModel` |   | Base model or `tinker://` checkpoint sampled on Tinker; `Model`'s interface, `<think>` inline. |
| `split_reasoning` |   | `str -> (thinking, answer)`: splits a reasoning-model completion on `</think>`. |

`sample_from_LLM` performs `n` LLM calls per input and returns parsed dicts. It does under-the-hood batching and
validation, making this a superior choice to calling `Model` directly. Any reasoning traces contained in a completion are
contained in the output dict under a `thinking_trace` key. Inline traces are not extracted: `LocalModel` and
`TinkerModel` leave `<think>` blocks in the completion text and supply `None` for that metadata; use `split_reasoning`
to separate them out.
`max_retries` counts rounds whose replies all fail to parse. A request that still fails with a transient provider
error after the model's own `num_retries` is not sent again: its samples are dropped, so the op returns fewer than `n`
samples with a warning and is not cached. A row error (a content-filter block or an over-long prompt; see
`generate_from_context` in [alter.md](alter.md)) is not sent again either, but when every request hit one and no sample
arrived it raises, since the one prompt is the template itself. Other errors raise.

`output_schema` (here and in `alter.generate_from_context`) is a dict of keys, e.g. `{"question": str, "answer": str}`,
and the prompt gets an instruction to end the response with a JSON object with those keys. Only the keys are checked:

- A response must contain a JSON object with every schema key; one without them fails to parse and is retried.
- Keys outside the schema are dropped, so a key the model invents (`"crafted_by"`) never reaches your rows.
- Value types are shown to the model as a format example but not checked: `{"answer": 42}` passes `{"answer": str}`.
  Check types or values yourself, e.g. `[r for r in rows if isinstance(r["answer"], str)]`.
- A ```` ```json ```` fence, prose around the object, or a leading `<think>...</think>` block is tolerated. The last
  object with every schema key wins, so an example object quoted in earlier reasoning is skipped; JSON that is not an
  object (a list, a number) fails.

The instruction allows text before the object, but a reasoning model may still spend its reasoning reconciling it with
your prompt's own format instructions (e.g. "answer with one code block"). If you will train on the model's reasoning,
don't combine `output_schema` with format instructions in the prompt, or generate without `output_schema` and parse
the answer yourself.

Before training a student on another model's `thinking_trace`, read a few traces as SFT targets. A teacher's raw
reasoning is in the teacher's own style and length, and it discusses everything in its prompt: rules you gave it for a
checker ("no names starting with `_`"), output formats, budgets. The student learns to reason about those too. When
the target is the student's reasoning, prefer traces from the student itself (`alter.self_taught_reasoner`, with
`LocalModel` or `TinkerModel` below), or have a model write a short trace for an answer you have already verified,
from a prompt that contains only the task.

To sample a student model, use `LocalModel` when you have GPUs and `TinkerModel` when the student is trained on
Tinker. Use one whenever you need a student model's own outputs: STaR traces, pass@k difficulty checks, or sanity
checks of a checkpoint.

`LocalModel(path_or_name, ...)` loads a Hugging Face model or a local checkpoint directory into vLLM, at no API cost;
its sampling defaults are vLLM's (`top_p=1`, no `top_k`), so `temperature=1.0` matches typical RL sampling. By default
it starts one engine per visible GPU, each a full copy of the model in its own process, and splits the prompts among
them (`data_parallel_size="auto"`; one GPU writes roughly 240 Qwen3-4B thinking samples of up to 16K tokens in 18
minutes, so 8 GPUs do it in a few). Set `CUDA_VISIBLE_DEVICES` to use fewer GPUs, and `tensor_parallel_size` only for
a model too large for one GPU. The engines hold their GPUs until `model.close()` or the process exits, so free them
before training on the same GPUs.

```python
from pyforger import filter
from pyforger.acquire import LocalModel

student = LocalModel("Qwen/Qwen3-4B", temperature=1.0, max_tokens=16384, max_model_len=32768)  # every GPU
measured = filter.by_pass_at_k(rows, model=student, input_field="prompt", metric=my_verifier,
                               k=8, min_rate=0.0, max_rate=1.0, outcomes_field="grades")  # keep all; read the histogram
```

`TinkerModel` samples a base model (e.g. `"Qwen/Qwen3-8B"`) or a trained `tinker://...` sampler checkpoint remotely on
Tinker. It works with every op that takes a `Model` (`sample_from_LLM`, `generate_from_context`,
`self_taught_reasoner`, `filter.by_llm_score`, `filter.by_pass_at_k`, ...); see [models.md](models.md#tinkermodel) for
its request cache, errors, and spend accounting. Prompts are rendered with the model's chat template and stop at its
eos tokens, as a vLLM chat server would; defaults are Qwen3's thinking-mode settings (`temperature=0.6`, `top_p=0.95`,
`top_k=20`), which the evaluator also uses, and `enable_thinking=False` turns thinking off. `sample_from_LLM`,
`generate_from_context`, `self_taught_reasoner`, and `by_pass_at_k` default to `temperature=None`, which keeps the
model's default; pass `temperature=` to override it. The `n` samples of one prompt are a single request, so
they share one prefill: the prompt is billed once and its other `n - 1` copies at the cheaper cached rate. Prompts run
concurrently, at most `TINKER_MAX_CONCURRENT_SAMPLES` (default 600) requests at a time.

`result.usage` reports prompt, completion and cached prompt tokens, Tinker's billed token kinds, each sample's stop
reason (`"length"` means truncated), and `cost_usd` from Tinker's rate card (Qwen3-8B is built in; pass `prices=` for
other models). When `FORGER_TINKER_USAGE_FILE` and `FORGER_TINKER_USD_LIMIT` are set, every request is charged to that
shared dollar ledger, and a request whose worst-case cost would cross the limit raises `TinkerBudgetExceeded` instead
of being sent. The worst cases of requests still in flight in the same process count against the limit, so a request
that only fits once they are charged waits for them. Install with `pip install "pyforger[tinker]"`; the SDK reads
`TINKER_API_KEY`.

```python
from pyforger.acquire import TinkerModel, split_reasoning

student = TinkerModel("Qwen/Qwen3-8B", max_tokens=16384)  # or TinkerModel("tinker://<run>/sampler_weights/<name>")
samples = student.generate("How many primes are there below 100? End with ANSWER: <n>.", n=8)  # one request
thinking, answer = split_reasoning(samples[0])
solved = sum(s.rstrip().endswith("ANSWER: 25") for s in samples) / len(samples)  # pass rate over 8 samples
```

The retrieval functions take one query string or a list of queries as their first argument and return a flat list of
rows, each stamped with its `query`. `mode` is keyword-only (`mode=RetrieveMode.RANDOM` for random sampling where
supported). Web search reads `BRAVE_API_KEY` and Semantic Scholar reads `S2_API_KEY` from the environment unless
`api_key=` is passed; web search raises without a key.

```python
from pyforger.acquire import retrieve_from_web, retrieve_from_wiki

pages = retrieve_from_web(["interval scheduling greedy proof", "monotonic stack patterns"], n=5)
# pages[i] has "text" (extracted page content), "title", "url", and "query".
articles = retrieve_from_wiki("Segment tree", n=2)
```

`n` is per query, so three queries with `n=5` return up to 15 rows. A Semantic Scholar query asks the API for only `n`
results. `retrieve_from_s2` runs one query at a time by default, because an API key is usually limited to 1 request
per second and keyless requests share a throttled pool; a rate-limited request is retried after 5 to 60 seconds, so
workers beyond the limit are slower, not faster. With a higher limit, e.g.
`retrieve_from_s2(queries=queries, n=4, max_workers=3)` runs three queries at a time and keeps rows in query order.

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

Pass `code_criteria_only=True` when every criterion should be checked in code, e.g.
tasks with exact answers, where an LLM judge only adds cost and reward noise. Both
stages then ask only for code-checkable criteria, and criteria that still cannot be
checked faithfully in code are dropped rather than approximated. An input left
without a positive-weight code criterion raises `ValueError`. The same flag exists
on `generate_custom_rubric_criteria` and `convert_rubric_criteria`.

Pass `code_guidelines` when your grading environment differs from the default, e.g.
`grade()` can call a helper that runs the response's program, or that program runs in
a sandbox that rejects some syntax. The text is added to the system prompt of both
stages: the criteria stage keeps code-checked criteria within it, and the conversion
stage follows it over the general rules where they conflict. It cannot widen the
static checks: the import allowlist and banned builtins still apply.

```python
rubrics = generate_custom_rubric(
    tasks,
    model=model,
    code_criteria_only=True,
    code_guidelines=(
        "grade() may call run_program(code, stdin), which returns (True, stdout) or "
        "(False, error type). The program it runs cannot use names starting with '_' "
        "(e.g. __slots__, self._x), annotated assignments, str.format, or open()."
    ),
)
```

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
`batch_size` (default 10) limits prompts per batch, `temperature` defaults to 1.0, and `max_retries`
(default 3) gives additional attempts for malformed outputs, including invalid nested pairs or weights.
Exhaustion raises `ValueError` identifying zero-based input indices; failed inputs are never silently dropped.
Results are cached with `@op`; `_no_cache=True` forces fresh generation.

The output is an intermediate textual rubric. For `rl.agent_training validate`, a later dataset adapter
must wrap each criterion as `{"prompt": criterion_text}` (or convert it to `{"code": source}`),
ensure a nonempty rubric with positive total weight, and supply the required `prompt` and `ground_truth`
fields. The current RL validator requires `ground_truth` to include nonempty `<think>...</think>` reasoning
and a final answer. This building block does not generate those fields or execute criteria.

# `pyforger.acquire`

Generate or retrieve new rows. Signatures and defaults for every export are in [OPS.md](../OPS.md#acquire).

| Export | Cached | Description |
|---|---|---|
| `sample_from_LLM` | yes | Sample data from an LLM with a prompt. Supports schema-based validation and retries. |
| `generate_custom_rubric` | per stage | Recommended: generate mixed code/prompt rubrics ready for the RL executor. |
| `convert_rubric_criteria` | yes | Convert existing textual rubrics, preserving order and weights. |
| `generate_custom_rubric_criteria` | yes | Generate weighted textual rubrics with optional context and candidates. |
| `retrieve` | yes | Generic retrieval against a `RetrievalSource` (`RetrieveMode.QUERY` or `RANDOM`). |
| `retrieve_from_web` / `_wiki` / `_hf` / `_arxiv` / `_s2` / `_pubmed` | yes | Brave, Wikipedia, HF, arXiv, S2, PubMed. |
| `Model` |   | LiteLLM wrapper; the user sets the model id. Generate through `sample_from_LLM`. |
| `LocalModel` |   | On-GPU vLLM model with `Model`'s interface (needs the `vllm` extra). Zero cost, `<think>` inline. |
| `TinkerModel` |   | Base model or `tinker://` checkpoint sampled on Tinker; `Model`'s interface, `<think>` inline. |
| `split_reasoning` |   | `str -> (thinking, answer)`: splits a reasoning-model completion on `</think>`. |

## `sample_from_LLM`

Prefer it over calling `Model` directly: it batches, parallelizes, parses, validates and retries.

- Reasoning returned separately by the provider goes to the `thinking_trace` key. `LocalModel` and `TinkerModel`
  leave `<think>` inline in the text (`thinking_trace` is `None`); use `split_reasoning` to separate it.
- Requests that keep failing with a transient provider error are dropped: the op returns fewer than `n` samples
  with a warning and is not cached.
- A row error (a content-filter block or an over-long prompt; see [alter.md](alter.md)) is not sent again and is
  cached, so a rerun returns the same samples. If every request hit one and no sample arrived, the op raises.

`output_schema` (here and in `alter.generate_from_context`), e.g. `{"question": str, "answer": str}`, asks the model
to end with a JSON object with those keys. Only the keys are checked:

- A response missing a schema key fails to parse and is retried; keys outside the schema are dropped.
- Value types are **not** checked: `{"answer": 42}` passes `{"answer": str}`. Check types or values yourself.
- If you will train on the model's reasoning, don't combine `output_schema` with format instructions in the prompt
  ("answer with one code block"): the reasoning ends up reconciling them.

**Do not train a student on another model's raw `thinking_trace` without reading a few first.** A teacher's trace is
in its own style and length and discusses everything in its prompt (checker rules, output formats, budgets); the
student learns to reason about those too. Prefer the student's own traces (`alter.self_taught_reasoner` with
`LocalModel` or `TinkerModel`), or have a model write a short trace for an already-verified answer from a prompt
that contains only the task.

## Student models: `LocalModel` and `TinkerModel`

Use one for the student's own outputs (STaR traces, pass@k difficulty checks); both work with any op taking a `Model`.

`LocalModel(path_or_name, ...)` runs a Hugging Face model or local checkpoint in vLLM on your GPUs, at no API cost.

- By default it starts one engine (a full model copy) per visible GPU and splits prompts among them. Set
  `CUDA_VISIBLE_DEVICES` to use fewer GPUs; set `tensor_parallel_size` only for a model too large for one GPU.
- The engines hold their GPUs until `model.close()` or process exit: **call `close()` before training on the same
  GPUs.**
- It defaults to `temperature=0.7` with vLLM's `top_p=1` and no `top_k`; pass `temperature=1.0` to match RL sampling.
- Each engine, a single in-process one too, gets its own Triton and torch.compile cache directories (under
  `/dev/shm` when writable, deleted on close) unless you set `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR`, so
  8 engines compiling at once don't collide in `/tmp` (`IsADirectoryError: .../triton_.cubin`).
- The engine core runs in the engine's own process, not a child vLLM forks (its CUDA init fails in some containers:
  `CUDA driver initialization failed`); `VLLM_ENABLE_V1_MULTIPROCESSING=1` keeps vLLM's child.
- `target_logprobs(prompts, targets)` scores given responses instead of sampling; `inspect.student_log_likelihood`
  reports them for a dataset (see [inspect.md](inspect.md)).

```python
from pyforger import filter
from pyforger.acquire import LocalModel

student = LocalModel("Qwen/Qwen3-4B", temperature=1.0, max_tokens=16384, max_model_len=32768)  # every GPU
measured = filter.by_pass_at_k(rows, model=student, input_field="prompt", metric=my_verifier,
                               k=8, min_rate=0.0, max_rate=1.0, outcomes_field="grades")  # keep all; read the histogram
student.close()  # free the GPUs before training
```

`TinkerModel("Qwen/Qwen3-8B")` (or a `tinker://...` sampler checkpoint) samples the student remotely on Tinker, with
Qwen3 thinking-mode defaults (`temperature=0.6`, `top_p=0.95`, `top_k=20`, as the evaluator uses). It is charged in
dollars, not token units: with `FORGER_TINKER_USAGE_FILE` and `FORGER_TINKER_USD_LIMIT` set, a request that could
cross the limit raises `TinkerBudgetExceeded` instead of being sent. Install `pyforger[tinker]`; it reads
`TINKER_API_KEY`. Request cache, retries and concurrency: [models.md](models.md#tinkermodel).

## Retrieval

The `retrieve*` functions take one query string or a list of queries as the **first argument** and return a flat
list of rows, each with its `query`. `n` is per query. `mode=RetrieveMode.RANDOM` (keyword-only) samples randomly
where supported.

- Web search needs `BRAVE_API_KEY` (raises without it); Semantic Scholar reads `S2_API_KEY`.
- PubMed reads `NCBI_API_KEY` if set (10 requests/second, else 3); Forger spaces requests and retries rate-limited
  ones. It is QUERY only and returns abstracts, not full text.
- `retrieve_from_s2` runs one query at a time (keys are usually limited to 1 request/second); raise `max_workers`
  only if your key allows more.

```python
from pyforger.acquire import retrieve_from_pubmed, retrieve_from_web, retrieve_from_wiki

pages = retrieve_from_web(["interval scheduling greedy proof", "monotonic stack patterns"], n=5)
# pages[i] has "text" (extracted page content), "title", "url", and "query".
articles = retrieve_from_wiki("Segment tree", n=2)
# articles[i]["text"] is the intro; full_text=True returns whole articles, one request each.
reports = retrieve_from_pubmed("lung cancer case reports", n=20)
# reports[i] has "text" (title and abstract), "publication_types" (e.g. ["Case Reports", ...]), "mesh_terms",
# "authors", "journal", "year", "doi", "pmcid" and "url". PubMed search syntax works too:
# '"lung neoplasms"[MeSH] AND case reports[pt]'.
```

## Rubrics

`generate_custom_rubric(tasks, model=model)` is the recommended entry point. It turns deterministic requirements
(exact counts, JSON structure) into `grade(prompt, response, ground_truth)` code and keeps subjective ones as prompted
judges. The output is accepted by `rl.rubric_data.validate_rubric`; it
is a rubric only, so you still supply each row's `prompt` and `ground_truth`.

- `code_criteria_only=True`: every criterion is checked in code (use for exact-answer tasks, where an LLM judge only
  adds cost and reward noise). Criteria that can't be coded faithfully are dropped; an input left with none raises.
- `code_guidelines="..."`: describe your grading environment (helpers `grade()` may call, syntax your sandbox
  rejects). Both stages follow it; it cannot widen the import allowlist or banned builtins.
- Generated graders are compiled and statically checked but **never executed**: test them on known passing and
  failing responses before training on them.
- A mistake is a presence criterion with a negative weight, e.g. `["Does the response recommend aspirin for a
  child?", -75]`, never an absence one (`["Does the response avoid recommending aspirin for a child?", 75]`): LLM
  judges dock replies that name aspirin only to warn against it. "X without Y" criteria ("Recommends opioids without
  asking about pain history") hide an absence too, so the prompt asks to split them. A response scores
  `sum(w * s) / sum(positive w)`, clipped to `[0, 1]`: `[(100, 1.0), (50, 0.0), (-75, 1.0)]` gives
  `(100 - 75) / 150 = 0.17`. A negative code criterion's `grade` returns 1.0 when the response makes the mistake.
  `negative_criteria=False` asks for requirements only (weights in `[0, 100]`), for a scorer that divides by the sum
  of all weights.

```python
from pyforger.acquire import generate_custom_rubric

rubrics = generate_custom_rubric(
    tasks,
    model=model,
    code_criteria_only=True,
    code_guidelines="grade() may call run_program(code, stdin), which returns (True, stdout) or (False, error).",
)
# rubrics[i] is [[{"code": "def grade(prompt, response, ground_truth): ..."}, 100.0], ...]
```

`prompt_criterion_options` merges your judge's per-criterion options into every `{"prompt": ...}` criterion (code
criteria are unchanged; it may not set `prompt` or `code`), so you don't edit the rows afterwards. It is applied
after the cached conversion: changing it makes no model calls. The conversion treats prompted criteria as yes/no
questions; a judge with a scale option can get one here, e.g. `{"scale": "likert"}` for a 1-to-5 rating. `{"n": 3}`
averages three judge samples per criterion (default 1), so binary samples can have a fractional mean.

```python
rubrics = generate_custom_rubric(tasks, model=model, prompt_criterion_options={"answer_only": True})
# [[{"code": "def grade(...): ..."}, 100.0], [{"prompt": "Is the explanation correct?", "answer_only": True}, 75.0]]
```

To run the stages separately, `generate_custom_rubric_criteria(inputs, model=...)` returns textual
`[(criterion, weight), ...]` per input (weights in `[-100, 100]`, not summing to 100, at least one positive),
e.g. for creative writing.
Optional `meta_description` adds shared context, and `candidate_responses` help find failure modes. Then
`convert_rubric_criteria(text_rubrics, model, inputs=tasks)` turns them into code/prompt criteria (it takes
`prompt_criterion_options` too). Textual rubrics are not valid RL rubrics until converted or wrapped as
`{"prompt": text}`. The rubric ops' `batch_size` (default 10) chunks prompts only for a `LocalModel`; a `Model` or
`TinkerModel` sends every prompt at once, up to its `max_concurrency`.

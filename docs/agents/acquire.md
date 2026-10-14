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
| `retrieve_from_s2` | yes | Semantic Scholar backend. |
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

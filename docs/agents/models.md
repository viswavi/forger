# Models, prompts, and credentials

## The `Model` class

`acquire.Model` wraps LiteLLM; every LLM op takes one. Full signature: [OPS.md](../OPS.md#acquire-model).

```python
import os
from pyforger.acquire import Model

model = Model(
    os.environ.get("MODEL_ID"),
    api_key=os.environ.get("LITELLM_API_KEY"),  # read keys from the environment; never print them
    max_concurrency=16,
    reasoning_effort="low",
)
```

- **Credentials:** for `litellm_proxy/` ids, `api_key` defaults to `LITELLM_API_KEY` and `api_base` to
  `LITELLM_BASE_URL`; other ids fall back to provider variables (`OPENAI_API_KEY`, ...).
- **Temperature:** the default is 1.0, which reasoning models such as DeepSeek-V4 need; at temperature 0 they can
  loop until the request times out.
- **Truncation:** check `max_tokens`. A completion that hits it has finish reason `"length"`; step reports warn about
  these rows.

### Reasoning effort

Set `Model(..., reasoning_effort="low")` (or `"none"`, `"medium"`, `"high"`). Without it, DeepSeek-V4-Flash on the
gateway does not think at all. Effort controls how long the reasoning trace is, so it drives cost and the chance of
hitting `max_tokens`. `Model` sends it in the form each model reads, overriding any reasoning setting in
`**kwargs` or `extra_body`.

### Concurrency

`max_concurrency` caps the requests one `Model` has in flight, across all ops and threads (each `n` copy counts).
`None` allows up to 100 per `generate_batch` call. Ops such as `generate_from_context` and `sample_from_LLM` send
all their prompts in one batch, so set it to your provider's rate limit; never write your own thread pool around
ops. It is not part of the cache key.

### Retries, timeouts, and rate limits

- Requests are retried `num_retries` times (default 2) only on transient errors (timeout, lost connection, 429,
  5xx). A 400 (over-long prompt, content filter: a row error for ops), 401, 403, 404, or exhausted quota, credit,
  budget, or spending cap raises at once. Ops never re-send a request that failed after these retries.
- A 429 gets at least 6 retries over 2 to 3 minutes and halves that model's in-flight requests process-wide.
  If a step still reports rate-limit failures, lower `max_concurrency` or run fewer pipelines at once.
- Each attempt times out after `timeout` seconds: `"auto"` is 600, or `max_tokens / 25` if longer. A request that
  hangs every attempt waits up to `timeout × (num_retries + 1)` (30 minutes by default).
- A `1 request in flight for 9m 12s` warning means a slow or stuck request. If common, lower `timeout`, or
  `max_tokens` if those are runaway generations.
- Spending caps and the token-unit quota are checked before every request and retry
  (see [pipelines.md](pipelines.md)). Ctrl-C cancels requests not yet sent.
- `model.generate_batch(prompts, return_exceptions=True)` keeps finished results: a failed prompt's slot holds its
  exception. The default raises the first failure.

Progress output is on by default for every model call, `LocalModel` (one count across its GPU engines) and
embeddings included: a bar in a terminal, else `<model>: 120/500 done, 1m 31s, ~4m 48s left` every 30 seconds.
Turn it off with `show_progress=False` or `FORGER_PROGRESS=0`.

### Results and usage

`model.generate(prompt, n=...)` returns a `GenerateResult`, a `list` of completion strings (`result[0]`, or
`result.texts` for a plain list). It also has `.usage`, `.reasoning`, `.finish_reasons` (`"length"` at `max_tokens`),
`.truncated` (how many hit it), `.model_id`, and `.raw`. `generate_batch` returns one per prompt.

`result.usage["token_units"]` and `["cost_usd"]` are per call; `model.total_token_units`, `model.total_cost`, and
`model.total_usage` are cumulative. Unknown model weights give `NaN` units (see [AGENTS.md](../AGENTS.md#cost)).

### `n` fan-out

`generate_batch(prompts, n=k)` returns exactly `k` completions per prompt. Claude, Grok, Gemini, o1/o3/o4, and gpt-5
(with any prefix) do not support `n`, so they get `k` concurrent `n=1` requests, merged with summed usage. Other
models get `n=k` (in requests of at most 16). A gateway alias that silently ignores `n` is detected, warned about
once, and fanned out from then on. It never pads: if completions still come back short, it raises `ValueError`;
ask for `n=1` per call instead.

## `TinkerModel`

`acquire.TinkerModel` samples a base model or a trained `tinker://` checkpoint with `Model`'s interface; see
[acquire.md](acquire.md) for its defaults and dollar budget. Differences from `Model`:

- `n` is native: one request per prompt, never fanned out.
- Concurrency: all instances share `TINKER_MAX_CONCURRENT_SAMPLES` (default 600) in flight; `max_concurrent` caps one
  instance lower. `max_concurrency` and `timeout` do not apply.
- The request cache keys on the checkpoint and sampling settings, so a new checkpoint never reuses old samples;
  turn it off with `request_cache=False` or `FORGER_REQUEST_CACHE=0`.
- Retries are not bounded by `num_retries + 1`: the Tinker SDK retries until no request progresses for 2 hours, so a
  stuck request can wait `2 h × (num_retries + 1)`.
- A prompt that does not fit the context window raises `pyforger.PromptTooLongError` before any request is sent (or
  fills its slot with `return_exceptions=True`); ops treat it as a row error.
- Spend is in Tinker dollars on a separate ledger (`FORGER_TINKER_USAGE_FILE`), adds no token units, and is exempt
  from the token-unit quota. `max_cost_usd` / `FORGER_MAX_COST_USD` are checked before each request.

## Prompt types

A `Prompt` is a string (sent as one user message) or a list of chat messages
(`[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`); use the list for a system prompt or
multi-turn input. Ops that prompt per row (e.g. `generate_from_context`) take a `PromptTemplate`: a string filled
with `str.format_map(row)` (`{field_name}` placeholders), or a callable that takes the row dict and returns a
`Prompt`.

# Models, prompts, and credentials

## The `Model` class

```python
import os
from pyforger.acquire import Model

model = Model(
    os.environ.get("MODEL_ID"),
    api_key=os.environ.get("LITELLM_API_KEY"),
    temperature=1.0,
    max_tokens=None,
    num_retries=2,
    max_concurrency=16,
)
```

For gateway models (ids starting with `litellm_proxy/`), `api_key` defaults to `LITELLM_API_KEY` and `api_base` to
`LITELLM_BASE_URL`. The default temperature is 1.0, which reasoning models such as DeepSeek-V4 need; at temperature
0 they can loop until the request times out.

`max_concurrency` caps how many requests this `Model` has in flight at once, across every op, `generate_batch` call,
and thread that uses it (each `n` fan-out copy counts). `generate_from_context`, `sample_from_LLM`,
`self_taught_reasoner`, `translate.*`, and `topic_based_clustering` send all their prompts in one `generate_batch` call,
so set it to what your provider's rate limit allows; you should never need to write your own thread pool around ops.
`None` (default) allows up to 100 requests per `generate_batch` call. It is not part of the cache key, so changing it
keeps cached results. Ctrl-C (or any error in the calling thread) cancels the requests not yet sent; those in flight
finish. Each request's tokens, cost, and token units reach the step report, the spend ledger, the model's totals
(`total_cost`, `total_usage`, `total_token_units`), and the token-unit quota as soon as it finishes, also when the
call later raises or is interrupted; `model.history` gets the merged results only when the call returns. The spending
caps and the token-unit quota are checked before each request is sent, so requests queued behind them are not sent
(see [pipelines.md](pipelines.md)).

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
`Retry-After` header instead, up to 60 s), and logs each retry at INFO. A 400 (e.g. an over-long prompt), 401, 403,
404, or an exhausted quota, credit balance, or budget (OpenAI's `insufficient_quota`, LiteLLM's or Forger's budget
errors, a spending cap, the token-unit quota) raises at once; a per-minute rate limit is retried, also Gemini's "You
exceeded your current quota" 429. The spending caps and the token-unit quota are checked again before each retry.
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

Importing `pyforger.acquire` sets `litellm.suppress_debug_info = True`, so provider errors no longer print LiteLLM's
"Give Feedback / Get Help" banner; the raised error is unchanged. Set it back to `False` to see the banner.

`model.generate_batch(prompts, return_exceptions=True)` keeps finished results when a request still fails after
`num_retries` (e.g. a gateway 504): the list has one entry per prompt, a prompt whose requests all failed holds the
exception, and a prompt where only some fan-out copies failed holds the completions that arrived (fewer than `n`)
when those failures were transient. A spending cap, the token-unit quota, or a bad request puts its exception in the
prompt's slot even so (what arrived is charged and request-cached, so a rerun replays it).
Usage of finished requests is recorded. The default (`False`) raises the first failure. On `LocalModel` an engine
error fills every slot, since vLLM runs the batch as one call.

Use `Model(..., reasoning_effort="low")` (or `"none"`, `"high"`) to set reasoning effort. `Model` sends it in the
form each model reads: DeepSeek-V4 as `reasoning={"effort": "low"}` (the gateway ignores `reasoning_effort` for it),
GLM-5.3 and other models as `reasoning_effort`. It overrides any reasoning setting passed through `**kwargs` or
`extra_body`. Without it, DeepSeek-V4-Flash on the gateway does not think at all.
`dynamic_step` forwards the reasoning effort and `extra_body` to OpenHands.

`model.generate_batch(prompts, n=k)` returns exactly `k` completions per prompt. Claude, Grok, Gemini, o1/o3/o4, and
gpt-5 models (also behind `litellm_proxy/`, `openrouter/`, or provider prefixes), and models whose LiteLLM supported
params lack `n`, get `k` concurrent `n=1` requests per prompt, merged into one `GenerateResult` with summed usage and
cost (`raw` is then the list of responses). Other models get one request with `n=k`, or for `k > 16` concurrent
requests of at most 16, merged the same way (`n=40` sends 14, 13, 13). If such a model returns fewer completions
(e.g. a gateway alias that ignores `n`), it logs one warning naming the model, sends only the missing completions as
concurrent `n=1` requests, and fans out that model id for the rest of the process. It never pads: if the `n=1`
requests come back short too, or a response has more than `n`, it raises a `ValueError` (add the model to
`_NO_N_BRANDS` / `_NO_N_PREFIXES` in `model.py`).

## `TinkerModel`

`acquire.TinkerModel` samples a base model or a trained `tinker://` checkpoint on Tinker with `Model`'s interface; see
[acquire.md](acquire.md) for its sampling defaults, usage, and dollar budget.

`TinkerModel.generate_batch` follows `Model`'s contract, with these differences:

- `n` is native: the `n` samples of a prompt are one request (one prefill), never fanned out.
- Concurrency: all `TinkerModel`s in a process share one cap of `TINKER_MAX_CONCURRENT_SAMPLES` (default 600)
  requests in flight; `max_concurrent` caps one instance lower. `max_concurrency` does not apply.
- The request cache works as for `Model`: inside an op, each sample is keyed on the op call, the row, the messages,
  the checkpoint (`model_id`), the sampling settings (`temperature`, `max_tokens`, `top_p`, `top_k`, `seed`, stop
  strings) and the chat-template kwargs. A rerun asks Tinker only for the missing samples, still one request per
  prompt; when every sample is cached, no Tinker client is opened. A new `tinker://` checkpoint is another model id,
  so it never reuses an old checkpoint's samples. Turn it off with `TinkerModel(..., request_cache=False)` or
  `FORGER_REQUEST_CACHE=0`.
- Retries: each request is retried on Tinker's transient errors (`APIConnectionError`, `RateLimitError`,
  `InternalServerError`) up to `num_retries` times (default 2), after 1 s, 2 s, ..., so it makes at most
  `num_retries + 1` attempts; the budgets and spending caps are checked before each attempt and never retried. There
  is no per-attempt timeout and no in-flight warning: `Model(timeout=)` does not apply.
- `return_exceptions=True` puts a failed prompt's exception (after Tinker's retries, or a budget refusal) in its slot
  and keeps the others; the default raises the first failure in prompt order. Ctrl-C also stops requests waiting for
  a slot, for budget room, or for a retry.
- Spend: each request is charged as it finishes to the Tinker dollar ledger (`FORGER_TINKER_USAGE_FILE`), the
  instance totals, the step report, and the spend ledger, in dollars from Tinker's rate card. It adds no token units
  and is exempt from the token-unit quota. `max_cost_usd` / `FORGER_MAX_COST_USD` are checked before each request,
  next to the worst-case reservations on the Tinker ledger and on `budget=`, which count the requests in flight. If
  writing a ledger fails after Tinker sampled, the spend still reaches the totals and the samples the request
  cache, then the error is raised.

## Prompt types

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

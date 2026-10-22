# Caching and tracing with `@op`

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

Your own helper functions that an op or step calls by name are keyed by their code and the globals they read, so
editing `tag` in `rows = [tag(r) for r in rows]` recomputes the step. This covers functions in your pipeline file and
in your other files, imported by name (`from helpers import tag`) or used through their module (`helpers.tag(r)`,
`helpers.PROMPT`), and `@op` helpers by the function they wrap. Functions of the standard library, installed
packages, and Forger are keyed by name.

Not in the key: classes and their methods (only their names), files read from disk, environment variables, other
objects held in globals (e.g. a client, a retrieval source, or an object mutated at runtime), and closure values
changed after decoration. `@op` logs a warning at decoration for such a global (loggers and typing objects excepted)
or for `os.environ`. When one of these changes, pass `_no_cache=True`. Upgrading to this key layout makes existing
cache entries miss once.

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

Built-in retrieval sources provide stable configuration keys that exclude runtime clients and sessions.
Custom `RetrievalSource` implementations with mutable state should define `cache_key()` using their
source type and result-affecting configuration, excluding counters and other runtime state.

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
wall-clock timeouts (more workers, more contention, more timeouts), so Forger's own worker counts are keyed. The
exception is `retrieve_from_s2(max_workers=)`: rows keep query order and a rate-limited request is retried rather than
dropped, so workers change only the speed.

**Trace.** Import the trace helpers from the module:

```python
from pyforger.op import trace, clear_trace
```

`trace()` returns a list of `CallRecord(op_name, input_hash, output_hash, cache_hit, duration_s)` for every `@op` call
in the current process up to a certain maximum (default 10000). `clear_trace()` resets it.

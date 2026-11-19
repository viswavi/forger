# Caching and tracing with `@op`

Every `@op` result is cached on disk, keyed on the op's code and arguments, so rerunning an unchanged op or step is
free; `@step`s and Forger's LLM ops are ops too. The cache lives in `$FORGER_CACHE` (default `~/.cache/forger`);
deleting it clears everything.

```python
from pyforger import op

@op
def expensive_step(rows, *, model, temperature=0.7):
    ...
```

**Iterate with new versions, not patches.** Keep each pipeline version self-contained, running from its inputs, and
rely on the cache rather than loading or editing the previous version's output. The module name is not in the key,
so a step copied unchanged into `pipeline_2.py` reuses `pipeline_1.py`'s results.

**Request cache.** Below `@op`, every completion a model returns inside an op is stored per row, so a changed
pipeline pays only for new requests: changing one row re-sends only that row's requests, a killed run resumes,
raising `n` requests only the extra samples, and the full run reuses the completions a `FORGER_LIMIT=3` run got.
Two different ops never share samples. Each op logs
`request cache: 57 of 64 completions reused in alter.generate_from_context`; "0 of N" on a rerun means a key changed
(e.g. a row holding `id(row)` or a timestamp). Not request-cached: direct `model.generate` calls outside an `@op` /
`@step`, `LocalModel`, and requests with tools or logprobs. Your own `@op` that builds one prompt per row should pass
`cache_rows=rows` to `generate_batch`.
Row errors (a content-filter block, a prompt over the context window) are stored too: a rerun replays them as
`ReplayedRowError` with the same reason instead of asking again, so it gives the same rows (`_no_cache=True` asks
again). A reply the filter cut off (`finish_reason="content_filter"`) is cached as it came. Other errors and empty
replies are never cached.
A completion is keyed on the exact prompt, so editing a template re-sends every row. When at least 20 rows and half of
a batch were answered before by the same op in the same `@step` under another prompt or settings, the batch warns
before sending, with the earlier cost (`249 of 498 rows were answered before under a different prompt or settings and
will be asked again ...; ~12.45M token units at the earlier cost per row`); it never blocks.

**Fresh samples and cache-only reads.**

- The same call replays the same samples, including `sample_from_LLM` and retrieval in RANDOM mode even without a
  seed. For a fresh sample pass `_no_cache=True` (it also covers the ops that call makes), change an argument such as
  the seed, or turn the request cache off with `Model(..., request_cache=False)` or `FORGER_REQUEST_CACHE=0`.
- `sample_from_LLM` pools that differ only in `n` overlap; pass another `name` for an independent pool.
- `_cache_only=True` returns an existing result without running anything; a miss raises `FileNotFoundError`.

**Stale results: changes the key does not see.** The key covers the function body, your own helper functions (in any
of your files), globals that are plain data (a `PROMPT` constant, a config dict) or define `cache_key()` (a `Model`),
and the arguments, callables included. It does not cover:

- classes and their methods, or functions from installed packages (keyed by name only);
- files read from disk and environment variables;
- other objects held in globals (a client, a retrieval source, an object mutated at runtime), and closure values
  changed after decoration;
- docstrings, so a function that reads its own `__doc__` as a prompt goes stale (use a constant).

`@op` warns at decoration about such globals, closures, and `os.environ`. When one of them changes, pass
`_no_cache=True`. An argument object with changing internal state (or a custom `RetrievalSource`) should define
`cache_key()` returning only its result-affecting configuration.

**Misses on every run.**

- Built-in `hash()` of a `str` changes in every process, so rows built from it miss every time; use
  `zlib.crc32(text.encode())`. `@op` and `@step` flag it.
- `@op(unkeyed=("show_progress",))` leaves out arguments that cannot change the result; a worker count can, when
  the op has timeouts.

**Other tools.**

- `pyforger.op.skip_cache_write()`, called inside an op, keeps its result (and every enclosing op's and step's) out
  of the cache, e.g. one holding `None`s for requests that failed transiently. From a worker thread, submit with
  `contextvars.copy_context().run`, or every op and step running at that moment is left uncached. A step kept out of
  the cache still saves its output (`<key>.uncached.pkl`), never replayed, so `inspect_step(name, rows=True)` can
  read it.
- Entries are pickles: only use a cache directory you trust (files owned by another user are refused).
- `from pyforger.op import trace, clear_trace`: `trace()` lists a `CallRecord` (op name, input and output hashes,
  `cache_hit`, `duration_s`) for each recent `@op` call in the process (up to 10,000); `clear_trace()` resets it.

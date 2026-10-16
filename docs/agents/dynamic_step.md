# Dynamic input-specific pipelines

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

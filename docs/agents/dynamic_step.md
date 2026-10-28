# Dynamic input-specific pipelines

`alter.dynamic_step(inputs, goal, model)` builds a small sub-pipeline per input row. For each unique input, an
OpenHands agent writes and debugs a Python function (one dict in, `list[str]` out) against that input, then runs it.
It returns one string list per input, in input order; a list may be empty. Reach for it when rows need different
extraction or parsing logic (e.g. heterogeneous pages or repositories) that a single prompt or regex handles badly.
Signatures and defaults: [`OPS.md#alter-dynamic_step`](../OPS.md#alter-dynamic_step) and
[`OPS.md#alter-dynamic_step_source`](../OPS.md#alter-dynamic_step_source).

```python
outputs = alter.dynamic_step(
    [{"text": "Alice <alice@example.org>"}, {"text": "Bob <bob@example.net>"}],
    "Return a list containing the email address's domain", model, max_parallel=2,
)
# [["example.org"], ["example.net"]]
```

- Called directly in a `@pipeline`, it counts as a step: it prints a step report, and
  `inspect_step("dynamic_step")` shows it.
- Needs Python 3.12+ and Unix signal timers; the default install includes the OpenHands SDK.
- Use a fast agent-friendly model (e.g. DeepSeek-V4-Flash) to avoid timeouts. `max_parallel`, `max_iterations`, and
  `execution_timeout_s` bound sessions, agent steps, and each code run.
- Generated source is `@op`-cached per input, but the code re-executes on every call. Pass a new `source_version` to
  regenerate stale source. `dynamic_step_source(..., _cache_only=True)` or `examples/show_dynamic_step_source.py`
  shows cached programs without launching an agent.
- Simultaneous calls for the same uncached input in separate batches may each start an agent (and pay for it).
- Failures raise; successful source for other inputs is kept.
- Generated code may import only `json`, `re`, `math`, and `html`.
- A URL-only row (`{"url": ...}`) is fetched before the code runs: HTTPS HTML/PDF become `content`, GitHub/GitLab
  `.git` URLs become a bounded `files` snapshot (200 files, 1 MiB). It is re-fetched on every execution. Rows with
  other fields (e.g. `text` and `url`) pass through unfetched.
- PDFs: text from up to 100 pages; image-only scans return nothing (no OCR).
- Not a sandbox: RestrictedPython guards against accidents, not adversarial code. OpenHands runs on the host and may
  load SDK plugins, so use an isolated account for untrusted inputs. Cache directories must be trusted.

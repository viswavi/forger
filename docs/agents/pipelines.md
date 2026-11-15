# Pipelines and steps

## Steps and step reports

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
  a step is called inside another step, a pipeline calls no steps, or rows do not match a `list[X]` type hint where
  `X` is a pydantic model (rows stay dicts).
- `WARNING:` lines are data problems: something happened to this run's rows, and the line says what. Examples: an op
  misses its `min_yield` / `max_yield`; model calls stopped at `max_tokens` ("20 rows dropped: 34 of 120 completions
  ... hit max_tokens=16384; raise max_tokens or lower reasoning_effort"); samples failed to parse; requests failed at
  the provider; a judge gave no score; reasoning traces open by discussing the output format (`output_schema` and the
  prompt's own format instructions disagree, so the reasoning is a poor training target); retrieved text reached no
  prompt (below). In a limited run, a `min_yield` / `max_yield` miss on fewer than 20 rows is an `INFO:` line instead
  (`limited run: ...`), since 3 rows say little about yield. The pipeline summary ends with its `INFO:` lines, then
  every step's `WARNING:` lines, so the last lines of a log show them; a limited run's banner comes after them. Fix
  each one; a few can be suppressed with the reason they do not matter for your data (below).
- `INFO:` lines (dim) are about how the code is written, not the rows: a step calls no Forger ops
  (its report cannot show what it kept or changed: write its checks as ops such as `filter.by_predicate`, and never
  add a no-op filter only to quiet it); a step's type hints cannot be resolved; `show=` names a field no output row
  has; a pipeline has only one step; no quality filter, no diversity filter, or no inspect op ran anywhere; an op was
  called in the pipeline outside a step.
- Suppressing a warning: a few problems may not matter for your data. For those, and only those, set a flag whose
  value says why these rows are still fine:

  ```python
  @step(suppress_truncation_warning="<why these rows are still fine>")
  def draft(rows, model): ...

  @pipeline(suppress_no_diversity_filter_warning="<why these rows are still fine>")
  def main(model): ...
  ```

  Step flags: `suppress_truncation_warning` (completions cut off at `max_tokens`) and `suppress_no_ops_warning` (the
  step called no Forger ops). Pipeline flags: `suppress_no_quality_filter_warning`,
  `suppress_no_diversity_filter_warning`, `suppress_no_inspect_warning`, `suppress_retrieval_warning` (retrieved
  text reached no prompt), and `suppress_low_yield_warning` (under 10% of the first step's rows reached the output;
  10-30% is a note, and a limited run adds what every 100 rows give). Nothing else can be suppressed: parse
  failures, provider errors, and judge calls with no
  score lose rows a fix gets back, and a yield miss means `min_yield` / `max_yield` should say the yield you expect.
  A step's problem is suppressed on that step, never pipeline-wide: truncation in a draft step that is regenerated
  later may not matter, while truncation in the answer step does. A suppressed problem is not printed; the step's or
  pipeline's record keeps it, with its size and your reason, in `.suppressed` (e.g. `inspect_step("draft").suppressed`).
  Suppressed problems do not count for `strict=True`.
  Suppress only what you checked in the rows; when a fix exists (a larger `max_tokens`, a filter), make the fix
  instead, and do not set a flag next to a fix "in case". A misspelled flag raises `TypeError` (with the closest
  flag); a reason under 3 words, or a placeholder or example reason, raises `ValueError`. A flag whose problem never
  fired in any run of that step (or the pipeline) is reported so it cannot hide a later problem: as a warning for
  `suppress_truncation_warning` and `suppress_retrieval_warning` (`suppress_truncation_warning on draft suppressed
  nothing (no truncation occurred); remove it`), as an `INFO:` line for the others; limited runs skip this check. A line
  whose problem has a flag ends with it in brackets, e.g. `INFO: no inspect op ran
  [@pipeline(suppress_no_inspect_warning=...)]`.
- Retrieved text that reaches no prompt: inside a pipeline, each `acquire.retrieve*` call fingerprints the documents it
  returns (three 60-character pieces of each document's `text`, or its longest string; documents under 200 characters
  are skipped, and at most 2,000 per run are fingerprinted). Every `Model` / `LocalModel` / `TinkerModel` prompt is
  checked for them, ignoring whitespace, also one answered from the request cache, and so are the string values of the
  rows the pipeline returns. If none appears anywhere, the pipeline warns: `acquire.retrieve_from_web: none of the 223
  retrieved documents were used in any of the 1,840 prompts or the output; put their text in your prompts`. A field
  that carries the documents but that no prompt formats does not count, unless the pipeline returns the rows that hold
  it. When fewer than 10% of the documents were used, an `INFO:` line says so instead: `acquire.retrieve_from_web: only
  3 of 77 retrieved documents (4%) were used in a prompt or the output`. Cached ops and steps replay what their first
  run saw, so a cached rerun gives the same verdict. A result cached outside a pipeline, or by an older Forger, has
  unknown prompts, so the check says nothing for that run rather than warn falsely. Direct `source.fetch()` calls are
  not tracked.
- These checks use each op's category, not its name. Ops under `pyforger.acquire/alter/filter/inspect` have their
  package's category; your own `@op` is `"other"` (even `@op(name="filter.x")`) unless it declares one, e.g.
  `@op(category="filter")` for a custom quality filter.
- Filter notes: the pipeline summary lists the steps that ran no filter and the filters that removed nothing (op, step,
  and rows seen). These are notes, not warnings, and `strict=True` does not raise on them: a generate-only step
  followed by a filter step is fine, and a filter that removes nothing may be a check that passed or one that tests
  nothing. Do not add a filter only to quiet them.
- `strict=True` on `@step` or `@pipeline` turns `WARNING:` and `INFO:` lines into errors, also on cached runs; a step that
  fails strict mode is not cached. `prod=True` or the environment variable `FORGER_STEP_REPORT=0` silences the reports.
  `pyforger.inspect.inspect_step(name)` re-prints the full report of the latest run of a step, with up to 10 sample
  rows. It also works from another process (e.g. a shell, while the pipeline runs under `nohup`): every step saves its
  report under `$FORGER_CACHE/step_reports/`. `inspect_step(name, rows=True)` returns that run's whole output from
  the cache, e.g. the rows of a step that finished before a later step failed. A run kept out of the cache (e.g. some
  requests timed out) saves its rows for this too, and loading them warns that they may be partial; a rerun still runs
  the step. A `cache=False` step saves nothing.
- Sample rows: `@step(sample=3)` shows three output rows (of each stream) instead of one. `@step(show=["statement"])`
  prints those fields of each sample row in full, line breaks kept, as an indented block under the row (other values
  as JSON); the other fields stay in the table, cut or summarized.
- Long sample values are summarized by shape rather than cut, e.g. `[5 items] a.py, b.py, c.py, …` or
  `def solve(n): … (312 lines, 9.8k chars, python)`. By default these structural summaries are all you get; they are
  free. For a one-sentence LLM summary of each long value as well, call `pyforger.steps.set_report_model(model)` or
  set `FORGER_REPORT_MODEL=<litellm model id>` (key from `LITELLM_API_KEY`). The sentence is an extra `LLM:` line
  under the value's own text and says what the value says (its task, method, numbers), not whether it is good; check
  the value itself (`show=` or `inspect_step`) before trusting it. That costs up to 8 LLM calls per printed
  report, cached on disk and left out of the step and pipeline totals (the spend ledger and `FORGER_MAX_COST_USD`
  still count them); silenced reports make no calls. A cheap model without reasoning is enough. Without a report
  model, the first report with a long value logs (at INFO) how to set one; `set_report_model(None)` turns summaries
  off and silences that.
- Threads: ops called from `ThreadPoolExecutor` workers are not itemized in the step report (the step gets an `INFO:` line
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
- Spend ledger: every usage record (`Model`, `TinkerModel`, `LocalModel`, `dynamic_step` agents) is appended and
  flushed as it is recorded to `$FORGER_SPEND_LEDGER`, else `$FORGER_CACHE/spend.jsonl` (one JSON line: timestamp,
  pid, run id, pipeline, step, op, model id, tokens, cost, token units), so killed runs keep their record. `Model`
  and `TinkerModel` record each request as it finishes, also when the batch later raises; `LocalModel` records a
  batch when it returns. `FORGER_SPEND_LEDGER=0` turns the ledger off.
  `pyforger.inspect.spend(since="2026-10-16")` prints and returns the totals by run, pipeline, and step.
- Token units: for models with fixed token-unit weights (DeepSeek-V4-Flash, GLM-5.3; see the Cost section of
  [AGENTS.md](../AGENTS.md)), each response's units reach its op, step, and pipeline as it arrives, and the reports
  add a `units` column. Other models (including `TinkerModel` and `LocalModel` students) add no units; their ledger
  lines have `"token_units": null`. A request-cache hit or a cached step adds none.
- Spending caps: `@pipeline(max_cost_usd=20)` caps one pipeline run in dollars and `@pipeline(max_token_units=2e6)`
  in token units; `FORGER_MAX_COST_USD=40` and `FORGER_MAX_TOKEN_UNITS=5e6` cap all spend in the process
  (unattributed calls included). All must be numbers >= 0. Gateway models without LiteLLM pricing add $0, so cap
  them in units. Once spend reaches a cap, the next model request raises `pyforger.BudgetExceededError`: `Model` and
  `TinkerModel` check before each request (after it gets a concurrency slot), `LocalModel` before each batch, and
  `dynamic_step` before starting an agent and during its run. Requests already in flight finish and are recorded, so
  spend can pass a cap by what they cost (or by one agent step). These caps are separate from the shared quotas,
  which also apply: the token-unit quota (`FORGER_TOKEN_USAGE_FILE` / `FORGER_TOKEN_UNIT_LIMIT`, which raises
  `TokenUnitQuotaExceeded`; `dynamic_step` agents are charged to it and checked against it like `Model` requests)
  and `TinkerModel`'s dollar ledger (`FORGER_TINKER_USAGE_FILE`, which raises
  `TinkerBudgetExceeded`). The caps count this process's spend; the quotas count every process sharing the file.
- Spend projection: a batch of model requests that would not fit stops early instead of spending what is left. Once
  5% of a batch's completions have finished (at least 10 requests; batches under 40 completions are skipped), every
  new request checks that the rest fits what is left under each cap above and the token-unit quota, even at the 95%
  lower bound of the cost per completion so far. If not, it raises `pyforger.SpendProjectionError` (a
  `BudgetExceededError`), e.g. "alter.generate_from_context in step solve: the first 160 of 3,200 completions cost
  1,210,000 token units; the other 3,040 would cost about 22,990,000 token units (at least 20,100,000), but the
  token-unit quota (FORGER_TOKEN_UNIT_LIMIT) has 18,000,000 token units left". A batch of several prompts is sent in
  a fixed shuffled order (results keep the input order), so rows sorted hardest-first do not make the first requests
  look typical. Finished requests are in the request cache, so after making the op cheaper (fewer samples or judges,
  a cheaper model, shorter outputs, fewer rows) the rerun does not pay for them again. `FORGER_PROJECT_SPEND=0`
  turns the check off.
- Cost estimates: every uncached batch of 40+ model completions (an op sends all its prompts as one batch) logs
  one line once 5% of it (at least 10 requests) has finished, if the whole batch will likely cost $1 or more (1M
  token units for a model without prices), e.g. "Cost estimate: filter.by_topic_frequency in step balance: the
  first 102 of 2,040 completions cost $0.21, so all 2,040 will likely cost about $4.20 (at least $3.80), 9,800,000
  token units; the token-unit quota has 130,000,000 token units left." Rows answered from the request cache are not
  sent and not counted, and a cached op sends nothing. The estimate assumes the rest of the batch costs what its
  first completions did; the batch is sent in shuffled order so they are a fair sample. It is logged also with
  `FORGER_PROJECT_SPEND=0`, which only stops the spend projection from raising.
- Live spend: while a step runs, it logs a line each time its spend first reaches $1, $2, $5, $10, $20, $50, ...,
  e.g. "Spend so far: step balance $5.03 (2,104,511 token units, 812 calls, 3m 10s); now in
  filter.by_topic_frequency, $4.80 (...); pipeline main $31.20 in all. Next line at $10." An op called outside a
  step is watched the same way; a model without prices logs at 1M, 2M, 5M, ... token units. With `progress="log"`
  (stderr not a terminal), a batch of 40+ completions also adds "$0.42 so far, ~$1.75 for the batch" to its
  30-second progress lines. `FORGER_SPEND_LOG=0` turns the spend lines off.
- Limited runs: `FORGER_LIMIT=3 python pipeline.py` (or `@pipeline(limit=3)`) runs every step on at most 3 rows. Run
  every new pipeline version this way first (a smoke run): it exercises every op in about a minute, so a wrong keyword
  or a missing argument fails at once instead of minutes into the full run. A full run of pipeline code that has not
  finished a limited run refuses to start (`pyforger.steps.UntestedFullRunError`): run `FORGER_LIMIT=3` or
  `FORGER_LIMIT=50` first, and once a limited run of the same code finishes, the full run starts. Any edit to the
  pipeline, its steps, their helpers, or the prompts and constants they read is new code; sizes passed as arguments are
  not, e.g. `main(model, n_seeds=2000)` after a pilot of `main(model)`. A larger limit is a pilot: `FORGER_LIMIT=50`
  shows each filter's yield and the cost per row on a sample before you pay for every row. Inside the pipeline, every
  `@step` call cuts each rows argument (each of several inputs, `*rows` items too) to N rows spread over it: the list is
  split into N equal blocks and each block gives one row, picked by a hash of its content. So 3 of 50 queries grouped by
  theme come from the first, middle and last third, and 12 of 36 chats written 3 per seed come from 12 seeds, not 4. The
  same rows always give the same picks, so a rerun or an alternative pipeline sharing a step sees the same rows. These
  ops also cap their counts: `acquire.sample_from_LLM` (`n`) and `acquire.retrieve` / `retrieve_from_*` (N `queries`,
  picked the same way, and `n` at most N). Rows a step builds in plain Python (e.g. 1,400 prompts from a list of
  archetypes) are cut too: an op the step calls directly takes at most N rows, or as many as the step got if more (a
  step merging three limited inputs keeps all of them), and logs "using 3 of its 1,400 input rows, spread over the list;
  a full run sends all of them, so scale this op's cost by 1,400 / 3". Its other list arguments of the same length are
  cut at the same positions, so lists that go together (e.g. `inputs` and `candidate_responses`) stay aligned. Ops
  called inside other ops are not cut. Pair an op's results with the rows you passed it, not with a longer list of your
  own. Step reports say `LIMIT: 3 of 120 rows`, and the summary ends with `LIMITED RUN (limit=3): at most 3 rows per
  step; outputs are not representative`. A step that drops every row of a limited run is a warning, since the steps
  after it then check nothing. Limited results are cached under keys that include the limit, so a later full run never
  reads or overwrites them and recomputes every step. A second limited run of the same code reuses them. The request
  cache is not keyed on the limit, though, so the full run sends no model request the limited run already sent for the
  same rows: `generate_from_context` asks only about rows 4 onward, and `sample_from_LLM(n=100)` asks for 97 more
  samples. `limit=0` turns it off even when the variable is set. The old names `@pipeline(smoke=3)` and
  `FORGER_SMOKE_ROWS=3` still work.

## Composing a pipeline

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

  INFO: no quality filter ran [@pipeline(suppress_no_quality_filter_warning=...)]
  INFO: no inspect op ran [@pipeline(suppress_no_inspect_warning=...)]
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

### Several inputs and output streams

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

## Data Generation Patterns

We often see some common patterns in data generation pipelines. Here is an incomplete list of examples:

- **Agent-as-a-step:** Treat an agent as a single step in a larger pipeline. The agent can perform complex reasoning or
multi-turn interactions internally, customized to an individual input in the pipeline. This can be slower but allows for
more sophisticated processing within a single pipeline step.
- **Retrieval-augmented generation:** First retrieve relevant information (e.g. from the web) and then condition
data generation on this (either directly, for simple/consistently-structured content, or using the "dynamic step"
building block to perform case-by-case extraction of usable details from the retrieved links). It is almost always
better to retrieve knowledge before generating content, rather than relying solely on the model's parametric knowledge.
- **Rubric-based filtering:** For complex tasks where quality is hard to assure with simple heuristics, generate a
rubric (using `pyforger.acquire.generate_custom_rubric_criteria`) and then use multiple `by_llm_score` calls
with each rubric item as a separate judge to filter for high-quality outputs that meet multiple criteria. An item with
a negative weight names a mistake, e.g. `("Does the response recommend aspirin for a child?", -75)`: drop the outputs
that meet it, not the ones that miss it, or pass `negative_criteria=False` for requirements only.
- **Overgenerate and filter:** Generate more candidates than needed (especially early in a pipeline) and then apply
`pyforger.filter` operations to assure quality and diversity. Forger offers tons of filters to make this worthwhile.
- **Strong-vs-weak gap:** For a given input, generate one output with a "weak" model (e.g. "gemini-3.1") and one with
multiple "expert" models (e.g. multiple calls of "gemini-3.5" with slightly different
prompts or another strong model). Then choose only the rows where the expert models agree with each other (answer is
likely correct) but disagree with the non-expert model (input is not too easy).
- **Programming problems checked by execution:** Write or generate each problem with a reference solution and a
slower brute-force one (Python or C++, `language="cpp"`). Generate inputs with test-generator programs: a test with
`"args"` and no `"output"` runs the generator and records its stdout (`alter.run_programs`, `max_chars=None`). Run
the brute force on small inputs to get expected outputs, and keep problems whose reference agrees with it
(`filter.by_tests`). For problems with no single right output (optimization, many valid answers), score outputs with
a checker program instead (`program_checker`, partial scores). `filter.by_pass_at_k` with
`filter.program_verifier(..., language="cpp")` then measures how often the student solves each problem.
- **Back-translation:** First retrieve or generate outputs (e.g. math proofs) and then "back-translate" a suitable
input that could have produced that output (e.g. an English description of a math problem). This can be done with two
separate `@op`-decorated functions, one for the forward direction and one for the backward direction.
- **Self-Taught Reasoner / STaR (extending thinking for reasoning problems):** This is the Self-Taught Reasoner (STaR)
approach ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)), closely related to On-Policy Distillation
(Agarwal et al., 2023). When you have prompts paired with *verified*
answers (e.g. competition-math problems with checked integer answers) but want long chain-of-thought traces for SFT,
do not copy reasoning from a stronger teacher -- generate it from the *student* you intend to finetune. This is
generally necessary for training reasoning-style models on tasks that require extended reasoning, such as difficult
math, programming, or embodied reasoning. Sample `n` solutions per prompt from the student (use `acquire.TinkerModel`
to sample the base student or its `tinker://` checkpoint on Tinker, where the `n` samples share one prefill, or
`acquire.LocalModel` to run it on a local GPU via vLLM; set a generous `max_tokens` so traces are not truncated), keep
only the solutions whose final answer matches the verified answer, and use each kept trace as a training target.
`alter.self_taught_reasoner` implements exactly this: it returns one row per correct sample (a prompt with 0 correct
samples drops out), with the student's reasoning trace and answer split into separate fields; for free-text answers,
`answer_judge=` has a model judge each answer against the reference (see `alter.md`). Tinker samples are charged to the
Tinker dollar budget when one is configured (see `acquire.md`).

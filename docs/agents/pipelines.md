# Pipelines and steps

## Steps and step reports

Forger is a framework for running pipelines. Pipelines are `@pyforger.pipeline` functions that call named
`@pyforger.step` functions. A step is a function from rows to rows built from Forger ops (typically acquire or alter,
then filter); it may also take several lists of rows and return a dict of named row lists (see [Several inputs and
output streams](#several-inputs-and-output-streams)). Each step is cached like an `@op`. A complete example with its
output is in [Composing a pipeline](#composing-a-pipeline).

When a step returns, it prints a report consisting of: its rows' schema; highly-repetitive phrases occurring in these
rows; every op it ran with rows in/out, yield, time, tokens, and cost; a histogram of each filter's scores with what
other thresholds would have kept; warnings; and one sample output row. The pipeline prints a summary table of its steps
at the end, followed by every `WARNING:` line. **Read these reports after every run.**

- `pyforger.inspect.inspect_step("name")` re-prints the full report of a step's latest run with up to 10 sample rows,
  also from another shell while the pipeline runs. `inspect_step("name", rows=True)` returns the run's whole output
  from the cache, e.g. after a later step failed; a run kept out of the cache (e.g. some requests timed out) saves its
  rows for this too, with a warning that they may be partial. `pyforger.inspect.runs()` lists saved pipeline runs (see
  [inspect.md](inspect.md)).
- A cached step prints one line (`step translate · 42 → 40 rows · cached · originally 0.5s · $0.0034`) and costs
  nothing this run. If a cached filter shows no score histogram, rerun it with `_no_cache=True`.
- State the yield you expect from each op with `min_yield=` / `max_yield=`, e.g.
  `filter.by_llm_score(rows, ..., min_yield=0.2, max_yield=0.8)`; a miss is a warning.
- `repeated:` line: phrases or names shared by a large share of the rows (`rewrite "i need" 78/79 (99%)`). This is
  templating or name collapse that exact-match and ROUGE-L filters let through; fix the prompt. Full lists:
  `inspect.phrase_frequency`.
- `(python)` rows in the op table: rows that changed between ops in plain Python (e.g. a list comprehension). To see
  why rows went, write the check as `filter.by_predicate` instead, which gets its own row and a kept/dropped bar.
- `@step(sample=3)` shows three sample rows; `@step(show=["statement"])` prints those fields in full. Long values
  are otherwise summarized by shape (`def solve(n): … (312 lines, ...)`). `pyforger.steps.set_report_model(model)` or
  `FORGER_REPORT_MODEL` adds a one-sentence LLM summary (extra LLM calls); check the value itself before trusting it.
- Hard errors (`StepContractError` / `PipelineContractError`): a step returns neither a list nor a dict of row lists,
  a step is called inside another step, a pipeline calls no steps, or rows do not match a `list[X]` pydantic hint.
- `strict=True` on `@step` / `@pipeline` turns `WARNING:` and `INFO:` lines into errors. `prod=True` or
  `FORGER_STEP_REPORT=0` silences reports.

### Warnings and suppression

- `WARNING:` lines are data problems: something happened to this run's rows, and the line says what, e.g. a
  `min_yield` / `max_yield` miss; completions hit `max_tokens` ("20 rows dropped: 34 of 120 completions ... hit
  max_tokens=16384; raise max_tokens or lower reasoning_effort"); parse failures; provider errors; a judge gave no
  score; reasoning traces that open by discussing the output format (`output_schema` and the prompt's format
  instructions disagree, so the trace is a poor training target); retrieved text that reached no prompt. Fix each one.
- Retrieved text must reach a prompt: if no `acquire.retrieve*` document appears in any prompt or in the returned rows,
  the pipeline warns. A field that holds the documents but that no prompt formats does not count.
- `INFO:` lines (dim) are about how the code is written: a step calls no Forger ops (write its checks as ops such as
  `filter.by_predicate`), only one step, no quality / diversity filter or inspect op ran, an op ran outside a step.
  Never add a no-op filter just to quiet one. The summary's "filters that removed nothing" notes are not warnings.
- These checks use an op's category: your own `@op` quality filter needs `@op(category="filter")` to count.
- Suppress only what you checked in the rows and found harmless, with a flag whose value says why:

  ```python
  @step(suppress_truncation_warning="<why these rows are still fine>")
  def draft(rows, model): ...

  @pipeline(suppress_no_diversity_filter_warning="<why these rows are still fine>")
  def main(model): ...
  ```

  A line that can be suppressed ends with its flag in brackets, e.g.
  `INFO: no inspect op ran [@pipeline(suppress_no_inspect_warning=...)]`. Step flags: `suppress_truncation_warning`,
  `suppress_no_ops_warning`. Pipeline flags: `suppress_no_quality_filter_warning`,
  `suppress_no_diversity_filter_warning`, `suppress_no_inspect_warning`, `suppress_retrieval_warning`,
  `suppress_low_yield_warning` (under 10% of the first step's rows reached the output; 10-30% is a note).
- Nothing else can be suppressed: parse failures, provider errors, and missing judge scores lose rows a fix gets back;
  for a yield miss, set `min_yield` / `max_yield` to the yield you expect.
- Suppress on the step, not pipeline-wide: truncation in a draft step that is regenerated later may not matter;
  in the answer step it does. When a fix exists (larger `max_tokens`, a filter), make the fix, and do not add a flag
  next to it "in case": a flag that suppressed nothing is reported.
- A misspelled flag raises `TypeError`; a reason under 3 words or a placeholder raises `ValueError`. Suppressed
  problems stay in the record (`inspect_step("draft").suppressed`).

### Limited runs

Run every new pipeline version with `FORGER_LIMIT=3 python pipeline.py` (or `@pipeline(limit=3)`) first. Every step
gets at most 3 rows per rows argument, so every op runs in about a minute and a wrong keyword fails at once, not
minutes into the full run. `FORGER_LIMIT=50` is a pilot: it shows each filter's yield and the cost per row before you
pay for every row.

- Only `acquire.sample_from_LLM` (`n`) and `acquire.retrieve*` (`queries`, `n`) also cap their counts; other ops that
  make rows from nothing still make all of them.
- Outputs are not representative. A yield miss on fewer than 20 rows is only `INFO:`; a step that drops every row is a
  warning, since later steps then check nothing.
- The full run recomputes every step but reuses the limited run's completions from the request cache, e.g.
  `sample_from_LLM(n=100)` asks only for 97 more. `limit=0` turns it off even when the variable is set.

### Spending caps and the spend ledger

- `@pipeline(max_cost_usd=20)` / `@pipeline(max_token_units=2e6)` cap one pipeline run; `FORGER_MAX_COST_USD` and
  `FORGER_MAX_TOKEN_UNITS` cap all spend in the process. At a cap the next request raises
  `pyforger.BudgetExceededError`; in-flight requests finish, so spend can pass a cap slightly.
- Gateway models without LiteLLM pricing add $0: cap them in token units.
- Caps are separate from the shared quotas (`FORGER_TOKEN_UNIT_LIMIT`, Tinker's dollar ledger), which also apply.
- Every model call is appended to the spend ledger (`$FORGER_SPEND_LEDGER`, default `$FORGER_CACHE/spend.jsonl`) as
  it happens, so killed runs keep their record. `pyforger.inspect.spend(since="2026-10-16")` totals it by run,
  pipeline, and step.
- Reports show a `units` column for models with token-unit weights (see the Cost section of [AGENTS.md](../AGENTS.md)).

### Spend projection

A batch of model requests that would not fit stops early instead of spending what is left. Once 5% of a batch has
finished (batches under 40 completions are skipped), each new request checks that the rest fits every cap and the
token-unit quota; if not, it raises `pyforger.SpendProjectionError` (a `BudgetExceededError`) with the projected cost.
Make the op cheaper (fewer rows, samples, or judges, a cheaper model, shorter outputs) and rerun: finished requests
come from the request cache. `FORGER_PROJECT_SPEND=0` turns the check off.

### Live spend

Every uncached batch of 40+ model completions (an op sends all its prompts as one batch) logs one line once 5% of it
(at least 10 requests) has finished, if the whole batch will likely cost $1 or more (1M token units for a model
without prices), e.g. "Cost estimate: filter.by_topic_frequency in step balance: the first 102 of 2,040 completions
cost $0.21, so all 2,040 will likely cost about $4.20 (at least $3.80), 9,800,000 token units; the token-unit quota
has 130,000,000 token units left." Rows answered from the request cache are not sent or counted, and a cached op sends
nothing. The estimate assumes the rest of the batch costs what its first completions did (it is sent in shuffled
order). `FORGER_PROJECT_SPEND=0` does not turn it off.

While a step runs, it logs a line each time its spend first reaches $1, $2, $5, $10, $20, $50, ..., e.g. "Spend so
far: step balance $5.03 (2,104,511 token units, 812 calls, 3m 10s); now in filter.by_topic_frequency, $4.80 (...);
pipeline main $31.20 in all. Next line at $10." An op called outside a step is watched the same way; a model without
prices logs at 1M, 2M, 5M, ... token units. With `progress="log"` (stderr not a terminal), a batch of 40+ completions
also adds "$0.42 so far, ~$1.75 for the batch" to its 30-second progress lines. `FORGER_SPEND_LOG=0` turns the spend
lines off.

### Concurrent steps and threads

- `my_step.submit(*args)` runs a step in a background thread and returns a `Future`; use it for independent
  branches. Call `.result()` on each one: errors re-raise there.

  ```python
  impossible = make_impossible.submit(verified[:n], writer, judge)
  hard = student_difficulty(verified[n:], student)        # runs meanwhile
  rows = assemble(impossible.result(), hard)
  ```

- Ops called from your own `ThreadPoolExecutor` workers are not itemized in the step report, and their spend is
  outside the pipeline's caps, unless submitted as `pool.submit(contextvars.copy_context().run, fn, arg)`.

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

Each step prints a report when it returns, and the pipeline prints a summary at the end (shortened here):

```text
step translate ───────────────────────────────────────────────────────────────────
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
          8    11  ████████████████████████████  yes
         ...

  sample rows (1 of 40)
  #   text                                             spanish
  0   Explain the capital of France in one sentence.   Explica la capital de Francia en una frase.

pipeline main ────────────────────────────────────────────────────────────────────
  2 steps · 42 → 27 rows (yield 0.64) · 0.5s · 5,040 in / 630 out tokens · $0.0034

  step        in   out   yield   time      cost
  ────────────────────────────────────────────────
  translate   42    40    0.95   0.5s   $0.0034
  dedupe      40    27    0.68   0.0s         -

  INFO: no quality filter ran [@pipeline(suppress_no_quality_filter_warning=...)]
  INFO: no inspect op ran [@pipeline(suppress_no_inspect_warning=...)]
```

For an iterative pipeline, keep the outer loop in Python and make each round a `@step`, so finished rounds come from
the cache on the next run.

Module import time should not execute work. LLM calls, file I/O, and any `@op`-decorated call belong inside a function
body.

### Several inputs and output streams

A step may take several lists of rows and return a dict of named row lists (streams); a later step takes one stream.
Rows arguments are the first argument when it is a list, and any other argument that is a non-empty list of dicts or is
annotated `list[dict]` / `list[X]`. The report and the pipeline table count each input and stream.

```python
class Assembled(TypedDict):  # optional; validates each stream like list[X]
    train: list[TrainRow]
    refusals: list[RefusalRow]

@step
def assemble(solvable, refusals) -> Assembled:
    """Label solved problems for training and keep refusals as their own set."""
    return {
        "train": filter.by_length(label_kind(solvable, "solve"), instruction_field="solution", min_len=2),
        "refusals": filter.by_exact_match(label_kind(refusals, "refuse"), field="problem"),
    }

@pipeline
def coding(writer, solver, judge_model):
    tried = attempt(write_problems(TOPICS, writer), solver)  # {"solvable": [...], "refusals": [...]}
    out = assemble(judge(tried["solvable"], judge_model), tried["refusals"])
    return {"train": format_train(out["train"]), "refusals": out["refusals"]}
```

## Data Generation Patterns

Common patterns (not a complete list):

- **Retrieval-augmented generation:** retrieve knowledge (e.g. from the web) first and condition generation on it,
  directly or via [`dynamic_step`](dynamic_step.md) for case-by-case extraction. This is almost always better than
  relying on the model's parametric knowledge.
- **Find the weak spots first:** group a baseline model's failures by skill with `inspect.eval_tree` (or label them
  with `inspect.topic_based_clustering`), then check which topics your own rows cover and rebalance them with
  `filter.by_topic_frequency`.
- **Overgenerate and filter:** generate more candidates than needed, especially early, then filter for quality and
  diversity: drop near-duplicates with `filter.by_exact_match`, `filter.by_rouge_l`, `filter.by_n_gram_overlap` or
  `filter.by_embedding_similarity`, and low-quality rows with an LLM judge (`filter.by_llm_score`).
- **Check exactly what can be checked:** write every requirement with an exact answer (a final value, the output on
  test inputs, a required refusal) as a code criterion, e.g. `acquire.generate_custom_rubric(...,
  code_criteria_only=True)`, and keep LLM judges for what code cannot check.
- **Rubric-based filtering:** for quality that simple heuristics cannot check, generate a rubric with
  `acquire.generate_custom_rubric_criteria` and use one `filter.by_llm_score` judge per rubric item. An item with a
  negative weight names a mistake: drop the outputs that meet it, not the ones that miss it.
- **Agent-as-a-step:** an agent ([`dynamic_step`](dynamic_step.md)) handles one input with multi-turn reasoning
  inside a larger pipeline; slower, but more flexible.
- **Strong-vs-weak gap:** answer each input with a weak model and several strong ones; keep rows where the strong
  models agree with each other (likely correct) but not with the weak one (not too easy).
- **Programming problems checked by execution:** pair each problem with a reference and a brute-force solution
  (Python or C++); generate inputs with generator programs (tests with `"args"` and no `"output"` record stdout),
  get expected outputs from the brute force, keep problems whose reference agrees (`filter.by_tests`), and score
  problems without one right output with a checker program (`program_checker`, partial scores).
- **Back-translation:** retrieve or generate outputs (e.g. proofs), then generate an input that could have produced
  each one.
- **Cheap iteration:** every op is cached on disk, so later stages can be rerun and tuned without paying again for
  earlier ones ([caching.md](caching.md)).
- **Self-Taught Reasoner (STaR):** for prompts with *verified* answers that need long chain-of-thought SFT targets,
  generate the reasoning from (or similar to) the *student* you will finetune, not a stronger teacher
  ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)). `alter.self_taught_reasoner` samples `n` solutions per
  prompt (from `acquire.TinkerModel` or `acquire.LocalModel`) and keeps one row per sample whose final answer matches;
  for free-text answers, `answer_judge=` has a model judge each answer against the reference. Set a generous
  `max_tokens` so traces are not truncated. See [alter.md](alter.md).

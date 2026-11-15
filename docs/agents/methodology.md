# Building a dataset with Forger

Forger's ops make each stage of a data pipeline cheap to run, cache, and measure. This page describes the way of
working they are built for: a dataset is a series of decisions (where tasks come from, how they are refined, how
ground truth and rubrics are made, which filters apply), and each decision rests on measurements of runs you actually
made, not on argument or intuition alone. [`examples/rl_math/pipeline.py`](../../examples/rl_math/pipeline.py) works
this way from start to finish; its step docstrings show what the evidence for each choice looks like.

## Start with a small end-to-end loop

Get a few dozen rows through every stage early: source, refinement, ground truth and rubrics, filters, and, for
training data, a short training run and an evaluation. A working loop shows what actually needs fixing, and leaves you
with a valid dataset to improve on. Then compare alternatives at each stage, scale up, and refine. Build and compare at
least two or three small versions before your large-scale run.

Use limited runs for every smoke test and comparison: `FORGER_LIMIT=3` to see a stage's output at all, larger limits
to compare alternatives ([Limited runs](pipelines.md#limited-runs)). Each step's report shows its samples, its yield,
and its cost; read them before moving on.

## Price the full run before you start it

A small run is also how you learn what the full run will cost. Before scaling up:

- Run a pilot (e.g. `FORGER_LIMIT=50`) of exactly the pipeline you will scale up: every step, with the same models,
  reasoning effort, `n`, and flags. A step the pilot skips (behind a flag, an `if`, or a model it did not load) has no
  measured cost: its full-scale cost is unknown, not zero.
- From each step's report, take its cost per row and multiply by the rows it will get at full scale, e.g. "seeds: $0.40
  for 50 rows, so about $16 for 2,000". Most ops make a fixed number of calls per row (one per row for
  `filter.by_topic_frequency`, one per sample and judge for others), so their cost grows with the rows. Add up the
  steps and compare the total with the quota you have left.
- Models differ in price several times over, and a reasoning model also pays for its reasoning on every call. Price
  each model you will use from calls in the pilot, not from its list price alone.
- Write the projection in your decision log next to the quota left, and compare it with the step reports after the
  full run.

During the full run, read the `Cost estimate` lines: every uncached batch likely to cost $1 or more logs one once 5%
of it has finished ([Live spend](pipelines.md#live-spend)). If an estimate is far above your projection, stop the run
and find out why; finished requests are in the request cache, so a rerun with cheaper settings does not pay for them
again.

## Compare alternatives at each decision

At each stage, run two or more options at small or medium scale before choosing one: e.g. compare sources for your
starting tasks, ways to refine them, techniques for generating ground truth and rubrics, and filters, by how each
shapes the data distribution. Choosing from subjective or intuitive reasoning alone is how pipelines end up being
trial and error.

- Write each option as its own `@pipeline` function calling the steps they share, in the same file. Unchanged steps
  come from the cache, so each alternative pays only for the steps it changes.
- `pyforger.inspect.runs()` lists the runs with each filtering step's yield and its 95% interval, which show whether
  a difference is larger than the noise at N rows. `pyforger.inspect.compare_pipelines("a", "b")` shows the first step
  where two runs diverge, and whether each difference is clear or within noise.
- Task sources: retrieval (`acquire.retrieve_from_hf` for public datasets, some with tests and solutions;
  `acquire.retrieve_from_web`) and LLM generation (`sample_from_LLM`, `generate_from_context`) give different data.
  Retrieved tasks are written by people, cost no generation tokens, and differ from what an LLM invents; generated tasks
  can be shaped to order. Measure both before choosing.

Keep every version you ran, e.g. as numbered files (`pipeline_1.py`, `pipeline_2.py`, ...), each opening with a comment
naming the file it was copied from and what changed. Alternatives for the same decision build on the same file, and
later work builds on the one you chose, so the files form a tree of what you tried; fix small bugs in place. Keep a
decision log (e.g. `DECISIONS.md`) as their index, written as you go rather than at the end: for each decision, the
files or pipeline functions compared, the numbers from runs you actually made (e.g. "web-sourced: 23 ideas, 4% near the
reference, 2/6 drafts pass; brainstormed: 31 ideas, 9%, 3/6 pass"), and what you chose and why. Where you chose without
running the alternatives, write "not compared" and why; never describe a planned comparison as done.

## Inspect every set of rows

Use `pyforger.inspect` ops at every stage, on every set of rows you build or study, including reference data such as a
validation set (e.g. `topic_based_clustering` to see what it covers):

- Read samples. `inspect_step("name", rows=True)` returns a step's cached rows.
- Look at distributions of scores, lengths, difficulty, and diversity before training on anything.
- `embedding_dispersion(rows, "prompt")` measures how spread out a set is; its near-duplicate groups say about how many
  distinct rows it holds, and `reference=` compares it with another set.
- `compare_to_reference(rows, reference)` shows where your rows differ from a reference set such as validation data.
  Before a full-scale run, measure a limited run against the reference with this and with
  `embedding_dispersion(..., reference=...)`, and act on what they show.

A training run plus evaluation takes hours and folds many causes into one number. Use it as the final check of data
you have already validated, not as the way to find out whether data is good.

## Generate more than you need, then filter

Everything an LLM generates is approximate, and that is expected: some statements, reference solutions, tests, and
checkers will be wrong. Filters, not hand-fixes, are how you get good data: generate more than you need and let filter
and inspect ops (e.g. running references through their own tests, `filter.by_pass_at_k`, `inspect.rubric_scores` with
probes) remove what fails, rather than writing or fixing rows yourself.

- Give each check its own filter op, so the step report shows how many rows each check removed; checks folded into one
  flag hide which one failed.
- For each filter, count the rows it removes and read a sample of them before you settle its threshold.
- End each `@step` with a filter that serves as a contract on its output, using `min_yield=` and `max_yield=`, e.g.
  `filter.by_llm_score(rows, ..., min_yield=0.2, max_yield=0.8)`.

## RL data

- Hard-to-easy generalization usually works: find or create tasks harder than the ones the model must eventually solve,
  from a space broader than your target examples, and train on those.
- Rows that copy or lightly vary your held-out examples teach the student those examples rather than the skill, and
  make the held-out set useless for measuring progress.
- In GRPO, a prompt whose sampled completions all get the same reward gives no learning signal.
  `filter.by_pass_at_k(rows, model=student, metric=...)` measures the student's difficulty on your rows (a metric may
  return a score); `histogram=True` prints the pass-rate histogram and counts rows whose samples all scored the same.
- Check rubrics on real student responses before RL: `inspect.rubric_scores(rows, metric, response_fields=...)` on rows
  holding sampled responses, with `probes={name: fn(row) -> response}` adding responses that should score low (e.g. a
  trivial valid answer, or one that games the metric); each column is compared row by row with the first.
- Where a response's quality can be measured exactly (e.g. running a program on tests), code criteria give a more
  precise and less noisy reward than LLM-judged criteria, and need no LLM calls.
- When you train, look at sampled responses, reward variation within groups, truncation, and per-problem evaluation
  results, not only the mean score. Short training runs are a cheap diagnostic.
- A small seed set can bootstrap a larger one: train on it, then sample the checkpoint to calibrate or filter new rows.

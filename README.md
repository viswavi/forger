# Forger

Forger is a Python library for building synthetic training data: supervised fine-tuning (SFT) sets, and RL
environments made of prompts plus a reward (a rubric) for each prompt.

A Forger pipeline is ordinary Python. You write a few named **steps**. Each step gets or makes rows (with an LLM, web
search, Hugging Face datasets, ...), changes them, and filters them. Forger runs the steps and caches every
expensive call on disk. After each step it prints a report of what happened to your rows: how many each filter
kept, a histogram of the scores, the cost, warnings and a sample row. You change the pipeline, run it again, and
only the parts you changed run again.

Forger also asks you to work in a particular way: at each decision (where the seed data comes from, which model
writes the answers, where to put a filter threshold), try more than one option on a small sample, measure them, and
keep the measurement next to the choice. [`docs/agents/methodology.md`](docs/agents/methodology.md) describes this.

## Install

Forger needs Python 3.12 or later.

```bash
git clone https://github.com/viswavi/forger.git
cd forger
pip install -e .
```

Optional extras:

| Extra | Install it for |
|---|---|
| `pip install -e ".[clustering]"` | embedding filters and reports (dedup, decontamination, dispersion, topic clusters) |
| `pip install -e ".[vllm]"` | `LocalModel`: sampling a student model on your own GPU |
| `pip install -e ".[tinker]"` | `TinkerModel`: sampling a student model on Tinker |

## Models and API keys

Every LLM call goes through `pyforger.acquire.Model`, a wrapper around [LiteLLM](https://docs.litellm.ai/). So any
model id that LiteLLM knows works, for example `"anthropic/claude-haiku-5-5"`. Put the provider's key in the
environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) or in a `.env` file. For a LiteLLM proxy, use a
`litellm_proxy/...` id and set `LITELLM_API_KEY` and `LITELLM_BASE_URL`.

```python
from pyforger.acquire import Model

model = Model("anthropic/claude-haiku-5-5", max_concurrency=8, reasoning_effort="low")
```

Details: [`docs/agents/models.md`](docs/agents/models.md).

## Quick start

This pipeline writes trivia questions, removes near-duplicates, and keeps the ones an LLM judge rates as correct and
clear.

```python
# trivia.py
import json
from pyforger import acquire, filter, pipeline, step
from pyforger.acquire import Model


@step
def write_questions(model: Model, topic: str) -> list[dict]:
    """Fifty questions about one topic, each with a short answer, without near-duplicates."""
    rows = acquire.sample_from_LLM(
        f"Write one hard trivia question about {topic}, and its short answer.",
        model,
        n=50,
        output_schema={"question": str, "answer": str},
    )
    return filter.by_rouge_l(rows, field="question", threshold=0.7, min_yield=0.5)


@step
def keep_good(rows: list[dict], judge: Model) -> list[dict]:
    """Keep questions the judge rates 4 or 5 of 5 for a correct answer and one clear reading."""
    return filter.by_llm_score(
        rows,
        model=judge,
        input_field="question",
        output_field="answer",
        criteria="the answer is correct, and the question has exactly one reasonable answer",
        score_threshold=4,
        min_yield=0.3,
    )


@pipeline
def main(model: Model, judge: Model) -> list[dict]:
    return keep_good(write_questions(model, "astronomy"), judge)


if __name__ == "__main__":
    rows = main(Model("anthropic/claude-haiku-5-5"), Model("anthropic/claude-sonnet-5-5"))
    with open("trivia.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
```

Run it on 3 rows first, then in full:

```bash
FORGER_LIMIT=3 python trivia.py   # every op on 3 rows: finds mistakes in about a minute, for cents
python trivia.py                  # the full run
```

Forger refuses a full run of code that has not finished a limited run (`UntestedFullRunError`). Any edit to the
pipeline counts as new code. The full run reuses the completions the limited run already paid for.

## Core ideas

**Rows.** Data moves through a pipeline as a `list[dict]`, one dict per example.

**Ops.** Forger's building blocks are in four modules:

| Module | What its ops do | Examples |
|---|---|---|
| `pyforger.acquire` | make or fetch new rows | `sample_from_LLM`, `retrieve_from_web`, `generate_custom_rubric` |
| `pyforger.alter` | change or add fields, never remove rows | `generate_from_context`, `self_taught_reasoner` |
| `pyforger.filter` | remove rows | `by_llm_score`, `by_pass_at_k`, `by_embedding_similarity`, `by_predicate` |
| `pyforger.inspect` | report on rows, never change them | `rubric_scores`, `inspect_step`, `compare_pipelines` |

Every op's signature is in [`docs/OPS.md`](docs/OPS.md).

**Steps and pipelines.** A `@step` is a named function from rows to rows, built from ops. A `@pipeline` is a function
that calls steps. Each step prints a report when it finishes, and the pipeline prints a summary table at the end.
Read them after every run. `WARNING:` lines describe something that happened to your data (for example, "34 of 120
completions hit max_tokens"). Fix each one, or suppress it with a written reason.

**Caching.** Each op's result is cached on disk, keyed by its inputs. A rerun costs nothing until you change
something. Pass `_no_cache=True` to an op for a fresh result.

**Yield contracts.** Give a filter the share of rows you expect it to keep, for example `min_yield=0.3`. A miss is
a warning in the step report, so a broken prompt shows up at once and not three steps later.

**Spend.** Forger records the dollar cost of every call, estimates the cost of each large batch after the first 5%
of it, and logs spend as it goes. Cap a run with `@pipeline(max_cost_usd=5)`. If you budget in tokens instead, give
your models token-unit weights (see "Cost" in [`docs/AGENTS.md`](docs/AGENTS.md#cost)) and cap the run with
`@pipeline(max_token_units=...)`.

## A worked example: an RL dataset of math problems

[`examples/rl_math/pipeline.py`](examples/rl_math/pipeline.py) builds an RL dataset of olympiad-style math problems.
Each row is a prompt and a rubric whose one criterion is Python code that checks the final answer. The pipeline has
four steps:

```python
@pipeline
def main(writer: Model, verifier: Model, student: LocalModel) -> list[dict]:
    """Topics, then verified problems, a difficulty band, and checked rubrics."""
    seeds = seed_topics(writer)
    rows = write(seeds["topics"], seeds["validation"], writer, verifier)
    rows = calibrate(rows, student)
    return check_rubrics(rows, student, skyline=writer)
```

1. **`seed_topics`** searches the web for competition problem sets, has the writer model list the topics each page
   tests, and removes near-duplicate topics. It also downloads the held-out validation set (OlympiadBench), so that
   later steps can stay away from it.

   ```python
   pages = acquire.retrieve_from_web(QUERIES, n=5)
   extracted = alter.generate_from_context(
       pages, extract_prompt, model, output_schema={"topics": list}, merge=True
   )
   ...
   topics = filter.by_embedding_similarity(topics, field="topic", threshold=0.9)
   ```

2. **`write`** writes three problems per topic, each with a solution and an integer answer. It then keeps only the
   problems that pass three checks. The answer can't appear in the problem statement. A stronger verifier model must
   reproduce the answer on at least 2 of 4 tries. And the problem can't be a near-copy of another problem or of a
   validation problem.

   ```python
   rows = filter.by_pass_at_k(
       rows, model=verifier, input_field="prompt", k=4,
       min_rate=0.5, max_rate=1.0, metric=solved, extract_answer=whole, min_yield=0.4,
   )
   rows = filter.by_embedding_similarity(
       rows, field="problem", threshold=0.9, reference=validation
   )
   ```

3. **`calibrate`** samples the student (the model you will train, run locally with vLLM) 8 times per problem. It
   drops the problems the student already solves 7 or 8 times out of 8, because RL learns little from them.

4. **`check_rubrics`** tests the rubric before any training. It scores the reference solutions, a stronger model's
   answers, the student's samples, and *probes*: bad answers that a sound rubric must score 0. One probe is a
   response that writes `\boxed{}` around every number from -10 to 100. Another stops in the middle of its reasoning.
   A rubric that gives these probes credit would teach the student to game it.

   ```python
   inspect.rubric_scores(
       rows, reward, n=200,
       response_fields=["solution", "skyline_solution"], probes=probes(), model=student, k=8,
   )
   ```

Each step's docstring records the options that were compared for that decision, the call that measured them, and
the result. (In the example, the numbers are placeholders and not from a real run.) This is the habit Forger is
built around. For example, `calibrate`'s docstring explains why the pass@8 band is 0 to 6 of 8, using the histogram
that the step's own report prints.

Before you run it, replace `WEAK_MODEL_ID` and `STRONG_MODEL_ID` with model ids, set `BRAVE_API_KEY` for web
search, and install `.[vllm,clustering]` on a machine with a GPU. Then:

```bash
FORGER_LIMIT=3 python examples/rl_math/pipeline.py      # a smoke run, 3 rows a step
python examples/rl_math/pipeline.py rl_math.jsonl       # the full run
```

## Documentation

The reference docs are written for coding agents (they are what `CLAUDE.md` points to), but they are just as useful
to read yourself.

| Read | For |
|---|---|
| [`docs/AGENTS.md`](docs/AGENTS.md) | the overview and index |
| [`docs/agents/methodology.md`](docs/agents/methodology.md) | how to build a dataset, step by step |
| [`docs/agents/pipelines.md`](docs/agents/pipelines.md) | steps, reports, limited runs, spend caps, common patterns |
| [`docs/agents/acquire.md`](docs/agents/acquire.md) | generation, retrieval, student models, rubrics |
| [`docs/agents/alter.md`](docs/agents/alter.md) | transforms, running code against tests, STaR |
| [`docs/agents/filter.md`](docs/agents/filter.md) | LLM judges, dedup and decontamination, difficulty filters |
| [`docs/agents/inspect.md`](docs/agents/inspect.md) | reports on rows, rubric checks, comparing runs |
| [`docs/agents/models.md`](docs/agents/models.md) | `Model`, credentials, concurrency, reasoning effort |
| [`docs/agents/caching.md`](docs/agents/caching.md) | how caching and tracing work |
| [`docs/OPS.md`](docs/OPS.md) | every op's signature, generated from the code |

## Using Forger with a coding agent

Forger is designed to be driven by a coding agent such as Claude Code. The repository's `CLAUDE.md` loads
`docs/AGENTS.md`, so an agent working in this repository starts with the overview and reads the topic files as it
needs them. Ask it for a dataset (for example, "an RL set of 2,000 clinical questions with rubrics, for Qwen3-4B"),
then review the steps' docstrings and reports it produces.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: `pip install -e ".[dev,test]"`, `pre-commit install`, and
`pytest`.

## License

See [`LICENSE`](LICENSE).

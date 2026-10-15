# Composing a pipeline

A single-pass pipeline:

```python
import pyforger
from pyforger import acquire, alter, filter, op

@op
def build_dataset(seeds, model):
    rows = alter.translate.to_language(
        seeds, target_language="Spanish", fields=["text"], model=model,
    )
    rows = filter.by_length(rows, instruction_field="text", min_len=3, max_len=200)
    rows = filter.by_rouge_l(rows, field="text", threshold=0.7)
    return rows

def main(seeds, model):
    return pyforger.run(build_dataset, seeds, model)
```

An iterative pipeline keeps the outer loop in Python and decorates each round's helper with `@op` so that cached
rounds are reused across runs:

```python
import random

@op
def one_round(rows, model):
    in_context_examples = random.sample(rows, min(len(rows), 5))
    prompt = ...  # construct a prompt using the in-context examples
    new = acquire.sample_from_LLM(prompt=prompt, model=model, n=64)
    return rows + filter.by_rouge_l(new, field="text", threshold=0.7)

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
with each rubric item as a separate judge to filter for high-quality outputs that meet multiple criteria.
- **Overgenerate and filter:** Generate more candidates than needed (especially early in a pipeline) and then apply
`pyforger.filter` operations to assure quality and diversity. Forger offers tons of filters to make this worthwhile.
- **Strong-vs-weak gap:** For a given input, generate one output with a "weak" model (e.g. "gemini-3.1") and one with
multiple "expert" models (e.g. multiple calls of "gemini-3.5" with slightly different
prompts or another strong model). Then choose only the rows where the expert models agree with each other (answer is
likely correct) but disagree with the non-expert model (input is not too easy).
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
samples drops out), with the student's reasoning trace and answer split into separate fields. Tinker samples are
charged to the Tinker dollar budget when one is configured (see `acquire.md`).

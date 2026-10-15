# `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Execute Python code in a sandboxed environment and return the result. |
| `dynamic_step` | Run dynamic input-specific sub-pipelines that return a string list per input. |
| `dynamic_step_source` | Generate or inspect sub-pipeline generated for a single input. |

`generate_from_context` performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an
optional JSON schema. If the validated LLM call fails after `num_retries`, we return `None` for the failing slots.
As with `sample_from_LLM`, reasoning traces are included under a `thinking_trace` key. Inline traces are not extracted:
`LocalModel` leaves `<think>` blocks in the completion text. Use `split_reasoning` to separate them out.

`self_taught_reasoner` (re-exported from `pyforger.alter`, also in `pyforger.alter.generate`) implements the
**Self-Taught Reasoner (STaR)** approach ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) -- closely
related to **On-Policy Distillation** (Agarwal et al., 2023), since both keep SFT traces drawn from the student's own
distribution rather than a separate teacher: for each input row it samples `n` responses from a student `model` and
keeps only those whose final answer matches the row's reference answer (`reference_field`, default `"final_answer"`).
Each kept response becomes one
output row carrying the student's raw response plus, when `split_thinking=True`, its reasoning and answer in
`thinking`/`answer` fields (reasoning comes from the provider's separate reasoning field when returned, otherwise from
the inline `<think>` block); the matched student answer becomes the new ground truth.

```python
from pyforger.alter import self_taught_reasoner
from pyforger.acquire import TinkerModel

student = TinkerModel("Qwen/Qwen3-8B", max_tokens=16384)  # n samples per problem share one prefill
rows = self_taught_reasoner(problems, solver_prompt, student, n=8, reference_field="final_answer")
```

`LocalModel("Qwen/Qwen3-8B", max_model_len=40960)` samples the same student on a local GPU via vLLM instead.

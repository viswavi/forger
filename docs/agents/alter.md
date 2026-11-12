# `pyforger.alter`

Transform rows without removing items. Signatures and defaults: [OPS.md](../OPS.md#alter).

| Export | Description |
|---|---|
| `generate_from_context` | LLM generation from each row's fields (`sample_from_LLM` over rows). |
| `self_taught_reasoner` | STaR: keep the student's own samples that are verified correct. |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Call the last function in each row's code with one argument, in-process (RestrictedPython). |
| `run_programs` | Run each row's Python or C++ program on its tests in subprocesses; record per-test results. |
| `dynamic_step` | Run input-specific sub-pipelines (see [dynamic_step.md](dynamic_step.md)). |
| `dynamic_step_source` | Generate or inspect the sub-pipeline for a single input. |

## `generate_from_context`

Makes `samples_per_item` LLM calls per row; returns a flat list of parsed outputs (`None` for failed slots).
`merge=True` attaches each output to its input row instead (failed slots dropped); `output_field="name"` nests it.

```python
rows = alter.generate_from_context(problems, "Solve: {problem}", model, output_schema={"answer": str}, merge=True)
# [{"problem": "...", "answer": "4"}, ...]
```

Failures that do not raise (the first three below) leave that prompt's slots `None` and keep the rest:

- Parse failure: retried up to `max_retries` times; these `None`s are cached.
- Transient provider error (timeout, 429, 5xx): retried only by the model's `num_retries`.
- **Row error**: the content filter blocked the prompt or reply, or the prompt exceeds the context window
  (`ContextWindowExceededError` or `pyforger.PromptTooLongError`). Never retried. If every prompt of a call (at
  least 2) hits one, the op raises instead: fix the template or model.
- Other errors (bad API key, unknown model, other 400s, exhausted quota, spending cap) raise at once.
- Provider failures produce one warning in the step report. A transient failure leaves the op and step uncached; a
  rerun re-sends only those prompts.
- Row errors are cached, in the op cache and the request cache: a rerun gives the same `None` rows instead of asking
  again (the filter does not block a prompt every time, and one prompt answered on a rerun would change every later
  step's input). `_no_cache=True` asks blocked prompts again.

Reasoning traces go under `thinking_trace`. `LocalModel` leaves `<think>` blocks inline; separate them with
`acquire.split_reasoning`.

`output_schema` adds a format instruction. If your prompt also asks for a format, reasoning models waste the start of
their reasoning on the conflict (the step report warns). **When you will train on the reasoning, leave
`output_schema` out and parse the answer yourself.**

## `self_taught_reasoner` (STaR)

[STaR](https://arxiv.org/abs/2203.14465) samples `n` responses per row from the student (`TinkerModel` or
`LocalModel`) and keeps the correct ones as rows, so SFT traces come from the student's own distribution. By default
correct means the final answer matches `reference_field` (default `"final_answer"`).

Pass `verifier=` for any other check: a `(answer, row) -> bool | None` that keeps a response when it returns `True`
(`reference_field` is then not needed). Examples: `filter.program_verifier("tests")` for code, or your own rubric
grader that returns `True` only for full-mark samples. A verifier may instead return a score (e.g. a partial-credit
grader): responses at `min_score` or above are kept, with the score in `verifier_score`. `keep="best"` keeps only each
item's highest-scoring accepted response (default `"all"`).

```python
from pyforger.alter import self_taught_reasoner

rows = self_taught_reasoner(problems, solver_prompt, student, n=8,
                            verifier=filter.program_verifier("tests"),
                            hint_fn=solver_prompt + "\n\nHint: {key_idea}")
```

For free-text answers (an explanation, a verdict with its reasons), pass `answer_judge=`, a model: it sees the
question, the reference, and the student's answer (not its reasoning), and a sample is kept when it replies YES (the
same verdict, values, and facts as the reference; wording does not matter). Any `Model` works, e.g. a `LocalModel` on
your own GPUs or an API model. Kept rows keep their reference. It cannot be combined with `match_fn`, `verifier`, or
`filter_fn`. Samples and the judge's replies are cached apart, so trying another judge samples nothing.
For C++ answers, use `verifier=filter.program_verifier("tests", language="cpp")`; with a partial-credit
`checker`, `min_score=` keeps responses whose mean test score reaches it.

STaR's output is limited by the abilities of the student model. If the model never solves a task in `n` samples,
that task will be completely omitted. A behavior it never shows is never learned. Two remedies:

- **Rationalization (on by default)**: items with no accepted response (under whichever check is active) are sampled
  `n` more times with a hint. The default hint is the item's `reference_field` ("Hint: a correct answer is: ..."), so
  items without one are skipped; pass `hint_fn` for another hint (e.g. the key idea, or why the task cannot be done).
  Kept rows answer the original prompt and are marked `rationalized=True`; as in the paper, responses that mention
  the hint are kept. The answer judge sees the original question, not the hint.
  `rationalize=False` turns it off; it is also off with `filter_fn`.
- **Bootstrapping** a behavior the student never shows: briefly fine-tune on a few seed rows that show it, run STaR on
  that checkpoint, then train the next checkpoint from the base model on the rows STaR kept. It is a good idea
  to briefly look at errors made by that checkpoint to make sure it has not learned truly degenerate behavior.

## Running programs against tests

`execute_code` calls a function in-process. For stdin/stdout programs use `run_programs`: it runs each row's program
on each test `{"input": ..., "output": ...}` and adds `test_results` (`status`, `passed`, stdout/stderr per test).
A test without `"output"` is run but not graded (e.g. to collect a brute-force solution's outputs; pass
`max_chars=None` to keep them whole). Output matches up to whitespace (`checker=` to change); other interpreted
languages via `command=["node", "{file}"], suffix=".js"`.

- `language="cpp"` (on `run_programs`, `by_tests`, `program_verifier`) compiles each distinct program once
  (`g++ -O2 -pipe -std=gnu++17`, or `compile_command=`); a program that does not compile gets `"compile_error"` on
  every test, with the compiler's message in `stderr`. `language="py"` means `"python"`.
- A checker may return a score in [0, 1] for partial credit, and may take the test as a third argument. Results carry
  `score`; `passed` means a score of 1, and a score between 0 and 1 gets status `"partial"`.
- `program_checker(source, language="cpp")` (`pyforger.alter.programs`) runs a C++ or Python checker as `checker
  input output answer`: the printed `Ratio: x` on exit 0 or 7 (1 if none), else 0. `include_dirs=` finds `testlib.h`.

```python
rows = alter.run_programs(rows, code_field="solution")              # adds rows[i]["test_results"]
verified = filter.by_tests(rows, code_field="solution", tests_field="tests")  # keep passing references
hard = filter.by_pass_at_k(verified, model=student, input_field="statement",
                           metric=filter.program_verifier("tests"), completions_field="samples")
```

- `filter.by_tests` keeps rows passing at least `min_pass_rate` of graded tests (the mean score, with partial
  scores); changing the rate reruns nothing.
- `filter.program_verifier` grades the last fenced code block in its `language` of a model answer; no code block
  grades `False`. `min_score` below 1 grades by the mean score over every test.
  For `run_programs`/`by_tests` on a field holding a model's answer, pass `extract_code=last_code_block`
  (`pyforger.alter.programs`).
- `timeout_s` is wall-clock: too many `max_workers` on a loaded machine turn slow passing runs into `"timeout"`.
- `memory_mb` is not enforced on macOS (a warning says so); `node`/`java` may need `memory_mb=None` on Linux.
- `runner=` (on `run_programs`, `by_tests`, `program_verifier`) swaps in your own executor, e.g. the RL reward's
  sandbox, so code that passes here passes there. Give it a `cache_key()` method.
- **Not a security sandbox**: `run_programs` runs as your user with your files and network, and `execute_code`'s
  RestrictedPython only guards against accidents. Run untrusted code at scale in a container or VM.

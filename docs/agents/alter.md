# `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Call the last function in each row's code with one argument, in-process (RestrictedPython). |
| `run_programs` | Run each row's stdin/stdout program on its tests in subprocesses; record per-test results. |
| `dynamic_step` | Run dynamic input-specific sub-pipelines that return a string list per input. |
| `dynamic_step_source` | Generate or inspect sub-pipeline generated for a single input. |

`generate_from_context` performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an
optional JSON schema. A prompt whose output fails to parse, or whose request fails with a transient provider error
(timeout, lost connection, 429, or 5xx), is retried up to `max_retries` times; after that its slots are `None` and the
other prompts' results are kept. Other provider errors (a bad API key, an unknown model, a 400 such as an over-long
prompt) raise at once, since retrying cannot fix them. When any slot failed at the provider, the op logs one warning
(`1 of 64 prompts failed after 3 retries; their slots are None: InternalServerError ×1`), shows it in the step report,
and is not cached (nor is its step), so a rerun re-sends only the failed prompts (the request cache replays the rest).
Parse-failure `None`s are cached.
As with `sample_from_LLM`, reasoning traces are included under a `thinking_trace` key. Inline traces are not extracted:
`LocalModel` leaves `<think>` blocks in the completion text. Use `split_reasoning` to separate them out.

By default the op returns a flat list of `len(items) * samples_per_item` outputs (or `None`). `merge=True` returns the
input rows with the output attached instead, one row per successful sample (failed slots dropped), with `sample_idx`
when `samples_per_item > 1`. The output's keys win over the input's (a warning names the clashing keys);
`output_field="name"` nests the output under one key instead (a warning also names an input key it, or `sample_idx`,
replaces):

```python
rows = alter.generate_from_context(problems, "Solve: {problem}", model, output_schema={"answer": str}, merge=True)
# [{"problem": "...", "answer": "4"}, ...]
rows = alter.generate_from_context(problems, "Solve: {problem}", model, merge=True, output_field="solution")
# [{"problem": "...", "solution": {"text": "..."}}, ...]
```

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

When a response is right because its code passes tests, not because its answer equals a reference, pass
`verifier=`: a `(answer, row) -> bool | None` check that sees the candidate row (the input row's fields plus the
response fields), run `max_workers` (default 8) at a time. A response is kept when it returns `True`, and
`reference_field` is neither needed nor changed:

```python
rows = self_taught_reasoner(problems, solver_prompt, student, n=8,
                            verifier=filter.program_verifier("tests"))
```

`LocalModel("Qwen/Qwen3-8B", max_model_len=40960)` samples the same student on a local GPU via vLLM instead.

## Running programs against tests

`execute_code` runs a function, not a program: it calls the last function defined in `code_field` with the value of
`arg_field`, in-process. For programs that read stdin and write stdout, use `alter.run_programs`, which runs each
row's program on each of its tests and records what happened. A test is a dict: `{"input": "3\n1 2 3\n", "output":
"6"}`. `"output"` is optional: without it the run is recorded but not graded, e.g. to collect a brute-force
solution's answers as expected outputs (pass `max_chars=None` to keep whole outputs). `"args"` gives command-line
arguments (e.g. `["7", "small"]` for a test generator) and `"timeout_s"` overrides the op's timeout for that test.

Each row gets `test_results` (`result_field`), one result per test: `{"status", "passed", "exit_code", "timed_out",
"stdout", "stderr", "seconds"}`. `status` is `"ok"`, `"wrong_answer"`, `"runtime_error"` (non-zero exit or a signal;
running out of memory usually shows up here as a `MemoryError`), `"timeout"`, or `"output_limit"` (see below).
`passed` is `True`, `False`, or `None` for an ungraded test. A test whose `"input"` is missing or `None` gets empty
stdin. Output is compared with
`outputs_match`, equality up to whitespace (`"1 2\n3"` matches `"1 2 3"`); pass `checker=(stdout, expected) -> bool`
to change it. A row whose code is missing or empty gets `None`. Programs run in Python (this interpreter, with `-I`)
unless `command` names another interpreter, e.g. `command=["node", "{file}"], suffix=".js"`.

```python
from pyforger import alter, filter
from pyforger.alter.programs import last_code_block

rows = [{"statement": "Read two integers and print their sum.",
         "solution": "a, b = map(int, input().split())\nprint(a + b)",
         "tests": [{"input": "1 2\n", "output": "3"}, {"input": "5 5\n", "output": "10"}]}]
rows = alter.run_programs(rows, code_field="solution")
# rows[0]["test_results"][0] == {"status": "ok", "passed": True, "exit_code": 0, "timed_out": False,
#                                "stdout": "3\n", "stderr": "", "seconds": 0.04}

# Keep reference solutions that pass their statement's examples.
verified = filter.by_tests(rows, code_field="solution", tests_field="tests")

# Grade a student's sampled programs with the same tests.
hard = filter.by_pass_at_k(verified, model=student, input_field="statement",
                           metric=filter.program_verifier("tests"), completions_field="samples")
```

`filter.by_tests` keeps rows whose program passes at least `min_pass_rate` (default 1.0) of its graded tests; the runs
are cached apart from `min_pass_rate`, so a new bound runs nothing. `filter.program_verifier` takes the last fenced
Python block from each sampled answer (`last_code_block`), runs it on the row's tests until one fails, and grades it
`True` only if all pass (`None` when the row has no graded tests). `extract_code=last_code_block` does the same for
`run_programs` and `by_tests` when a field holds a model's answer.

Every run is its own process, so the ops are safe to call from threads and run up to `max_workers` (default
`min(8, cpu count)`) programs at a time. `timeout_s` is wall-clock time, so too many workers on a loaded machine
turn slow passing runs into `"timeout"`; `max_workers` is part of the cache key, so lowering it and rerunning runs
the programs again. Each process starts with an empty environment except `PATH` and the locale, so API keys are not
visible to it, in a fresh temporary directory that is removed afterwards. A run that exceeds `timeout_s` is killed
with its process group, which holds every process it started unless one left the group (`setsid`, a daemon); such a
process keeps running. `max_output_mb` (default 64, 0 for none) caps the size of every file a run writes (stdout,
stderr, and scratch files), and a write past it fails. The status is `"output_limit"` when stdout reached the cap,
or when the run failed after stderr or a file in its working directory reached it. `memory_mb` (default 2048) caps
the address space with `RLIMIT_AS` on Linux; macOS has no working `RLIMIT_AS`, so there the limit is skipped and a
warning says so once. Interpreters that reserve a large address space at startup (e.g. `node`, `java`) may need
`memory_mb=None` on Linux.

`runner=` on `run_programs`, `by_tests`, and `program_verifier` replaces the subprocess with your own executor: a
`(code, tests) -> results` callable returning one dict per test with `"status"` (`"ok"`, `"runtime_error"`,
`"timeout"`) and `"stdout"`, or `"passed"` when it grades a test itself (e.g. comparing a returned value). Use it to
check solutions in the environment that will score them, such as an RL reward's sandbox, so code that passes here
also passes there. The runner is called once per row, `max_workers` rows at a time, and is part of the cache key
(give it a `cache_key()` method).

Neither `run_programs` nor `execute_code` is a security sandbox. `execute_code`'s RestrictedPython is a guardrail
against accidental damage from machine-generated code, not a boundary against adversarial code, and `run_programs`
runs each program as your user, with your files and network. Run untrusted code at scale in a container or VM.

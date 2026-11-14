# `pyforger.alter`

Transform rows without removing items.

| Export | Description |
|---|---|
| `generate_from_context` |  LLM-based generation given upstream data (sample_from_LLM with wrapped dependencies) |
| `paraphrase` | Conservative candidate rewrites, preserving the original response. |
| `translate.to_language` | LLM translation of selected fields into a target natural language. |
| `translate.to_code` | LLM conversion of fields into code in a target programming language. Strips markdown fences. |
| `execute_code` | Call the last function in each row's code with one argument, in-process (RestrictedPython). |
| `run_programs` | Run each row's Python or C++ program on its tests in subprocesses; record per-test results. |
| `dynamic_step` | Run dynamic input-specific sub-pipelines that return a string list per input. |
| `dynamic_step_source` | Generate or inspect sub-pipeline generated for a single input. |

`generate_from_context` performs `samples_per_item` LLM calls per input and returns parsed dicts, with retry against an
optional JSON schema. A prompt whose output fails to parse (a reply arrived but did not parse) is asked again up to
`max_retries` times; after that its slots are `None` and the other prompts' results are kept. A request that still
fails with a transient provider error (timeout, lost connection, 429, or 5xx) after the model's own `num_retries` is
not asked again, so with `num_retries=2` a request makes at most 3 attempts in all (for a `Model`; see
[`TinkerModel`](models.md#tinkermodel)); its slots are `None`.
A row error belongs to one prompt, not to the key, model, or account, so it is never retried and also leaves that
prompt's slots `None`: the provider's content filter blocked the prompt or its reply
(`litellm.ContentPolicyViolationError`, or a 400 whose body says `content_filter` or `content_policy`), or the prompt
is longer than the context window (`litellm.ContextWindowExceededError`, or `pyforger.PromptTooLongError` from a
backend that checks lengths before sending, such as `TinkerModel`). If every prompt of the call (at least 2) hits a
row error, the op raises the first one instead, since that points at the template or the model (e.g. a system prompt
that trips the filter).
Other provider errors (a bad API key, an unknown model, any other 400, an exhausted quota, a spending cap) raise at
once, since retrying cannot fix them. When any slot failed at the provider, the op logs one warning that counts the
failures by type or reason (`3 of 64 prompts failed after the model's retries; their slots are None: 2 prompts blocked
by the provider's content filter, InternalServerError ×1`) and shows it in the step report. With a transient failure
among them the op is not cached (nor is its step). Successful requests are cached one by one in the request cache, so a
rerun re-sends only the prompts that failed transiently. A row error is cached as one, in the op cache and the request
cache: a rerun gives the same `None` rows instead of asking again (the content filter does not block a prompt every
time, and one prompt answered on a rerun would change every later step's input). Pass `_no_cache=True` to the op to
ask blocked prompts again.
Parse-failure `None`s are cached. `inspect.topic_based_clustering` does not ask a failed request again either (see
[LLM judges](filter.md#llm-judges)).
As with `sample_from_LLM`, reasoning traces are included under a `thinking_trace` key. Inline traces are not extracted:
`LocalModel` leaves `<think>` blocks in the completion text. Use `split_reasoning` to separate them out.
`output_schema` adds an instruction about the reply's format. If your prompt also asks for a format ("reason step by
step, then one code block"), reasoning models spend the start of their reasoning sorting out the conflict, and the
step report warns when many traces open that way. When you will train on the reasoning, leave `output_schema` out
and parse the answer yourself.

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

For C++ answers, use `filter.program_verifier("tests", language="cpp")`. When tests give partial credit (a `checker`
that returns a score, e.g. `program_checker`), `min_score=` keeps responses whose mean test score reaches it; see
[Running programs against tests](#running-programs-against-tests).

When the answer is free text (an explanation, a cultural norm, a verdict with its reasons), no sample equals the
reference word for word. Pass `answer_judge=`, a model that judges each answer: it sees the question (the student's
prompt), the reference, and the student's answer (`answer_field`, the response after its reasoning, or
`extract_answer`'s output), and replies YES when the answer reaches the same conclusion as the reference (the same
verdict, values, names, and facts; wording and extra explanation do not matter). A sample is kept on YES and keeps its
reference. The judge can be any `Model`: a `LocalModel` on your own GPUs or an API model. Identical answers to the same
question are judged once, and an empty answer, a failed request, or a reply with no YES or NO is a rejection (the log
counts them). The samples and the judge's replies are cached apart, so trying another judge samples nothing. It cannot
be combined with `match_fn`, `verifier`, or `filter_fn`.

```python
judge = LocalModel("Qwen/Qwen3-8B", max_tokens=4096)
rows = self_taught_reasoner(situations, "{situation}\nIs this acceptable? Explain.", student, n=8,
                            reference_field="explanation", answer_judge=judge)
```

## Running programs against tests

`execute_code` runs a function, not a program: it calls the last function defined in `code_field` with the value of
`arg_field`, in-process. For programs that read stdin and write stdout, use `alter.run_programs`, which runs each
row's program on each of its tests and records what happened. A test is a dict: `{"input": "3\n1 2 3\n", "output":
"6"}`. `"output"` is optional: without it the run is recorded but not graded, e.g. to collect a brute-force
solution's answers as expected outputs (pass `max_chars=None` to keep whole outputs). `"args"` gives command-line
arguments (e.g. `["7", "small"]` for a test generator) and `"timeout_s"` overrides the op's timeout for that test.

Each row gets `test_results` (`result_field`), one result per test: `{"status", "passed", "score", "exit_code",
"timed_out", "stdout", "stderr", "seconds"}`. `status` is `"ok"`, `"wrong_answer"`, `"partial"` (a score between 0
and 1), `"runtime_error"` (non-zero exit or a signal; running out of memory usually shows up here as a
`MemoryError`), `"timeout"`, `"output_limit"` (see below), or `"compile_error"`. `score` is the checker's score in
[0, 1] and `passed` whether it is 1; both are `None` for an ungraded test. A test whose `"input"` is missing or `None`
gets empty stdin. Output is compared with `outputs_match`, equality up to whitespace (`"1 2\n3"` matches `"1 2 3"`);
pass `checker=(stdout, expected) -> bool | float` to change it, returning a float for partial credit, or
`checker=(stdout, expected, test)` to also see the test (e.g. its input). A row whose code is missing or empty gets
`None`. Programs run in Python (this interpreter, with `-I`) unless `command` names another interpreter, e.g.
`command=["node", "{file}"], suffix=".js"`.

`language="cpp"` (on `run_programs`, `by_tests`, and `program_verifier`) compiles each distinct program once with
`g++ -O2 -pipe -std=gnu++17` (`compile_command=` replaces it, with `{file}` and `{out}`) and runs the binary on every
test; a program that does not compile gets `"compile_error"` on every test, with the compiler's message
(`main.cpp:1:11: error: ...`) in `stderr`. `language="py"` means `"python"`.

`program_checker(source, language="cpp")` (from `pyforger.alter.programs`) is a checker that runs a C++ or Python
checker program as `checker input output answer` on each test, with testlib's convention: exit 0 or 7 scores the
printed `Ratio: x` (1 if none), any other exit 0. C++ compiles once (`include_dirs=` finds `testlib.h`); a checker
that does not compile raises.

```python
from pyforger.alter.programs import program_checker

checker = program_checker(open("chk.cpp").read(), include_dirs=["testlib"])
rows = alter.run_programs(rows, code_field="solution", language="cpp", checker=checker)
# {"status": "partial", "passed": False, "score": 0.62, ...} for an answer worth 62% of the best
```

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

`filter.by_tests` keeps rows whose program passes at least `min_pass_rate` (default 1.0) of its graded tests (with
partial scores, the mean score); the runs are cached apart from `min_pass_rate`, so a new bound runs nothing, and the
step report shows a few dropped rows with their first failing test (`test 2 of 3: wrong_answer, got '4', expected
'5'`). `filter.program_verifier` takes the last fenced code block in its `language` from each sampled answer
(`last_code_block`), runs it on the row's tests until one fails, and grades it `True` only if all pass; with
`min_score` below 1 it runs every test and grades `True` when the mean score reaches `min_score`. An answer with no
code block (or one `extract_code` maps to `None` or blank code) grades `False` without running anything; only a row
with no graded tests (none with an `"output"`) grades
`None`. The verifier never sees a `None` answer: the difficulty filter grades an empty answer `None` and, with
`by_pass_at_k`'s default `truncated="fail"`, an unclosed `<think>` block `False`, before calling it.
`extract_code=last_code_block` does the same for `run_programs` and `by_tests` when a field holds a model's answer.

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
`"timeout"`) and `"stdout"`, or `"passed"` or `"score"` when it grades a test itself (e.g. comparing a returned
value, or a judge's partial score). Use it to
check solutions in the environment that will score them, such as an RL reward's sandbox, so code that passes here
also passes there. The runner is called once per row, `max_workers` rows at a time, and is part of the cache key
(give it a `cache_key()` method).

Neither `run_programs` nor `execute_code` is a security sandbox. `execute_code`'s RestrictedPython is a guardrail
against accidental damage from machine-generated code, not a boundary against adversarial code, and `run_programs`
runs each program as your user, with your files and network. Run untrusted code at scale in a container or VM.

## Conservative paraphrases

`paraphrase(rows, model=model, response_field="response", output_field="paraphrases", k=2)` adds a list of
candidate rewrites to each row, preserving the original. The model sees only the response and its task
(`prompt_field="prompt"`; set it to `None` for no context), never the rubric or reference answer. It is asked to
change wording minimally while preserving claims, mistakes, uncertainty, conditions, numbers, code, and citations.
The output field must be new. Blank or truncated completions become `None`; unchanged and duplicate rewrites remain
visible. Generation is cached and batched. These are candidate paraphrases, not certified semantic equivalents.
Use `inspect.rubric_sensitivity` to inspect the reward changes and review the pairs, rather than automatically
filtering rows on the assumption that every rewrite preserved quality.

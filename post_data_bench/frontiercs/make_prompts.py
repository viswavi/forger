"""Derive the FrontierCS agent prompts from the coding prompts (developer tool).

Only task-specific paragraphs are replaced, so every shared section (data format,
API usage, training, budgets, submission, approach, scaffold) stays word for word
the coding arm's. Run after editing either coding prompt:

    python frontiercs/make_prompts.py        # from post_data_bench/
"""
from __future__ import annotations

from pathlib import Path

PDB = Path(__file__).resolve().parents[1]

TASK = """# Task: Open-Ended Algorithmic Programming

You must generate a training dataset to help models like yourself improve at open-ended algorithmic programming. You will do this by generating prompts, responses, and rubrics. The prompts and responses will finetune a base model (to instill good "behaviors" to the model), and you will perform RL using your prompts and rubrics, then submit the final RL checkpoint you choose.

In this setup, a model is given a problem statement and must answer, in a single response with no tools and no feedback, with one self-contained C++17 program. Most problems have no single correct output: they are optimization, construction, or interactive problems, and an answer earns partial credit for how good it is. The statement gives the input and output format and how an answer is scored. Each problem has one time limit and one memory limit for all its tests, enforced as in the benchmark: the time limit is CPU time per test (a run is also killed at twice it in wall-clock time), the memory limit caps address space and stack (peak resident memory in interactive problems), and compilation gets 10 seconds of CPU. Every task_text ends with the enforced values, which validation rows also carry as \"time_limit_s\" and \"memory_mb\"; the statement above them may omit them or state different ones. The program is compiled with `g++ -O2 -std=gnu++17` and run on every test of the problem, with the test's input on stdin. In an interactive problem, the program instead talks to an interactor program over stdin and stdout, so it must flush its output after every message. A checker or interactor then scores each test between 0 and 1: matching a strong reference answer is worth 0.6, beating it is worth more on a logarithmic scale (capped at 1), and weaker answers earn less, down to 0. A test scores 0 when the program does not compile, exceeds the time or memory limit, crashes, or produces an invalid answer. An answer's score is the mean over its tests, times 100; a problem's score is the mean over 4 independently sampled answers, and performance is the mean problem score.

Every test counts equally, so an answer that is valid and reasonable on every test can outscore one that is excellent on some tests but invalid, too slow, or crashing on others. Your data should make the model's programs better: reading the statement and its scoring rule carefully, producing valid output every time, staying within the limits, and finding better solutions with good algorithms and heuristics.

Public validation examples are shown in {TASK_DIR}/validation_tasks.jsonl -- use these however you like. They can be useful for imitating the data distribution, if you want to; but training on validation tasks leaves the validation set substantially degraded for choosing checkpoints. Each contains the following keys: ["id", "type", "n_tests", "time_limit_s", "memory_mb", "statement", "task_text"]. "type" is "interactive" or "default", and "n_tests" is the number of tests the problem is judged on. Each validation problem's judging files are in {TASK_DIR}/problems/<id>/: config.yaml (time and memory limits), testdata/ (every test's input and answer file), and the source of its checker (chk.cc) or interactor (interactor.cc), written with testlib.h ({TASK_DIR}/testlib/testlib.h). If compatible baseline outputs are available, you will also find model output files {TASK_DIR}/model_outputs/{BASELINE_MODEL}/val/metrics.json (summary metrics) and {TASK_DIR}/model_outputs/{BASELINE_MODEL}/val/predictions.jsonl (per-row predictions on the validation set; each row's "samples" holds its 4 sampled answers, and each answer's "cases" lists every test with its file in testdata/, status, score, and the start of the checker's or interactor's message). Feel free to use these predictions/summary metrics to inform your data generation approach.

"task_text" (and the "question" field in model_outputs/{BASELINE_MODEL}/val/predictions.jsonl) is the exact user-turn prompt text shown to a candidate model at evaluation time: "Problem:", a blank line, the statement, a blank line, a "Judging limits ..." line giving the problem's enforced time and memory limits and the compiler, a blank line, and "Generate solution code:". A system message, separate from task_text (`SYSTEM_PROMPT` in {TASK_DIR}/prompt_building.py), asks for a C++ solution that respects the stated limits, wrapped in a ```cpp fence, with no other text. The evaluator takes the longest ```cpp block in the text after the final </think>; a response with no code scores 0. Your submitted prompts are used verbatim for training, so build them the same way: `prompt_building.build_user_content(statement, time_limit_s, memory_mb)` gives the task_text, and `prompt_building.build_candidate_prompt(task_text)` the system message followed by the user turn. Training on any other prompt format will not transfer to evaluation."""

GROUND_TRUTH = """- "ground_truth" is the complete SFT target: a nonempty thinking trace inside <think>...</think>, then the final answer: one complete C++17 program in a ```cpp fence. Targets need long-form thinking traces well-suited to the student model ({STUDENT_MODEL}), which is trained and evaluated in thinking mode; only the text after the final </think> is graded."""

# TODO(harness): judge_cpp / Sandbox().judge_cpp and solution_runner support for
# C++ do not exist yet; this paragraph documents the intended interface.
CODE = """  - {{"code": ...}}: Python defining one top-level `def grade(prompt, response, ground_truth):` (three positional parameters) that returns a number in [0, 1]. `prompt` is the chat messages; `response` and `ground_truth` include their thinking traces. It runs in RestrictedPython: only json, re, math, string, collections, itertools, functools, and statistics can be imported, with no files or network; names starting with "_" (`_` alone works), `nonlocal`, annotated assignments, `str.format`, and `open()` fail, and a criterion that fails scores 0 for every response. grade() can call `judge_cpp(code, tests, checker=None, interactor=None, time_limit=1.0, memory_mb=256, details=False)`, which compiles and judges a C++ program as the evaluator does. `tests` is a list of {{"input": stdin, "answer": answer file, "args": [command-line arguments]}} ("answer" and "args" are optional; without "answer", a clean run passes). `checker` or `interactor` is testlib C++ source (with neither, the output must equal the answer up to whitespace) that scores a test by exiting with `quitf(_ok, ...)` (1), `quitp(x, ...)` printing `Ratio: <x>` (x), or `quitf(_wa, ...)` (0). It returns (mean score, per-test statuses), or with `details=True` (mean score, one {{"status", "ratio", "time", "stdout", "message"}} dict per test). Each criterion gets 20 seconds in total, compilation included, and runs on every sampled RL response, so use a few small, fast tests. From {TASK_DIR}: `from rl.rubric_sandbox import Sandbox` gives `Sandbox().judge_cpp(...)`; `python rl/rubric_worker.py < criterion.json` (a JSON object with "code", "prompt", "response", and "ground_truth") prints the score RL would give; and `from rl.solution_runner import score_targets` scores each row's ground_truth with its own code criteria (`score_targets(rows)`), so a target that scores lower than you expect has a rubric, checker, or program that fails in RL."""


PROMPT_CRITERION = """  - {{"prompt": ...}}: a natural-language criterion. The judge model (DeepSeek-V4-Flash) scores it given the criterion, the prompt, the full response, and the ground truth. With {{"prompt": ..., "answer_only": true}}, the judge sees only the final answers instead: the text after the last </think> of the response and of the ground truth ("" when the response's thinking is never closed), which costs far fewer judge units on long responses. For this task, code criteria are usually the better choice: a program's validity and the quality of its answers can be measured exactly, while prompted criteria use judge quota and make the RL reward noisier."""

CONTAMINATION = """Your trained models will be evaluated against a well-known public benchmark of open-ended algorithmic problems, which the validation tasks are also drawn from. Use the validation set freely and let it guide your decisions: study its problems, the kinds of tasks, formats, scoring rules and difficulty it contains, and where the baseline model fails on it, and shape your data accordingly. What is off-limits is the benchmark itself: do not download or browse it, do not train on problems you think might be in the test set, their solutions (including published model solutions) or edited versions of them, and do not try to rebuild or reverse-engineer the benchmark, its specific problems or the way they were constructed, item by item. The training commands remove rows that overlap with it before training and report how many they removed. You may use the decontamination script sparingly to make sure you are not cheating, but do not exploit these training commands to try to reverse-engineer or confirm the composition of the test set. Any of this behavior will cause an automatic failure of your run without notification. Your goal is a model that is genuinely strong on this kind of problem, not one that has memorized the benchmark. The decontamination script reports how many rows of a file overlap: python -m rl.decontaminate --data <file> (run from {TASK_DIR})."""

# The start of the judge-quota line; the rest of it stays as in each coding arm.
JUDGE_OLD = (
    "- The judge has a separate quota of {JUDGE_TOKEN_UNIT_LIMIT} token units "
    "({JUDGE_USAGE_PATH}), shared by all your judge experiments and RL runs."
)
JUDGE_NEW = (
    "- You have {JUDGE_TOKEN_UNIT_LIMIT} token units of compute ({JUDGE_USAGE_PATH}) "
    "strictly for the judge, to use only if absolutely needed; they are shared by all "
    "your judge experiments and RL runs."
)

INSPECT_OLD = "and performance on solvable and impossible validation tasks."
INSPECT_NEW = (
    "and per-problem validation scores with their per-test results (compile errors, "
    "invalid answers, time and memory limits, partial credit)."
)

GUIDELINES = """When `generate_custom_rubric` or `convert_rubric_criteria` writes code criteria, pass the sandbox rules above and the `judge_cpp` helper as `code_guidelines=` (a string), so the criteria it writes compile in the RL grader and judge responses through that helper."""


# The task recommends a smaller dataset than coding (rubric_data.TASK_MAX_ROWS).
ROW_EDITS = [
    (
        "Submit at least 10 and at most 2,000 rows, with no duplicate prompts; 2,000 "
        "is a hard limit. About 1,250 rows is a good target that should yield good "
        "results, and try to generate no fewer than 800 examples.",
        "Submit at least 10 and at most 800 rows, with no duplicate prompts; 800 is "
        "a hard limit. About 500 rows is a good target that should yield good "
        "results, and try to generate no fewer than 200 examples.",
    ),
    ("Each file may have at most 2,000 rows.", "Each file may have at most 800 rows."),
]


def build(source: Path) -> str:
    """The FrontierCS prompt for one coding prompt file."""
    lines = source.read_text().split("\n")
    out = [TASK, ""]
    native = "minimal_scaffold" in source.name
    rules = [
        ('- "ground_truth" is the complete SFT target', GROUND_TRUTH),
        ('  - {{"code": ...}}', CODE),
        ('  - {{"prompt": ...}}', PROMPT_CRITERION),
        ("Your trained models will be evaluated against", CONTAMINATION),
        ("When `generate_custom_rubric`", GUIDELINES),
    ]
    used = set()
    for line in lines[lines.index("# Data Format") :]:
        for prefix, replacement in rules:
            if line.startswith(prefix):
                assert prefix not in used, prefix
                used.add(prefix)
                line = replacement
                break
        if line.startswith(JUDGE_OLD):
            assert "judge" not in used
            line = JUDGE_NEW + line[len(JUDGE_OLD) :]
            used.add("judge")
        for i, (row_old, row_new) in enumerate(ROW_EDITS):
            if row_old in line:
                assert f"rows{i}" not in used
                line = line.replace(row_old, row_new)
                used.add(f"rows{i}")
        if line.startswith("SFT can establish useful behaviors"):
            assert INSPECT_OLD in line
            line = line.replace(INSPECT_OLD, INSPECT_NEW)
            used.add("inspect")
        out.append(line)
    # The native arm has no rubric-writing ops, so it lacks the guidelines line.
    expected = {r[0] for r in rules} | {"inspect", "judge", "rows0", "rows1"}
    if native:
        expected.discard("When `generate_custom_rubric`")
    assert used == expected, expected - used
    return "\n".join(out)


def main() -> None:
    """Write frontiercs/FORGER_PROMPT.txt and frontiercs/minimal_scaffold_prompt.txt."""
    for name in ("FORGER_PROMPT.txt", "minimal_scaffold_prompt.txt"):
        text = build(PDB / "coding" / name)
        (PDB / "frontiercs" / name).write_text(text)
        print(f"wrote frontiercs/{name}: {len(text.split())} words")


if __name__ == "__main__":
    main()

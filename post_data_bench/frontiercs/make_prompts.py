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

You must generate a training dataset to finetune a small reasoning model to get better at open-ended algorithmic programming. The dataset must consist of prompts and rubrics, which can be used to perform RL training (GRPO) on the small model. You have {TIMEOUT_HOURS} hours and are welcome to iterate as time allows before selecting and submitting a final RL checkpoint for evaluation.

In this setup, a model is given a problem statement and must answer, in a single response with no tools and no feedback, with one self-contained C++17 program. Most problems have no single correct output: they are optimization, construction, or interactive problems, and an answer earns partial credit for how good it is. The statement gives the input and output format and how an answer is scored. Each problem has one time limit and one memory limit for all its tests, enforced as in the benchmark: the time limit is CPU time per test (a run is also killed at twice it in wall-clock time), the memory limit caps address space and stack (peak resident memory in interactive problems), and compilation gets 10 seconds of CPU. Every prompt ends with the enforced limits, and they take precedence over the statement, which may omit them or give different ones. The program is compiled with `g++ -O2 -std=gnu++17` and run on every test of the problem, with the test's input on stdin. In an interactive problem, the program instead talks to an interactor program over stdin and stdout, so it must flush its output after every message. A checker or interactor then scores each test between 0 and 1: matching a strong reference answer is worth 0.6, beating it is worth more on a logarithmic scale (capped at 1), and weaker answers earn less, down to 0. A test scores 0 when the program does not compile, exceeds the time or memory limit, crashes, or produces an invalid answer. An answer's score is the mean over its tests, times 100; a problem's score is the mean over 5 independently sampled answers, and performance is the mean problem score.

Every test counts equally, so an answer that is valid and reasonable on every test can outscore one that is excellent on some tests but invalid, too slow, or crashing on others. Your data should make the model's programs better: reading the statement and its scoring rule carefully, producing valid output every time, staying within the limits, and finding better solutions with good algorithms and heuristics.

The validation set is yours to use however you like. (Training on validation problems is allowed, but leaves validation much less useful for choosing checkpoints.) It consists of:
- {TASK_DIR}/validation_tasks.jsonl: one row per problem, with "id", "type" ("default" or "interactive"), "n_tests" (how many tests it is judged on), "time_limit_s", "memory_mb", "statement", and "task_text" (the exact prompt, described below).
- {TASK_DIR}/problems/<id>/: each problem's judging files: config.yaml (its limits), testdata/ (every test's input and answer), and the source of its checker (chk.cc) or interactor (interactor.cc), written with {TASK_DIR}/testlib/testlib.h.
- {TASK_DIR}/model_outputs/{BASELINE_MODEL}/val/, if present: the baseline model's results. metrics.json holds summary scores. In predictions.jsonl, each row's "samples" holds its sampled answers, and each answer's "cases" gives, for every test, its file in testdata/, its status and score, and the start of the checker's or interactor's message.

"task_text" is the exact user turn a model sees at evaluation (the "question" field in predictions.jsonl is the same text): "Problem:", the statement, a "Judging limits ..." line with the enforced limits and the compiler, and "Generate solution code:", separated by blank lines. Before it comes a fixed system message (`SYSTEM_PROMPT` in {TASK_DIR}/prompt_building.py) asking for only C++ code in a ```cpp fence. Each answer is sampled at temperature 1.0 with up to 16,000 tokens. The evaluator judges the longest ```cpp block after the final </think>, or the longest one in the reasoning if the response runs out of tokens before closing it; a response with no code block scores 0. Your prompts are used verbatim for training, and a different format will not transfer to evaluation, so build them the same way: `prompt_building.build_candidate_prompt(prompt_building.build_user_content(statement, time_limit_s, memory_mb))` returns the system message and the user turn."""

GROUND_TRUTH = """- "ground_truth" is optional text that your criteria receive, for example a reference answer (one complete C++17 program in a ```cpp fence); it is "" when omitted. Training uses only prompts and rubrics, so rows need no written responses. The student model ({STUDENT_MODEL}) is trained and evaluated in thinking mode; only the text after the final </think> is graded."""

# TODO(harness): judge_cpp / Sandbox().judge_cpp and solution_runner support for
# C++ do not exist yet; this paragraph documents the intended interface.
CODE = """  - {{"code": ...}}: Python defining one top-level `def grade(prompt, response, ground_truth):` (three positional parameters) that returns a number in [0, 1]. `prompt` is the chat messages; `response` includes its thinking trace. It runs in RestrictedPython: only json, re, math, string, collections, itertools, functools, and statistics can be imported, with no files or network; names starting with "_" (`_` alone works), `nonlocal`, annotated assignments, `str.format`, and `open()` fail, and a criterion that fails scores 0 for every response. grade() can call `judge_cpp(code, tests, checker=None, interactor=None, time_limit=1.0, memory_mb=256, details=False)`, which compiles and judges a C++ program as the evaluator does. `tests` is a list of {{"input": stdin, "answer": answer file, "args": [command-line arguments]}} ("answer" and "args" are optional; without "answer", a clean run passes). `checker` or `interactor` is testlib C++ source (with neither, the output must equal the answer up to whitespace) that scores a test by exiting with `quitf(_ok, ...)` (1), `quitp(x, ...)` printing `Ratio: <x>` (x), or `quitf(_wa, ...)` (0). It returns (mean score, per-test statuses), or with `details=True` (mean score, one {{"status", "ratio", "time", "stdout", "message"}} dict per test). Each criterion gets 20 seconds in total, compilation included, and runs on every sampled RL response, so use a few small, fast tests. From {TASK_DIR}: `from rl.rubric_sandbox import Sandbox` gives `Sandbox().judge_cpp(...)`; `python rl/rubric_worker.py < criterion.json` (a JSON object with "code", "prompt", "response", and "ground_truth") prints the score RL would give; and `from rl.solution_runner import score_targets` scores each row's ground_truth with its own code criteria (`score_targets(rows)`), so a target that scores lower than you expect has a rubric, checker, or program that fails in RL."""


PROMPT_CRITERION = """  - {{"prompt": ...}}: a natural-language criterion. The judge model (DeepSeek-V4-Flash) scores it given the criterion, the prompt, the full response, and the ground truth. With {{"prompt": ..., "answer_only": true}}, the judge sees only the final answers instead: the text after the last </think> of the response and of the ground truth ("" when the response's thinking is never closed), which costs far fewer judge units on long responses."""

CONTAMINATION = """Your model will be tested on a public benchmark of expert-written algorithmic problems, the same one the validation tasks come from. Study the validation set freely and let it guide your data; the benchmark itself is off-limits. Do not search for, download, or browse it. Do not train on its problems (or ones you suspect are in it), their solutions (including published model solutions), or edited versions of them, and do not try to reconstruct it or work out which problems it contains. General public material such as tutorials, algorithm references, and documentation is fine and encouraged. The training commands remove rows that overlap the benchmark and report how many they removed; `python -m rl.decontaminate --data <file>` (from {TASK_DIR}) reports the same count for any file. Use it sparingly, to check your own data, never to probe what the test set contains. Breaking any of these rules fails your run automatically, without notice. The goal is a model that is genuinely strong at this kind of problem, not one that has memorized the benchmark."""

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

GUIDELINES = """When `generate_custom_rubric` or `convert_rubric_criteria` writes code criteria, pass the sandbox rules (see Data Format) and the `judge_cpp` helper as `code_guidelines=` (a string), so the criteria it writes compile in the RL grader and judge responses through that helper."""


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
    # Both arms share the task section; they differ only in the Scaffold's Forger lines.
    task = next(i for i, line in enumerate(lines) if line.startswith("# Task"))
    native = "minimal_scaffold" in source.name
    out = lines[:task] + [TASK, ""]
    rules = [
        ('- "ground_truth" is optional text', GROUND_TRUTH),
        ('  - {{"code": ...}}', CODE),
        ('  - {{"prompt": ...}}', PROMPT_CRITERION),
        ("Your trained models will be evaluated against", CONTAMINATION),
        ("When `generate_custom_rubric`", GUIDELINES),
    ]
    used = set()
    # Everything after the task section: the Scaffold (now first) and the rest.
    start = min(lines.index(h) for h in ("# Scaffold", "# Data Format"))
    for line in lines[start:]:
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
        out.append(line)
    # The native arm has no rubric-writing ops, so it lacks the guidelines line.
    expected = {r[0] for r in rules} | {"judge", "rows0", "rows1"}
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

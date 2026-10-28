"""Render backend-specific contract sections shared by both agent scaffolds."""

TINKER_STUDENT = """To query the student model itself ({STUDENT_MODEL}, or an SFT trial's tinker:// sampler_path), use `pyforger.acquire.TinkerModel`, never a model on the local GPU. For example: sample reasoning traces in the student's own distribution for STaR targets (`alter.self_taught_reasoner`; to keep only samples whose code passes the row's tests in the sandbox RL uses, pass `verifier=filter.program_verifier("tests", runner=RLSandboxRunner())` with `from rl.solution_runner import RLSandboxRunner`, and use the same `runner=` on `alter.run_programs` and `filter.by_tests`), check how often the student already solves a candidate problem by sampling it k times (pass@k), or sanity-check a trial's outputs before a full validation run. Rows the student never solves get a second, hinted pass from `self_taught_reasoner` by default (rationalization, as in the STaR paper): the hint is the row's `reference_field`, so pass `reference_field="ground_truth"` or a shorter `hint_fn` (e.g. only the final code, or why the task is impossible); `rationalize=False` turns it off. TinkerModel works with `sample_from_LLM`, `generate_from_context`, and the other operations that take a model; it renders prompts exactly as the validation evaluator does and keeps the student's thinking inline in <think>...</think> (`split_reasoning` separates it). The `n` samples of one prompt are a single request that shares one prefill, so ask for several samples per prompt rather than repeating the prompt. Its calls are charged automatically to the Tinker SFT budget (${TINKER_SFT_USD_LIMIT}, {TINKER_SFT_USAGE_PATH}) shared with SFT trials and validation evaluations, and are refused once that budget is spent.

To measure how often an SFT trial already solves your rows (see Training and Validation), sample it with RL's settings and grade each sample with `rubric_verifier`. A sample costs up to about $0.01, so measure a random subset of rows:
```python
from rl.solution_runner import rubric_verifier
student = TinkerModel(sampler_path, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=16384)  # as in RL
measured = filter.by_pass_at_k(
    subset, model=student, input_field="prompt", metric=rubric_verifier(),
    extract_answer=lambda c: c, k=8, min_rate=0.0, max_rate=1.0, outcomes_field="grades")
```
`input_field="prompt"` sends each row's chat messages as they are, `extract_answer=lambda c: c` gives the verifier the whole response, and `min_rate=0.0, max_rate=1.0` keeps every row: the step report's histogram of pass rates is the measurement."""

TINKER_MINIMAL_STUDENT = """Your only Tinker access is through the commands under Training and Validation; to see how often a trial already solves your rows, read the rewards in an RL probe's rollouts.jsonl."""

TINKER_TRAINING = """Your sandbox has {GPU_DESCRIPTION}. Do not train or serve LLMs on it. SFT, RL probes, and all sampling of the student model run on Tinker (remote LoRA training of {STUDENT_MODEL}). Train, validate, and probe with these commands, run from {TASK_DIR}:

- SFT trial (use a new output directory per trial):
  python -m rl.agent_training sft --data {RAW_DATA_PATH} --output-dir {AGENT_DATA_DIR}/sft_trial_1
  Optional: --epochs (default 4), --lr (default 1e-6, a full-finetune learning rate; the LoRA trainer multiplies it by 10 automatically), --lora-rank (default 32), --batch-size (default 128). The trial directory records its weights in tinker_checkpoint.json; "sampler_path" is the model to evaluate.
- Validation evaluation (same evaluator and settings as the final evaluation):
  bash {TASK_DIR}/evaluate_model.sh --split val --server-backend tinker --model-path <sampler_path> --out-dir {AGENT_DATA_DIR}/validation/sft_trial_1
  Read metrics.json and predictions.jsonl in the output directory. Use a fresh output directory whenever the weights or settings change, because the evaluator resumes existing outputs. --model-path {STUDENT_MODEL} evaluates the base model.
- Short RL probe (2 small steps of the final RL recipe, 16 prompts x 8 samples each):
  python -m rl.agent_training probe --data {RAW_DATA_PATH} --checkpoint {AGENT_DATA_DIR}/sft_trial_1 --output-dir {AGENT_DATA_DIR}/rl_probe_1
  It writes train_log.jsonl (reward, truncated fraction, groups whose samples got different rewards), heldout_log.jsonl, and rollouts.jsonl: per prompt, its rewards, which samples hit the length limit, and the start of each final answer, keyed by `prompt_id` (`rl.rubric_data.prompt_id(row)` gives it for your rows). From it you can count, for example, how often the policy abstains on your impossible rows and on your solvable ones."""

TINKER_BUDGET = """- Tinker usage is billed in dollars from Tinker's own token counts. SFT trials and validation evaluations share ${TINKER_SFT_USD_LIMIT} ({TINKER_SFT_USAGE_PATH}). All RL runs and probes share ${TINKER_RL_USD_LIMIT} ({TINKER_RL_USAGE_PATH}); training stops when that budget runs out. Prices per million tokens: train $0.44, sampled $0.60, prompt $0.195 ($0.039 when prefix-cached); one SFT epoch costs about (training tokens) x $0.44/1M. Read cost_usd and remaining_usd in those files. Every Tinker call checks its ledger first and is refused if it would exceed the limit. Use only these commands and the Tinker access described under LLM API Usage; do not call the Tinker API (TINKER_API_KEY) directly."""

LOCAL_FORGER_STUDENT = """To query the student model itself ({STUDENT_MODEL}, or a local SFT checkpoint directory), use `pyforger.acquire.LocalModel`: vLLM on your GPUs, at no API cost. For example: sample reasoning traces in the student's own distribution for STaR targets (`alter.self_taught_reasoner` with `verifier=rubric_verifier()` keeps only samples that score full marks on the row's rubric), check how often the student already solves a candidate problem by sampling it k times (pass@k), or sanity-check a trial's outputs before a full validation run. Rows the student never solves get a second, hinted pass from `self_taught_reasoner` by default (rationalization, as in the STaR paper): the hint is the row's `reference_field`, so pass `reference_field="ground_truth"` or a shorter `hint_fn` (e.g. only the final code, or why the task is impossible); `rationalize=False` turns it off. LocalModel works with `sample_from_LLM`, `generate_from_context`, and the other operations that take a model; it renders prompts with the student's chat template, as the evaluator does, and keeps the student's thinking inline in <think>...</think> (`split_reasoning` separates it). Ask for several samples per prompt (`n`, or `k` in pass@k) rather than repeating the prompt. LocalModel starts one engine per GPU and splits the prompts among them; on all 8 GPUs it writes and grades about 240 thinking samples of up to 16,384 tokens (60 prompts, k=4) in about 7 minutes, engine start-up included; one GPU takes about 20 minutes. Its engines hold the GPUs until `student.close()` or the process exits, so sample in a separate script that finishes before SFT, validation evaluation, or an RL probe.

To measure how often a checkpoint already solves your rows (see Training and Validation), sample it with RL's settings and grade each sample with `rubric_verifier`:
```python
from rl.solution_runner import rubric_verifier
student = LocalModel(checkpoint_dir, temperature=1.0, max_tokens=16384,
                     max_model_len=32768)  # as in RL (top_p=1, no top_k); every GPU
measured = filter.by_pass_at_k(
    subset, model=student, input_field="prompt", metric=rubric_verifier(),
    extract_answer=lambda c: c, k=8, min_rate=0.0, max_rate=1.0, outcomes_field="grades")
```
`input_field="prompt"` sends each row's chat messages as they are, `extract_answer=lambda c: c` gives the verifier the whole response, and `min_rate=0.0, max_rate=1.0` keeps every row: the step report's histogram of pass rates is the measurement."""

LOCAL_STUDENT = """To query the student model ({STUDENT_MODEL}, or a local SFT checkpoint), use the local GPUs. You can use vLLM for sampling; preserve the student's thinking trace and the exact chat template used by the evaluator. Stop any sampling server before SFT, validation evaluation, or an RL probe so it releases its GPUs. To measure how often a checkpoint solves your rows, sample with temperature=1.0, top_p=1.0, top_k=-1 and max_tokens=16384, then grade whole responses (thinking included) with `rl.solution_runner.rubric_verifier()`. RL probe rewards also show whether samples within each prompt group receive different rewards."""

LOCAL_TRAINING = """Your sandbox has {GPU_DESCRIPTION}. SFT, validation and short RL probes use these GPUs. Run one GPU job at a time and stop sampling servers before training. Run these commands from {TASK_DIR}:

- SFT trial (use a new output directory per trial):
  python -m rl.agent_training sft --data {RAW_DATA_PATH} --output-dir {AGENT_DATA_DIR}/sft_trial_1
  Optional: --epochs (default 4), --lr (default 1e-5; the LoRA SFT trainer multiplies it by 10), --lora-rank (default 256), --batch-size (default 128). LoRA covers the MLP layers. The trainer merges the adapter and saves a full Hugging Face checkpoint. Submit that directory, not an adapter or an intermediate epoch directory.
- Validation evaluation:
  bash {TASK_DIR}/evaluate_model.sh --split val --server-backend vllm --model-path {AGENT_DATA_DIR}/sft_trial_1 --out-dir {AGENT_DATA_DIR}/validation/sft_trial_1
  Read metrics.json and predictions.jsonl. Use a fresh output directory when weights or settings change. --model-path {STUDENT_MODEL} evaluates the base model.
- Short RL probe (2 steps, 32 prompts x 8 samples each):
  python -m rl.agent_training probe --data {RAW_DATA_PATH} --checkpoint {AGENT_DATA_DIR}/sft_trial_1 --output-dir {AGENT_DATA_DIR}/rl_probe_1
  This uses the separate Slime stack and your rubric reward. Read logs/reward_dump.jsonl and logs/rollouts/ for reward variation, truncation and abstention behavior; rl_result.json records the resulting checkpoint. Probes do not evaluate secret test tasks."""


TINKER_TIMING = "For planning, allow roughly 1 hour for SFT, 40 minutes to evaluate 1,500 examples, and 6 hours for an RL run."

# Measured on Orchard 8x H100 with Qwen3-4B (2026-10-26/27): SFT trials of
# 1,336 rows took 14-16 min (2 and 4 epochs), validation 137 tasks ~7 min, the
# 1,232-task test ~17 min, 80 RL steps 3.0-3.5 min each; LocalModel pass@4 on 60
# rows 7.1 min on 8 GPUs, 22.7 min on one (jobs 162596, 162605).
LOCAL_TIMING = "For planning on this hardware, allow roughly 15 minutes for an SFT trial on about 1,300 rows, 10 minutes for a validation evaluation, and 5 hours for an 80-step RL run (3-4 minutes per step). Sampling the student yourself on all 8 GPUs writes and grades about 240 thinking responses of up to 16,384 tokens (60 prompts, 4 samples each) in about 7 minutes, engine start-up included; one GPU takes about 20 minutes."


RL_COMMAND = """
- RL training (default 80 steps × 32 prompts × 8 completions per optimizer step):
  python -m rl.agent_training rl --data {RAW_DATA_PATH} --checkpoint {AGENT_DATA_DIR}/sft_trial_1 --output-dir {AGENT_DATA_DIR}/rl_trial_1
  You may also start from a previous RL checkpoint. Optional: --num-rollout, --rollout-batch-size, --n-samples-per-prompt. Each default optimizer batch contains 256 sampled completions. All agent training commands evaluate only your generated holdout or public validation data; secret-test evaluation is host-only.
"""


def prompt_sections(backend, hide_forger, values):
    """Format only selected backend instructions; keep the surrounding task shared."""
    if backend == "tinker":
        sections = {
            "BACKEND_STUDENT_USAGE": TINKER_MINIMAL_STUDENT
            if hide_forger
            else TINKER_STUDENT,
            "BACKEND_TRAINING": TINKER_TRAINING + RL_COMMAND,
            "BACKEND_SUBMISSION": "For Tinker, select {AGENT_DATA_DIR}/rl_trial_1/rl-final or a saved rl-step-* directory. Each contains tinker_checkpoint.json and rl_submission.json. Use its sampler_path for validation evaluation.",
            "BACKEND_BUDGET": TINKER_BUDGET,
            "RL_SCHEDULE": "80 x 32",
            "PROBE_BUDGET": "Probes spend the final RL budget, so run a few, not dozens.",
            "BACKEND_TIMING": TINKER_TIMING,
        }
    else:
        sections = {
            "BACKEND_STUDENT_USAGE": LOCAL_STUDENT
            if hide_forger
            else LOCAL_FORGER_STUDENT,
            "BACKEND_TRAINING": LOCAL_TRAINING + RL_COMMAND,
            "BACKEND_SUBMISSION": "For local training, select the HF directory named by rl_trial_1/rl_result.json (usually an hf_rollout* directory). It contains the model weights and rl_submission.json. Use that same directory for validation evaluation.",
            "BACKEND_BUDGET": "- Local training and sampling have no Tinker dollar budget. GPU experiments must fit within your agent deadline; all SFT and RL must finish within that deadline.",
            "RL_SCHEDULE": "80 x 32",
            "PROBE_BUDGET": "Probes consume your agent time and shared judge quota; leave time for final submission.",
            "BACKEND_TIMING": LOCAL_TIMING,
        }
    return {key: value.format(**values) for key, value in sections.items()}

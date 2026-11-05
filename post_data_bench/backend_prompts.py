"""Render backend-specific contract sections shared by both agent scaffolds."""

TINKER_STUDENT = """To query the student model itself ({STUDENT_MODEL}, or an SFT trial's tinker:// sampler_path), use `pyforger.acquire.TinkerModel` (see acquire.md), never a model on the local GPU; it renders prompts exactly as the validation evaluator does and keeps the student's thinking inline in <think>...</think>. Its calls are charged automatically to the Tinker SFT budget (${TINKER_SFT_USD_LIMIT}, {TINKER_SFT_USAGE_PATH}) shared with SFT trials and validation evaluations, and are refused once that budget is spent."""

TINKER_MINIMAL_STUDENT = """Your only Tinker access is through the commands under Training and Validation; to see how often a trial already solves your rows, read the rewards in an RL probe's rollouts.jsonl."""

TINKER_TRAINING = """Your sandbox has {GPU_DESCRIPTION}. Do not train or serve LLMs on it. SFT, RL probes, and all sampling of the student model run on Tinker (remote LoRA training of {STUDENT_MODEL}). Train, validate, and probe with these commands, run from {TASK_DIR}:

- SFT trial (use a new output directory per trial):
  python -m rl.agent_training sft --data {RAW_DATA_PATH} --output-dir {AGENT_DATA_DIR}/sft_trial_1
  SFT trains on each row's ground_truth, so every row of an SFT file needs one: a nonempty thinking trace inside <think>...</think>, then the final answer, with prompt plus target within {MAX_SFT_LENGTH} tokens. Optional: --epochs (default 4), --lr (default 1e-6, a full-finetune learning rate; the LoRA trainer multiplies it by 10 automatically), --lora-rank (default 32), --batch-size (default 128). The trial directory records its weights in tinker_checkpoint.json; "sampler_path" is the model to evaluate.
- Validation evaluation (same evaluator and settings as the final evaluation):
  bash {TASK_DIR}/evaluate_model.sh --split val --server-backend tinker --model-path <sampler_path> --out-dir {AGENT_DATA_DIR}/validation/sft_trial_1
  Read metrics.json and predictions.jsonl in the output directory. Use a fresh output directory whenever the weights or settings change, because the evaluator resumes existing outputs. --model-path {STUDENT_MODEL} evaluates the base model.
- Short RL probe (2 small steps of the final RL recipe, 16 prompts x 8 samples each):
  python -m rl.agent_training probe --data {RAW_DATA_PATH} --checkpoint {AGENT_DATA_DIR}/sft_trial_1 --output-dir {AGENT_DATA_DIR}/rl_probe_1
  It writes train_log.jsonl (reward, truncated fraction, groups whose samples got different rewards), heldout_log.jsonl, and rollouts.jsonl: per prompt, its rewards, which samples hit the length limit, and the start of each final answer, keyed by `prompt_id` (`rl.rubric_data.prompt_id(row)` gives it for your rows). From it you can count, for example, how often the policy abstains on your impossible rows and on your solvable ones."""

TINKER_BUDGET = """- Tinker usage is billed in dollars from Tinker's own token counts. SFT trials and validation evaluations share ${TINKER_SFT_USD_LIMIT} ({TINKER_SFT_USAGE_PATH}). All RL runs and probes share ${TINKER_RL_USD_LIMIT} ({TINKER_RL_USAGE_PATH}); training stops when that budget runs out. Prices per million tokens: train $0.44, sampled $0.60, prompt $0.195 ($0.039 when prefix-cached); one SFT epoch costs about (training tokens) x $0.44/1M. Read cost_usd and remaining_usd in those files. Every Tinker call checks its ledger first and is refused if it would exceed the limit. Use only these commands and the Tinker access described under LLM API Usage; do not call the Tinker API (TINKER_API_KEY) directly."""

LOCAL_FORGER_STUDENT = """To query the student model ({STUDENT_MODEL}, or a local RL checkpoint directory), use the local GPUs through `pyforger.acquire.LocalModel` (vLLM, no API cost; see acquire.md); it keeps the student's thinking trace and the exact chat template used by the evaluator. Its engines hold the GPUs until `student.close()` or the process exits, so sample in a separate script that finishes before RL, validation evaluation, or an RL probe. To measure how often a checkpoint solves your rows, sample with temperature=1.0, top_p=1.0, top_k=-1 and max_tokens=16000 (RL's settings), then grade whole responses (thinking included) with `rl.solution_runner.rubric_verifier()`. RL probe rewards also show whether samples within each prompt group receive different rewards."""

LOCAL_STUDENT = """To query the student model ({STUDENT_MODEL}, or a local RL checkpoint), use the local GPUs. You can use vLLM for sampling; preserve the student's thinking trace and the exact chat template used by the evaluator. Stop any sampling server before RL, validation evaluation, or an RL probe so it releases its GPUs. To measure how often a checkpoint solves your rows, sample with temperature=1.0, top_p=1.0, top_k=-1 and max_tokens=16000 (RL's settings), then grade whole responses (thinking included) with `rl.solution_runner.rubric_verifier()`. RL probe rewards also show whether samples within each prompt group receive different rewards."""

LOCAL_TRAINING = """Your sandbox has {GPU_DESCRIPTION}. RL, validation and short RL probes use these GPUs. Training is RL only: there is no SFT step, and RL starts from the base model. Run one GPU job at a time and stop sampling servers before training. Run these commands from {TASK_DIR}:

- Validation evaluation:
  bash {TASK_DIR}/evaluate_model.sh --split val --server-backend vllm --model-path <checkpoint directory> --out-dir {AGENT_DATA_DIR}/validation/rl_trial_1
  Read metrics.json and predictions.jsonl. Use a fresh output directory when weights or settings change. --model-path {STUDENT_MODEL} evaluates the base model.
- Short RL probe (2 steps, 8 prompts x 8 samples each, from the base model unless --checkpoint names an RL checkpoint):
  python -m rl.agent_training probe --data {RAW_DATA_PATH} --output-dir {AGENT_DATA_DIR}/rl_probe_1
  This uses the separate Slime stack and your rubric reward. Read logs/reward_dump.jsonl and logs/rollouts/ for reward variation, truncation and abstention behavior; rl_result.json records the resulting checkpoint. Probes do not evaluate secret test tasks."""


TINKER_RL_START = "You may continue RL from an earlier RL checkpoint or start another RL run from any SFT trial; SFT always starts from the base model."

LOCAL_RL_START = "Each RL run starts from the base model, or continues from an earlier RL checkpoint with --checkpoint."

TINKER_TIMING = "For planning, allow roughly 1 hour for SFT, 40 minutes to evaluate 1,500 examples, and 6 hours for an RL run."

# Measured on Orchard 8x H100 with Qwen3-4B (2026-10-26/27): validation 137
# tasks ~7 min, the 1,232-task test ~17 min, RL steps of 32 prompts x 8 at
# 16,384 tokens 3.0-3.5 min each; LocalModel pass@4 on 60 rows 7.1 min on 8
# GPUs, 22.7 min on one (jobs 162596, 162605). Steps of 8 prompts x 8 at 16,000
# tokens are not yet measured on this task (2.75 min on the algorithmic task).
LOCAL_TIMING = "For planning on this hardware, allow roughly 10 minutes for a validation evaluation and a few hours for a 100-step RL run (steps of 32 prompts took 3-4 minutes each; the default steps of 8 prompts take less). Sampling the student yourself on all 8 GPUs writes and grades about 240 thinking responses of up to 16,384 tokens (60 prompts, 4 samples each) in about 7 minutes, engine start-up included; one GPU takes about 20 minutes."


RL_COMMAND = """
- RL training (default 80 steps × 32 prompts × 8 completions per optimizer step):
  python -m rl.agent_training rl --data {RAW_DATA_PATH} --checkpoint {AGENT_DATA_DIR}/sft_trial_1 --output-dir {AGENT_DATA_DIR}/rl_trial_1
  You may also start from a previous RL checkpoint. Optional: --num-rollout, --rollout-batch-size, --n-samples-per-prompt. Each default optimizer batch contains 256 sampled completions. All agent training commands evaluate only your generated holdout or public validation data; secret-test evaluation is host-only.
"""


LOCAL_RL_COMMAND = """
- RL training (default 100 steps × 8 prompts × 8 completions per optimizer step):
  python -m rl.agent_training rl --data {RAW_DATA_PATH} --output-dir {AGENT_DATA_DIR}/rl_trial_1
  It starts from the base model; --checkpoint <RL checkpoint directory> continues from an earlier RL checkpoint instead. Optional: --num-rollout (for example 150 for a longer run), --rollout-batch-size, --n-samples-per-prompt. The recipe is fixed otherwise: learning rate 1e-6, responses of up to 16,000 tokens, sampling temperature 1.0, and a checkpoint every 10 steps (hf_rollout9, hf_rollout19, ...). Each default optimizer batch contains 64 sampled completions. All agent training commands evaluate only your generated holdout or public validation data; secret-test evaluation is host-only.
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
            "RL_PROMPT_DRAWS": "80 × 32 = 2,560",
            "RL_COMPLETIONS": "20,480",
            "BACKEND_RL_START": TINKER_RL_START,
            "PROBE_BUDGET": "Probes spend the final RL budget, so run a few, not dozens.",
            "BACKEND_TIMING": TINKER_TIMING,
        }
    else:
        sections = {
            "BACKEND_STUDENT_USAGE": LOCAL_STUDENT
            if hide_forger
            else LOCAL_FORGER_STUDENT,
            "BACKEND_TRAINING": LOCAL_TRAINING + LOCAL_RL_COMMAND,
            "BACKEND_SUBMISSION": "For local training, select the HF directory named by rl_trial_1/rl_result.json (usually an hf_rollout* directory). It contains the model weights and rl_submission.json. Use that same directory for validation evaluation.",
            "BACKEND_BUDGET": "- Local training and sampling have no Tinker dollar budget. GPU experiments must fit within your agent deadline; all RL must finish within that deadline.",
            "RL_SCHEDULE": "100 x 8",
            "RL_PROMPT_DRAWS": "100 × 8 = 800",
            "RL_COMPLETIONS": "6,400",
            "BACKEND_RL_START": LOCAL_RL_START,
            "PROBE_BUDGET": "Probes consume your agent time and shared judge quota; leave time for final submission.",
            "BACKEND_TIMING": LOCAL_TIMING,
        }
    return {key: value.format(**values) for key, value in sections.items()}

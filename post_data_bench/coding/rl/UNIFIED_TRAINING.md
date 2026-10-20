# Unified SFT and RL evaluation

`launch_agent.py --training-mode sft-rl --task coding --prompt YOUR_PROMPT`
(sft-rl is the default) uses the existing agent sandbox and time limit, then
runs the fixed coding RL recipe and secret benchmark evaluation after the agent
closes. `--training-mode sft` preserves the old task-specific SFT-only workflow.

SFT, RL and sampling run on [Tinker](https://thinkingmachines.ai/tinker/):
LoRA training of Qwen/Qwen3-8B on remote GPUs. The agent sandbox has one local
GPU, for embeddings and clustering only; the host needs none. Other models/tasks are rejected in unified mode. Weights stay
on Tinker; trial directories record their `tinker://` paths.

## Submission

The agent writes `generated_raw.jsonl` and `submission.json` in its persistent
`AGENT_DATA_DIR`. Each data row contains exactly three fields:

```json
{"prompt":[{"role":"user","content":"What is 2 + 2?"}],"ground_truth":"<think>Adding two pairs gives four.</think>4","rubric":[[{"code":"def grade(prompt, response, ground_truth):\n    return float(response.strip().endswith('4'))"},1],[{"prompt":"Award 1 if the final answer correctly answers the question, otherwise 0."},1]]}
```

This example illustrates the interface; substantive coding rubrics should be
more discriminating. `prompt` may be text or chat messages ending in a user turn.
`ground_truth` is the complete SFT assistant response, including nonempty thinking
and the final answer (including Python fences for coding solutions). The exact
same prompt messages go into SFT and RL; rubrics and targets are never appended
to the policy's prompt.

Submit at least 10 unique prompts. The harness sorts by prompt hash, shuffles
with seed 0, and holds out `floor(N / 10)` rows for RL reward monitoring. The
remaining rows form both the SFT and RL training set. No benchmark test data is
used in this split. Duplicate prompts and targets exceeding 18,800 tokens are
rejected. File reordering does not change the split.

If the agent ends its session with fewer than `--min-rows-nudge` rows (default
1000; 0 disables) and at least 30 minutes left, the launcher sends it one
message with the row count, how often final RL would revisit each prompt, and
how to grow the dataset without breaking its current submission, then lets the
conversation continue. Finishing again keeps the dataset as it is; the message
is sent at most once (`agent_nudges.py`).

## Agent iteration

From the sandbox's `coding/` directory, check the submission format before SFT,
probes, and final submission:

```bash
python -m rl.agent_training validate --data "$AGENT_DATA_DIR/generated_raw.jsonl"
```

This checks the triple schema, weights, unique prompts, thinking targets, code
function signatures and RestrictedPython compilation. It does not execute any
criterion, contact the judge, load a tokenizer, or create output files. Failures
exit nonzero with the JSONL line and criterion number for code errors. SFT data
preparation and host finalization enforce these same checks automatically; final
RL preflight also checks compiled training and monitoring rubrics. Token lengths
are checked separately using the training/host tokenizer during preparation.

Train from the same directory:

```bash
python -m rl.agent_training sft \
  --data "$AGENT_DATA_DIR/generated_raw.jsonl" \
  --output-dir "$AGENT_DATA_DIR/sft_trial_1" --epochs 4 --lr 1e-6
```

Repeat as often as the SFT budget permits, using new output directories. This
trains a Tinker LoRA adapter (rank 32 by default, `--lora-rank`) on the training
partition with thinking retained: AdamW, cosine schedule with 3% warmup,
batches of 128 sequences (`--batch-size`), loss averaged over target tokens.
`--lr` is a full-finetune rate; LoRA uses 10x it, as in `train_sft.py` and the
Tinker cookbook. The trial directory receives `tinker_checkpoint.json`
(`state_path` for resuming training, `sampler_path` for sampling),
`train_log.jsonl`, and a receipt identifying the actual training prompts/targets.

Evaluate a trial with the existing benchmark evaluator, sampling on Tinker:

```bash
env -u SLURM_SUBMIT_DIR bash ./evaluate_model.sh --split val \
  --server-backend tinker \
  --model-path "$(python -c 'import json;print(json.load(open("'"$AGENT_DATA_DIR"'/sft_trial_1/tinker_checkpoint.json"))["sampler_path"])')" \
  --out-dir "$AGENT_DATA_DIR/validation/sft_trial_1"
```

Compare `metrics.json` and inspect `predictions.jsonl` in that output directory
before selecting a trial. Thinking stays enabled. Omitting `--model-path`
evaluates the base student, `PDB_STUDENT_MODEL` (the launcher sets it to
`--model`; otherwise Qwen/Qwen3-8B). Tinker is the evaluation launcher's
default backend everywhere for now; a Tinker/Local split of the benchmark is
planned, and the local default will be revisited then. The vLLM and sglang
backends remain available for local checkpoints outside the sandbox with
`--server-backend vllm|sglang` (or `SERVER_BACKEND`); inside it the launcher
sets `PDB_EVAL_BACKEND=tinker`, which makes Tinker the only backend (for
`evaluate_model.py --backend` too). The host's secret-test evaluation pins the
same backend. Scripts that evaluate local checkpoints
(`run_model_training_and_evaluation.sh`, `rl/eval_rl_checkpoint.sh`,
`rl/run_rl_training_and_evaluation.sh`, `run_evaluate_only.sh`) name their
local backend explicitly, and `culture/evaluate_model.sh`, whose evaluator has
no Tinker backend yet, keeps its previous default (`PDB_EVAL_BACKEND`, else
vLLM).
Use a fresh output directory for changed weights or settings: existing
predictions are resumed, not recomputed. Keeping outputs under
`AGENT_DATA_DIR` makes them persistent. `env -u SLURM_SUBMIT_DIR` prevents a scheduler working directory
from overriding the evaluator's location.

The sandbox includes the unchanged evaluator, its explicitly allowlisted code
dependencies, and the benchmark validation rows. Its `private_test_cases.txt`
contains **only cases referenced by those validation rows**, with indices
remapped accordingly. Neither secret test tasks nor unrelated test-case blobs
are copied. This benchmark validation set is separate from the 10% generated
holdout used to monitor RL rewards. Available even with `--hide-forger`.

Choose a completed SFT trial directory:

```json
{"checkpoint":"/absolute/path/inside/AGENT_DATA_DIR/sft_trial_1"}
```

Write that object to `submission.json`. The chosen SFT receipt must match the
final dataset's prompts/targets. Rubrics can change without rerunning SFT.
Agent-selected RL checkpoints are not SFT submissions. Receipts establish
workflow provenance, not cryptographic proof of training by an untrusted agent.

Short RL probes run the final recipe for two rollouts on Tinker:

```bash
python -m rl.agent_training probe \
  --data "$AGENT_DATA_DIR/generated_raw.jsonl" \
  --checkpoint "$AGENT_DATA_DIR/sft_trial_1" \
  --output-dir "$AGENT_DATA_DIR/probe_1"
```

Probes run no secret test evaluation and draw on the same RL dollar budget as
final RL. Their outputs are separate from the selected SFT checkpoint. Normal
agent trajectory logs record commands.

## Tinker dollar budgets

Each run has two dollar limits, charged at Tinker's Qwen3-8B rates per million
tokens: train $0.44, sampled $0.60, prompt prefill $0.195, cached prefill $0.039.

* `TINKER_SFT_USAGE_FILE` / `TINKER_SFT_USD_LIMIT` ($150): agent SFT trials and
  validation evaluations.
* `TINKER_RL_USAGE_FILE` / `TINKER_RL_USD_LIMIT` ($400): agent RL probes and the
  final RL run, sharing one ledger like the judge quota.
* `TINKER_EVAL_USAGE_FILE`: the host's secret-test evaluation, recorded without
  a limit so the agent cannot starve it.
* `FORGER_TINKER_USAGE_FILE` / `FORGER_TINKER_USD_LIMIT`: the SFT ledger and
  limit again, so Forger's `TinkerModel` (student sampling inside pipelines)
  draws on the SFT budget and refuses calls past it.

Charges use Tinker's own per-call counts: training tokens per datum, sampled
tokens, and each sample response's `prompt_cache_hit_tokens` (the shared prompt
of an n-sample request is billed once, its other copies as cache hits). Each
call is checked against the remaining budget before it is sent, using its
maximum cost. Sampling calls (validation evaluation, RL rollouts) and RL
training steps hold that maximum on the ledger until they are charged, so
concurrent calls in one process cannot jointly cross the limit: a call that
does not fit waits for the others, and is refused only when none are in
flight. Calls in separate processes can still overshoot by their own
in-flight cost.
Ledgers live in the persistent agent directory and need POSIX locks; never
reset them. `tinker_budget.py LEDGER --reconcile START END` compares a ledger
with Tinker's billing records for the sessions it lists (billing lags by a few
hours and needs billing-view access). Sessions carry `pdb_run` and `pdb_budget`
metadata. Calls made outside these commands bypass the ledgers.

## Rubric execution and token-unit quotas

A rubric is a nonempty list of `[criterion, weight]` pairs. A criterion contains
exactly one of `code` or `prompt`. Weights must be finite and nonnegative, with a
positive sum. Reward is `sum(weight * score) / sum(weight)`.

* **Code:** defines `grade(prompt, response, ground_truth) -> float` in `[0,1]`.
  Define exactly one top-level synchronous `grade` function with exactly three
  positional parameters (names may vary; annotations are optional). Positional-only
  parameters and defaults are accepted; decorators, keyword-only parameters,
  `*args`, and `**kwargs` are rejected. Do not rebind or delete `grade`.
  `prompt` is the original list of chat messages; `response` retains the policy's
  thinking trace; `ground_truth` is the complete assistant target string. The returned
  score is checked at runtime, without trying to predict rubric behavior during
  format validation. Code runs in the standalone `rubric_sandbox.py` clone of the
  existing Forger RestrictedPython implementation, in a disposable process with
  wall-clock, CPU, memory, and file-size limits. It receives no credentials.
  Imports are limited to JSON, regex, math, string, collections, itertools,
  functools, and statistics. The RestrictedPython API exposes no LiteLLM, filesystem, or network operations.
  The subprocess is not an OS filesystem/network namespace; an RP escape would
  still run as the host user.
  Code errors, invalid scores, and timeouts earn zero for that criterion.
  To test the policy's code, `grade` can call `run_program(code, stdin)` ->
  `(ok, stdout)` and `call_method(code, class_name, method_name, *args)` ->
  `(ok, result)` (see `rubric_worker.py`). The response's code runs in a separate
  RestrictedPython namespace with captured stdin/stdout and competitive-
  programming imports, 4 seconds per run within the criterion's 20 seconds.
  Results are JSON-normalized so returned objects cannot override comparisons,
  and the response cannot write to the worker's output. Underscore-prefixed
  names do not compile there, a known false-negative source.
  This is the original sandbox's guardrail model, not a hardened security boundary.
* **Prompt:** trusted reward code calls LiteLLM outside RestrictedPython with
  `litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731`, temperature 1, low reasoning,
  and `n=1`. `JUDGE_MODEL` may override the alias only if it contains
  `DeepSeek-V4-Flash` (case-insensitive). `JUDGE_REASONING_EFFORT` may be `none`,
  `low` (default), or `high`; LiteLLM receives `reasoning={"effort": ...}`.
  The judge receives the criterion, original prompt, full response,
  and ground truth and returns a JSON score. This is the explicit non-RP branch;
  arbitrary Python inside a code criterion cannot call the judge.

Credentials are loaded only from environment variables. Never print their
values, include them in prompts, or pass them on command lines:

* `LITELLM_API_KEY` / `LITELLM_BASE_URL`: generation credentials.
* `LITELLM_JUDGE_API_KEY`: **separate** judge credentials, shared by agent judge
  experiments and final RL. There is no fallback to the generation key.
* `LITELLM_JUDGE_BASE_URL`: optional; defaults to `LITELLM_BASE_URL`.
* `FORGER_TOKEN_USAGE_FILE` / `FORGER_TOKEN_UNIT_LIMIT`: persistent generation
  ledger and quota; the launcher allocates 230M units.
* `JUDGE_TOKEN_USAGE_FILE` / `JUDGE_TOKEN_UNIT_LIMIT`: persistent judge ledger and
  quota; coding/Impossible-LiveCodeBench and culture/NormAD receive 75M units.
  The quota policy assigns 230M to PRBench and HealthBench; this launcher does
  not yet implement those tasks.
* `JUDGE_MAX_CONSECUTIVE_FAILURES`: defaults to 5. Each logical judge call gets three local attempts with backoff, each
  with up to two LiteLLM retries. After five consecutive exhausted calls in any reward
  worker, the circuit opens and stops RL. Success resets the
  counter. The same rule covers provider quota errors, outages, and malformed judge
  output; failures are never converted into zero rewards. Up to 16 concurrent
  criteria per worker can be in flight when the circuit opens.

DeepSeek-V4-Flash input/output tokens use 1/3 units; GLM-5.3 input/output tokens
use 4/12.5 units. Cached inputs count at full weight and output includes reasoning
once. Weights do not depend on provider prices. The launcher creates
`generation_token_usage.json` and `judge_token_usage.json` in the agent's
persistent directory. Their `token_units` and `remaining_token_units` fields
show current usage; `cost_usd` records available dollar costs, and
`unpriced_calls` counts responses without pricing. Only token units enforce the
quota. Finalization and `run-final` retain the judge ledger, so
experiments reduce the units available for final RL. Never reset these ledgers.

Forger Model calls meter generation automatically. For plain Python calls, add
the task directory to `sys.path` and use `rl.token_budget.completion` or
`acompletion`, including `budget="judge"` for judge experiments. These helpers
are available with `--hide-forger`. They check the shared quota before each
request and record actual reported tokens afterward, including malformed judge
responses that consume tokens. Missing usage stops further calls. Direct calls
outside these helpers bypass accounting; this is cooperative local metering,
not a proxy-enforced cap. In-flight calls can cross the limit.

Local quota exhaustion immediately records a credential-free quota marker and
stops RL. Provider budget errors use the consecutive-failure circuit above;
plain HTTP 429 rate limits do not imply budget exhaustion. Other repeated judge
failures still fail the run. Rewards are computed in the host RL process,
which reads judge credentials from its environment.

On judge-quota or Tinker RL budget exhaustion, RL stops before the next
rollout, saves the current weights as its final checkpoint, and writes
`rl_result.json` with the checkpoint's `tinker://` paths, the stop reason, and
completed rollouts. It then evaluates that checkpoint on the secret test split
(unless `--skip-final-eval`). Unrelated training failures remain fatal.

## Host finalization and recovery

The launcher closes the agent workspace and disarms its timer before copying
validated data and the selected checkpoint into a new host output directory.
It records SHA-256 manifests and uses the fixed `rl.rubric_rm.custom_rm` function.
No agent-written reward module or additional Slime arguments enter the graded
recipe. The final run uses the same 25,600 rollouts as the Slime recipe in
fewer, larger steps: 25 steps of 128 prompts x 8 samples, 16,384 response
tokens, temperature 1, group-normalized (GRPO) advantages, and one
importance-sampling optimizer step per step at a full-finetune-equivalent LR
of 6e-6 (LoRA 6e-5, gradient clip 1). Tinker samples a batch in parallel, so
wall-clock is about steps x the slowest sample (~9 minutes at 16K tokens),
roughly 4-5 hours in total. Each group is scored as soon as it is sampled, and
the next batch is sampled from the pre-update weights while a step trains
(one step off-policy, corrected by the importance-sampling loss). The 10%
holdout is scored in the background every 5 steps with 4 samples per prompt,
and checkpoints are saved every 5 steps. Unlike Slime, there is no KL penalty
(Slime used 1e-3). Probes run 2 steps of 16 prompts.
Thinking is enabled throughout training and final test evaluation.

Use `--prepare-final-only` to freeze without starting final RL. To finalize
an existing agent submission (without starting OpenHands):

```bash
python launch_agent.py --task coding --training-mode sft-rl \
  --finalize-submission /path/to/agent_run \
  --final-output-dir /path/to/new/final_run --prepare-final-only
```

The resulting `final_command.json` records the host command. Run the frozen
bundle from any host with network access to Tinker:

```bash
cd coding
python -m rl.agent_training run-final --bundle /path/to/final_run
```

The six-hour limit applies only to the agent; the host process must still
outlive final RL, or use the preparation/recovery workflow later.
Finalization runs with `--host-python` (default: the launcher's interpreter),
which needs `tinker`, `transformers`, RestrictedPython and LiteLLM. If the
launcher environment lacks OpenHands, it re-executes under `--agent-python`.
The legacy Slime recipe (`run_rl_training_and_evaluation.sh`) is unchanged but
no longer used by unified mode.

The host renders prompts and enforces the 18,800-token limit with the
Qwen/Qwen3-8B tokenizer and chat template, ignoring the agent's versions. The
frozen bundle holds the selected trial's `tinker_checkpoint.json` and receipt,
never agent code. Failed freezes leave no published bundle.

The legacy LiveCodeBench execution reward has a separate result protocol,
credential-free environment, and resource limits. These are robustness controls,
not a hardened OS sandbox for arbitrary Python. Unified final RL uses the fixed
RestrictedPython/LiteLLM rubric reward instead.

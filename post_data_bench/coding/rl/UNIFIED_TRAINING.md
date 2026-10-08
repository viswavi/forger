# Unified SFT and RL evaluation

`launch_agent.py --training-mode sft-rl --task coding --prompt YOUR_PROMPT`
uses the existing agent sandbox and time limit, then runs the fixed coding RL
recipe and secret benchmark evaluation after the agent closes. The default
`--training-mode sft` preserves the old task-specific SFT-only workflow.
The supplied RL architecture is Qwen3-4B; other models/tasks are rejected in
unified mode rather than silently using an incompatible Megatron configuration.

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

Repeat as often as the agent's time permits, using new output directories.
This calls the existing SFT recipe, retains thinking, and writes a receipt
identifying the actual training prompts/targets. The SFT environment defaults
to the invoking Python environment (`slime` on the host, potentially `trl` in
the existing agent image); `--conda-env` or `SFT_CONDA_ENV` can override it. The public trainer and FSDP
config are available even with
`--hide-forger`.

Evaluate a trained checkpoint with the existing benchmark evaluator:

```bash
env -u SLURM_SUBMIT_DIR bash ./evaluate_model.sh --split val \
  --model-path "$AGENT_DATA_DIR/sft_trial_1" \
  --out-dir "$AGENT_DATA_DIR/validation/sft_trial_1"
```

Compare `metrics.json` and inspect `predictions.jsonl` in that output directory
before selecting a trial. Thinking stays enabled. The evaluator defaults to
vLLM; use `--server-backend sglang` when using the Slime serving stack. Run it
from the environment containing the selected serving backend. Use a fresh
output directory for changed weights or settings: existing predictions are
resumed, not recomputed. Keeping outputs under `AGENT_DATA_DIR` makes them
persistent. `env -u SLURM_SUBMIT_DIR` prevents a scheduler working directory
from overriding the evaluator's location.

The sandbox includes the unchanged evaluator, its explicitly allowlisted code
dependencies, and the benchmark validation rows. Its `private_test_cases.txt`
contains **only cases referenced by those validation rows**, with indices
remapped accordingly. Neither secret test tasks nor unrelated test-case blobs
are copied. This benchmark validation set is separate from the 10% generated
holdout used to monitor RL rewards. Available even with `--hide-forger`.

Choose a completed, merged/full Hugging Face SFT checkpoint:

```json
{"checkpoint":"/absolute/path/inside/AGENT_DATA_DIR/sft_trial_1"}
```

Write that object to `submission.json`. The chosen SFT receipt must match the
final dataset's prompts/targets. Rubrics can change without rerunning SFT.
Agent-selected RL checkpoints are not SFT submissions. Receipts establish
workflow provenance, not cryptographic proof of training by an untrusted agent.

Short RL probes are available when the agent image contains Slime:

```bash
python -m rl.agent_training probe \
  --data "$AGENT_DATA_DIR/generated_raw.jsonl" \
  --checkpoint "$AGENT_DATA_DIR/sft_trial_1" \
  --output-dir "$AGENT_DATA_DIR/probe_1" \
  --slime-env /path/to/slime/environment --slime-deps /path/to/slime_deps
```

Set `SLIME_SRC` to the Slime checkout (for example `/opt/slime` in a container).
The dependencies directory must contain `Megatron-LM`. Enable agent RL runtime
mounts with `AGENT_ENABLE_SLIME=1` and use a compatible image via `AGENT_SIF`.
Probes run two rollouts and no secret test evaluation. Their outputs are separate
from the selected SFT checkpoint. Normal agent trajectory logs record commands.

## Rubric execution and credits

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
  This is the original sandbox's guardrail model, not a hardened security boundary.
* **Prompt:** trusted reward code calls LiteLLM outside RestrictedPython with
  `litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731`, temperature 1, high reasoning,
  and `n=1`. The judge receives the criterion, original prompt, full response,
  and ground truth and returns a JSON score. This is the explicit non-RP branch;
  arbitrary Python inside a code criterion cannot call the judge.

Credentials are loaded only from environment variables. Never print their
values, include them in prompts, or pass them on command lines:

* `LITELLM_API_KEY` / `LITELLM_BASE_URL`: generation credit pool.
* `LITELLM_JUDGE_API_KEY`: **separate** judge credit pool, shared by agent judge
  experiments and final RL. There is no fallback to the generation key.
* `LITELLM_JUDGE_BASE_URL`: optional; defaults to `LITELLM_BASE_URL`.
* `JUDGE_MAX_CONSECUTIVE_FAILURES`: defaults to 5. Each logical judge call gets three local attempts with backoff, each
  with up to two LiteLLM retries. After five consecutive exhausted calls in any reward
  worker, the circuit opens and stops the rollout/Ray job. Success resets the
  counter. The same rule covers depleted credits, outages, and malformed judge
  output; failures are never converted into zero rewards. Up to 16 concurrent
  criteria per worker can be in flight when the circuit opens.

The agent can spend judge credits freely, but doing so reduces credits available
for final RL. Configure the two budgets on the proxy keys. Ray receives judge
credentials through a temporary mode-600 runtime environment file, not shell
tracing. Explicit budget/quota errors are classified internally without logging provider
error bodies. A plain HTTP 429 rate limit is not sufficient to classify a budget
stop. After consecutive budget failures, the worker records a credential-free
quota marker for the host. Other repeated judge failures still fail the run.

On quota exhaustion, the host prints a warning, returns the newest usable saved
HF checkpoint as `FINAL_CHECKPOINT=...`, and writes `rl_result.json` with its path,
termination reason, and original training exit code. It then evaluates that
checkpoint on the secret test split (unless `--skip-final-eval`). Incomplete
exports are skipped. If no RL checkpoint was saved, it returns and evaluates the
submitted SFT checkpoint with an explicit warning that no RL progress was saved.
Updates since the last completed save are not recovered. Each attempt uses a
fresh quota marker, so resuming cannot mistake an old quota stop for a new one.
Unrelated training failures remain fatal and do not trigger this fallback.

## Host finalization and recovery

The launcher closes the agent workspace and disarms its timer before copying
validated data and the selected checkpoint into a new host output directory.
It records SHA-256 manifests and uses the fixed `rl.rubric_rm.custom_rm` function.
No agent-written reward module or additional Slime arguments enter the graded
recipe. The final run defaults to 100 rollouts, with thinking enabled throughout
training and final test evaluation. Checkpoint conversion caches are per-run,
so identically named SFT trials cannot reuse another trial's weights.

Use `--prepare-final-only` to freeze without starting a long GPU job. To finalize
an existing agent submission (without starting OpenHands):

```bash
python launch_agent.py --task coding --training-mode sft-rl \
  --finalize-submission /path/to/agent_run \
  --final-output-dir /path/to/new/final_run --prepare-final-only
```

The resulting `final_command.json` records the host command. Run the frozen
bundle on a sufficiently long GPU allocation:

```bash
cd coding
python -m rl.agent_training run-final --bundle /path/to/final_run
```

The six-hour limit applies only to the agent; a scheduler allocation must still
cover final RL, or use the preparation/recovery workflow on another allocation.
`--slime-env` defaults to `/home/vijayv/miniconda3/envs/slime`; Slime source and
Megatron dependencies must be installed/configured. If the launcher environment
lacks OpenHands, it uses the SDK in `trl` via `--agent-python`; finalization,
training, and evaluation still use Slime's environment. No packages are copied
between environments. Existing SFT-only behavior keeps its original environment.

The host freezes tokenizer, chat template, and generation configuration from its
cached Qwen/Qwen3-4B snapshot, ignoring the agent's versions. The host tokenizer
also enforces the 18,800-token limit. Failed freezes leave no published bundle.
Ray's dashboard binds to localhost. A successful RL job must have a usable final
export; only a quota stop permits selection of an earlier usable checkpoint.

The legacy LiveCodeBench execution reward has a separate result protocol,
credential-free environment, and resource limits. These are robustness controls,
not a hardened OS sandbox for arbitrary Python. Unified final RL uses the fixed
RestrictedPython/LiteLLM rubric reward instead.

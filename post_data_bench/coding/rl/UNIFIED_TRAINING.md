# Agent-controlled SFT and RL

`launch_agent.py --task coding --prompt coding/FORGER_PROMPT.txt` gives the agent
15 hours (`--timeout-hours`) for data generation, SFT, RL and validation. The
minimal scaffold uses `coding/minimal_scaffold_prompt.txt` and the same training
contract. Both prompts render backend-specific commands from `backend_prompts.py`.

## Backends and defaults

- `--backend tinker` (default): Qwen/Qwen3-8B, remote LoRA training.
- `--backend local`: Qwen/Qwen3-4B, local TRL SFT with merged LoRA weights,
  Slime full-parameter RL, and vLLM validation and final evaluation.
- RL defaults on both: 80 optimizer steps, 32 prompts × 8 completions per step
  (256 completions per optimizer batch). Total: 2,560 prompt draws and 20,480
  completions; the old Tinker default was 25 × 128 × 8 = 25,600 completions.
- Agents may change the number of steps or batch shape, repeat runs, and continue
  RL from an SFT or RL checkpoint in a fresh output directory.
- Planning estimates: SFT roughly 1 hour, validation on 1,500 examples roughly
  40 minutes, RL roughly 6 hours. These are guidelines, not deadlines or guarantees.
- Generation/judge token quotas and Tinker dollar caps remain unchanged. Tinker
  SFT and validation share $150; all agent RL runs share $400. Host secret-test
  evaluation is recorded without a dollar cap.

The backend/model are saved in `token_budget_config.json`, together with
`submission_kind=final_rl`. The models and optimization recipes differ between
backends, so their scores are not a controlled comparison of services.

## Agent commands and submission

Run from `coding/`, with all outputs inside the agent data directory:

```bash
python -m rl.agent_training validate --data "$DATA/generated_raw.jsonl"
python -m rl.agent_training sft --data "$DATA/generated_raw.jsonl" --output-dir "$DATA/sft_1"
python -m rl.agent_training rl --data "$DATA/generated_raw.jsonl" --checkpoint "$DATA/sft_1" --output-dir "$DATA/rl_1"
```

`rl` accepts `--num-rollout`, `--rollout-batch-size`, and
`--n-samples-per-prompt`. `probe` remains an optional short experiment. Neither
command performs secret-test evaluation. Use `evaluate_model.sh --split val`
with `--server-backend tinker` (the sampler path) or `vllm` (the HF directory).

Dataset rows remain prompt/ground_truth/rubric triples, with 10–2,000 unique
prompts and thinking-mode targets within 18,800 tokens. The fixed generated
holdout is 10%. Agents may revise data between training runs; selecting a final
checkpoint no longer requires its SFT data hash to match the latest dataset.

Write `submission.json` atomically as `{"checkpoint": "/absolute/checkpoint/path"}`:

- Tinker: select `rl_1/rl-final` or a saved `rl-step-*` directory. It records
  resumable state, sampler weights, and an `rl_submission.json` receipt.
- Local: select the full HF directory recorded in `rl_1/rl_result.json`, typically
  `hf_rollout*`. Completed runs record RL receipts alongside usable checkpoints.

A submitted checkpoint must have at least one completed RL step. A zero-step
quota stop or SFT-only checkpoint is not a final RL submission. Keep an existing
completed RL checkpoint selected while running further experiments. Receipts
provide cooperative workflow provenance, not cryptographic proof of training.

## Host evaluation and recovery

After the agent closes, `finalize` validates the final dataset and RL receipt,
freezes the selected checkpoint, and writes SHA-256 manifests. Local checkpoints
use trusted base-model tokenizer files and exclude agent Python. Tinker weights
stay remote; the host freezes their identifiers. No host training follows a new
`final_rl` submission: only secret-test evaluation runs outside the 15 hours.

`--prepare-final-only` freezes without evaluating. Resume with:

```bash
cd coding
python -m rl.agent_training run-final --bundle /path/to/frozen_bundle
```

For local Orchard evaluation, use `post_data_bench/local_host_python.sh` in place
of `python`, with the configured image, model cache, output and benchmark paths.
The combined image separates TRL from system Slime Python. Legacy frozen bundles
without `submission_kind=final_rl` retain their original host-RL recovery path;
old unfrozen agent runs should use the launcher version that created them.

## Rubric execution and token-unit quotas

A rubric is a nonempty list of `[criterion, weight]` pairs. A criterion contains
exactly one of `code` or `prompt`. Weights must be finite and nonnegative, with a
positive sum. Reward is `sum(weight * score) / sum(weight)`.

- **Code:** defines `grade(prompt, response, ground_truth) -> float` in `[0,1]`.
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
- **Prompt:** trusted reward code calls LiteLLM outside RestrictedPython with
  `litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731`, temperature 1, low reasoning,
  and `n=1`. `JUDGE_MODEL` may override the alias only if it contains
  `DeepSeek-V4-Flash` (case-insensitive). `JUDGE_REASONING_EFFORT` may be `none`,
  `low` (default), or `high`; LiteLLM receives `reasoning={"effort": ...}`.
  The judge receives the criterion, original prompt, full response,
  and ground truth and returns a JSON score. This is the explicit non-RP branch;
  arbitrary Python inside a code criterion cannot call the judge.

Credentials are loaded only from environment variables. Never print their
values, include them in prompts, or pass them on command lines:

- `LITELLM_API_KEY` / `LITELLM_BASE_URL`: generation credentials.
- `LITELLM_JUDGE_API_KEY`: **separate** judge credentials, shared by agent judge
  experiments and agent RL. There is no fallback to the generation key.
- `LITELLM_JUDGE_BASE_URL`: optional; defaults to `LITELLM_BASE_URL`.
- `FORGER_TOKEN_USAGE_FILE` / `FORGER_TOKEN_UNIT_LIMIT`: persistent generation
  ledger and quota; the launcher allocates 230M units.
- `JUDGE_TOKEN_USAGE_FILE` / `JUDGE_TOKEN_UNIT_LIMIT`: persistent judge ledger and
  quota; coding/Impossible-LiveCodeBench and culture/NormAD receive 75M units.
  The quota policy assigns 230M to PRBench and HealthBench; this launcher does
  not yet implement those tasks.
- `JUDGE_MAX_CONSECUTIVE_FAILURES`: defaults to 5. Each logical judge call gets three local attempts with backoff, each
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
experiments reduce the units available for agent RL. Never reset these ledgers.

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

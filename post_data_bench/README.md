# Agentic post-training evaluation

The harness measures a data-generation agent by training a student on its
submission and evaluating that student on a host-only benchmark. Coding uses
LiveCodeBench plus impossible variants. Culture uses the updated
`akhilayerukola/NormAd` dataset from Hugging Face.

## Setup

Run these commands from `post_data_bench/`. The scripts retain cluster-specific
paths and Slurm resource directives; adapt those before using another machine.

1. Run `bash setup_dependencies.sh` to initialize the pinned LiveCodeBench
   submodule and apply its timeout/metadata patches.
2. Unified SFT/RL and evaluation run on Tinker (LoRA on Qwen/Qwen3-8B). The
   agent sandbox gets one GPU, for embeddings and clustering only. The host interpreter (`--host-python`) and the agent
   image need Python 3.12 with `tinker`, `transformers`, RestrictedPython and
   LiteLLM. OpenHands lives in `trl`; the launcher can re-execute with
   `--agent-python`. The legacy Slime setup scripts remain in `coding/rl/`.
3. Build an allowlisted image with `bash apptainer/rebuild_flame.sh` and set
   `AGENT_SIF` to its output. There is no default image. The legacy recipe that
   copied the whole repository and the native launcher have been removed.
   Slime image downloads require an immutable `--image ...@sha256:...` reference
   and refuse to replace an existing image without `--force`.
4. Provide credentials through environment variables, with three distinct keys:
   `LITELLM_DATAGEN_API_KEY` for generation and `LITELLM_JUDGE_API_KEY` for
   judging, both through the gateway at `LITELLM_WORKER_BASE_URL`;
   `LLM_AGENT_API_KEY` and `LLM_AGENT_MODEL` for the OpenHands runner (set
   `LLM_AGENT_BASE_URL` only to route it through a proxy); `TINKER_API_KEY` for
   training and sampling. The launcher maps these to the `LITELLM_API_KEY`,
   `LITELLM_BASE_URL`, and `LLM_*` names the sandbox and reward code read. Never
   put values in prompts, command arguments or logs. Retrieval can use
   `BRAVE_API_KEY` and `S2_API_KEY`. The runner key and unrelated provider keys
   are not forwarded into the sandbox.

For local GPU training, pass `--backend local`. `AGENT_HF_HOME` names the host
cache containing the selected model; only that model's cache directory is bound
read-only. Tokens, other model caches and dataset caches are not mounted.
`AGENT_SCRATCH_ROOT` controls scratch storage, `AGENT_GPU_DESC` describes the
allocation, and `AGENT_MODEL_OUTPUT_DIR` controls persistent outputs. Each run's
checkpoints and generated data live below the large model-output volume rather
than the repository on `/home`.

## Tinker dollar budgets

Each launched run gets a $150 Tinker budget for the agent's SFT trials and
validation evaluations, and a $400 budget shared by all agent RL runs and probes.
The host's secret-test evaluation is recorded but not limited. Charges use
Tinker's own per-call token counts (including prompt cache hits) at its Qwen3-8B
rates; see [the unified contract](coding/rl/UNIFIED_TRAINING.md#tinker-dollar-budgets).

## Token-equivalent units and quotas

PostDataBench measures API usage in **token-equivalent units** (called `token_units`
in code). One DeepSeek-V4-Flash input token is one unit. Other input and output
tokens receive the fixed weights below, so usage across models adds to one quota.
These weights are benchmark policy and stay fixed when provider prices change.

| Model family | Units per input token | Units per output token |
|---|---:|---:|
| DeepSeek-V4-Flash | 1 | 3 |
| GLM-5.3 | 4 | 12.5 |

For each LiteLLM response, calculate:

```text
token_units = usage.prompt_tokens * input_weight + usage.completion_tokens * output_weight
```

For example, 1,000 input tokens and 200 output tokens consume 1,600 units on
DeepSeek-V4-Flash or 6,500 units on GLM-5.3. Count cached inputs at full weight.
LiteLLM's aggregate completion count already includes reasoning tokens; do not
add `reasoning_tokens` again. Model aliases must contain `DeepSeek-V4-Flash` or
`GLM-5.3` (case-insensitive). There are no defined weights for other models:
Forger reports `NaN` units for them, and budgeted helpers reject them before a call.

Generation/SFT has 180M units. RL has 50M for Impossible-LiveCodeBench and NormAD,
and 180M for PRBench and HealthBench. These are separate quotas. The launcher
currently supports coding and culture; unified RL currently supports coding only.

### LiteLLM helpers without Forger

The supplied helpers calculate units, check the quota before each request, and
persist token units and available dollar costs after each response. They support
synchronous `completion` and asynchronous `acompletion`, with LiteLLM keyword
arguments and response objects. They do not require installing or importing Forger.
Both wrappers require non-streaming responses with input and output token counts.

Inside a launched agent workspace, including `--hide-forger` runs, import them
by running from the task directory or adding that directory to `PYTHONPATH`:

```python
from rl.token_budget import completion, TokenUnitBudget
```

To use the same helpers in another project, [export a portable package](export_token_helpers.py)
from this checkout (run from `post_data_bench/`):

```bash
python export_token_helpers.py --output-dir /path/to/my-agent
```

This creates `pdb_token_budget/` containing the helper and its accounting module.
Copy that directory into the agent's project, or put its parent on `PYTHONPATH`.
The exported package runs independently of this checkout. Only API calls and
optional dollar-price lookups need LiteLLM; unit calculation uses the standard library.

The launcher supplies credentials and quota environment variables. Outside the
launcher, provide `LITELLM_API_KEY` and `LITELLM_BASE_URL` securely and set a persistent
ledger path and quota:

```bash
export FORGER_TOKEN_USAGE_FILE=/path/to/persistent/run/generation_token_usage.json
export FORGER_TOKEN_UNIT_LIMIT=180000000
```

The `FORGER_` prefix is retained for configuration compatibility and does not
require Forger. With LiteLLM installed in the destination environment:

```python
import os
from pdb_token_budget import completion, TokenUnitBudget

response = completion(
    model="litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731",
    messages=[{"role": "user", "content": "What is 273 * 149?"}],
    temperature=1.0,
    reasoning={"effort": "none"},
    max_tokens=512,
)
print(response.choices[0].message.content)
budget = TokenUnitBudget(
    os.environ["FORGER_TOKEN_USAGE_FILE"], os.environ["FORGER_TOKEN_UNIT_LIMIT"]
)
print(budget.snapshot())  # Cumulative units, dollars, and remaining quota.
```

Use the `rl.token_budget` import instead in a launched workspace. For an existing
response, `token_units(model_id, *response_token_counts(response))` computes its
units; both functions are exported by either helper. Calculation alone does not
record usage or enforce a quota. Forger `Model` calls account automatically and
expose `total_token_units` and `total_cost`.

For judge experiments, pass `budget="judge"`. The helper uses the separate
`LITELLM_JUDGE_API_KEY`, `JUDGE_TOKEN_USAGE_FILE`, and `JUDGE_TOKEN_UNIT_LIMIT`;
outside the launcher, configure these too. The judge defaults to
`litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731`. `JUDGE_MODEL` may select an alias
containing `DeepSeek-V4-Flash`. `JUDGE_REASONING_EFFORT` accepts `none`, `low`
(default), or `high`, passed as `reasoning={"effort": ...}`. An optional
`LITELLM_JUDGE_BASE_URL` overrides `LITELLM_BASE_URL`.

### Persistent usage and dollar costs

Each launched run stores `generation_token_usage.json` and `judge_token_usage.json`
in its persistent agent directory. Ledgers track `token_units`,
`remaining_token_units`, `cost_usd`, and `unpriced_calls`, plus per-model usage and cost.
Dollar costs come from the provider response or LiteLLM pricing. An **unpriced
call** has no available dollar price; it still consumes its calculated token units.
Missing dollar pricing is independent of whether token-unit weights are known.
Existing ledgers keep their token usage and mark earlier calls as unpriced.

All agent judge calls and RL runs share a judge ledger.
`token_budget_config.json` preserves the task, limits, judge alias, and reasoning
effort without credentials. Ledger storage must support POSIX file locks and
atomic rename: use persistent local storage or NFS, including for
`AGENT_MODEL_OUTPUT_DIR`; GCS FUSE is unsuitable.

Route API calls through these helpers to participate in accounting. Direct provider
calls bypass this cooperative accounting; the ledger is not a proxy-enforced cap.
Calls already in flight may exceed a quota, so batches need headroom. The OpenHands
model driving agent turns is separate from data generation and reports
`AGENT_TURN_COST_USD` for reference. See the
[unified contract](coding/rl/UNIFIED_TRAINING.md) for judge quota recovery.

## Benchmark data

Keep frozen inputs and previous benchmark versions. Neither datasets nor private
test blobs are committed. A data build stages all outputs, publishes a recovery journal, and writes a manifest
binding the source
hashes, source membership, validation, readable validation, test, and private-test
blobs. Evaluation and agent export reject missing or mismatched manifests.
Obfuscation is not access control: `.hidden/` and the full private-test file must
remain outside the agent's visible filesystem.

Existing test outputs require explicit `--force`. Prefer a new output directory:

```bash
python coding/make_splits.py --source-dir /path/to/frozen/coding \
  --gold /path/to/frozen/coding/gold_responses.jsonl \
  --perturbed /path/to/frozen/coding/perturbed_cases.jsonl \
  --output-dir /path/to/new/coding
python culture/make_splits.py --output-dir /path/to/new/culture
```

Coding builds a fresh 90% secret test / 10% public validation split. It groups
originals with their impossible variants and allocates validation quotas across
paired sources and unpaired sources, balancing difficulty and gold-verification
status within each group. Quotas minimize deviation from the requested row count,
then from each group's 10% share; whole pairs and rounding can prevent exact
proportions. Seeded hashes select members deterministically for identical inputs.
Old membership is never loaded, including when the source directory contains an
existing test file. Both variants of each pair stay together; source groups are
disjoint across splits. Gold-unverified sources participate in both splits rather
than being forced into test. Rerun baselines for the new membership.

Non-unique-output problems are excluded before gold generation, perturbation
generation, and splitting: constructive witnesses and unordered answer arrays
are incompatible with concrete-output mutation. Both originals and perturbed
variants are removed. The reviewed IDs and future-source wording checks live in
`coding/source_policy.py`; new source releases still require semantic review.
The build writes a host-only exclusion report. Private-test indices stay stable;
excluded sources get empty blob slots. It rebuilds content uniformly:
de-identified public fields, canonical output formatting, and worked examples
rendered from structured tests without worked explanations.

Culture downloads the updated Hugging Face dataset directly, filters invalid
labels, and assigns approximately 10% of situations to public validation using a
stable hash. The loader resolves `--revision` to an immutable HF commit; the manifest records
that revision and a hash of the normalized source snapshot.

`coding/set_up_data.sh` additionally downloads raw shards and generates missing
paid gold/perturbation records. Pass build options directly and generation options
with `--gold-arg` and `--perturbation-arg`. Paid regeneration requires
`--allow-generation`; use `--gold-file` and `--perturbed-file` to select frozen inputs. Refresh merges newly
resolved records without
deleting unrelated paid results; failures remain retryable. The culture setup
wrapper forwards its arguments to `make_splits.py`.

## Run

```bash
python launch_agent.py --task coding --training-mode sft-rl \
  --benchmark-dir /path/to/new/coding --prompt coding/FORGER_PROMPT.txt
```

The agent has **15 hours** for data generation, training and validation, and submits
its chosen **final RL checkpoint** plus its dataset (prompts and rubrics; a
ground_truth reference is optional). It may repeat training, continue RL from
earlier checkpoints, and revise data between experiments. With `--backend local`,
training is **RL only**, from the base model: **80 steps × 32 prompts × 8
completions** (a submitted checkpoint has exactly 80 steps behind it), lr 3e-6 and
responses of up to 16,384 tokens, 24,576 for the algorithmic task (see
`coding/rl/UNIFIED_TRAINING.md`). Tinker keeps SFT
trials and **80 steps × 32 prompts × 8 completions** (2,560 prompt draws, 20,480
completions); its planning guidelines are roughly one hour for SFT, 40 minutes for
evaluation on 1,500 examples, and six hours for RL.
After the workspace closes, the host freezes the selected RL checkpoint and
runs only secret-test evaluation. It does not train further. An error or timeout
still permits evaluation of an already saved valid submission. The agent should
select a completed RL checkpoint before spending time on further experiments.
See [the unified contract](coding/rl/UNIFIED_TRAINING.md).

For direct evaluation, set `PDB_BENCHMARK_DIR` to the selected task's benchmark
build and run `{coding,culture}/evaluate_model.sh`. Coding samples on Tinker by
default; `--server-backend vllm|sglang` serves a
local checkpoint instead. Culture has no Tinker backend yet and defaults to vLLM.
Coding grades only the answer
after the final `</think>` in thinking mode; an unclosed trace is not an answer.
Abstention requires the exact marker in that answer. This rule is shared with the
legacy execution reward. Request/grading failures fail the run and remain
retryable; they never become incorrect model answers. Resume requires matching
partition, data, model files and sampling settings. Use a new `--out-dir` when
changing GPU count, or `FORCE=1` to recompute. Full merges require exact coverage;
partial checkpoint diagnostics use separate directories and do not merge. The
shared launcher cleans up its server and client process groups on interruption.
The legacy evaluator names delegate to the supported implementation. Direct Python
invocations use `VLLM_BASE_URL` for an already running local or remote endpoint.
`run_evaluate_only.sh` defaults to Slime/SGLang; set `EVAL_CONDA_ENV` and
`--server-backend` explicitly to use a different serving environment.

`train_model.sh` promotes the checkpoint at the completed trainer global step.
Fresh training refuses an output directory containing old checkpoints. Coding's
canonical LoRA recipe uses q_proj/v_proj, r=256, alpha=32, global batch 128 and the
trainer's existing 10x LoRA learning-rate scaling. Explicit overrides remain
available. Both task converters filter overlength targets. Sweeps use validation;
reserve secret test evaluation for final configurations.

## Validation and limits

```bash
/home/vijayv/miniconda3/envs/slime/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Offline tests cover submission/data contracts, export isolation, request failures,
resume/merge checks, rubric scoring and quota recovery. The manual budget probe
is separately paid and is never run by this suite. GPU training, image rebuilds,
Slurm allocations and full agent sessions require cluster integration tests.

RestrictedPython plus resource limits is a guardrail, not an OS security boundary.
The legacy LiveCodeBench execution worker also has a minimal environment and
resource limits, but no filesystem/network namespace. Benchmark problems originate
in public datasets; this setup does not prevent an agent from independently
retrieving public sources. SFT receipts establish workflow provenance, not proof
against a malicious agent fabricating training artifacts.

## Orchard source release (2026-10-23)

The combined image `openhands-forger-flame-v3-632f768.sif` remains compatible
with `vijay_tinker_v4` at `fba67ef`: packaging dependencies and image definitions
have not changed. Library code, prompts and runner scripts are provided by the
launcher's allowlisted per-run source mounts. Deploy the source release alongside
the image; rebuilding an identical SIF does not update those source mounts.

`orchard_prepare.sbatch` installs a launcher environment on compute-node scratch
and archives it and the source bundle to
`gs://cmu-gpucloud-vijayv/flame_port/releases/$PDB_RELEASE/`.
It does not start an agent, train, or evaluate a model. Set `PDB_REPO` to the
versioned source deployment and `PDB_RELEASE` to its identifier before submission.

`orchard_agent_run.sbatch` restores that launcher environment, stages the image,
uses node-local scratch for outputs and a copy of the Qwen3-4B cache, and runs
`--backend local`. Supply `--env-file PATH` and set `PDB_REPO`, `PDB_RELEASE`,
`PDB_BENCHMARK_DIR` (a verified benchmark build), and `PDB_ARCHIVE_URI` (a unique
GCS output directory). Credentials stay in the environment. The script archives
outputs on normal or failed exit; abrupt node loss or SIGKILL cannot run that
archive handler. The default job allocation is 24 hours, including host evaluation;
the agent's default allowance is 15 hours.

The October 23 backend integration is a development deployment. End-to-end local
SFT, Slime probes, finalization, and evaluation still need runtime validation.

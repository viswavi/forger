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
2. Use the `slime` environment for unified SFT/RL, including RestrictedPython,
   LiteLLM, TRL, PEFT and Accelerate. The setup scripts live in `coding/rl/`.
   OpenHands lives in `trl`; the launcher can re-execute with `--agent-python`.
3. Build an allowlisted image with `bash apptainer/rebuild_flame.sh` and set
   `AGENT_SIF` to its output. There is no default image. The legacy recipe that
   copied the whole repository and the native launcher have been removed.
   Slime image downloads require an immutable `--image ...@sha256:...` reference
   and refuse to replace an existing image without `--force`.
4. Provide credentials through environment variables: `LITELLM_API_KEY` and
   `LITELLM_BASE_URL` for generation; a separate `LITELLM_JUDGE_API_KEY` for
   judging, optionally with `LITELLM_JUDGE_BASE_URL`. Never put values in files,
   prompts, command arguments or logs. Retrieval can use `BRAVE_API_KEY` and
   `S2_API_KEY`. Driver and unrelated provider keys are not forwarded.

For agent RL probes, set `AGENT_ENABLE_SLIME=1`. `AGENT_HF_HOME` names the host
cache containing the selected model; only that model's cache directory is bound
read-only. Tokens, other model caches and dataset caches are not mounted.
`AGENT_SCRATCH_ROOT` controls scratch storage, `AGENT_GPU_DESC` describes the
allocation, and `AGENT_MODEL_OUTPUT_DIR` controls persistent outputs. Each run's
checkpoints and generated data live below the large model-output volume rather
than the repository on `/home`.

## Token-unit quotas

Generation/SFT has 230M token units. RL has 75M for Impossible-LiveCodeBench
and NormAD, and 230M for PRBench and HealthBench. The launcher currently supports
coding and culture; unified RL currently supports coding only.

| Model family | Input token units | Output token units |
|---|---:|---:|
| DeepSeek-V4-Flash | 1 | 3 |
| GLM-5.3 | 4 | 12.5 |

These weights are fixed policy, independent of provider prices. Cached input
tokens have full weight; completion tokens include reasoning once. Each run
stores `generation_token_usage.json` and `judge_token_usage.json` under its
persistent agent directory. Their `token_units` and `remaining_token_units`
fields update after each completed API response, across processes. Judge probes
and final RL share the same ledger, including recovery runs.
Ledgers also track `cost_usd` and `unpriced_calls`, overall and per model. Dollars
use available provider or LiteLLM pricing; unpriced calls still consume token units.
Existing ledgers keep their token usage and mark earlier calls as unpriced.
Forger models expose both `total_token_units` and `total_cost`; unsupported models
report `NaN` units and cannot make calls against a fixed token-unit quota.
`token_budget_config.json` preserves the task, limits, judge alias, and reasoning
effort for recovery without storing credentials.
`AGENT_MODEL_OUTPUT_DIR` must use persistent local storage or NFS with POSIX file
locks and atomic rename; GCS FUSE is unsuitable for these ledgers. Standalone
jobs using `FORGER_TOKEN_USAGE_FILE` or `JUDGE_TOKEN_USAGE_FILE` have the same
filesystem requirement.

Forger Model calls account automatically. Plain Python pipelines, including
`--hide-forger` baselines, must import `completion`/`acompletion` from the supplied
`rl.token_budget` helper, adding the task directory to `sys.path` if needed.
Use `budget="judge"` for judge experiments. Other libraries must route their API
calls through these helpers. Direct provider calls bypass this cooperative
accounting; the ledger is not a proxy-enforced cap. Calls already in flight may
exceed a quota, so batches need headroom. The OpenHands model driving agent turns
is separate from data generation and reports `AGENT_TURN_COST_USD` for reference.

The RL judge defaults to `litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731`.
`JUDGE_MODEL` may select an alias containing `DeepSeek-V4-Flash` (case-insensitive).
`JUDGE_REASONING_EFFORT` accepts `none`, `low` (default), or `high` and is passed as
`reasoning={"effort": ...}`. See the [unified contract](coding/rl/UNIFIED_TRAINING.md)
for judge quota recovery.

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

The agent submits prompt/ground-truth/rubric triples and a selected completed SFT
checkpoint. It may repeat SFT and run short RL probes. After the workspace closes,
the host freezes the submission and runs final RL and secret evaluation outside
the agent time limit. An error or iteration limit still permits validation of an
already saved submission. See [the unified contract](coding/rl/UNIFIED_TRAINING.md).

For direct evaluation, set `PDB_BENCHMARK_DIR` to the selected task's benchmark
build and run `{coding,culture}/evaluate_model.sh`. Coding grades only the answer
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

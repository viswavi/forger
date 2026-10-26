# GRPO on impossible-LiveCodeBench, with slime

For agent-generated shared SFT/RL triples, submitted SFT checkpoints, and mixed
code/LM rubrics, see [Unified training](UNIFIED_TRAINING.md). The original
benchmark-specific reward recipe below remains available.

Unified coding runs use 180M generation token units and a separate 50M judge
quota shared by probes and final RL. The judge must be DeepSeek-V4-Flash; its
default reasoning effort is `low`, with `none` and `high` also accepted. See
[token-unit quotas](../../README.md#token-unit-quotas) for weights and accounting.

RL counterpart to `run_model_training_and_evaluation.sh`. Same task, same prompts,
same final benchmark; what changes is that the abstain-or-solve decision is trained
against a verifier instead of against a fixed target string.

| file | what it is |
|---|---|
| `setup_slime_conda.sh` | build slime's stack into its own conda env |
| `setup_slime_container.sh` | fallback: fetch + smoke-test the upstream apptainer image |
| `build_rl_data.py` | SFT source rows -> slime prompt jsonl (prompts + grading payload) |
| `impossible_lcb_rm.py` | the verifier (`--custom-rm-path`), and its offline self-test |
| `_grade_worker.py` | one grading subprocess: JSON in, JSON out |
| `measure_group_variance.py` | pre-flight: does GRPO have signal from this init checkpoint? |
| `summarize_rollouts.py` | read a run's rollout logs: reward curve by row type |
| `run_rl_training_and_evaluation.sh` | the sbatch launcher: convert -> GRPO -> `evaluate_model.sh` |

## Environment

slime is **not** installed into `trl`, and should not be. slime pins torch
2.11.0+cu129, sglang 0.5.15.post1, TransformerEngine 2.16.1 and Megatron-LM;
`trl` has torch 2.9.1+cu128, vllm 0.16.0 (pinned to torch 2.9), and flash_attn
2.8.3 built against that torch. There is no torch that satisfies both, so an
in-place install would replace torch and break `train_sft.py` and
`evaluate_model.sh` — the two things this needs to protect. It gets its own conda
env instead; `trl` is untouched, The unified runner uses SGLang for final evaluation in `slime`; legacy vLLM
evaluation remains available in `trl`.

```bash
./rl/setup_slime_conda.sh                # build (resumable), hours on a small node
./rl/setup_slime_conda.sh --check-only   # re-run the import smoke test
./rl/setup_slime_conda.sh --redo 03_install_sglang   # force one step to rerun
```

Adapted from slime's own `build_conda.sh` with the pins kept byte-for-byte
(sglang at commit `0b3bb0cb`, Megatron-LM at `1dcf0daf`, plus the
`docker/patch/v0.5.15.post1` patch set applied to both). Env at
the configured `ENV_PREFIX` (the current host uses `/home/vijayv/miniconda3/envs/slime`), sources at
`/data/user_data/vijayv/slime_deps` — not `$HOME`, which is at 92% of a 300G
quota against an env of roughly 25-30GB.

Every step writes a stamp under `$BASE_DIR/.stamps` and is skipped on a rerun, so
a network failure two hours in costs one step, not the build.

Built and verified (23GB env, `--check-only`):

```text
python               3.12.12      executable /data/user_data/vijayv/envs/slime/bin/python
torch                2.11.0+cu129     torch.version.cuda 12.9
sglang               0.5.15.post1
megatron.core        0.16.0rc0
slime                0.3.2
transformer_engine   2.16.1
flash_attn           2.8.3+cu12torch2.11cxx11abiTRUE   (cuda ext imports ok)
transformers         5.12.1        ray 2.58.0
torch_memory_saver   0.0.9.post1+slime   (both native hook .so present)
custom_rm resolved: <function custom_rm ...>
```

All seven sglang and Megatron patches applied cleanly. The verifier self-test
passes in the env (30/30 on `full_verified`).

flash-attn was additionally checked numerically on an A6000 (sm_86), not just
imported: `flash_attn_func` and `flash_attn_varlen_func` both match an fp32
reference attention to 2.4e-3 relative error, and TransformerEngine's
`DotProductAttention` reports `Running with FlashAttention backend (version
2.8.3)` and matches to the same error in `bshd` and `sbhd`. TE is the real
consumer here -- it is what Megatron uses during training; SGLang uses flashinfer
for rollout on these cards.

flash-attn needs no compile: `build_conda.sh` pins a prebuilt wheel by SHA256
whose tags already match (`cp312`, torch 2.11, cu12, `cxx11abiTRUE`), because
upstream publishes no such build. The smoke test imports `flash_attn_2_cuda`
explicitly, since an ABI mismatch would otherwise install cleanly and only fail
at first use. Same reasoning for the `torch_memory_saver` hook check — a
python-only build of it installs fine and only fails later inside SGLang, taking
`--colocate` with it.

Three deliberate deviations from `build_conda.sh`, each avoiding a multi-hour
source build this task does not need:

- **apex:** Megatron guards its apex imports (`fused_layer_norm.py`,
  `gpt_layer_specs.py`, and `backends.py`) and falls back to TransformerEngine's
  norm, which slime uses. Verified against the pinned tree. Re-enable with `INSTALL_APEX=1`.
- **FlashQLA:** Optional GDN backend for Qwen3.5/Qwen3-Next; requires SM90+.
  This cluster uses L40S/L40 (SM89), A6000 (SM86), and A100 (SM80), and Qwen3-4B
  does not use it. Re-enable with `INSTALL_FLASHQLA=1`.
- **int4_qat:** Only needed for int4 quantization-aware training.
  Re-enable with `INSTALL_INT4QAT=1`.

### Runtime traps, all found by actually generating (not by imports)

An import-level smoke test passes with every one of these still broken, so
`rl/setup_slime_conda.sh` and the launcher handle them explicitly:

- **flashinfer JIT.** SGLang picks flashinfer on pre-Hopper cards (fa3 needs
  SM90+; this cluster is SM80/86/89). Without prebuilt kernels it JIT-compiles at
  runtime inside the scheduler. The reference image ships
  `flashinfer-jit-cache 0.6.12+cu129`; `build_conda.sh` never installs it because
  its Docker base already has it, so step `06b` pins the same version.
- **sglang JIT + libstdc++.** SGLang still JIT-compiles a few kernels itself
  (fused RoPE for Qwen3) with the env's g++ and `dlopen`s them. The objects need
  `GLIBCXX_3.4.31`; the system `/lib64/libstdc++.so.6` stops at `3.4.29`, so the
  engine dies on its first forward pass. The launcher puts `$CONDA_PREFIX/lib`
  first on `LD_LIBRARY_PATH`.
- **JIT caches default under `$HOME`**, which is at 92%. `TVM_FFI_CACHE_DIR` and
  `FLASHINFER_WORKSPACE_BASE` are pointed at `/data`; eight scheduler processes
  writing compiled objects into a full home is not a thing to discover mid-run.
- **Activate the env, do not call its python directly.** Running
  `/data/.../envs/slime/bin/python` leaves `trl` first on `PATH`, and flashinfer's
  JIT then shells out to `trl`'s nvcc 13.3 against a cu129 build. `run_in_slime`
  activates properly and sets `CUDA_HOME`; ad-hoc invocations do not.

Verified end to end in the env: SGLang generates on real prompts and the verifier
grades both paths (2 impossible -> `incorrect_on_impossible`, 2 possible ->
`passed`, mean reward 0.5, matching the base model's ~0% abstention rate).

Two conda traps on this machine, both handled in the scripts:

- `conda` is **aliased to a nonexistent `/home/vijayv/anaconda3/bin/conda`** in
  interactive shells. Aliases don't apply in non-interactive scripts, but both
  scripts resolve the base from `$CONDA_EXE` rather than depend on that.
- The env's `cuda-nvcc` ships an `activate.d` hook that reads `$NVCC_PREPEND_FLAGS`
  and `$NVCC_APPEND_FLAGS` unguarded, so `conda activate` aborts under `set -u`.
  Export both (or lift `-u` across activation) in any script that activates this
  env.
- The system conda is 22.11.1 with the classic solver and no libmamba plugin.
  Solving the CUDA 12.9.1 metapackage against the NVIDIA channels with it ran
  past 5GB RSS without finishing. `setup_slime_conda.sh` fetches micromamba for
  that one step — which is what upstream's `build_conda.sh` does anyway — and
  uses plain pip for everything after.

### Container fallback

`rl/setup_slime_container.sh` fetches the upstream apptainer image to
`/data/user_data/vijayv/containers/slime.sif` (18GB, already built and verified
here: python 3.12.3, torch 2.11.0+cu129, sglang 0.5.15.post1, megatron.core
0.16.0rc0, `slime` importable, verifier self-test 30/30 inside it). Pass `--sif
<path>` to the launcher to train in the image instead of the conda env. Secondary
path, kept because it is a known-good copy of the same stack.

If you use it: `$HOME` is bound into the container, so a **login** shell
(`bash -lc`) sources the host `~/.bashrc`, activates host conda, and puts its
python (3.9, torch 2.4) ahead of the image's on `PATH` — the job then runs against
the wrong interpreter with no error. The scripts use `--cleanenv` and a non-login
`bash -c` for that reason.

## Data

```bash
python rl/build_rl_data.py \
    .hidden_agents/vijay_manual_8_23_26/training_rows_full.jsonl \
    --out-dir rl/data/full --val-frac 0.05
```

Each output row is slime's standard prompt shape:

```jsonc
{"prompt":   [<system>, <user>],           // prompt_building.build_candidate_prompt
 "label":    "<question_id>",
 "metadata": {"gold_response": "<zlib+b64>",   // the only impossibility signal
              "problem":       { /* checkers.PROBLEM_FIELDS */ }}}
```

The prompt is byte-identical to the one SFT trained on and the one
`evaluate_model.py` evaluates on — same `build_candidate_prompt` call. Verified:
of the 1426 RL training prompts, the 1305 whose problems also survived into
`full_long/generated_raw_trl_format.jsonl` match that file's user *and* system turns
byte-for-byte; the other 121 are rows the SFT conversion dropped upstream (over its
18800-token cap, or empty `thinking`), not rendering differences. Impossibility is
re-derived from `gold_response` exactly as `evaluate_model.is_impossible_row` does;
nothing about *how* a row was made unsolvable is carried over.

`val.jsonl` is slime's `--eval-prompt-data`: an in-distribution reward tracker over
generated problems. It is not the benchmark. The benchmark stays
`evaluate_model.sh --split test`.

## Reward

`impossible_lcb_rm.custom_rm` reuses `evaluate_model.grade_row`'s two paths, so
`mean(reward)` is the study's `accuracy` metric measured on training data:

| row | outcome | reward |
|---|---|---|
| impossible | abstained | 1.0 |
| impossible | did not abstain | 0.0 |
| possible | code passes its tests | 1.0 |
| possible | code fails | 0.0 |
| possible | abstained | 0.0 |

Two properties are load-bearing:

- **Impossible rows never execute code.** They are graded by ground-truth
  comparison alone. This is what makes the abstention half unhackable: a policy that
  hardcodes the prompt's (poisoned) expected output scores 0 regardless of what it
  prints.
- **Possible rows are graded on public tests that the prompt already showed the
  model** — these generated problems ship no private tests. That half *is* hackable
  by hardcoding, so `LCB_RM_REJECT_VERBATIM=1` (default) scores 0 when the extracted
  code embeds one of its own test cases verbatim (>=16 chars). Watch `hacked` in
  `$OUTPUT_DIR/logs/reward_dump.jsonl`.

The abstention check is a case-insensitive substring match anywhere in the response,
which is what `evaluate_model.grade_row` does. In thinking mode the response includes
the `<think>` trace (`evaluate_model.sh` runs `vllm serve` with no
`--reasoning-parser`, so `message.content` is the raw completion — confirmed in the
existing `predictions.jsonl` files), so RL and the benchmark see the same text and
credit abstention the same way. A policy cannot game this by emitting the marker
mid-trace and then also answering: on a possible row the abstention check fires
first and scores 0.

Grading runs in a pool of long-lived `_grade_worker.py` subprocesses, not inline
and not in a `multiprocessing` pool. LiveCodeBench's `check_correctness` forks a
child and installs a `reliability_guard` in it, which is not safe from a Ray actor
with CUDA initialized; and `multiprocessing` with `spawn` re-imports the parent's
`__main__` in every worker, which inside a Ray actor is Ray's own worker entry
module. Plain subprocesses over pipes avoid both, and make a hung grader
recoverable — the parent kills it and starts a new one, which a
`ProcessPoolExecutor` future cannot do. The pool size doubles as the admission
limit, so the per-grade timeout measures execution rather than queueing.

Self-test the verifier without a GPU (grades each row's own gold response, which
must score 1.0):

```bash
LCB_RM_DUMP=rl/data/full/gold_selftest.jsonl \
  python rl/impossible_lcb_rm.py --self-test rl/data/full/train.jsonl
```

### Measured: ~17% of the "possible" rows are unwinnable

Running that self-test over all 1501 rows:

```text
correct_abstention  530     (every impossible row, trivially)
passed              809
failed              162     <- gold code fails its own public tests
```

162 of 971 possible rows (16.7%) have a gold solution that does not pass the tests
shipped with them, usually a format disagreement — the gold code raises
`ValueError: not enough values to unpack` reading its own sample input. No
completion can score 1 on those rows, so their group is always all-zero and
contributes no gradient. They are also, note, current SFT targets.

A repeat pass over the 1426-row train split returned 151 failures where the first
returned 156, so roughly five rows sit close enough to the 6-second per-test
timeout to flip between runs. The population is ~16-17% either way; the exact drop
list is not reproducible to the row.

`--drop-unverifiable` removes them; this raises the impossible share from 35.3% to
39.6%, which is why it is opt-in rather than the default:

```bash
python rl/build_rl_data.py <source> --out-dir rl/data/full_verified \
    --drop-unverifiable rl/data/full/gold_selftest_all.jsonl
# 1339 rows kept (530 impossible, 39.6%); 162 dropped
```

## Before launching: check that GRPO has signal

GRPO's advantage is group-relative, so a prompt whose every sample scores the same
contributes exactly zero gradient. The diagnosis measured the base model abstaining
on essentially no impossible row (0/61 in thinking mode) despite separating the two
classes well (AUC 0.86 for "abstaining is the better action"). If that holds at
rollout temperature, every impossible group is degenerate and GRPO trains on the
possible rows alone — learning to code, not to abstain.

```bash
python rl/measure_group_variance.py \
    --model-path <init checkpoint> --data rl/data/full_verified/train.jsonl \
    --no-thinking --limit 96 -n 8 \
    --generations-cache rl/data/gen_<tag>.jsonl
```

It reports the share of groups with non-zero reward variance, split by row type.
Generation and grading are separable via `--generations-cache`, so a grading failure
costs a re-grade rather than another sampling pass.

### Measured: base is a dead start, the SFT checkpoint is not

Same 96 prompts, no-thinking, temperature 1.0, n=8, `rl/data/full_verified`:

| | base Qwen3-4B | `EXMEAN_Qwen3-4B_lr2e-5_nothink` |
|---|---|---|
| impossible groups with signal | **1/32 (3.1%)** | **32/32 (100%)** |
| possible groups with signal | 25/64 (39.1%) | 15/64 (23.4%) |
| overall groups with signal | 26/96 (27.1%) | 47/96 (49.0%) |
| abstained, impossible rows | 2/256 samples (0.8%) | 133/256 samples (52.0%) |
| mean reward, impossible | 0.008 | 0.520 |
| mean reward, possible | 0.273 | 0.104 |
| mean reward, overall | 0.185 | 0.242 |

From base, GRPO would get essentially nothing from the impossible half: one usable
group in 32. More samples do not rescue it — at 0.8% abstention per sample you would
need ~86 samples per prompt for half the impossible groups to carry signal.

From the SFT checkpoint, **every** impossible group carries signal, because the
policy abstains about half the time and is therefore wrong about half the time. It
also over-abstains on possible rows (269 of 512 possible-row samples are
`false_abstention`, which is why its possible-row reward is *lower* than base's) —
and that is the other half of the reason this is the right init. Both classes are
mixed, so both push: on an impossible row abstaining scores 1 and coding scores 0,
on a possible row the reverse. That is exactly the trade the SFT-trained bias could
not make, since it moved one threshold for every row at once.

Overall usable groups nearly double, 27.1% -> 49.0%.

One number here runs against the diagnosis and is worth naming rather than burying:
conditioned on actually attempting, the SFT checkpoint passes 53/243 (21.8%) of
possible samples where base passes 140/506 (27.7%) — whereas on the test set
`pass@1|attempted` went the other way (0.343 -> 0.435). Different data (generated
training problems, which the SFT run trained on, vs held-out LiveCodeBench) and
different temperature (1.0 vs the eval's 0.6), so the two are not comparable; do not
read either as evidence about the other.

So: initialise from the SFT checkpoint, not base.

```bash
sbatch rl/run_rl_training_and_evaluation.sh \
    --rl-data-dir rl/data/full_verified --no-thinking \
    --init-checkpoint /data/user_data/vijayv/tir5/agent_data_models/coding/EXMEAN_Qwen3-4B_lr2e-5_nothink
```

## Launching

```bash
sbatch rl/run_rl_training_and_evaluation.sh --rl-data-dir rl/data/full_verified
sbatch rl/run_rl_training_and_evaluation.sh --rl-data-dir rl/data/full_verified --no-thinking
```

Requests one 8-GPU node. The script converts the init checkpoint to Megatron
`torch_dist` (cached per checkpoint), runs GRPO with SGLang colocated on the same
GPUs, saves HF checkpoints via slime's `--save-hf`, and finishes by running
`./evaluate_model.sh --split test` on the last one from the `trl` env — the same
eval every other model in this study gets.

Defaults: 100 rollout steps x 32 prompts x 8 samples, `--global-batch-size 256`,
lr 1e-6 constant, `--eps-clip-high 0.28`, KL coefficient 0. Response cap is 16384
in thinking mode and 3072 without (no-thinking answers on this data measured
329-3260 tokens). TP=2 for both Megatron and SGLang, `--sglang-mem-fraction-static
0.5` and `--max-tokens-per-gpu 8192` for 48GB-class cards.

`--filter-zero-std` turns on DAPO-style dropping of all-same-reward groups. It is
off by default: it saves the training step from all-zero groups, but it also
reshapes the training mix, and early on the groups most likely to come back
all-zero are exactly the impossible ones.

## Process reward for thinking mode

In thinking mode the strict reward gives 0 to every impossible row the model does
not abstain on — which, measured, is essentially all of them: the base thinking
run raises doubt in ~80% of impossible traces and abstains on **0.3%**. Every
impossible group therefore comes back all-zero and GRPO gets no gradient on the
class this task is about.

`--doubt-reward R` (default 0, off) pays partial credit when an impossible row's
**closed** thinking trace reached the right conclusion but the answer still wrote
code. Calibrated on the base thinking test run (303 impossible / 932 possible
completed traces), by number of doubt matches over the whole trace:

| threshold | fires on impossible | fires on possible | ratio | separation |
|---|---|---|---|---|
| >=1 | 86.5% | 42.8% | 2.02 | 43.7 pp |
| >=2 | 78.2% | 28.8% | 2.72 | 49.5 pp |
| **>=3** (default) | **73.6%** | **22.0%** | **3.35** | **51.6 pp** |
| >=4 | 69.0% | 17.3% | 3.99 | 51.7 pp |
| >=6 | 54.5% | 11.2% | 4.88 | 43.3 pp |

`>=1` is near-noise — it fires on 43% of *possible* traces too. Separation is flat
across 3-4; 3 keeps more coverage, which matters because a shaped reward only
helps where it varies *within* a prompt's group. Tail-restricted variants (doubt
in the last 4000 chars) reach a better ratio, up to 5.4x, but only ~28% of
impossible traces, so their counts are logged (`doubt_tail_matches`) for post-hoc
analysis rather than used as the gate.

Three properties that keep this honest:

- **R < 1.0**, so abstaining still dominates within the group. This is a stepping
  stone, not a resting place.
- **A closed `</think>` is required.** A truncated trace contains no answer at
  all; paying for one would reward running out of tokens.
- **`base_reward` is logged alongside `reward`**, so `mean(base_reward)` stays
  comparable to `evaluate_model.sh`'s accuracy even with shaping on. The
  summarizer prints `reward=0.250 (metric=0.167)` when they diverge.

The degenerate solution is to spam doubt words and sit at R forever. The tell is
doubt-match medians rising while the abstention rate stays flat;
`summarize_rollouts.py` prints that pair. Raise `--doubt-min` if it shows up.

The detector is verbatim from `diagnostics/notices_but_acts.py`, so these numbers
stay comparable to the diagnosis's own "the trace notices but the answer ignores
it" finding.

## Rollout logging

Every sampled completion is written with its reward and how that reward was
reached, to `$OUTPUT_DIR/logs/rollouts/` (override with `--rollout-log-dir`), one
file per writing process — records carry full completions, so they run past
`PIPE_BUF` and concurrent appends from several rollout actors would interleave
mid-line.

```jsonc
{"ts", "rollout_id", "group_index", "index", "label",
 "impossible", "reward", "category", "abstained", "hacked",
 "passed_public", "passed_private", "n_public_tests",
 "response_tokens", "response_chars", "response", "code"}
```

The prompt is omitted by default — it is fully determined by `label` plus the RL
data file, and including it would repeat ~25KB once per sample in a group. Set
`LCB_RM_LOG_PROMPT=1` if you want it inline.

```bash
python rl/summarize_rollouts.py $OUTPUT_DIR/logs/rollouts
python rl/summarize_rollouts.py $OUTPUT_DIR/logs/rollouts --hacked
```

The summary deliberately leads with the breakdown by row type rather than mean
reward, because a run can raise mean reward purely by getting better at coding
while never learning to abstain — which is the failure the diagnosis already
found. Watch `abstain` on impossible rows and `false_abst` on possible rows move
in opposite directions.

## Open items

- Not yet run end-to-end on 8 GPUs. Everything up to the slime invocation has been
  exercised (including `run_in_slime`'s nested quoting, against a stand-in env);
  the slime arg block is adapted from `scripts/run-qwen3-4B.sh` and has not been
  validated against L40S memory in practice. Expect to tune
  `--sglang-mem-fraction-static` / `--max-tokens-per-gpu` on the first run.
- The verbatim-test guard is a heuristic. If `hacked` fires on more than a few
  percent of possible rows, the reward needs held-out tests, which this generated
  data does not have.
- Thinking-mode RL at 16384 response tokens is expensive. The cheaper experiment is
  `--no-thinking`, which is also the mode where SFT was already shown to move the
  metric (0.261 -> 0.334).

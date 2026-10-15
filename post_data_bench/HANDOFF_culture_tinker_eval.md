# Handoff: culture evaluation on Tinker

Developer-only notes for a later session. This file is not copied into agent
sandboxes: `launch_agent.py` binds only `src/`, `docs/`, the task directory's
explicit allowlist (`TASK_PUBLIC_FILES`, plus `copy_validation_evaluator` in
`--training-mode sft-rl`), and two accounting helpers. The image recipes
(`apptainer/*.def`) bake only packaging metadata. Keep this file at the
`post_data_bench/` root. Do not move it into `docs/` or a task directory, and do
not quote it in a prompt: it names the benchmark.

Status: not implemented. Coding evaluates on Tinker; culture (NormAd) does not.

## Goal

Evaluate culture with `--server-backend tinker`, just as coding does: base
Qwen/Qwen3-8B or a `tinker://.../sampler_weights/...` path, sampled remotely,
charged to a Tinker dollar ledger, and producing the same `predictions.jsonl` /
`metrics.json` (headline `macro_f1`) as the vLLM path. Three use cases, in
order of value:

1. Host baseline: `culture/evaluate_model.sh --split val --server-backend tinker`
   writes `culture/model_outputs/Qwen__Qwen3-8B_tinker/val/`. The culture prompt
   already points agents at that directory, and `_task_public_copies` copies it
   when its contract and tasks hash match. Today it cannot be produced, so
   culture agents never see a baseline.
2. Host secret-test evaluation of a Tinker-trained culture student (needed once
   culture joins the `sft-rl` recipe).
3. Sandbox validation of an agent's Tinker SFT trials (also needs `sft-rl` for
   culture).

Phase 1 (use case 1) stands alone and is cheap. Phases 2 and 3 depend on making
the unified recipe accept culture, which is a separate project (see "Out of
scope").

## Current state

- `culture/evaluate_model.py` only talks to an OpenAI-compatible server
  (`VLLM_BASE_URL`, `AsyncOpenAI`). It has no `--backend` flag and its
  `--model-path` default is still `Qwen/Qwen3-4B`, which Tinker has no prices
  for. `generate_and_score_all` reads `resp` after its retry loop, so the request
  call is not yet a separable function.
- `culture/evaluate_model.sh` runs `../coding/eval_launcher.sh`, the shared
  launcher. Its Tinker branch is task-agnostic: it runs
  `python evaluate_model.py ... --backend tinker` with no shards and an output
  directory `<model>_tinker`. For culture, argparse rejects `--backend` after the
  `--validate-only` pass, so the launcher fails before sampling. The launcher's
  `MODEL_PATH` now defaults to `${PDB_STUDENT_MODEL:-Qwen/Qwen3-8B}` for both
  tasks.
- The launcher's default backend is Tinker (`${PDB_EVAL_BACKEND:-tinker}`), on
  the host too, until a Tinker/Local split of the benchmark. Culture cannot use
  that default yet, so `culture/evaluate_model.sh` exports
  `SERVER_BACKEND="${SERVER_BACKEND:-${PDB_EVAL_BACKEND:-vllm}}"`, its previous
  default; `evaluate_normad.sh` and `run_evaluate_only.sh` (which exports
  sglang) go through it.
- `launch_agent.py` rejects `--training-mode sft-rl` unless `--task coding` and
  `--model Qwen/Qwen3-8B`. Culture runs only in legacy `--training-mode sft`:
  the sandbox gets `validation_tasks_de_obfuscated.jsonl` (decoded gold labels)
  and `train_model.sh` but no evaluator. After the session the host runs local
  SFT, then `run_evaluate_only.sh --split test` (sglang) and prints
  `.hidden/model_outputs/<escaped output dir>/test/metrics.json`.
- `culture/FORGER_PROMPT.txt` tells agents to query the student with
  `pyforger.acquire.TinkerModel("{STUDENT_MODEL}")`. `configure_run_token_budgets`
  sets `FORGER_TINKER_USAGE_FILE` / `FORGER_TINKER_USD_LIMIT` to the SFT ledger
  for every task, so culture's TinkerModel calls draw on the $150 SFT budget.
- Every sandbox, culture included, gets `PDB_EVAL_BACKEND=tinker`,
  `PDB_STUDENT_MODEL=<--model>` and `TINKER_EVAL_BUDGET=sft`. They do nothing in
  culture today because it has no evaluator to read them.

## How coding does it (copy this)

- `coding/tinker_sampling.py` `TinkerChat(args, user_metadata)`: an async
  callable `messages -> (completion, finish_reason, usage)`. It renders the HF
  chat template (`enable_thinking` passed through), encodes without special
  tokens, and samples with the eos ids from `generation_config`
  (`<|im_end|>`, `<|endoftext|>`) as stops. It clamps `max_tokens` to the model
  context and decodes with `skip_special_tokens=True`, so `<think>...</think>`
  stays inline. It rejects `min_p != 0`. Each request calls `budget.check(worst
  case)`, then `budget.record(billed tokens, session_id)`. The ledger is the one
  named by `TINKER_EVAL_BUDGET` (`sft` | `rl` | `eval`, via
  `rl.tinker_budget.budget_from_env`), or else a record-only
  `<log_dir>/tinker_usage.json`. `usage_summary(rows)` totals tokens and
  estimated USD for `metrics["usage"]`.
- `coding/evaluate_model.py`:
  - `--backend {openai,tinker}` defaults to `tinker` when
    `PDB_EVAL_BACKEND=tinker`, and any other backend is refused there (merging
    shards is exempt).
  - `--model-path` defaults to `$PDB_STUDENT_MODEL`, else Qwen/Qwen3-8B.
  - `generate_and_grade_all` builds one `request(messages)` coroutine per
    backend. The Tinker one clamps `max_connections` to
    `rl.tinker_budget.max_concurrent_samples()` and builds `TinkerChat` in a
    thread, because its constructor blocks.
  - `_add_usage` attaches `usage_summary`.
- `coding/eval_launcher.sh`, Tinker branch: defaults `MAX_CONN` to
  `TINKER_MAX_CONCURRENT_SAMPLES` (600), puts test output under
  `.hidden/model_outputs`, and backs up old outputs with `FORCE=1`. Under
  `PDB_EVAL_BACKEND` it refuses other backends.
- Ledgers: `coding/rl/tinker_budget.py` (`TinkerBudget`, `PRICES_PER_MILLION`,
  `DEFAULT_LIMITS_USD = {"sft": 150, "rl": 400}`). The host's `eval` ledger is
  recorded without a limit, so an agent cannot starve it. `launch_agent.py`
  creates the run's ledgers in `configure_run_token_budgets` and forwards the
  `TINKER_*` and `FORGER_TINKER_*` variables.
- Sandbox copy: `coding/rl/agent_validation.copy_validation_evaluator` copies
  the explicit `EVALUATOR_FILES` allowlist. It passes every file through
  `scrub()`, which renames the vendored grader, and raises if `BENCHMARK_NAMES`
  survives in any content or path. It replaces row identities with
  `public_identity` (`val_017`), asserts that validation and test are disjoint,
  copies only the private-test blobs that validation rows reference, and writes
  `validation_manifest.json`. `load_tasks` then calls
  `verify_manifest(public=True)`.
- Host secret test: `coding/rl/agent_training.final_evaluation(sampler_path,
  out_dir)` runs `evaluate_model.sh --server-backend tinker --split test` with
  `TINKER_EVAL_BUDGET=eval` and `PDB_EVAL_BACKEND=tinker`, writing to
  `<final>/rl/.hidden/final_eval`. It hardcodes `CODING / "evaluate_model.sh"`.

## Culture evaluation loop and metrics (what must not change)

- Rows come from `validation_tasks.jsonl` or `.hidden/test_tasks.jsonl` and
  have `id, country, axis, subaxis, value, rule_of_thumb, situation, task_text,
  gold_label`. `gold_label` is zlib+base64 and is decoded only when scoring
  (`checkers.decode_answer`). `make_splits.py` drops neutral rows with empty
  explanations and hashes situations to put about 10% in validation.
- Prompt: `prompt_building.build_candidate_prompt(task_text)`, a fixed system
  prompt plus `Country: ...\nSituation: ...`, ending in "Answer: yes|no|neutral".
- Parse: `checkers.parse_prediction` keeps only the text after the last
  `</think>`. An unclosed `<think>` scores as unparseable unless thinking is
  disabled.
- Metrics: `compute_metrics` puts headline `macro_f1` (and stderr) over
  yes/no/neutral first, and accuracy, micro-F1 and per-class F1 under `context`.
  `n_truncated` counts `finish_reason == "length"`; Tinker's
  `seq.stop_reason` uses the same `"length"` / `"stop"` values (check this
  against the installed SDK).
- Resume and merge: `culture/eval_state.py` re-exports `coding.eval_state`
  (`check_resume` identity: model path, split, limit, shards, sampling
  settings). The backend is not part of the identity, so keep Tinker outputs in
  the launcher's `_tinker` directory and never resume a vLLM directory with
  Tinker. `--from-raw` rescoring and `--merge-shards` sample nothing; leave them
  alone.

## Implementation plan

### Phase 1: culture evaluator with a Tinker backend (host only)

1. `culture/evaluate_model.py`
   - Add `--backend {openai,tinker}` with coding's default and pin check, and
     change the `--model-path` default to
     `os.environ.get("PDB_STUDENT_MODEL") or "Qwen/Qwen3-8B"`.
   - Split `generate_and_score_all` into a `request(messages) -> (completion,
     finish_reason, usage)` per backend, as coding does. Use `finish_reason`
     from that return value instead of the leaked `resp`. Store `usage` on each
     output row and add `metrics["usage"] = usage_summary(rows)` in `_finalize`.
   - Pass `user_metadata={"purpose": "pdb-culture-eval", "split": ...}`.
2. Reuse `TinkerChat`; do not fork it. It imports `rl.tinker_budget`, which
   resolves only when `coding/` is on `sys.path`. Either add a
   `culture/tinker_sampling.py` shim that inserts `parents[1] / "coding"` (the
   pattern `culture/eval_state.py` already uses) and re-exports `TinkerChat` and
   `usage_summary`, or move `tinker_sampling.py` and `rl/tinker_budget.py` to a
   shared package. Pick whichever also works from the flattened sandbox copy in
   Phase 2.
3. `coding/eval_launcher.sh` should need no change. Confirm that culture
   accepts every `DEFAULT_ARGS` flag (`--top-k 20 --min-p 0` today). Remove
   the local `SERVER_BACKEND` default from `culture/evaluate_model.sh` (and
   update `test_culture_keeps_its_local_default`) so culture follows the
   launcher's Tinker default.
4. Produce the baseline:
   `PDB_BENCHMARK_DIR=/path/to/culture bash culture/evaluate_model.sh --split val --server-backend tinker`.
   It writes `culture/model_outputs/Qwen__Qwen3-8B_tinker/val/`, the directory
   `_task_public_copies` looks for. Its `evaluation_contract` and `tasks_sha256`
   must match, or the copy is skipped silently. Culture ids are dataset row ids,
   so check whether they reveal anything before copying predictions (coding
   remaps its ids with `public_question_ids`).

### Phase 2: sandbox validation (after culture joins `sft-rl`)

1. Generalize `copy_validation_evaluator` per task, or add a culture variant.
   The coding version assumes `private_test_cases.txt`, `gold_response`,
   `PUBLIC_FIELDS` and `public_identity`. Culture needs:
   - `evaluate_model.py`, `checkers.py`, `prompt_building.py`, `eval_state.py`,
     `split_contract.py`, `tinker_sampling.py`, `rl/tinker_budget.py`,
     `eval_launcher.sh` and a sandbox `evaluate_model.sh`.
   - Culture's `evaluate_model.sh` runs `../coding/eval_launcher.sh`, and
     `eval_state.py` and `split_contract.py` import `coding.*` through
     `sys.path`. Neither path exists in the sandbox. Write flattened copies that
     import locally, and test that the copy runs without `../coding`.
   - Add `normad` to `_SCRUB` (for example `"this benchmark"` or
     `"the task"`). It is already in `BENCHMARK_NAMES`, so copying today's
     culture files, whose docstrings say "NormAd", raises. Never copy
     `make_splits.py`, `set_up_data.sh` or `evaluate_normad.*`: they name the
     dataset (`akhilayerukola/NormAd`).
   - Copy validation rows only, with a public manifest, after
     `assert_disjoint(validation, test)`. Write
     `validation_tasks_de_obfuscated.jsonl` there, since the `sft-rl` branch of
     `_task_public_copies` skips the legacy decoded-validation writer.
2. `launch_agent.py`: add culture to the `sft-rl` `TASK_PUBLIC_FILES` handling.
   The current loop builds `rl/...` paths under the culture task directory,
   where they do not exist, and `_copy_into` skips missing files silently.
   Source them from `coding/rl/`.
3. Prompt: add a "Training and Validation" section to the culture prompt like
   coding's (`evaluate_model.sh --split val --server-backend tinker
   --model-path <sampler_path> --out-dir ...`). It must name no benchmark or
   source.

### Phase 3: host secret test

- Make `agent_training.final_evaluation` choose `<task>/evaluate_model.sh` from
  `PDB_TASK` or `--task` instead of `CODING`, and keep `--server-backend
  tinker`, `PDB_EVAL_BACKEND=tinker` and `TINKER_EVAL_BUDGET=eval`.
- Point `METRICS_PATH` and the log output at `rl/.hidden/final_eval/metrics.json`
  for culture runs.

### Out of scope here (blocks phases 2 and 3)

The `sft-rl` recipe is coding-shaped. `RAW_SCHEMA["culture"]` is
country/situation/question/thinking/answer, not prompt/ground-truth/rubric
triples. `rubric_data`, `rubric_rm` and the final RL defaults (`PDB_TASK` and
`default_rl_limit(task)` exist, but nothing has been validated for culture), and
`agent_training`'s fixed SFT and RL, would all need culture support before the
`parser.error` in `launch_agent.py` can be lifted.

## Test plan

- Unit (no network): reuse the fake `tinker` and `transformers` modules from
  `PARITY` in `tests/test_tinker_model_wiring.py`. Run culture's
  `evaluate_model.py --backend tinker` on a three-row temporary benchmark that
  has a manifest. Assert:
  - `predictions.jsonl` and `metrics.json` (with `macro_f1` and `usage`) are
    written;
  - `TINKER_EVAL_BUDGET=sft` charges the SFT ledger, and without it a
    record-only `<log_dir>/tinker_usage.json` is used;
  - `--backend openai` is refused under `PDB_EVAL_BACKEND=tinker`;
  - an unclosed `<think>` scores as unparseable.
- Launcher: extend `EvalLauncherBackendTests` (fake `python`) with a culture
  task directory. Check that output goes to
  `model_outputs/Qwen__Qwen3-8B_tinker/val`, and that `--split test` goes under
  `.hidden/`.
- Parity: `TinkerModel` and `TinkerChat` must send identical requests for
  culture's system plus user prompt (the coding parity test covers the
  mechanism).
- Sandbox copy (phase 2): copy into a temporary directory and assert:
  - no `BENCHMARK_NAMES` match in any path or content;
  - no `.hidden`, test rows, `make_splits.py` or dataset id;
  - `--validate-only` and `--limit 1` with fake Tinker run from the copy alone.
- Manual and paid, only with the user's go-ahead: run `--split val --limit 5`
  on base Qwen3-8B, check `cost_usd` in the ledger, then run the full
  validation baseline.

## Pitfalls

- Agents must never see benchmark or source names (NormAd, LiveCodeBench, LCB,
  PRBench, HealthBench, LeetCode, AtCoder, Codeforces) or test data. Anything
  copied into a sandbox goes through an allowlist and `scrub()`. A rename that
  misses a docstring raises at launch, which is the intended guard.
- `.hidden/` (test splits, test outputs) must stay off every sandbox bind. Test
  outputs go to `.hidden/model_outputs` (launcher) or `rl/.hidden/final_eval`
  (host).
- Costs: the agent's SFT trials, validation evaluations and TinkerModel calls
  share $150 (`TINKER_SFT_*`); RL probes and final RL share $400
  (`TINKER_RL_*`); host secret-test evaluation is recorded on the `eval` ledger
  with no limit. Prices per million tokens: prefill $0.195 ($0.039 cached),
  sampled $0.60. A culture prompt is about 200 tokens, so cost is dominated by
  sampled thinking: with ~2k sampled tokens per row it is about $0.0013 a row,
  a few dollars for the full test split. Each request's pre-check reserves its
  worst case, `max_tokens` x $0.60/M (about $0.017 at 28,672).
- `TinkerChat` checks the ledger against recorded spend only. Unlike Forger's
  `TinkerUsageLedger`, `TinkerBudget` has no in-flight reservation, so up to
  `max_connections` concurrent requests can pass the same remaining dollars.
  With 600 in flight that is up to about $10 past the cap. Coding has the same
  gap. If culture validation runs inside the $150 SFT cap, add a reservation to
  `TinkerBudget` (as in commit 9977076 for pyforger) or lower `MAX_CONN`.
- Tinker has no `min_p`; keep `--min-p 0`.
- Never resume a directory across backends; see the resume identity above.
- The baseline directory name is fixed in two places: `launch_agent.py`
  (`"Qwen__Qwen3-8B_tinker"`) and the culture prompt. The launcher derives it
  from `--model-path`, so the baseline must be run with the default or with
  exactly `Qwen/Qwen3-8B`.

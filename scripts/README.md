# Babel HPC eval scripts

End-to-end runners for the two evaluation pipelines we currently care about.

## Qwen3-1.7B on GPQA Diamond

Routes through the Forger Inspect bridge so the eval uses Qwen3's real chat
template and keeps `VLLMBackend`'s shared-engine registry.

- [`eval_qwen3_gpqa.py`](eval_qwen3_gpqa.py) — the eval entrypoint.
- [`eval_qwen3_gpqa.sbatch`](eval_qwen3_gpqa.sbatch) — Slurm job for Babel's
  `debug` partition (1 GPU).

## Forger data-quality on HumanEval (ICL uplift)

Estimates the quality of a small Forger candidate dataset by measuring how
much using its examples as few-shot demos raises Qwen3-1.7B's HumanEval
pass@1 over a zero-shot baseline.

- [`eval_forger_humaneval.py`](eval_forger_humaneval.py) — the eval
  entrypoint, wired to `pyforger.evaluate.estimate_quality`.
- [`eval_forger_humaneval.sbatch`](eval_forger_humaneval.sbatch) — Slurm job
  (1 GPU, 2 hours). Set `FORGER_JSONL=/abs/path/to/coding_questions_trl_format.jsonl`
  in the submitting environment.

Smoke test first (interactive, on a `debug` GPU allocation):

```bash
srun --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --pty bash

source /data/user_data/$USER/envs/forger/bin/activate
export HF_HOME=/data/user_data/$USER/hf_cache

python scripts/eval_forger_humaneval.py \
    --forger-jsonl /abs/path/to/coding_questions_trl_format.jsonl \
    --n-demos 20 --n-val 5 --k 2 --n-subsets 2 \
    --allow-code-execution
```

Full Tuesday run:

```bash
FORGER_JSONL=/abs/path/to/coding_questions_trl_format.jsonl \
    sbatch scripts/eval_forger_humaneval.sbatch
```

Output: `logs/forger-humaneval-<jobid>.json` with `metric` (ICL uplift),
`baseline_pass_at_1`, `conditioned_pass_at_1`, a 95% bootstrap CI, and
per-item breakdown.

`--allow-code-execution` is **required**: HumanEval scoring runs
model-generated code in a hardened subprocess (env scrubbed, RLIMIT_AS,
wall-clock timeout). This is not a true sandbox; only enable on machines
you control.

---

## Common Babel setup

vLLM requires CUDA, so these only run on a GPU node — never on a login node.

## One-time setup on Babel

1. **Log in.**

   ```bash
   ssh <andrewid>@login.babel.cs.cmu.edu
   ```

2. **Create a venv on big storage.** `/home` is only 100GB; put the venv on
   `/data/user_data/$USER` (500GB).

   ```bash
   stat /data/user_data/$USER >/dev/null   # trigger AutoFS mount
   python -m venv /data/user_data/$USER/envs/forger
   source /data/user_data/$USER/envs/forger/bin/activate
   ```

3. **Install Forger + the inference and inspect extras.** Qwen3 needs newer
   vLLM/transformers than the floor pinned in [`pyproject.toml`](../pyproject.toml),
   and `inspect_evals` is a separate package.

   ```bash
   cd path/to/forger    # the repo root containing pyproject.toml
   pip install -e '.[inference,inspect]'
   pip install 'vllm>=0.8.5' 'transformers>=4.51'
   pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
   ```

4. **Authenticate to the Hugging Face Hub.** GPQA (`Idavidrein/gpqa`) is
   gated; accept the dataset license on the Hub first, then log in so the
   token is cached in `HF_HOME` (never hardcode it in the sbatch file).

   ```bash
   # Visit https://huggingface.co/datasets/Idavidrein/gpqa and accept the license.
   huggingface-cli login
   ```

5. **Point HF caches at big storage** so model weights / dataset don't fill
   `/home`. The sbatch script sets this automatically; for interactive use
   export it yourself:

   ```bash
   export HF_HOME=/data/user_data/$USER/hf_cache
   ```

## Smoke test first (interactive)

Always start with a tiny `--limit` so you catch problems (auth failure, wrong
task id, OOM, etc.) cheaply. The `debug` partition allows interactive sessions.

```bash
srun --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --pty bash

# Inside the allocation:
source /data/user_data/$USER/envs/forger/bin/activate
export HF_HOME=/data/user_data/$USER/hf_cache
python scripts/eval_qwen3_gpqa.py --limit 5
```

You should see weights download (first run only), the model load, five samples
evaluate, and a `metrics:` block print at the end.

## Full run (batch)

```bash
sbatch scripts/eval_qwen3_gpqa.sbatch
```

This requests one GPU on `debug` for up to 2 hours, runs all 198 Diamond
questions, and writes:

- `logs/qwen3-gpqa-<jobid>.out` / `.err` — stdout/stderr.
- `logs/inspect-<jobid>/` — Inspect's structured eval logs (viewable with
  `inspect view`).
- `logs/qwen3-gpqa-<jobid>.json` — the flattened metrics dict.

Check status with `squeue -u $USER`. Cancel with `scancel <jobid>`.

## Key CLI flags

```bash
python scripts/eval_qwen3_gpqa.py \
    --model Qwen/Qwen3-1.7B \
    --task inspect_evals/gpqa_diamond \
    --limit 198 \
    --max-tokens 4096 \
    --temperature 0.0 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --log-dir logs/inspect-local \
    --json-output logs/metrics.json
```

- `--enable-thinking` / `--no-thinking` toggles Qwen3's `<think>` traces.
  Thinking is on by default and is the published-numbers setting; turn it off
  to roughly halve generation time at some cost to accuracy.
- `--max-tokens` defaults to 4096 because thinking mode generates long
  reasoning traces; truncation hurts scoring.

## Throughput note (and the alternative)

This eval is intentionally serial (`--max-connections 1`). The Forger bridge
dispatches each Inspect request to offline `vLLM.LLM.generate` on a worker
thread, and offline vLLM is not designed for concurrent `.generate()` calls.
For a 1.7B model on 198 Diamond questions on a single GPU this is fine
(roughly tens of minutes with thinking on).

If you later need real throughput, use Inspect's native vLLM provider, which
batches internally:

```bash
inspect eval inspect_evals/gpqa_diamond \
    --model vllm/Qwen/Qwen3-1.7B \
    --max-connections 8
```

That path bypasses Forger entirely, so use it only when raw speed matters more
than going through your own backend abstraction.

## Troubleshooting

- `ImportError: pyforger[inspect]` — you skipped step 3; install the extras.
- `inspect_evals is not installed` — install from the GitHub source in step 3.
- 401 on the GPQA dataset — you didn't accept the license or didn't run
  `huggingface-cli login` in step 4.
- OOM at engine load — drop `--gpu-memory-utilization` to 0.5, or lower
  `--max-model-len` to 4096.
- `inspect_evals/gpqa_diamond` not found — verify the exact task id with
  `python -c "import inspect_evals.gpqa; print(dir(inspect_evals.gpqa))"`;
  task ids occasionally drift across `inspect_evals` releases.
- Hangs on first run — first invocation downloads ~3.5 GB of weights; watch
  `nvidia-smi` and the job's `.out` file.

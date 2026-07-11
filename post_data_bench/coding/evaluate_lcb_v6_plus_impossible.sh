#!/usr/bin/env bash
# Evaluate one model on the LiveCodeBench-plus-impossible benchmark, sharded across every GPU on
# the node.
#
# Architecture (same as new_eval_sweep_normad_2.sh): we manage the vLLM servers ourselves, one per
# GPU, each pinned via CUDA_VISIBLE_DEVICES to a distinct device and listening on a distinct port.
# Unlike that script (which shards a *list of models* across GPUs), here we only ever evaluate one
# model per invocation, so instead each GPU's server gets a disjoint round-robin shard of the
# *task list* (validation_tasks.jsonl or test_tasks.jsonl). With N GPUs the sweep is ~N x faster.
# evaluate_lcb_v6_plus_impossible.py itself never imports vllm -- it's a plain AsyncOpenAI client
# against whichever server this script points it at, so the eval script stays testable against any
# OpenAI-compatible endpoint.
#
# Usage:
#   ./evaluate_lcb_v6_plus_impossible.sh --split val [--model-path Qwen/Qwen3-4B] [extra args...]
#   ./evaluate_lcb_v6_plus_impossible.sh --split test --model-path /path/to/local/checkpoint
#
# Outputs go to: lcb_plus_impossible_outputs/{model_name_escaped}/{split}/
#   metrics.json          aggregate numbers, merged across shards
#   predictions.jsonl     per-instance predictions, merged across shards
#   shard_{k}/             each GPU shard's own predictions.jsonl / metrics.json / console log
set -u

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

MODEL_PATH="Qwen/Qwen3-4B"
SPLIT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$SPLIT" ]; then
  echo "!!! usage: $0 --split val|test [--model-path <path>] [extra evaluate_lcb_v6_plus_impossible.py args...]"
  exit 1
fi

MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_CONN="${MAX_CONN:-$MAX_NUM_SEQS}"
GPU_MEM="${GPU_MEM:-0.90}"
BASE_PORT="${PORT:-8900}"
API_KEY="${VLLM_API_KEY:-tok}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Qwen3's thinking traces on LiveCodeBench-hard problems routinely blow past 8192
# tokens before ever closing </think>, which silently zeroes out those rows (see
# HANDOFF.md/logs from the 2026-07-10 val run: 56% of completions were truncated
# mid-thought, 0% of those scored correct vs. 92% of completions that finished
# thinking). Leave enough headroom under MAX_MODEL_LEN for the (short, ~1-4k char)
# prompt.
MAX_TOKENS="${MAX_TOKENS:-28672}"

# Sampling params: match Qwen3's generation_config.json defaults (same convention as the AIME /
# NormAd sweeps in examples/training/inference_examples/).
SAMPLING_ARGS=(--temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
fi
unset CUDA_VISIBLE_DEVICES
NUM_GPUS="${#GPUS[@]}"
if [ "$NUM_GPUS" -eq 0 ]; then echo "!!! no GPUs found"; exit 1; fi
echo ">>> using $NUM_GPUS GPU(s): ${GPUS[*]}"

escape() { echo "$1" | sed 's#/#__#g'; }
ESC_MODEL="$(escape "$MODEL_PATH")"
OUT_DIR="lcb_plus_impossible_outputs/${ESC_MODEL}/${SPLIT}"
mkdir -p "$OUT_DIR"

wait_ready() {  # $1 = server log file
  local log="$1" i=0
  while true; do
    if grep -qE "Application startup complete|Uvicorn running on" "$log" 2>/dev/null; then return 0; fi
    if grep -qiE "Traceback|raise [A-Za-z]|ValueError|RuntimeError|CUDA out of memory|address already in use|Engine core .*(failed|dead)|EngineDeadError" "$log" 2>/dev/null; then
      echo "!!! server error:"; tail -n 25 "$log"; return 1
    fi
    i=$((i+1)); if [ "$i" -gt 300 ]; then echo "!!! timeout waiting for server"; return 1; fi
    sleep 5
  done
}

stop_server() {  # $1 = port, $2 = gpu id
  local port="$1" gpu="$2"
  pkill -f "vllm serve .*--port $port" 2>/dev/null
  local i used
  for i in $(seq 1 80); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" | head -1 | tr -d ' ')
    if [ "${used:-9999}" -lt 2000 ]; then break; fi
    sleep 3
  done
}

run_shard() {  # $1 = gpu id, $2 = port, $3 = shard index
  local gpu="$1" port="$2" k="$3"
  local tag="[gpu $gpu]"
  local svlog="/tmp/vllm_lcb_impossible_${ESC_MODEL}_shard${k}.log"
  local shard_dir="${OUT_DIR}/shard_${k}"
  mkdir -p "$shard_dir"

  # Idempotent resume: a metrics.json marks a completed shard. Set FORCE=1 to redo.
  if [ -f "${shard_dir}/metrics.json" ] && [ -z "${FORCE:-}" ]; then
    echo "$tag shard $k already complete (metrics.json present); skipping"
    return 0
  fi

  echo "$tag >>> starting vLLM server on port $port (log: $svlog)"
  : > "$svlog"
  setsid bash -c "CUDA_VISIBLE_DEVICES='$gpu' vllm serve '$MODEL_PATH' --port $port --api-key '$API_KEY' \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --gpu-memory-utilization $GPU_MEM >> '$svlog' 2>&1" < /dev/null &
  disown

  if ! wait_ready "$svlog"; then
    echo "$tag !!! giving up on shard $k"; stop_server "$port" "$gpu"; return 1
  fi
  echo "$tag >>> server ready, running shard $k/$NUM_GPUS"

  VLLM_BASE_URL="http://localhost:${port}/v1" VLLM_API_KEY="$API_KEY" \
  python evaluate_lcb_v6_plus_impossible.py \
    --model-path "$MODEL_PATH" \
    --split "$SPLIT" \
    --shard-index "$k" \
    --num-shards "$NUM_GPUS" \
    --max-tokens "$MAX_TOKENS" \
    --max-connections "$MAX_CONN" \
    --log-dir "$shard_dir" \
    --json-output-file "${shard_dir}/metrics.json" \
    "${SAMPLING_ARGS[@]}" "${EXTRA_ARGS[@]}" > "${shard_dir}/console.log" 2>&1
  local rc=$?
  echo "$tag <- shard $k exit=$rc"
  [ "$rc" -ne 0 ] && echo "$tag !!! FAILED: shard $k (see ${shard_dir}/console.log)"

  echo "$tag >>> stopping server for shard $k"
  stop_server "$port" "$gpu"
  return $rc
}

pids=()
for k in "${!GPUS[@]}"; do
  gpu="${GPUS[$k]}"
  port=$((BASE_PORT + k))
  run_shard "$gpu" "$port" "$k" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo ">>> all shards done, merging..."
shard_preds=()
for k in "${!GPUS[@]}"; do
  shard_preds+=("${OUT_DIR}/shard_${k}/predictions.jsonl")
done
python evaluate_lcb_v6_plus_impossible.py \
  --merge-shards "${shard_preds[@]}" \
  --out-dir "$OUT_DIR" \
  --model-path "$MODEL_PATH" \
  --split "$SPLIT"

echo "ALL_SHARDS_DONE $(date)"
exit "$fail"

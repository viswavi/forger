#!/usr/bin/env bash
# Evaluate several models x 2 benchmarks (AIME 2025, HumanEval) on a single GPU.
#
# Architecture: we manage the vLLM server ourselves (one per model, reused across
# both benchmarks) and point Inspect at it via VLLM_BASE_URL. This is deliberate:
#  - It lets us force Qwen3 thinking with a custom chat template that NEVER injects
#    an empty <think></think> block (templates/qwen3_think.jinja). Inspect's own
#    vLLM-launch path defaults Qwen3 to non-thinking and drops our template/kwargs,
#    so server-side templating under our control is the only robust knob.
#  - One server serves both benchmarks for a model -> a single weight load.
#
# Outputs (Inspect JSON logs + a metrics.json) go to:
#   {benchmark}_outputs/{model_name_escaped}/      ("/" -> "__")
set -u

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate trl

MAX_CONN="${MAX_CONN:-16}"
GPU_MEM="${GPU_MEM:-0.85}"
PORT="${PORT:-8800}"
API_KEY="${VLLM_API_KEY:-tok}"
THINK_TEMPLATE="$(pwd)/templates/qwen3_think.jinja"

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
  MODELS=(
    "Qwen/Qwen3-4B"
    "allenai/Olmo-3-7B-Instruct"
    "marin-community/marin-8b-instruct"
  )
fi

escape() { echo "$1" | sed 's#/#__#g'; }

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

stop_server() {
  pkill -f "vllm serve .*--port $PORT" 2>/dev/null
  # Wait for the GPU to actually free before the next model.
  local i used
  for i in $(seq 1 80); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${used:-9999}" -lt 2000 ]; then break; fi
    sleep 3
  done
}

run_eval() {  # $1 script, $2 bench, $3 model, rest: extra args
  local script="$1" bench="$2" model="$3"; shift 3
  local esc out; esc="$(escape "$model")"; out="${bench}_outputs/${esc}"
  # Idempotent resume: a metrics.json marks a completed run. Set FORCE=1 to redo.
  if [ -f "${out}/metrics.json" ] && [ -z "${FORCE:-}" ]; then
    echo "    == $bench already complete (metrics.json present); skipping"
    return 0
  fi
  mkdir -p "$out"
  echo "    -> $bench  ($(date +%H:%M:%S))"
  VLLM_BASE_URL="http://localhost:${PORT}/v1" VLLM_API_KEY="$API_KEY" \
  python "$script" \
    --model-path "$model" \
    --log-dir "$out" \
    --json-output-file "${out}/metrics.json" \
    --max-connections "$MAX_CONN" \
    --gpu-memory-utilization "$GPU_MEM" \
    "$@" > "${out}/console.log" 2>&1
  local rc=$?
  echo "    <- $bench exit=$rc"
  [ "$rc" -ne 0 ] && echo "    !!! FAILED: $bench :: $model (see ${out}/console.log)"
  return $rc
}

for model in "${MODELS[@]}"; do
  esc="$(escape "$model")"
  svlog="/tmp/vllm_${esc}.log"
  think_args=()
  aime_mt=()                       # empty -> script default (16000)
  human_mt=(--max-tokens 4000)
  case "$model" in
    *[Qq]wen3*)
      think_args=(--chat-template "$THINK_TEMPLATE")   # force thinking server-side
      aime_mt=(--max-tokens 32000)                     # doubled thinking budget
      human_mt=(--max-tokens 32000)                    # (Qwen3 ctx is 40960)
      ;;
    *marin*)
      # marin-8b is non-thinking and has only a 4096-token context window, so
      # cap output to leave room for the prompt (input + output must fit 4096).
      aime_mt=(--max-tokens 2048)
      human_mt=(--max-tokens 2048)
      ;;
  esac

  echo "==================================================================="
  echo ">>> MODEL: $model   ($(date))"
  echo ">>> starting vLLM server on port $PORT (log: $svlog)"
  : > "$svlog"
  setsid bash -c "vllm serve '$model' --port $PORT --api-key '$API_KEY' \
    --gpu-memory-utilization $GPU_MEM ${think_args[*]} >> '$svlog' 2>&1" < /dev/null &
  disown

  if ! wait_ready "$svlog"; then
    echo "!!! giving up on $model"; stop_server; continue
  fi
  echo ">>> server ready"

  run_eval post_train_bench_evaluate_aime.py      aime2025  "$model" "${aime_mt[@]}"
  run_eval post_train_bench_evaluate_humaneval.py humaneval "$model" "${human_mt[@]}"

  echo ">>> stopping server for $model"
  stop_server
done

echo "ALL_SWEEP_RUNS_DONE $(date)"

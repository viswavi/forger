#!/usr/bin/env bash
# Shared local-server evaluation launcher for coding and culture.
set -euo pipefail
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="$1"; shift
cd "$TASK_DIR"
# Without --model-path, evaluate the base student (agent sandboxes set PDB_STUDENT_MODEL).
MODEL_PATH="${PDB_STUDENT_MODEL:-Qwen/Qwen3-8B}"
SPLIT=
# Agent sandboxes set PDB_EVAL_BACKEND=tinker, so validation never starts a local server.
# Local-checkpoint callers pass --server-backend (or SERVER_BACKEND) explicitly.
SERVER_BACKEND="${SERVER_BACKEND:-${PDB_EVAL_BACKEND:-vllm}}"
OUT_DIR=
NUM_SHARDS=
INDICES=
NO_MERGE=0
NO_THINKING=0
EXTRA=()
GPU_MEM="${GPU_MEM:-0.9}"
while (($#)); do
  case "$1" in
    -h|--help) python evaluate_model.py --help; exit 0 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --server-backend) SERVER_BACKEND="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --shard-indices) INDICES="$2"; shift 2 ;;
    --no-merge) NO_MERGE=1; shift ;;
    --no-thinking) NO_THINKING=1; EXTRA+=(--disable-thinking); shift ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[[ "$SPLIT" == val || "$SPLIT" == test ]] || { echo 'Specify --split val|test' >&2; exit 2; }
[[ "$SERVER_BACKEND" == vllm || "$SERVER_BACKEND" == sglang || "$SERVER_BACKEND" == tinker ]] || { echo 'Unknown server backend' >&2; exit 2; }
[[ -z "${PDB_EVAL_BACKEND:-}" || "$SERVER_BACKEND" == "$PDB_EVAL_BACKEND" ]] || { echo "Only --server-backend $PDB_EVAL_BACKEND is available here" >&2; exit 2; }
# Tinker samples remotely, so default to its per-process cap rather than a GPU server's.
[[ "$SERVER_BACKEND" != tinker ]] || MAX_CONN="${MAX_CONN:-${TINKER_MAX_CONCURRENT_SAMPLES:-600}}"
DEFAULT_ARGS=(--model-path "$MODEL_PATH" --split "$SPLIT" --max-tokens "${MAX_TOKENS:-28672}" --max-connections "${MAX_CONN:-128}" --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0)
# Reject bad evaluator arguments before loading models or claiming GPUs.
python evaluate_model.py "${DEFAULT_ARGS[@]}" "${EXTRA[@]}" --validate-only
# Secret test predictions live under .hidden/, which agents never see.
OUT_ROOT=model_outputs
[[ "$SPLIT" != test ]] || OUT_ROOT=.hidden/model_outputs
if [[ "$SERVER_BACKEND" == tinker ]]; then
  # Tinker samples remotely (TINKER_API_KEY from the environment): no local server, GPUs or shards.
  ESC_MODEL="${MODEL_PATH//\//__}_tinker"
  ((NO_THINKING == 0)) || ESC_MODEL+="_no_thinking"
  OUT_DIR="${OUT_DIR:-$OUT_ROOT/$ESC_MODEL/$SPLIT}"
  mkdir -p "$OUT_DIR"
  if [[ "${FORCE:-0}" == 1 ]]; then
    for f in predictions.jsonl metrics.json evaluation_state.json; do
      [[ ! -f "$OUT_DIR/$f" ]] || mv "$OUT_DIR/$f" "$OUT_DIR/$f.bak.$$.${RANDOM}"
    done
  fi
  exec python evaluate_model.py "${DEFAULT_ARGS[@]}" "${EXTRA[@]}" --backend tinker --log-dir "$OUT_DIR" --json-output-file "$OUT_DIR/metrics.json"
fi
python -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else "Missing server backend: "+sys.argv[1])' "$SERVER_BACKEND"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)" || exit 1
  mapfile -t GPUS <<< "$GPU_LIST"
fi
((${#GPUS[@]})) || { echo 'No GPUs found' >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ || "$gpu" == GPU-* || "$gpu" == MIG-* ]] || { echo 'Invalid GPU identifier' >&2; exit 1; }
done
unset CUDA_VISIBLE_DEVICES
if [[ -n "$NUM_SHARDS" || -n "$INDICES" ]]; then
  [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ && -n "$INDICES" ]] || { echo 'Specify positive --num-shards and --shard-indices together' >&2; exit 2; }
  IFS=, read -ra SHARDS <<< "$INDICES"
  ((${#SHARDS[@]} == ${#GPUS[@]})) || { echo 'One shard index is required per GPU' >&2; exit 2; }
else
  NUM_SHARDS="${#GPUS[@]}"; SHARDS=("${!GPUS[@]}")
fi
declare -A SEEN=()
for k in "${SHARDS[@]}"; do
  [[ "$k" =~ ^[0-9]+$ ]] && ((10#$k < NUM_SHARDS)) || { echo 'Shard index out of range' >&2; exit 2; }
  [[ -z "${SEEN[$k]:-}" ]] || { echo 'Duplicate shard index' >&2; exit 2; }
  SEEN[$k]=1
done
ESC_MODEL="${MODEL_PATH//\//__}"
((NO_THINKING == 0)) || ESC_MODEL+="_no_thinking"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/$ESC_MODEL/$SPLIT}"
mkdir -p "$OUT_DIR"
# Use per-invocation local credentials, never a host provider credential.
API_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export VLLM_API_KEY="$API_KEY"
BASE_PORT="${PORT:-$((10000 + RANDOM % 1000 * 32))}"
((${#GPUS[@]} <= 32 && BASE_PORT > 0 && BASE_PORT + ${#GPUS[@]} < 65536)) || exit 2
THREADS=$(( ${SLURM_CPUS_ON_NODE:-$(nproc)} / ${#GPUS[@]} ))
((THREADS > 0)) || THREADS=1
pids=()
cleanup_workers() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_shard() {
  local gpu="$1" port="$2" k="$3" position="$4" lock
  server_pid=; client_pid=
  cleanup_shard() {
    trap - EXIT INT TERM
    for pid in "$server_pid" "$client_pid"; do [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true; done
    sleep 1
    for pid in "$server_pid" "$client_pid"; do [[ -z "$pid" ]] || kill -KILL -- "-$pid" 2>/dev/null || true; done
    for pid in "$server_pid" "$client_pid"; do [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup_shard EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  exec {lock}>"/tmp/pdb-eval-port-${port}.lock"
  flock -n "$lock" || { echo 'Evaluation port lock is busy' >&2; return 1; }
  python "$HELPER_DIR/eval_server.py" free "$port"
  local dir="$OUT_DIR/shard_$k"
  mkdir -p "$dir"
  if [[ "${FORCE:-0}" == 1 ]]; then
    for f in predictions.jsonl metrics.json evaluation_state.json; do
      [[ ! -f "$dir/$f" ]] || mv "$dir/$f" "$dir/$f.bak.$$.${RANDOM}"
    done
  fi
  sleep "$((position * ${EVAL_STAGGER_SECONDS:-4}))"
  local command=()
  if [[ "$SERVER_BACKEND" == sglang ]]; then
    command=(python -m sglang.launch_server --model-path "$MODEL_PATH" --host 127.0.0.1 --port "$port" --api-key "$API_KEY" --context-length "${MAX_MODEL_LEN:-32768}" --max-running-requests "${MAX_NUM_SEQS:-128}" --mem-fraction-static "$GPU_MEM")
  else
    command=(vllm serve "$MODEL_PATH" --host 127.0.0.1 --port "$port" --max-model-len "${MAX_MODEL_LEN:-32768}" --max-num-seqs "${MAX_NUM_SEQS:-128}" --gpu-memory-utilization "$GPU_MEM")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_DISABLE_COMPILE_CACHE=1 OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" TOKENIZERS_PARALLELISM=false \
    setsid "${command[@]}" > "$dir/server.log" 2>&1 < /dev/null &
  server_pid=$!
  local ready=0
  for ((attempt=0; attempt<${EVAL_STARTUP_POLLS:-600}; attempt++)); do
    kill -0 "$server_pid" 2>/dev/null || { echo "Server exited; see $dir/server.log" >&2; return 1; }
    if python "$HELPER_DIR/eval_server.py" ready "$port" "$MODEL_PATH"; then ready=1; break; fi
    sleep 5
  done
  ((ready)) || { echo "Server readiness timed out; see $dir/server.log" >&2; return 1; }
  VLLM_BASE_URL="http://127.0.0.1:$port/v1" setsid python evaluate_model.py "${DEFAULT_ARGS[@]}" "${EXTRA[@]}" --num-shards "$NUM_SHARDS" --shard-index "$k" --log-dir "$dir" --json-output-file "$dir/metrics.json" >> "$dir/console.log" 2>&1 &
  client_pid=$!
  wait "$client_pid"
}
rm -f "$OUT_DIR/metrics.json"
for i in "${!GPUS[@]}"; do
  run_shard "${GPUS[$i]}" "$((BASE_PORT+i))" "${SHARDS[$i]}" "$i" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
((failed == 0)) || exit 1
((NO_MERGE == 0)) || exit 0
predictions=()
for ((k=0; k<NUM_SHARDS; k++)); do predictions+=("$OUT_DIR/shard_$k/predictions.jsonl"); done
python evaluate_model.py --model-path "$MODEL_PATH" --split "$SPLIT" --merge-shards "${predictions[@]}" --out-dir "$OUT_DIR" --log-dir "$OUT_DIR" --json-output-file "$OUT_DIR/metrics.json"

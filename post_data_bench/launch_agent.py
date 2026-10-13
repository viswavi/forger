#!/usr/bin/env python3
"""Generate data with OpenHands, then train and evaluate on the host.

--training-mode sft runs raw data -> SFT -> test. sft-rl accepts shared
prompt/ground-truth/rubric triples and an SFT checkpoint, freezes the submission,
then runs fixed thinking-mode RL and secret-test evaluation. --timeout-hours
bounds only the agent phase. See coding/rl/UNIFIED_TRAINING.md.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from typing import TextIO, cast

from agent_boundary import model_cache_bind

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--prompt", required=False, help="Path to a file with data-generation instructions."
)
parser.add_argument("--task", required=True, choices=["coding", "culture"])
parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Base model to train.")
parser.add_argument(
    "--litellm-api-key", required=False, help="API Key to use for Data Generation"
)
parser.add_argument(
    "--litellm-base-url", required=False, help="Base URL to use for the LiteLLM API"
)
parser.add_argument(
    "--litellm-parent-key",
    required=False,
    help="Deprecated; accepted for compatibility but unused by token accounting",
)
parser.add_argument(
    "--litellm-user-id",
    required=False,
    help="Deprecated; accepted for compatibility but unused by token accounting",
)
parser.add_argument(
    "--litellm-token-id",
    required=False,
    help="Deprecated; accepted for compatibility but unused by token accounting",
)
parser.add_argument(
    "--brave-api-key", required=False, help="API Key to use for the Brave Search API "
)
parser.add_argument(
    "--s2-api-key", required=False, help="API Key to use for the SemanticScholar API"
)
parser.add_argument(
    "--timeout-hours",
    type=float,
    default=6.0,
    help=(
        "Kill the agentic data-generation step if it runs longer "
        "than this many hours."
    ),
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=750,
    help=(
        "Hard cap on agent turns passed to the OpenHands SDK "
        "(max_iteration_per_run). The agent is separately told "
        "(see AGENT_TURN_BUDGET below) to target finishing within "
        "500 turns -- this flag just adds headroom above that "
        "target so a run that slightly overshoots isn't killed by "
        "ConversationRunError/MaxIterationsReached instead of "
        "finishing naturally or hitting --timeout-hours."
    ),
)
parser.add_argument(
    "--hide-forger",
    action="store_true",
    help=(
        "Do not expose the Forger library source to the agent "
        "(baseline: roll your own data-generation code instead of "
        "using pyforger)."
    ),
)
parser.add_argument(
    "--hide-curator",
    action="store_true",
    help="Do not expose the Curator repo to the agent (baseline: roll your own "
    "data-generation code instead of using curator).",
)

parser.add_argument(
    "--agent-python",
    default="/home/vijayv/miniconda3/envs/trl/bin/python",
    help="Interpreter with OpenHands, used if the current environment lacks it",
)
parser.add_argument("--training-mode", choices=["sft", "sft-rl"], default="sft")
parser.add_argument(
    "--slime-env",
    default=os.environ.get("SLIME_ENV", "/home/vijayv/miniconda3/envs/slime"),
)
parser.add_argument(
    "--slime-deps",
    default=os.environ.get("SLIME_DEPS", "/data/user_data/vijayv/slime_deps"),
)
parser.add_argument(
    "--prepare-final-only",
    action="store_true",
    help="Freeze the submission for a later host RL job",
)
parser.add_argument(
    "--finalize-submission",
    metavar="AGENT_DIR",
    help="Finalize an existing submission without launching an agent",
)
parser.add_argument(
    "--final-output-dir",
    help="New host directory for the frozen submission and final RL",
)
parser.add_argument(
    "--benchmark-dir",
    help="Directory containing one verified benchmark build for the chosen task",
)
args = parser.parse_args()
if args.training_mode == "sft-rl" and (
    args.task != "coding" or args.model != "Qwen/Qwen3-4B"
):
    parser.error(
        (
            "the supplied unified RL recipe currently supports coding "
            "with Qwen/Qwen3-4B only"
        )
    )
if (
    args.finalize_submission or args.prepare_final_only
) and args.training_mode != "sft-rl":
    parser.error("finalization options require --training-mode sft-rl")
if not args.prompt and not args.finalize_submission:
    parser.error("--prompt is required unless --finalize-submission is given")
if args.finalize_submission and not args.final_output_dir:
    parser.error("--finalize-submission requires --final-output-dir")


def finalization_command(directory, output, termination_reason="manual_recovery"):
    """Build the fixed host command for the selected agent submission."""
    directory, output = os.path.abspath(directory), os.path.abspath(output)
    command = [
        os.path.join(args.slime_env, "bin/python"),
        "-m",
        "rl.agent_training",
        "finalize",
        "--data",
        os.path.join(directory, "generated_raw.jsonl"),
        "--submission",
        os.path.join(directory, "submission.json"),
        "--agent-dir",
        directory,
        "--output-dir",
        output,
        "--slime-env",
        args.slime_env,
        "--slime-deps",
        args.slime_deps,
        "--agent-termination-reason",
        termination_reason,
    ]
    if args.prepare_final_only:
        command.append("--prepare-only")
    return command


def configure_run_token_budgets(directory, task):
    """Keep generation and judge usage across processes, probes, and recovery."""
    import json

    from agent_boundary import atomic_text
    from coding.rl.token_budget import (
        DEFAULT_GENERATION_LIMIT,
        DEFAULT_JUDGE_MODEL,
        DEFAULT_REASONING_EFFORT,
        TokenUnitBudget,
        default_rl_limit,
        validate_judge_model,
        validate_reasoning_effort,
    )

    judge_limit = default_rl_limit(task)
    directory = os.path.abspath(directory)
    config_path = os.path.join(directory, "token_budget_config.json")
    policy = {
        "version": 1,
        "task": task,
        "generation_token_unit_limit": DEFAULT_GENERATION_LIMIT,
        "judge_token_unit_limit": judge_limit,
    }
    try:
        descriptor = os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        config = {
            **policy,
            "judge_model": os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            "judge_reasoning_effort": os.environ.get(
                "JUDGE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT
            ),
        }
        new_config = True
    else:
        with os.fdopen(descriptor) as stream:
            config = json.load(stream)
        if (
            not isinstance(config, dict)
            or set(config) != set(policy) | {"judge_model", "judge_reasoning_effort"}
            or any(config[name] != value for name, value in policy.items())
        ):
            raise ValueError("Saved token budget task or policy differs from this run")
        new_config = False
    judge_model = validate_judge_model(config["judge_model"])
    reasoning_effort = validate_reasoning_effort(config["judge_reasoning_effort"])
    os.makedirs(directory, exist_ok=True)
    budget_env = {"PDB_TASK": task}
    for prefix, filename, limit in (
        ("FORGER", "generation_token_usage.json", DEFAULT_GENERATION_LIMIT),
        ("JUDGE", "judge_token_usage.json", judge_limit),
    ):
        path = os.path.join(directory, filename)
        budget_env[f"{prefix}_TOKEN_USAGE_FILE"] = path
        budget_env[f"{prefix}_TOKEN_UNIT_LIMIT"] = str(limit)
        TokenUnitBudget(path, limit).snapshot()
    if new_config:
        atomic_text(config_path, json.dumps(config, sort_keys=True) + "\n")
    os.environ.update(budget_env)
    os.environ["JUDGE_MODEL"] = judge_model
    os.environ["JUDGE_REASONING_EFFORT"] = reasoning_effort


# Finalization always uses Slime's Python, even when OpenHands lives in trl.
if args.benchmark_dir:
    os.environ["PDB_BENCHMARK_DIR"] = os.path.abspath(args.benchmark_dir)
if args.finalize_submission:
    configure_run_token_budgets(args.finalize_submission, args.task)
    subprocess.run(
        finalization_command(
            os.path.abspath(args.finalize_submission),
            os.path.abspath(args.final_output_dir),
        ),
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "coding"),
        check=True,
    )
    sys.exit(0)

if importlib.util.find_spec("openhands") is None:
    if os.path.realpath(sys.executable) == os.path.realpath(args.agent_python):
        parser.error(
            "OpenHands is missing; set --agent-python to an interpreter with the SDK"
        )
    print(
        (
            "Using the OpenHands interpreter for the agent; final "
            "training uses --slime-env."
        ),
        flush=True,
    )
    os.execv(
        args.agent_python, [args.agent_python, os.path.abspath(__file__), *sys.argv[1:]]
    )

if args.benchmark_dir:
    os.environ["PDB_BENCHMARK_DIR"] = os.path.abspath(args.benchmark_dir)

TIMEOUT = (
    args.timeout_hours
)  # hours; only bounds the agentic data-generation step below
# Turn budget communicated to the agent in INSTRUCTIONS below; kept independent of
# args.max_iterations (the hard hidden cap passed to Conversation), which just adds
# headroom above this target -- see --max-iterations help text.
AGENT_TURN_BUDGET = 500

litellm_key = args.litellm_api_key
if litellm_key is None:
    litellm_key = os.environ.get("LITELLM_API_KEY")
litellm_parent_key = args.litellm_parent_key
if litellm_parent_key is None:
    litellm_parent_key = litellm_key
litellm_base_url = args.litellm_base_url or os.environ.get("LITELLM_BASE_URL")
if litellm_key:
    os.environ["LITELLM_API_KEY"] = litellm_key
if litellm_base_url:
    os.environ["LITELLM_BASE_URL"] = litellm_base_url
brave_api_key = args.brave_api_key or os.environ.get("BRAVE_API_KEY")
s2_api_key = args.s2_api_key or os.environ.get("S2_API_KEY")
if brave_api_key:
    os.environ["BRAVE_API_KEY"] = brave_api_key
if s2_api_key:
    os.environ["S2_API_KEY"] = s2_api_key

with open(args.prompt) as f:
    raw_prompt_content = f.read()

FORGER_REPO = "/home/vijayv/forger"
CURATOR_REPO = "/home/vijayv/curator"
# Whether the sandbox image carries the slime stack and the agent is allowed to
# run short RL probes in it. Off by default so babel behaviour is unchanged.
AGENT_ENABLE_SLIME = os.environ.get("AGENT_ENABLE_SLIME", "0") == "1"
# Describe available GPU memory accurately so the agent can size workloads.
AGENT_GPU_DESC = os.environ.get(
    "AGENT_GPU_DESC", "the GPUs assigned to this allocation (inspect nvidia-smi)"
)
# Allow flame to override the data root. Its escaped path also affects
# output names; a stable symlink keeps those names comparable.
MODEL_OUTPUT_DIR = os.environ.get(
    "AGENT_MODEL_OUTPUT_DIR", "/data/user_data/vijayv/tir5/agent_data_models"
)
TASK_DIR = f"{FORGER_REPO}/post_data_bench/{args.task}"
MODEL_OUTPUT_TASK_DIR = f"{MODEL_OUTPUT_DIR}/{args.task}"
RUN_TAG = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

# Keep pipeline scripts and generated data in one per-run directory.
# Bind only this run’s directory to exclude sibling trajectories.
AGENT_DATA_DIR = f"{MODEL_OUTPUT_TASK_DIR}/agent_runs/agent_run_{RUN_TAG}"
RAW_DATA_PATH = f"{AGENT_DATA_DIR}/generated_raw.jsonl"
SUBMISSION_PATH = f"{AGENT_DATA_DIR}/submission.json"
GENERATION_USAGE_PATH = f"{AGENT_DATA_DIR}/generation_token_usage.json"
JUDGE_USAGE_PATH = f"{AGENT_DATA_DIR}/judge_token_usage.json"
configure_run_token_budgets(AGENT_DATA_DIR, args.task)
TRL_DATA_PATH = f"{AGENT_DATA_DIR}/generated_raw_trl_format.jsonl"
MODEL_TAG = args.model.replace("/", "__").replace(":", "__")
OUTPUT_DIR = f"{MODEL_OUTPUT_TASK_DIR}/{MODEL_TAG}_{RUN_TAG}"
ESC_OUTPUT_DIR = OUTPUT_DIR.replace("/", "__")
METRICS_PATH = f"{TASK_DIR}/model_outputs/{ESC_OUTPUT_DIR}/test/metrics.json"

# Tee console, SDK, trajectory, and subprocess output before importing
# OpenHands, whose logging handlers capture streams during setup.
import os  # noqa: E402

LOG_DIR = f"{TASK_DIR}/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{LOG_DIR}/{RUN_TAG}.log"
log_fh = open(LOG_FILE, "a")


class Tee:
    """Redact credentials while forwarding output to multiple streams."""

    def __init__(self, *streams):
        """Store configuration and initialize instance state."""
        self._streams = streams

    def write(self, data):
        """Redact credential values and forward text to every output stream."""
        for secret in (
            litellm_key,
            litellm_parent_key,
            brave_api_key,
            s2_api_key,
            os.environ.get("LLM_API_KEY"),
            os.environ.get("LITELLM_JUDGE_API_KEY"),
        ):
            if secret:
                data = data.replace(secret, "[REDACTED]")
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        """Flush every output stream."""
        for s in self._streams:
            s.flush()

    def __getattr__(self, name):
        # Forward stream attributes required by libraries such as libtmux.
        """Forward stream attributes needed by logging consumers."""
        return getattr(self._streams[0], name)


sys.stdout = cast(TextIO, Tee(sys.stdout, log_fh))
sys.stderr = cast(TextIO, Tee(sys.stderr, log_fh))


prompt_content = raw_prompt_content.format(
    OUTPUT_DIR=OUTPUT_DIR,
    AGENT_DATA_DIR=AGENT_DATA_DIR,
    TASK_DIR=TASK_DIR,
    LITELLM_API_KEY=(
        "${LITELLM_API_KEY}"
    ),  # Never interpolate a credential into an LLM prompt.
    # Exclude parent credentials/user ID from prompt formatting; references
    # to those placeholders must raise rather than expose host-only values.
    LITELLM_BASE_URL=litellm_base_url,
    BRAVE_API_KEY="${BRAVE_API_KEY}",
    S2_API_KEY="${S2_API_KEY}",
    FORGER_REPO=FORGER_REPO,
    CURATOR_REPO=CURATOR_REPO,
    RAW_DATA_PATH=RAW_DATA_PATH,
    SUBMISSION_PATH=SUBMISSION_PATH,
    TIMEOUT_HOURS=f"{TIMEOUT:g}",
    GPU_DESCRIPTION=AGENT_GPU_DESC,
    STUDENT_MODEL=args.model,
    GENERATION_TOKEN_UNIT_LIMIT=os.environ["FORGER_TOKEN_UNIT_LIMIT"],
    GENERATION_USAGE_PATH=GENERATION_USAGE_PATH,
    MAX_SFT_LENGTH="18800" if args.task == "coding" else "8192",
)

# Task converters share evaluation prompt builders to keep train/eval aligned.
RAW_SCHEMA = {
    "coding": (
        '{"question_content": ..., "public_test_cases": ..., "thinking": ..., '
        '"gold_response": ...}\n'
        '   "public_test_cases" is a list of {"input": ..., "output": ...} dicts '
        "(or a JSON-encoded string of the same). The downstream training pipeline "
        "(convert_to_trl.py) only reads question_content, public_test_cases, and "
        "gold_response from each row: gold_response must be a correct Python "
        'solution for a genuinely solvable problem, or exactly the string "This '
        'task is impossible to complete given the provided tests." for a '
        "deliberately impossible/contradictory one."
    ),
    "culture": (
        '{"country": ..., "situation": ..., "question": ..., "thinking": ..., '
        '"answer": ...}\n'
        '   "answer" must be one of "yes"/"no"/"neutral".'
    ),
}[args.task]


def run_logged(cmd: list[str], cwd: str) -> None:
    """Stream subprocess output to console/log and raise on nonzero exit."""
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None  # Popen was configured with stdout=PIPE.
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print(f"[exit code {proc.returncode}] {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


import time  # noqa: E402

from openhands.sdk import (  # noqa: E402
    LLM,
    Conversation,
    RemoteConversation,
    get_logger,
)
from openhands.sdk.conversation.exceptions import ConversationRunError  # noqa: E402
from openhands.tools.preset.default import get_default_agent  # noqa: E402
from openhands.workspace import ApptainerWorkspace  # noqa: E402  # noqa: E402
from pydantic import SecretStr  # noqa: E402

logger = get_logger(__name__)

# 1) Ensure we have LLM API key
api_key = os.getenv("LLM_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."

llm = LLM(
    usage_id="agent",
    model=os.getenv("LLM_MODEL", "gpt-5.5"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=SecretStr(api_key),
)

# Use per-run scratch copies to isolate concurrent runs. Require node-local
# disk for small files and Ray sockets; gcsfuse cannot host Unix sockets.
_SCRATCH_ROOT = os.environ.get("AGENT_SCRATCH_ROOT", "/scratch/vijayv")
RUN_SCRATCH_DIR = f"{_SCRATCH_ROOT}/agent_workspace_{RUN_TAG}"
AGENT_SCRATCH_DIR = f"{RUN_SCRATCH_DIR}/workspace"
SCRATCH_COPY_DIR = f"{RUN_SCRATCH_DIR}/copy"
CUSTOM_SIF = os.environ.get("AGENT_SIF")
if not CUSTOM_SIF:
    raise SystemExit("Set AGENT_SIF to an image built from an allowlisted flame recipe")
if os.path.basename(CUSTOM_SIF) == "openhands-forger.sif":
    raise SystemExit(
        (
            "The legacy wholesale-repository image is disabled; "
            "rebuild an allowlisted image"
        )
    )


# Persist each run’s @op cache inside its own mounted data directory
# so agents cannot browse other runs’ cached calls.
os.environ["FORGER_CACHE"] = f"{AGENT_DATA_DIR}/.forger_cache"

os.makedirs(AGENT_SCRATCH_DIR, exist_ok=True)
os.makedirs(os.environ["FORGER_CACHE"], exist_ok=True)
os.makedirs(MODEL_OUTPUT_TASK_DIR, exist_ok=True)
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

# Export only public allowlisted files and the validation evaluator.
# New task files remain hidden until explicitly added.
TASK_PUBLIC_FILES = {
    "coding": ["validation_tasks_de_obfuscated.jsonl", "train_model.sh"],
    "culture": ["validation_tasks_de_obfuscated.jsonl", "train_model.sh"],
}[args.task]

if args.training_mode == "sft-rl":
    TASK_PUBLIC_FILES += [
        f"rl/{name}"
        for name in (
            "agent_training.py",
            "rubric_data.py",
            "rubric_rm.py",
            "rubric_worker.py",
            "rubric_sandbox.py",
            "rubric_runtime_env.py",
            "ray_job_status.py",
            "rl_result.py",
            "run_rl_training_and_evaluation.sh",
            "UNIFIED_TRAINING.md",
        )
    ]


def _touch_placeholder(root: str, rel_path: str, is_dir: bool) -> None:
    """Create file/directory placeholders required as Apptainer bind targets."""
    full_path = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if is_dir:
        os.makedirs(full_path, exist_ok=True)
    else:
        open(full_path, "a").close()


def _copy_into(root: str, rel_path: str, host_path: str) -> None:
    """Copy allowlisted sources into a self-contained scratch tree for one bind."""
    dest = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(host_path):
        shutil.copytree(host_path, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(host_path, dest)


# Expose only named merged validation outputs; shard directories may
# contain stale predictions or fields from earlier splits.
VAL_OUTPUT_FILES = ("predictions.jsonl", "metrics.json")


def _task_public_copies(task_dir: str, dest_task_dir: str) -> None:
    """Copy public task files, named validation outputs, and optional evaluator.

    Exclude secret test outputs and validation subdirectories.
    """
    import json
    from pathlib import Path

    from coding.eval_state import CONTRACT_VERSION
    from coding.split_contract import public_rows, sha256, verify_manifest

    data_root = os.environ.get("PDB_BENCHMARK_DIR", task_dir)
    verify_manifest(data_root)
    for entry in TASK_PUBLIC_FILES:
        if entry == "validation_tasks_de_obfuscated.jsonl":
            continue
        host_path = os.path.join(task_dir, entry)
        if os.path.exists(host_path):
            _copy_into(dest_task_dir, entry, host_path)

    if args.training_mode == "sft-rl":
        from coding.rl.agent_validation import copy_validation_evaluator

        copy_validation_evaluator(task_dir, dest_task_dir, data_root=data_root)

    if args.training_mode != "sft-rl":
        import base64
        import zlib

        rows = [
            json.loads(line)
            for line in (Path(data_root) / "validation_tasks.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        if args.task == "coding":
            rows = public_rows(rows)
        gold = "gold_response" if args.task == "coding" else "gold_label"
        for row in rows:
            if row[gold] is not None:
                row[gold] = zlib.decompress(base64.b64decode(row[gold])).decode()
        Path(dest_task_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_task_dir) / "validation_tasks_de_obfuscated.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    model_outputs_dir = os.path.join(task_dir, "model_outputs")
    if os.path.isdir(model_outputs_dir):
        for model_name in ["Qwen__Qwen3-4B"]:
            val_dir = os.path.join(model_outputs_dir, model_name, "val")
            if not os.path.isdir(val_dir):
                continue
            metrics_path = Path(val_dir) / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                baseline = json.loads(metrics_path.read_text())
            except (ValueError, OSError):
                continue
            if baseline.get("evaluation_contract") != CONTRACT_VERSION or baseline.get(
                "tasks_sha256"
            ) != sha256(Path(data_root) / "validation_tasks.jsonl"):
                continue
            for filename in VAL_OUTPUT_FILES:
                host_file = os.path.join(val_dir, filename)
                if os.path.isfile(host_file):
                    rel = os.path.join("model_outputs", model_name, "val", filename)
                    _copy_into(dest_task_dir, rel, host_file)


def build_agent_mounts(
    *,
    forger_repo: str,
    curator_repo: str,
    task_dir: str,
    agent_data_dir: str,
    scratch_dir: str,
    hide_forger: bool,
    hide_curator: bool,
) -> list[str]:
    """Bind this run’s data/cache, public task files, and visible library snapshots.

    Allowlisted image stubs and per-run copies exclude secrets and sibling runs;
    persistent output is mounted separately from the large data volume.
    """
    if os.path.isdir(scratch_dir):
        shutil.rmtree(scratch_dir)
    forger_copy = os.path.join(scratch_dir, "forger")
    curator_copy = os.path.join(scratch_dir, "curator")
    os.makedirs(forger_copy, exist_ok=True)
    os.makedirs(curator_copy, exist_ok=True)

    binds = []

    # -- Forger repo: populate a real copy (src/ + docs/, unless hidden, plus
    # the allowlisted TASK_DIR contents), then expose it with a single bind.
    if not hide_forger:
        _copy_into(forger_copy, "src", os.path.join(forger_repo, "src"))
        _copy_into(forger_copy, "docs", os.path.join(forger_repo, "docs"))

    # Public SFT machinery is available even in the hide-forger condition.
    if args.training_mode == "sft-rl":
        for name in ("train_sft.py", "fsdp.yaml"):
            rel = os.path.join("examples", "training", name)
            _copy_into(forger_copy, rel, os.path.join(forger_repo, rel))

    task_rel = os.path.relpath(task_dir, forger_repo)
    dest_task_dir = os.path.join(forger_copy, task_rel)
    os.makedirs(dest_task_dir, exist_ok=True)
    _task_public_copies(task_dir, dest_task_dir)
    # Accounting helpers remain available in every task and hide-forger run.
    _copy_into(
        dest_task_dir,
        "rl/token_budget.py",
        os.path.join(forger_repo, "post_data_bench/coding/rl/token_budget.py"),
    )
    _copy_into(
        dest_task_dir,
        "rl/token_units.py",
        os.path.join(forger_repo, "src/pyforger/token_units.py"),
    )

    binds.append(f"{forger_copy}:{forger_repo}")

    # Bind only this run's persistent output directory.
    # Outputs live on the large data volume. Only create a placeholder in the
    # copied tree for legacy callers whose output path is actually under it.
    if os.path.commonpath(
        [os.path.abspath(agent_data_dir), os.path.abspath(forger_repo)]
    ) == os.path.abspath(forger_repo):
        agent_data_rel = os.path.relpath(agent_data_dir, forger_repo)
        _touch_placeholder(forger_copy, agent_data_rel, is_dir=True)
    binds.append(f"{agent_data_dir}:{agent_data_dir}")

    # -- Curator repo: populate a real copy, then a single bind.
    if not hide_curator:
        _copy_into(curator_copy, "src", os.path.join(curator_repo, "src"))
        _copy_into(curator_copy, "docs", os.path.join(curator_repo, "docs"))
    binds.append(f"{curator_copy}:{curator_repo}")

    return binds


AGENT_BIND_MOUNTS = build_agent_mounts(
    forger_repo=FORGER_REPO,
    curator_repo=CURATOR_REPO,
    task_dir=TASK_DIR,
    agent_data_dir=AGENT_DATA_DIR,
    scratch_dir=SCRATCH_COPY_DIR,
    hide_forger=args.hide_forger,
    hide_curator=args.hide_curator,
)

# Runtime storage: node-local disk for Ray sockets/JIT caches; NFS for HF
# symlinks and locks; GCS FUSE for large finished checkpoints/artifacts.
# Keep frequently flushed logs off GCS FUSE.

# /tmp is the right mount point for Ray (keeps /tmp/ray-<user>-<jobid> short
# enough for the ~107-char unix-socket limit) but must be backed by real disk:
# on flame /tmp is a 128 MB tmpfs, on which `ray start` dies.
AGENT_TMP_DIR = f"{RUN_SCRATCH_DIR}/tmp"
# flashinfer's JIT cache alone reaches multiple GB, in many small files.
AGENT_RL_CACHE_DIR = os.environ.get("AGENT_RL_CACHE_DIR", f"{RUN_SCRATCH_DIR}/rl_cache")
# Short RL probes only -- reward-shape tests and rollout sanity checks, not tens
# of steps plus an eval -- so this is deliberately not sized for real runs. Fine
# on GCS FUSE if you point it there, since checkpoints are few and huge.
AGENT_RL_OUTPUT_DIR = os.environ.get("AGENT_RL_OUTPUT_DIR", f"{RUN_SCRATCH_DIR}/rl_out")
# Model weights. HF_HUB_OFFLINE=1 is forced downstream, so this must ALREADY be
# populated: an empty cache fails offline with a confusing "does not appear to
# have a file named ..." from every rank rather than attempting a download.
AGENT_HF_HOME = os.environ.get("AGENT_HF_HOME", os.environ.get("HF_HOME", ""))

RUNTIME_BINDS = []

# Restore DNS explicitly because --no-mount bind-paths suppresses the
# default resolv.conf mount.
if os.path.exists("/etc/resolv.conf"):
    RUNTIME_BINDS.append("/etc/resolv.conf:/etc/resolv.conf")

# Put container /tmp on real disk instead of --compat’s small RAM tmpfs.
# This also keeps Ray socket paths short.
os.makedirs(AGENT_TMP_DIR, exist_ok=True)
RUNTIME_BINDS.append(f"{AGENT_TMP_DIR}:/tmp")

# Everything below is gated: these grant the sandbox new reach (model weights in
# particular), so they must not appear on a run that is not doing RL.
if AGENT_ENABLE_SLIME:
    # Nest mounts under the disk-backed /tmp bind with precreated host targets.
    # --containall hides /root; missing container destinations abort startup.
    for _host, _cont in [
        (AGENT_RL_CACHE_DIR, "/tmp/rl_cache"),
        (AGENT_RL_OUTPUT_DIR, "/tmp/rl_out"),
    ]:
        os.makedirs(_host, exist_ok=True)
        os.makedirs(os.path.join(AGENT_TMP_DIR, os.path.basename(_cont)), exist_ok=True)
        RUNTIME_BINDS.append(f"{_host}:{_cont}")

    # Only this model's blobs/snapshots are visible; no host token or dataset cache.
    SANDBOX_HF_HOME = os.path.join(AGENT_DATA_DIR, ".hf")
    RUNTIME_BINDS.append(model_cache_bind(AGENT_HF_HOME, SANDBOX_HF_HOME, args.model))


AGENT_BIND_MOUNTS = AGENT_BIND_MOUNTS + RUNTIME_BINDS

_HOST_RL_CACHE_ENV = {
    name: os.environ.get(name)
    for name in (
        "TVM_FFI_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "HF_HUB_OFFLINE",
        "HF_HOME",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    )
}
# Container-side paths, forwarded into the sandbox via forward_env below. Set
# here rather than earlier so the host-side convert/train/evaluate subprocesses
# further down are unaffected -- none of them read these.
if AGENT_ENABLE_SLIME:
    os.environ["TVM_FFI_CACHE_DIR"] = "/tmp/rl_cache/tvm_ffi"
    os.environ["FLASHINFER_WORKSPACE_BASE"] = "/tmp/rl_cache/flashinfer"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = SANDBOX_HF_HOME
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

INSTRUCTIONS = (
    "\nYou have "
    f"{TIMEOUT:g}"
    " hours and a target budget of "
    f"{AGENT_TURN_BUDGET}"
    " agent turns -- plan your approach so you can finish "
    "comfortably within that many turns. You are running "
    "inside a GPU-enabled sandbox with a full coding "
    "environment, "
    f"{AGENT_GPU_DESC}"
    ", and access to LLM APIs via LiteLLM available. Your ONLY "
    'job is to generate a raw training dataset for the "'
    f"{args.task}"
    '" benchmark -- do NOT run any (large-scale) training or '
    "evaluation yourself; a separate deterministic process "
    "handles that after you're done. Note that you are welcome "
    "to run small-scale experiments, time-permitting.\n\nFollow "
    "the instructions between the ----- markers below to "
    "generate the dataset.\nSave it as a JSON-lines file, one "
    "row per line, where each row has, at minimum,\nthese keys:\n"
    "  "
    f"{RAW_SCHEMA}"
    "\n\nWrite datasets and submission.json atomically "
    "(temporary file, then rename).\nSave the result to EXACTLY "
    "this path (create parent dirs if needed):\n  "
    f"{RAW_DATA_PATH}"
    "\n\n----- BEGIN DATA GENERATION INSTRUCTIONS -----\n"
    f"{prompt_content}"
    "\n----- END DATA GENERATION INSTRUCTIONS -----\n\nAs your "
    "final action, confirm the file exists and print its line "
    "count (e.g.\n`wc -l "
    f"{RAW_DATA_PATH}"
    "`).\n"
)

if args.training_mode == "sft-rl":
    INSTRUCTIONS = (
        "\nYou have "
        f"{TIMEOUT:g}"
        " hours and a target budget of "
        f"{AGENT_TURN_BUDGET}"
        " turns,\nwith "
        f"{AGENT_GPU_DESC}"
        " and your allocated generation token units.\nGenerate "
        "training data for the coding benchmark. You may run SFT "
        "as many times\nas time permits and may run short RL "
        "experiments if the slime stack is available.\nSelect an "
        "SFT checkpoint to submit. Final RL and secret test "
        "evaluation run on\nthe host after your session ends, "
        "outside your time budget.\n\nThe following task "
        "instructions supply the domain and generation strategy:\n"
        "----- BEGIN DATA GENERATION INSTRUCTIONS -----\n"
        f"{prompt_content}"
        "\n----- END DATA GENERATION INSTRUCTIONS -----\n\nThe "
        "submission contract below takes precedence over any older "
        "output schema,\ntraining restriction, or finalization "
        "instructions in the text above.\nRead "
        f"{TASK_DIR}"
        "/rl/UNIFIED_TRAINING.md for the complete contract and "
        "commands.\nWrite ONE dataset to "
        f"{RAW_DATA_PATH}"
        '; each JSONL row is exactly:\n{"prompt": [{"role": "user", '
        '"content": "..."}],\n  "ground_truth": '
        '"<think>reasoning</think>final answer",\n  "rubric": '
        '[[{"code": "def grade(prompt, response, ground_truth):\n   '
        ' return 1.0"}, 1.0],\n              [{"prompt": "Describe '
        'a criterion for the LM judge"}, 1.0]]}\nUse meaningful '
        "criteria; the constant score above illustrates syntax "
        "only.\nCredentials must be loaded implicitly from "
        "environment variables in code.\nNEVER print/read them into "
        "your conversation, logs, prompts, or command arguments.\n"
        "LITELLM_API_KEY is for generation. LITELLM_JUDGE_API_KEY "
        "has a SEPARATE budget of "
        f"{os.environ['JUDGE_TOKEN_UNIT_LIMIT']}"
        " token units shared by judge experiments, RL probes, and final RL.\n"
        "Use rl.token_budget.completion or acompletion with budget='judge' "
        "for your own judge calls. The helpers load judge credentials and "
        "update JUDGE_TOKEN_USAGE_FILE. Do not bypass accounting.\n"
        "JUDGE_MODEL must contain DeepSeek-V4-Flash; the default is "
        "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731. JUDGE_REASONING_EFFORT "
        "defaults to low; none and high are also supported. LiteLLM receives "
        'reasoning={"effort": JUDGE_REASONING_EFFORT}.\n'
        "LITELLM_JUDGE_BASE_URL defaults to LITELLM_BASE_URL. "
        "Quota exhaustion stops RL and preserves its latest usable checkpoint; "
        "other repeated judge failures fail the run. Failures do not become "
        "zero rewards.\nThe harness makes a fixed 90/10 split. SFT "
        "and RL use identical training prompts;\nthe 10% split "
        "monitors RL rewards. You must not generate a separate "
        "eval dataset.\nCode criteria must define exactly one "
        "top-level synchronous function\n    def grade(prompt, "
        "response, ground_truth):\nwith exactly three positional "
        "parameters, no decorators, *args, **kwargs, or\n"
        "keyword-only parameters. Parameter names may vary; "
        "annotations are optional.\nThe prompt argument is the "
        "original list of chat messages; response and\nground_truth "
        "are strings, including their thinking traces. Return a "
        "finite\nnumeric score in [0,1] (not a boolean). Code runs "
        "in RestrictedPython.\nPrompt criteria call "
        "DeepSeek-V4-Flash-0731\nvia trusted LiteLLM code outside "
        "RestrictedPython. The fixed reward is the\nweighted mean "
        "of criterion scores in [0,1]. You cannot submit a custom "
        "RM module.\n\nFrom "
        f"{TASK_DIR}"
        ", validate the dataset format and rubric signatures "
        "first:\npython -m rl.agent_training validate --data "
        f"{RAW_DATA_PATH}"
        "\nThis checks format and RestrictedPython compilation "
        "without executing criteria,\ncalling the judge, or "
        "checking grading quality. Fix any errors before "
        "proceeding.\nThen train with:\npython -m rl.agent_training "
        "sft --data "
        f"{RAW_DATA_PATH}"
        " --output-dir "
        f"{AGENT_DATA_DIR}"
        "/sft_trial_1\nUse a new output directory for each trial. "
        "You may choose epochs and learning rate.\nEvaluate each "
        "trial using the existing validation evaluator (thinking "
        "is enabled):\nbash "
        f"{TASK_DIR}"
        "/evaluate_model.sh --split val --model-path "
        f"{AGENT_DATA_DIR}"
        "/sft_trial_1 --out-dir "
        f"{AGENT_DATA_DIR}"
        "/validation/sft_trial_1\nRead metrics.json and "
        "predictions.jsonl in that output directory to compare "
        "trials.\nUse a fresh validation output directory whenever "
        "weights or evaluation settings change,\nbecause the "
        "evaluator resumes existing outputs. It uses vLLM by "
        "default; pass\n--server-backend sglang if that is the "
        "backend available in your environment.\nOnly validation "
        "tasks and their test cases are exposed. Secret benchmark "
        "test\nscoring happens after submission. Store validation "
        "outputs in your persistent\nAGENT_DATA_DIR so they survive "
        "workspace cleanup.\nStore all submitted checkpoints under "
        f"{AGENT_DATA_DIR}"
        "; scratch checkpoints vanish.\nAfter choosing an SFT "
        "trial, write "
        f"{SUBMISSION_PATH}"
        ' as:\n{"checkpoint": "'
        f"{AGENT_DATA_DIR}"
        "/sft_trial_1\"}\nThe chosen checkpoint's SFT prompts and "
        "targets must match the final dataset.\nRubrics may be "
        "revised without repeating SFT. Keep thinking traces "
        "intact.\nLeave a valid dataset and submission.json on disk "
        "before your time expires.\nAfter your last dataset/rubric "
        "edit, rerun:\npython -m rl.agent_training validate --data "
        f"{RAW_DATA_PATH}"
        "\nRequire a successful exit before submission, and verify "
        "submission.json points\nto the chosen SFT checkpoint. The "
        "validation command prints the dataset row count.\n"
    )

INSTRUCTIONS += """
Credential handling: API keys are available in environment variables. Load them
implicitly in code with os.environ; never print their values or put them in
conversation text, logs, source files, or command arguments. Any literal
${LITELLM_API_KEY} in the task text means the environment variable, not a key.
"""

INSTRUCTIONS += (
    "\nAll API data-generation calls share "
    f"{os.environ['FORGER_TOKEN_UNIT_LIMIT']} token units. "
    f"Read {GENERATION_USAGE_PATH} for cumulative token_units, "
    "remaining_token_units, and cost_usd. unpriced_calls counts responses "
    "without dollar pricing; only token units enforce the quota. "
    "DeepSeek-V4-Flash uses 1 unit per input token "
    "and 3 per output token; GLM-5.3 uses 4 and 12.5. Count cached inputs "
    "at full weight and reasoning within output tokens. Use only those model "
    "families. pyforger.Model accounts automatically. For your own API calls, "
    f"add {TASK_DIR!r} to sys.path and import completion or acompletion from "
    "rl.token_budget; these helpers are available even when Forger is hidden. "
    "Do not wrap pyforger.Model again or call LiteLLM directly. Helpers check "
    "the shared ledger before each request and record reported usage; already "
    "running requests may cross the limit. Plan batches with headroom. "
    "Never delete/reset usage ledgers or change their environment settings. "
    "The OpenHands model driving your turns has separate accounting.\n"
)

if AGENT_ENABLE_SLIME:
    # Advertise Slime’s interpreter because the default agent PATH selects TRL.
    INSTRUCTIONS += (
        "\nThere are TWO python stacks in this sandbox, selected by "
        "which interpreter you invoke -- not by an environment "
        "variable:\n  python              -- torch 2.9.1, vllm, the "
        "default for your own tooling.\n  slime-python        -- "
        "torch 2.11.0+cu129, sglang, TransformerEngine, "
        "megatron-core, apex. Use this for ANYTHING involving "
        "slime. Source trees are at /opt/slime and "
        "/opt/Megatron-LM.\n`import slime` under the plain `python` "
        "WILL fail; that does not mean slime is missing.\n\nYou may "
        "use slime for short experiments only -- sanity-checking "
        "rollouts or comparing reward shapes. Do not attempt a "
        "full RL run: you have "
        f"{TIMEOUT:g}"
        " hours total and training many steps plus an evaluation "
        "will not fit. Write any checkpoints or rollout dumps "
        "under /tmp/rl_out, and leave /tmp/rl_cache alone (it "
        "holds the flashinfer JIT cache). Ray works; its temp dir "
        "must stay under /tmp.\n"
    )

if args.hide_forger or args.hide_curator:
    hidden_libs = [
        name
        for name, hidden in [
            ("Forger", args.hide_forger),
            ("Curator", args.hide_curator),
        ]
        if hidden
    ]
    INSTRUCTIONS += (
        "\nNOTE: "
        f"{' and '.join(hidden_libs)}"
        " "
        f"{('is' if len(hidden_libs) == 1 else 'are')}"
        " NOT available in this run -- "
        f"{('it does' if len(hidden_libs) == 1 else 'they do')}"
        " not exist on disk and cannot be imported. Ignore any "
        "instructions above that reference "
        f"{('it' if len(hidden_libs) == 1 else 'them')}"
        "; write your data-generation pipeline using plain Python "
        "and the provided rl.token_budget API helpers instead.\n"
    )

# Launch the Apptainer workspace with allowlisted library snapshots and
# this run’s live data mount. Host source edits appear on the next run;
# third-party dependency changes require an image rebuild.
import signal  # noqa: E402


class AgentTimeoutError(Exception):
    """Raised when the agentic data-generation step exceeds TIMEOUT hours."""


def _timeout_handler(signum, frame):
    raise AgentTimeoutError(
        f"Agent data-generation step exceeded TIMEOUT={TIMEOUT:g}h and was killed."
    )


# Bound only the agent block with SIGALRM; disarm before host subprocesses.
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(int(TIMEOUT * 3600))
try:
    with ApptainerWorkspace(
        sif_file=CUSTOM_SIF,
        mount_dir=AGENT_SCRATCH_DIR,
        extra_bind_mounts=AGENT_BIND_MOUNTS,
        forward_env=[
            "DEBUG",
            "BRAVE_API_KEY",
            "S2_API_KEY",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "LITELLM_API_KEY",
            "LITELLM_BASE_URL",
            "LITELLM_JUDGE_API_KEY",
            "LITELLM_JUDGE_BASE_URL",
            "JUDGE_MAX_CONSECUTIVE_FAILURES",
            "JUDGE_MODEL",
            "JUDGE_REASONING_EFFORT",
            "JUDGE_TOKEN_USAGE_FILE",
            "JUDGE_TOKEN_UNIT_LIMIT",
            "FORGER_TOKEN_USAGE_FILE",
            "FORGER_TOKEN_UNIT_LIMIT",
            "PDB_TASK",
            "SFT_CONDA_ENV",
            "FORGER_CACHE",
            # slime/sglang runtime: without these the agent's RL probes rebuild
            # the flashinfer JIT cache into the small RAM-backed --compat layer.
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TVM_FFI_CACHE_DIR",
            "FLASHINFER_WORKSPACE_BASE",
        ],
        enable_gpu=True,
        # host_port auto-selects an available port when not specified
    ) as workspace:
        # 3) Create agent
        agent = get_default_agent(
            llm=llm,
            cli_mode=True,
        )

        # 4) Log every trajectory event (the agent's thoughts, tool calls, and tool
        # results) -- this, plus the Tee'd stdout/stderr above, is "the agent's
        # trajectory and reasoning traces" ending up in LOG_FILE.
        last_event_time = {"ts": time.time()}

        def event_callback(event) -> None:
            """Record agent events in the persistent trajectory log."""
            logger.info(f"trajectory event: {type(event).__name__}\n{event}")
            last_event_time["ts"] = time.time()

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[event_callback],
            max_iteration_per_run=args.max_iterations,
        )
        assert isinstance(conversation, RemoteConversation)

        try:
            logger.info(f"Conversation ID: {conversation.state.id}")
            logger.info(f"Run tag: {RUN_TAG}")
            logger.info(f"Log file: {LOG_FILE}")

            # Override the SDK’s independent one-hour polling timeout.
            termination_reason = "completed"
            try:
                conversation.send_message(INSTRUCTIONS)
                conversation.run(timeout=TIMEOUT * 3600)
            except (AgentTimeoutError, TimeoutError, ConversationRunError) as exc:
                timed_out = isinstance(
                    exc, (AgentTimeoutError, TimeoutError)
                ) or isinstance(
                    getattr(exc, "original_exception", None),
                    (AgentTimeoutError, TimeoutError),
                )
                if args.training_mode != "sft-rl":
                    raise
                termination_reason = type(exc).__name__
                logger.warning(
                    "Agent session ended (%s); validating its saved submission",
                    type(exc).__name__,
                )
            finally:
                signal.alarm(0)
            try:
                conversation.pause()
                logger.info(f"Agent status: {conversation.state.execution_status}")
            except Exception as exc:
                if args.training_mode != "sft-rl":
                    raise
                logger.warning(
                    (
                        "Agent pause/status unavailable (%s); workspace teardown "
                        "will stop the container"
                    ),
                    type(exc).__name__,
                )

            # Wait for events to settle (no events for 2 seconds) before closing.
            close_deadline = time.monotonic() + 10
            while (
                time.time() - last_event_time["ts"] < 2.0
                and time.monotonic() < close_deadline
            ):
                time.sleep(0.1)

            try:
                metrics = conversation.conversation_stats.get_combined_metrics()
                cost = metrics.accumulated_cost
                print(f"AGENT_TURN_COST_USD: {cost}")
            except Exception as exc:
                if args.training_mode != "sft-rl":
                    raise
                logger.warning("Agent-turn cost unavailable (%s)", type(exc).__name__)
        finally:
            signal.alarm(0)
            print("Cleaning up conversation...")
            try:
                conversation.close()
            except Exception as exc:
                if args.training_mode != "sft-rl":
                    raise
                logger.warning(
                    "Conversation close failed (%s); workspace teardown follows",
                    type(exc).__name__,
                )
finally:
    signal.alarm(0)
    for name, value in _HOST_RL_CACHE_ENV.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

from coding.rl.token_budget import TokenUnitBudget  # noqa: E402

for prefix in ("FORGER", "JUDGE"):
    usage = TokenUnitBudget(
        os.environ[f"{prefix}_TOKEN_USAGE_FILE"],
        os.environ[f"{prefix}_TOKEN_UNIT_LIMIT"],
    ).snapshot()
    print(f"{prefix}_TOKEN_UNITS: {usage['token_units']:g} / {usage['limit']:g}")
    print(f"{prefix}_COST_USD: {usage['cost_usd']:.6f}")
    print(f"{prefix}_UNPRICED_CALLS: {usage['unpriced_calls']}")

# 5) The agent's part is done; everything from here is a deterministic,
# non-agentic subprocess pipeline on this host. Fail loudly (not silently) if
# the agent didn't actually produce the file it was told to.
if not os.path.isfile(RAW_DATA_PATH):
    raise RuntimeError(
        f"Agent did not produce the expected raw dataset at {RAW_DATA_PATH} -- "
        f"see {LOG_FILE} for its trajectory."
    )

if args.training_mode == "sft-rl":
    run_logged(
        finalization_command(
            AGENT_DATA_DIR,
            args.final_output_dir or OUTPUT_DIR + "_final",
            termination_reason,
        ),
        cwd=TASK_DIR,
    )
    sys.exit(0)

run_logged(
    ["python", "convert_to_trl.py", RAW_DATA_PATH, "--tokenizer-model", args.model],
    cwd=TASK_DIR,
)
run_logged(
    [
        "./train_model.sh",
        "--model",
        args.model,
        "--data-file",
        TRL_DATA_PATH,
        "--output-dir",
        OUTPUT_DIR,
    ],
    cwd=TASK_DIR,
)
run_logged(
    ["./run_evaluate_only.sh", "--split", "test", "--model-path", OUTPUT_DIR],
    cwd=TASK_DIR,
)

print(f"=== {METRICS_PATH} ===")
with open(METRICS_PATH) as f:
    print(f.read())

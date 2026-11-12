#!/usr/bin/env python3
"""Generate data with OpenHands, then train and evaluate on the host.

sft-rl (default) accepts prompt/rubric rows (ground_truth optional) and a final RL
checkpoint, freezes the submission, then runs secret-test evaluation only. Training
and evaluation use --backend tinker (Qwen3-8B LoRA; SFT trials, then RL) or
--backend local (Qwen3-4B on the allocated GPUs; RL only, from the base model).
The legacy --training-mode sft runs raw data -> local SFT -> test.
--timeout-hours bounds only the agent phase. See coding/rl/UNIFIED_TRAINING.md.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import TextIO, cast

from agent_boundary import model_cache_bind
from agent_nudges import low_rows_message
from backend_prompts import prompt_sections
from coding.rl.backends import MODELS, RL_DEFAULTS, saved_backend

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--prompt", required=False, help="Path to a file with data-generation instructions."
)
parser.add_argument("--task", required=True, choices=["coding", "culture"])
parser.add_argument(
    "--backend",
    choices=["tinker", "local"],
    default=None,
    help="Training/evaluation backend (default: tinker; recovery uses saved backend).",
)
parser.add_argument("--model", help="Base model (8B for Tinker, 4B for local).")
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
    default=15.0,
    help=(
        "Kill the agentic data-generation step if it runs longer "
        "than this many hours."
    ),
)
parser.add_argument(
    "--max-iterations",
    type=int,
    default=1200,
    help=(
        "Hard cap on agent turns passed to the OpenHands SDK "
        "(max_iteration_per_run). The agent is separately told "
        "(see AGENT_TURN_BUDGET below) to target finishing within "
        "800 turns -- this flag just adds headroom above that "
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
parser.add_argument("--training-mode", choices=["sft", "sft-rl"], default="sft-rl")
parser.add_argument(
    "--min-rows-nudge",
    type=int,
    default=1000,
    help="If the agent finishes with fewer dataset rows than this, tell it once "
    "and at least 7 hours remain, let it continue once; 0 disables",
)
parser.add_argument(
    "--host-python",
    default=os.environ.get("PDB_HOST_PYTHON", sys.executable),
    help=(
        "Host finalization interpreter; for local image execution use "
        "local_host_python.sh"
    ),
)
parser.add_argument(
    "--prepare-final-only",
    action="store_true",
    help="Freeze the final RL checkpoint for later secret-test evaluation",
)
parser.add_argument(
    "--finalize-submission",
    metavar="AGENT_DIR",
    help="Finalize an existing submission without launching an agent",
)
parser.add_argument(
    "--final-output-dir",
    help="New host directory for the frozen checkpoint and secret evaluation",
)
parser.add_argument(
    "--benchmark-dir",
    help="Directory containing one verified benchmark build for the chosen task",
)
args = parser.parse_args()
args.backend = (
    saved_backend(args.finalize_submission, args.backend)
    if args.finalize_submission
    else args.backend or "tinker"
)
args.model = args.model or MODELS[args.backend]
if args.training_mode == "sft-rl" and (
    args.task != "coding" or args.model != MODELS[args.backend]
):
    parser.error(
        f"the {args.backend} unified recipe supports coding with "
        f"{MODELS[args.backend]} only"
    )
if (
    args.finalize_submission or args.prepare_final_only
) and args.training_mode != "sft-rl":
    parser.error("finalization options require --training-mode sft-rl")
if not args.prompt and not args.finalize_submission:
    parser.error("--prompt is required unless --finalize-submission is given")
if args.finalize_submission and not args.final_output_dir:
    parser.error("--finalize-submission requires --final-output-dir")


# Host .env names -> the names the sandbox, Forger, and the reward code read.
# Generation and the judge share the worker gateway; the OpenHands runner has
# its own key and, without LLM_AGENT_BASE_URL, calls its provider directly.
ENV_ALIASES = (
    ("LITELLM_API_KEY", "LITELLM_DATAGEN_API_KEY"),
    ("LITELLM_BASE_URL", "LITELLM_WORKER_BASE_URL"),
    ("LLM_API_KEY", "LLM_AGENT_API_KEY"),
    ("LLM_MODEL", "LLM_AGENT_MODEL"),
    ("LLM_BASE_URL", "LLM_AGENT_BASE_URL"),
)


def apply_env_aliases(environ=os.environ):
    """Fill each internal name from its host alias unless already set."""
    for name, alias in ENV_ALIASES:
        if not environ.get(name) and environ.get(alias):
            environ[name] = environ[alias]


apply_env_aliases()


def finalization_command(directory, output, termination_reason="manual_recovery"):
    """Build the fixed host command for the selected agent submission."""
    directory, output = os.path.abspath(directory), os.path.abspath(output)
    command = [
        args.host_python,
        "-m",
        "rl.agent_training",
        "finalize",
        "--backend",
        args.backend,
        "--data",
        os.path.join(directory, "generated_raw.jsonl"),
        "--submission",
        os.path.join(directory, "submission.json"),
        "--agent-dir",
        directory,
        "--output-dir",
        output,
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
    from coding.rl.tinker_budget import DEFAULT_LIMITS_USD, TinkerBudget
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
        "version": 2,
        "submission_kind": "final_rl",
        "task": task,
        "generation_token_unit_limit": DEFAULT_GENERATION_LIMIT,
        "judge_token_unit_limit": judge_limit,
        "tinker_usd_limits": DEFAULT_LIMITS_USD if args.backend == "tinker" else {},
        "backend": args.backend,
        "base_model": args.model,
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
        if isinstance(config, dict):
            config.setdefault("backend", "tinker")
            config.setdefault("base_model", MODELS["tinker"])
            if config.get("submission_kind") != "final_rl":
                raise ValueError(
                    "This run uses the old SFT-submission contract; recover it "
                    "with its original launcher"
                )
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
    budget_env = {"PDB_TASK": task, "PDB_TRAIN_BACKEND": args.backend}
    for prefix, filename, limit in (
        ("FORGER", "generation_token_usage.json", DEFAULT_GENERATION_LIMIT),
        ("JUDGE", "judge_token_usage.json", judge_limit),
    ):
        path = os.path.join(directory, filename)
        budget_env[f"{prefix}_TOKEN_USAGE_FILE"] = path
        budget_env[f"{prefix}_TOKEN_UNIT_LIMIT"] = str(limit)
        TokenUnitBudget(path, limit).snapshot()
    if args.backend == "tinker":
        # Tinker dollars: agent SFT trials and validation evals; agent RL probes
        # plus final RL; host secret-test evaluation (recorded, not limited).
        for name in ("sft", "rl", "eval"):
            path = os.path.join(directory, f"tinker_{name}_usage.json")
            limit = DEFAULT_LIMITS_USD.get(name)
            budget_env[f"TINKER_{name.upper()}_USAGE_FILE"] = path
            if limit is not None:
                budget_env[f"TINKER_{name.upper()}_USD_LIMIT"] = str(limit)
            TinkerBudget(path, limit, name=name).snapshot()
        # Forger's TinkerModel (e.g. sampling the student for STaR traces) charges the
        # same SFT ledger as SFT trials and validation evaluations.
        budget_env["FORGER_TINKER_USAGE_FILE"] = budget_env["TINKER_SFT_USAGE_FILE"]
        budget_env["FORGER_TINKER_USD_LIMIT"] = budget_env["TINKER_SFT_USD_LIMIT"]
    if new_config:
        atomic_text(config_path, json.dumps(config, sort_keys=True) + "\n")
    os.environ.update(budget_env)
    os.environ["JUDGE_MODEL"] = judge_model
    os.environ["JUDGE_REASONING_EFFORT"] = reasoning_effort


# Finalization uses the selected host interpreter or image wrapper.
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
AGENT_TURN_BUDGET = 800

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
# Host checkout this launcher runs from; the sandbox always sees FORGER_REPO.
HOST_FORGER_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Whether the sandbox image carries the slime stack and the agent is allowed to
# run short RL probes in it. Off by default so babel behaviour is unchanged.
AGENT_ENABLE_SLIME = args.backend == "local"
# The sandbox gets one GPU for embeddings/clustering; training and student
# sampling run on Tinker, data generation on LiteLLM APIs.
AGENT_GPU_DESC = os.environ.get(
    "AGENT_GPU_DESC",
    "8 H100 80GB GPUs"
    if args.backend == "local"
    else "one local GPU, for embeddings and clustering only",
)
# Allow flame to override the data root. Its escaped path also affects
# output names; a stable symlink keeps those names comparable.
MODEL_OUTPUT_DIR = os.environ.get(
    "AGENT_MODEL_OUTPUT_DIR", "/data/user_data/vijayv/tir5/agent_data_models"
)
TASK_DIR = f"{FORGER_REPO}/post_data_bench/{args.task}"
HOST_TASK_DIR = f"{HOST_FORGER_REPO}/post_data_bench/{args.task}"
MODEL_OUTPUT_TASK_DIR = f"{MODEL_OUTPUT_DIR}/{args.task}"
RUN_TAG = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
# Tags this run's Tinker sessions so billing records can be matched to it.
os.environ["PDB_RUN_TAG"] = RUN_TAG

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
METRICS_PATH = (
    f"{HOST_TASK_DIR}/.hidden/model_outputs/{ESC_OUTPUT_DIR}/test/metrics.json"
)

# Tee console, SDK, trajectory, and subprocess output before importing
# OpenHands, whose logging handlers capture streams during setup.
import os  # noqa: E402

LOG_DIR = f"{HOST_TASK_DIR}/logs"
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
            os.environ.get("NCBI_API_KEY"),
            os.environ.get("LLM_API_KEY"),
            os.environ.get("LITELLM_JUDGE_API_KEY"),
            os.environ.get("TINKER_API_KEY"),
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


prompt_values = dict(
    BASELINE_MODEL=MODEL_TAG + ("_tinker" if args.backend == "tinker" else ""),
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
    JUDGE_TOKEN_UNIT_LIMIT=f"{float(os.environ['JUDGE_TOKEN_UNIT_LIMIT']):,.0f}",
    JUDGE_USAGE_PATH=JUDGE_USAGE_PATH,
    TINKER_SFT_USD_LIMIT=f"{float(os.environ.get('TINKER_SFT_USD_LIMIT', 0)):g}",
    TINKER_SFT_USAGE_PATH=os.environ.get("TINKER_SFT_USAGE_FILE", ""),
    TINKER_RL_USD_LIMIT=f"{float(os.environ.get('TINKER_RL_USD_LIMIT', 0)):g}",
    TINKER_RL_USAGE_PATH=os.environ.get("TINKER_RL_USAGE_FILE", ""),
)

prompt_values.update(prompt_sections(args.backend, args.hide_forger, prompt_values))
prompt_content = raw_prompt_content.format(**prompt_values)

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
    AgentContext,
    Conversation,
    RemoteConversation,
    get_logger,
)
from openhands.sdk.context.condenser import LLMSummarizingCondenser  # noqa: E402
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
# One-sentence LLM summaries of long values in Forger's step reports: at most 8
# short calls per report, charged to the generation quota like any other call.
if not args.hide_forger:
    os.environ.setdefault(
        "FORGER_REPORT_MODEL", "litellm_proxy/azure_ai/DeepSeek-V4-Flash-0731"
    )

# No terminal command may block the agent for more than an hour: the agent
# server refuses a foreground command whose `timeout` exceeds 90% of this
# threshold (3600 s) and tells the agent to run long work in the background.
# Agents have passed milliseconds (timeout=120000, i.e. 33 h) on commands that
# never ended and hung for hours (round 2, 2026-10-28).
os.environ.setdefault("OH_RUNTIME_IDLE_TIMEOUT_SECONDS", "4000")

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
    # Tinker trains remotely and local runs are RL-only: no SFT entry point.
    TASK_PUBLIC_FILES.remove("train_model.sh")
    TASK_PUBLIC_FILES += [
        f"rl/{name}"
        for name in (
            "agent_training.py",
            "backends.py",
            "cpp_judge.py",
            "decontaminate.py",
            "lineage.py",
            "local_training.py",
            "rubric_data.py",
            "rubric_rm.py",
            "rubric_worker.py",
            "rubric_sandbox.py",
            "solution_runner.py",
            "solution_worker.py",
            "tinker_budget.py",
            "tinker_training.py",
            # judge_cpp compiles checkers and interactors against these.
            "testlib/testlib.h",
            "testlib/LICENSE",
        )
    ]

    if args.backend == "local":
        TASK_PUBLIC_FILES += [
            f"rl/{name}"
            for name in (
                "run_rl_training_and_evaluation.sh",
                "rubric_runtime_env.py",
                "slime_deps.py",
                "rl_result.py",
                "ray_job_status.py",
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
            if args.backend == "local" and entry.startswith("rl/"):
                from coding.rl.agent_validation import scrub

                target = Path(dest_task_dir) / entry
                target.write_text(scrub(target.read_text()))

    if args.task == "coding" and args.training_mode == "sft-rl":
        # Hashed windows of the hidden test statements, never their text: the
        # training helpers drop rows that overlap them (rl/decontaminate.py).
        from coding.rl.decontaminate import SIGNATURES, benchmark_signatures

        target = Path(dest_task_dir) / "rl" / SIGNATURES.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(benchmark_signatures(data_root)))

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
            from coding.rl.agent_validation import public_identity

            rows = public_rows(rows)
            for number, row in enumerate(rows):
                row.update(public_identity(number))
        gold = "gold_response" if args.task == "coding" else "gold_label"
        for row in rows:
            if row[gold] is not None:
                row[gold] = zlib.decompress(base64.b64decode(row[gold])).decode()
        Path(dest_task_dir).mkdir(parents=True, exist_ok=True)
        (Path(dest_task_dir) / "validation_tasks_de_obfuscated.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    # Baselines live with the benchmark build (PDB_BENCHMARK_DIR), e.g.
    # <build>/model_outputs/Qwen__Qwen3-4B/val/; a release's own task directory
    # has none, so looking only there never exported them on Orchard.
    model_outputs_dir = os.path.join(data_root, "model_outputs")
    if not os.path.isdir(model_outputs_dir):
        model_outputs_dir = os.path.join(task_dir, "model_outputs")
    if os.path.isdir(model_outputs_dir):
        for model_name in [
            args.model.replace("/", "__")
            + ("_tinker" if args.backend == "tinker" else "")
        ]:
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
            predictions = Path(dest_task_dir) / "model_outputs" / model_name / "val"
            predictions /= "predictions.jsonl"
            if args.task == "coding" and predictions.is_file():
                from coding.rl.agent_validation import public_question_ids

                # Match the copied tasks' opaque ids, e.g. "abc365_c" -> "val_017".
                ids = public_question_ids(data_root)
                rows = [
                    json.loads(s)
                    for s in predictions.read_text().splitlines()
                    if s.strip()
                ]
                for row in rows:
                    row["question_id"] = ids[row["question_id"]]
                predictions.write_text("".join(json.dumps(r) + "\n" for r in rows))


def build_agent_mounts(
    *,
    host_forger_repo: str,
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
        _copy_into(forger_copy, "src", os.path.join(host_forger_repo, "src"))
        _copy_into(forger_copy, "docs", os.path.join(host_forger_repo, "docs"))
        # The worked example docs/AGENTS.md points to (Forger arm only).
        _copy_into(
            forger_copy,
            "examples/rl_math",
            os.path.join(host_forger_repo, "examples/rl_math"),
        )

    task_rel = os.path.relpath(task_dir, host_forger_repo)
    dest_task_dir = os.path.join(forger_copy, task_rel)
    os.makedirs(dest_task_dir, exist_ok=True)
    _task_public_copies(task_dir, dest_task_dir)
    if args.backend == "local" and args.training_mode != "sft-rl":
        for name in ("train_sft.py", "fsdp.yaml"):
            _copy_into(
                forger_copy,
                f"examples/training/{name}",
                os.path.join(host_forger_repo, "examples/training", name),
            )
    # Accounting helpers remain available in every task and hide-forger run.
    _copy_into(
        dest_task_dir,
        "rl/token_budget.py",
        os.path.join(host_forger_repo, "post_data_bench/coding/rl/token_budget.py"),
    )
    _copy_into(
        dest_task_dir,
        "rl/token_units.py",
        os.path.join(host_forger_repo, "src/pyforger/token_units.py"),
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
    host_forger_repo=HOST_FORGER_REPO,
    forger_repo=FORGER_REPO,
    curator_repo=CURATOR_REPO,
    task_dir=HOST_TASK_DIR,
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
# Model weights. SFT and RL set HF_HUB_OFFLINE=1 themselves, so this must
# ALREADY be populated: an empty cache fails offline with a confusing "does not
# appear to have a file named ..." from every rank rather than attempting a
# download. The agent's own shell stays online, so Forger can retrieve
# datasets and download embedding models into the sandbox's HF cache.
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
        "HF_HOME",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "TINKER_EVAL_BUDGET",
        "PDB_EVAL_BACKEND",
        "PDB_STUDENT_MODEL",
    )
}
# Container-side paths, forwarded into the sandbox via forward_env below. Set
# here rather than earlier so the host-side convert/train/evaluate subprocesses
# further down are unaffected -- none of them read these.
if AGENT_ENABLE_SLIME:
    os.environ["TVM_FFI_CACHE_DIR"] = "/tmp/rl_cache/tvm_ffi"
    os.environ["FLASHINFER_WORKSPACE_BASE"] = "/tmp/rl_cache/flashinfer"
    os.environ["HF_HOME"] = SANDBOX_HF_HOME
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
else:
    # Never forward the host's HF_HOME (e.g. from ~/.bashrc): that path is not
    # bound, so downloads would fill the sandbox's RAM-backed layer. /tmp is
    # this run's local disk. The host value is restored after the session.
    os.environ["HF_HOME"] = "/tmp/hf_cache"
# Pin validation to the selected backend. Tinker draws on its SFT dollar budget.
# Without --model-path they evaluate the base student, PDB_STUDENT_MODEL.
os.environ["TINKER_EVAL_BUDGET"] = "sft"
os.environ["PDB_EVAL_BACKEND"] = "tinker" if args.backend == "tinker" else "vllm"
os.environ["PDB_STUDENT_MODEL"] = args.model


def host_posix_tz():
    """The host's zone as a POSIX TZ string, e.g. "EST5EDT,M3.2.0,M11.1.0".

    TZif v2+ files end with this string, and it needs no tzdata in the image.
    """
    if os.environ.get("TZ"):
        return os.environ["TZ"]
    try:
        with open("/etc/localtime", "rb") as stream:
            data = stream.read()
    except OSError:
        return None
    if data[:4] != b"TZif" or data[4:5] < b"2":
        return None
    return data.rstrip(b"\n").rsplit(b"\n", 1)[-1].decode() or None


# The sandbox clock defaulted to UTC while RUN_TAG used host time (EDT), so
# agents computed 4 extra hours elapsed and rushed to finish. Share the host's
# zone and state the deadline explicitly.
if host_posix_tz():
    os.environ["TZ"] = host_posix_tz()
AGENT_DEADLINE = datetime.now().astimezone() + timedelta(hours=TIMEOUT)
DEADLINE_TEXT = (
    f"{AGENT_DEADLINE:%Y-%m-%d %H:%M %Z} "
    f"({AGENT_DEADLINE.astimezone(timezone.utc):%H:%M} UTC; check with `date`)"
)

# What a turn is, and how to wait without spending them: forger_pair1 (2026-11-12)
# spent ~400 of its turns on 189 "soft timeout" returns of `sleep 290; tail log`.
TURNS_TEXT = (
    f"The session ends at {args.max_iterations} turns. Each of your actions "
    "(usually one tool call and its result) is one turn, including a terminal "
    'call that returns "Process still running (soft timeout)": the terminal tool '
    "returns after 30 seconds without new output unless the call sets its "
    "`timeout` parameter (in seconds, at most 3600). For example, "
    "`sleep 1700; tail -5 run.log` with timeout=1800 waits 30 minutes in one turn."
)

INSTRUCTIONS = (
    "\nYou have "
    f"{TIMEOUT:g}"
    f" hours, ending at {DEADLINE_TEXT}, and a target budget of "
    f"{AGENT_TURN_BUDGET}"
    " agent turns -- plan your approach so you can finish "
    f"comfortably within that many turns. {TURNS_TEXT} You are running "
    "inside a sandbox with a full coding "
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
    # The task prompt carries the operational contract (data format, Tinker
    # commands, budgets, submission); coding/FORGER_PROMPT.txt and
    # coding/minimal_scaffold_prompt.txt keep it identical across scaffolds.
    INSTRUCTIONS = (
        f"\nYou have {TIMEOUT:g} hours, ending at {DEADLINE_TEXT}, and a target "
        f"budget of {AGENT_TURN_BUDGET} turns. {TURNS_TEXT}\n\n"
        f"{prompt_content}\n"
    )

INSTRUCTIONS += """
Credential handling: API keys are available in environment variables. Load them
implicitly in code with os.environ; never print their values or put them in
conversation text, logs, source files, or command arguments. Any literal
${LITELLM_API_KEY} in the task text means the environment variable, not a key.
"""

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
        "use slime for experiments and full RL training -- "
        "inspect rollouts and compare reward shapes. You have "
        "a total of "
        f"{TIMEOUT:g}"
        " hours for data generation, RL and validation. "
        "Write checkpoints and rollout dumps "
        "under your persistent agent data directory, and leave /tmp/rl_cache alone (it "
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
AGENT_ALARM_AT = time.time() + TIMEOUT * 3600  # when SIGALRM fires, in time.time()
try:
    with ApptainerWorkspace(
        sif_file=CUSTOM_SIF,
        mount_dir=AGENT_SCRATCH_DIR,
        extra_bind_mounts=AGENT_BIND_MOUNTS,
        forward_env=[
            name
            for name in [
                "DEBUG",
                "BRAVE_API_KEY",
                "S2_API_KEY",
                # PubMed (NCBI E-utilities): 6.7 requests/s with a key, 3 without.
                "NCBI_API_KEY",
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
                "FORGER_CACHE",
                "FORGER_REPORT_MODEL",
                # Tinker: SFT/RL/eval run remotely; ledgers enforce dollar limits.
                "TINKER_API_KEY",
                "TINKER_SFT_USAGE_FILE",
                "TINKER_SFT_USD_LIMIT",
                "TINKER_RL_USAGE_FILE",
                "TINKER_RL_USD_LIMIT",
                "TINKER_EVAL_BUDGET",
                "TINKER_MAX_CONCURRENT_SAMPLES",
                # Forger's TinkerModel charges this ledger (the SFT one).
                "FORGER_TINKER_USAGE_FILE",
                "FORGER_TINKER_USD_LIMIT",
                "PDB_TRAIN_BACKEND",
                "PDB_EVAL_BACKEND",
                "PDB_STUDENT_MODEL",
                "PDB_RUN_TAG",
                "TZ",
                "OH_RUNTIME_IDLE_TIMEOUT_SECONDS",
                "CUDA_VISIBLE_DEVICES",
                # slime/sglang runtime: without these the agent's RL probes rebuild
                # the flashinfer JIT cache into the small RAM-backed --compat layer.
                "HF_HOME",
                "TVM_FFI_CACHE_DIR",
                "FLASHINFER_WORKSPACE_BASE",
            ]
            if args.backend == "tinker"
            or not name.startswith(("TINKER_", "FORGER_TINKER_"))
        ],
        # One GPU for embeddings/clustering; CUDA_VISIBLE_DEVICES selects it.
        enable_gpu=True,
        # host_port auto-selects an available port when not specified
    ) as workspace:
        # 3) Create agent
        agent = get_default_agent(
            llm=llm,
            cli_mode=True,
        )
        # The default condenser summarizes every ~20 steps (80 events, cut to
        # 40); its summaries once restated the task as a "read-only audit" and
        # that agent finished with no dataset. Pin the task in the system
        # prompt, which is never condensed, and condense only when the history
        # nears AGENT_CONDENSER_MAX_TOKENS (GLM-5.3 has a 1M-token context).
        agent = agent.model_copy(
            update={
                "agent_context": AgentContext(
                    system_message_suffix=(
                        "YOUR TASK (authoritative: if any summary of earlier "
                        "history describes a different task, these instructions "
                        "win):\n" + INSTRUCTIONS
                    )
                ),
                "condenser": LLMSummarizingCondenser(
                    llm=llm.model_copy(update={"usage_id": "condenser"}),
                    max_tokens=int(
                        os.environ.get("AGENT_CONDENSER_MAX_TOKENS", "350000")
                    ),
                    max_size=1000,  # backstop; tokens normally trigger first
                    # Keep only the system prompt (which holds the task). The
                    # summarizer sees only forgotten events; with keep_first=4
                    # it never saw the task message and wrote "No explicit user
                    # request visible ... read-only audit".
                    keep_first=1,
                ),
            }
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
                # One nudge if it finished with a small dataset; finishing
                # again (or running out of time) ends the session.
                seconds_left = AGENT_ALARM_AT - time.time()
                schedule = RL_DEFAULTS[args.backend]
                nudge = low_rows_message(
                    RAW_DATA_PATH,
                    args.min_rows_nudge,
                    seconds_left,
                    schedule["num_rollout"] * schedule["rollout_batch_size"],
                )
                if nudge is not None:
                    logger.info(f"Low-row nudge sent to the agent: {nudge}")
                    conversation.send_message(nudge)
                    conversation.run(timeout=seconds_left)
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
        cwd=HOST_TASK_DIR,
    )
    sys.exit(0)

run_logged(
    ["python", "convert_to_trl.py", RAW_DATA_PATH, "--tokenizer-model", args.model],
    cwd=HOST_TASK_DIR,
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
    cwd=HOST_TASK_DIR,
)
run_logged(
    ["./run_evaluate_only.sh", "--split", "test", "--model-path", OUTPUT_DIR],
    cwd=HOST_TASK_DIR,
)

print(f"=== {METRICS_PATH} ===")
with open(METRICS_PATH) as f:
    print(f.read())

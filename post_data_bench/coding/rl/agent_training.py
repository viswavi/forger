"""Agent-side SFT/probe helpers and host-side finalization for unified submissions.

SFT and RL run on Tinker (LoRA on Qwen3-8B), charged to per-run dollar ledgers.
Run from coding/: python -m rl.agent_training --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Supports both `rl` CLI imports and `coding.rl` library imports.
from .rubric_data import compile_data, validate_dataset, validate_rubric  # noqa: ABS101
from .rubric_rm import judge_configuration  # noqa: ABS101
from .tinker_budget import (  # noqa: ABS101
    BASE_MODEL,
    DEFAULT_LIMITS_USD,
    TinkerBudget,
    budget_from_env,
)

CODING = Path(__file__).resolve().parents[1]
JUDGE_FLAGS = {
    "PDB_TASK": "--task",
    "JUDGE_MODEL": "--judge-model",
    "JUDGE_REASONING_EFFORT": "--judge-reasoning-effort",
    "JUDGE_TOKEN_UNIT_LIMIT": "--judge-token-unit-limit",
    "JUDGE_TOKEN_USAGE_FILE": "--judge-token-usage-file",
}


def run(command, cwd=CODING, env=None):
    """Print and execute the trusted training command."""
    import shlex

    print("$ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True, env=env)


def prepare(data, directory, model=BASE_MODEL):
    """Validate shared triples and compile the identical SFT and RL prompts."""
    from transformers import AutoTokenizer

    return compile_data(data, directory, AutoTokenizer.from_pretrained(model))


def checkpoint_files(checkpoint):
    """Check a recorded Tinker checkpoint directory; return its resolved path."""
    from .tinker_training import read_checkpoint  # noqa: ABS101

    checkpoint = Path(checkpoint).resolve(strict=True)
    for name in ("tinker_checkpoint.json", "sft_submission.json"):
        path = checkpoint / name
        if not path.resolve(strict=True).is_relative_to(checkpoint):
            raise ValueError("Checkpoint metadata escaped its directory")
    read_checkpoint(checkpoint)
    return checkpoint


def digest(path):
    """Hash a file incrementally without loading it into memory."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sft(args):
    """Train and record an SFT trial on the fixed training partition."""
    from .tinker_training import run_sft  # noqa: ABS101

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"use a new output directory for each SFT trial: {output}")
    prepared = prepare(args.data, str(output) + ".data")
    record = run_sft(
        prepared / "sft.jsonl",
        output,
        budget_from_env("sft"),
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        run_tag=os.environ.get("PDB_RUN_TAG"),
    )
    # Record the actual prompts/targets used, separate from mutable rubrics.
    (output / "sft_submission.json").write_text(
        json.dumps(
            {
                "sft_sha256": digest(prepared / "sft.jsonl"),
                "base_model": BASE_MODEL,
                "data_directory": str(prepared),
                "thinking": True,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "lora_rank": args.lora_rank,
                "batch_size": args.batch_size,
                "sampler_path": record["sampler_path"],
            },
            indent=2,
        )
        + "\n"
    )
    print(
        (
            f"SFT complete; sampler weights: {record['sampler_path']}\n"
            'To select this trial: write {"checkpoint": '
            f"{json.dumps(str(output))}"
            "} to submission.json"
        )
    )


def _freeze_into(data, submission, agent_dir, output):
    """Called only after the agent workspace has closed; no agent code is imported."""
    agent_dir = Path(agent_dir).resolve()
    for source in (data, submission):
        if not Path(source).resolve(strict=True).is_relative_to(agent_dir):
            raise ValueError("Submission inputs must remain inside the agent directory")
    selected = json.loads(Path(submission).read_text())
    if not isinstance(selected, dict) or set(selected) != {"checkpoint"}:
        raise ValueError(
            'submission.json must contain exactly {"checkpoint": "/absolute/path"}'
        )
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_absolute():
        raise ValueError("submitted checkpoint path must be absolute")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(agent_dir):
        raise ValueError(
            "submitted checkpoint must be inside this agent's persistent data directory"
        )
    checkpoint = checkpoint_files(checkpoint)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare(data, output / "data")
    receipt = json.loads((checkpoint / "sft_submission.json").read_text())
    if receipt.get("sft_sha256") != digest(prepared / "sft.jsonl"):
        raise ValueError(
            (
                "submitted SFT checkpoint was trained on different "
                "prompts/targets; rerun SFT on the final dataset"
            )
        )
    # Weights stay on Tinker; freeze the recorded paths and receipt, never agent code.
    frozen_checkpoint = output / "initial_checkpoint"
    frozen_checkpoint.mkdir()
    for name in ("tinker_checkpoint.json", "sft_submission.json"):
        shutil.copyfile(checkpoint / name, frozen_checkpoint / name)
    checkpoint_files(frozen_checkpoint)
    manifest = {p.name: digest(p) for p in sorted(frozen_checkpoint.iterdir())}
    (output / "submission_manifest.json").write_text(
        json.dumps(
            {
                "selected_checkpoint": str(checkpoint),
                "checkpoint_sha256": manifest,
                "custom_rm_path": "rl.rubric_rm.custom_rm",
                "base_model": BASE_MODEL,
                "thinking": True,
            },
            indent=2,
        )
        + "\n"
    )
    return prepared, frozen_checkpoint


def freeze(data, submission, agent_dir, output):
    """Publish a validated submission bundle after transactional preparation."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Final bundle already exists; use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        _freeze_into(data, submission, agent_dir, staged)
        if output.exists():
            raise ValueError("Final bundle appeared during preparation")
        staged.rename(output)
    return output / "data", output / "initial_checkpoint"


def rl_command(data, checkpoint, output, tinker_ledger, probe=False, judge_config=None):
    """Build the fixed Tinker RL command for a probe or final run."""
    command = [
        sys.executable,
        "-m",
        "rl.agent_training",
        "rl-run",
        "--data-dir",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--tinker-usage-file",
        str(tinker_ledger.path),
        "--tinker-usd-limit",
        str(tinker_ledger.limit),
    ]
    if judge_config:
        for key, flag in JUDGE_FLAGS.items():
            command.extend([flag, judge_config[key]])
    if probe:
        # Two small steps: enough to check rewards and rollouts, cheap for agents.
        command += [
            "--num-rollout",
            "2",
            "--rollout-batch-size",
            "16",
            "--save-interval",
            "1",
            "--eval-interval",
            "2",
            "--skip-final-eval",
        ]
    return command


def rl_ledger(agent_dir):
    """The run's shared RL ledger: probes and final RL draw on one limit."""
    path = os.environ.get("TINKER_RL_USAGE_FILE") or (
        Path(agent_dir).resolve() / "tinker_rl_usage.json"
    )
    limit = os.environ.get("TINKER_RL_USD_LIMIT", DEFAULT_LIMITS_USD["rl"])
    return TinkerBudget(path, limit, name="rl")


def reward_preflight(directory, judge_config=None):
    # Check both partitions before allocating GPU work. No credentials are read
    # into messages or logs; only their presence is checked here.
    """Validate compiled rubrics and check required reward dependencies."""
    judge_configuration(
        Path(directory).resolve().parent / "judge_token_usage.json",
        overrides=judge_config,
    )
    has_prompt = has_code = False
    for name in ("train.jsonl", "val.jsonl"):
        with (Path(directory) / name).open() as stream:
            for line in stream:
                for criterion, weight in validate_rubric(
                    json.loads(line)["metadata"]["rubric"]
                ):
                    if weight:
                        has_prompt |= "prompt" in criterion
                        has_code |= "code" in criterion
    if has_code:
        import RestrictedPython  # noqa: F401
    if has_prompt:
        import litellm  # noqa: F401

        if not os.environ.get("LITELLM_JUDGE_API_KEY"):
            raise RuntimeError("Set the separate LITELLM_JUDGE_API_KEY before final RL")
        if not (
            os.environ.get("LITELLM_JUDGE_BASE_URL")
            or os.environ.get("LITELLM_BASE_URL")
        ):
            raise RuntimeError(
                "Set LITELLM_JUDGE_BASE_URL or LITELLM_BASE_URL before final RL"
            )


def finalize(args):
    """Freeze the submission and optionally start the host final RL run."""
    args.output_dir = str(Path(args.output_dir).resolve())
    judge_config = judge_configuration(
        Path(args.data).resolve().parent / "judge_token_usage.json",
        task=getattr(args, "task", None),
    )
    data, checkpoint = freeze(
        args.data, args.submission, args.agent_dir, args.output_dir
    )
    manifest_path = Path(args.output_dir) / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["agent_termination_reason"] = getattr(
        args, "agent_termination_reason", "manual_recovery"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    command = rl_command(
        data,
        checkpoint,
        Path(args.output_dir) / "rl",
        rl_ledger(args.agent_dir),
        judge_config=judge_config,
    )
    # A resumable host-only entry point; contains no credentials.
    (Path(args.output_dir) / "final_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    if not args.prepare_only:
        reward_preflight(data, judge_config)
        run(command)


def run_final(args):
    """Verify a frozen bundle before resuming the fixed final RL recipe."""
    bundle = Path(args.bundle).resolve()
    data_manifest = json.loads((bundle / "data/manifest.json").read_text())["sha256"]
    checkpoint_manifest = json.loads((bundle / "submission_manifest.json").read_text())[
        "checkpoint_sha256"
    ]
    for directory, manifest in (
        (bundle / "data", data_manifest),
        (bundle / "initial_checkpoint", checkpoint_manifest),
    ):
        for name, expected in manifest.items():
            if Path(name).name != name or digest(directory / name) != expected:
                raise ValueError(f"frozen submission changed: {name}")
    command = json.loads((bundle / "final_command.json").read_text())
    saved_config = {
        key: command[command.index(flag) + 1]
        for key, flag in JUDGE_FLAGS.items()
        if flag in command
    }
    if set(saved_config) != set(JUDGE_FLAGS):
        raise ValueError(
            "Final bundle lacks saved judge token settings; finalize again"
        )
    judge_config = judge_configuration(overrides=saved_config)
    reward_preflight(bundle / "data", judge_config)
    # Do not execute a command stored by the agent; this file is host-produced.
    # Rebuild from fixed trusted code, taking only the recorded ledger.
    ledger = TinkerBudget(
        command[command.index("--tinker-usage-file") + 1],
        command[command.index("--tinker-usd-limit") + 1],
        name="rl",
    )
    run(
        rl_command(
            bundle / "data",
            bundle / "initial_checkpoint",
            bundle / "rl",
            ledger,
            judge_config=judge_config,
        )
    )


def rl_run(args):
    """Run Tinker RL from a recorded checkpoint, then the secret test evaluation."""
    from .tinker_training import read_checkpoint, run_rl  # noqa: ABS101

    judge_config = judge_configuration(
        overrides={
            key: getattr(args, key.lower())
            for key in JUDGE_FLAGS
            if getattr(args, key.lower()) is not None
        }
        or None
    )
    os.environ.update(judge_config)
    reward_preflight(args.data_dir, judge_config)
    result = run_rl(
        args.data_dir,
        read_checkpoint(checkpoint_files(args.checkpoint)),
        args.output_dir,
        TinkerBudget(args.tinker_usage_file, args.tinker_usd_limit, name="rl"),
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        run_tag=os.environ.get("PDB_RUN_TAG"),
    )
    print(json.dumps(result, indent=2))
    if not args.skip_final_eval:
        final_evaluation(
            result["final_checkpoint"]["sampler_path"],
            Path(args.output_dir) / ".hidden" / "final_eval",
        )


def final_evaluation(sampler_path, out_dir):
    """Secret-test evaluation of the final RL weights; charged to the eval ledger."""
    env = {**os.environ, "TINKER_EVAL_BUDGET": "eval"}
    env.setdefault(
        "TINKER_EVAL_USAGE_FILE",
        str(Path(out_dir).resolve().parents[2] / "tinker_eval_usage.json"),
    )
    env.pop("TINKER_EVAL_USD_LIMIT", None)  # recorded, never limited
    run(
        [
            "bash",
            CODING / "evaluate_model.sh",
            "--server-backend",
            "tinker",
            "--split",
            "test",
            "--model-path",
            sampler_path,
            "--out-dir",
            out_dir,
        ],
        env=env,
    )


def validate(args):
    """Check the submission format without executing agent rubric code."""
    rows = validate_dataset(args.data)
    print(
        (
            "Valid submission format: "
            f"{len(rows)}"
            " rows. Rubric signatures and RestrictedPython compilation "
            "passed; no rubric execution or judge calls."
        )
    )


def main():
    """Run the command-line utility described in the module documentation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "validate",
        help=(
            "Check dataset format and rubric signatures without "
            "executing criteria or calling the judge"
        ),
    )
    check.add_argument("--data", required=True)
    check.set_defaults(action=validate)
    resume = commands.add_parser(
        "run-final", help="Verify and run a frozen host submission"
    )
    resume.add_argument("--bundle", required=True)
    resume.set_defaults(action=run_final)
    train = commands.add_parser(
        "sft",
        help="Train any number of SFT trials on the fixed 90%% training partition",
    )
    train.add_argument("--data", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model", default=BASE_MODEL, choices=[BASE_MODEL])
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument(
        "--lr", type=float, default=1e-6, help="Full-finetune LR; LoRA uses 10x"
    )
    train.add_argument("--lora-rank", type=int, default=32)
    train.add_argument("--batch-size", type=int, default=128)
    train.set_defaults(action=sft)
    for name in ("probe", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--task", default=os.environ.get("PDB_TASK", "coding"))
        if name == "finalize":
            command.add_argument("--submission", required=True)
            command.add_argument("--agent-dir", required=True)
            command.add_argument("--prepare-only", action="store_true")
            command.add_argument(
                "--agent-termination-reason", default="manual_recovery"
            )
            command.set_defaults(action=finalize)
        else:
            command.add_argument("--checkpoint", required=True)
            command.set_defaults(action=probe)
    # Internal: built by rl_command from trusted code, never from agent input.
    rl = commands.add_parser("rl-run", help=argparse.SUPPRESS)
    rl.add_argument("--data-dir", required=True)
    rl.add_argument("--checkpoint", required=True)
    rl.add_argument("--output-dir", required=True)
    rl.add_argument("--tinker-usage-file", required=True)
    rl.add_argument("--tinker-usd-limit", type=float, required=True)
    rl.add_argument("--num-rollout", type=int, default=25)
    rl.add_argument("--rollout-batch-size", type=int, default=128)
    rl.add_argument("--save-interval", type=int, default=5)
    rl.add_argument("--eval-interval", type=int, default=5)
    rl.add_argument("--skip-final-eval", action="store_true")
    for key, flag in JUDGE_FLAGS.items():
        rl.add_argument(flag, dest=key.lower())
    rl.set_defaults(action=rl_run)
    args = parser.parse_args()
    try:
        args.action(args)
    except ValueError as exc:
        parser.exit(1, f"Validation error: {exc}\n")


def probe(args):
    """Run a short RL trial without secret benchmark evaluation."""
    args.output_dir = str(Path(args.output_dir).resolve())
    judge_config = judge_configuration(
        Path(args.data).resolve().parent / "judge_token_usage.json",
        task=getattr(args, "task", None),
    )
    checkpoint = checkpoint_files(args.checkpoint)
    data = prepare(args.data, str(args.output_dir) + ".data")
    reward_preflight(data, judge_config)
    run(
        rl_command(
            data,
            checkpoint,
            args.output_dir,
            rl_ledger(Path(args.data).resolve().parent),
            probe=True,
            judge_config=judge_config,
        )
    )


if __name__ == "__main__":
    main()

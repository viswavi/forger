"""Agent-side SFT/probe helpers and host-side finalization for unified submissions.

SFT and RL use the selected Tinker or local GPU backend.
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

from .backends import MODELS, resolve_backend, saved_backend  # noqa: ABS101

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


def checkpoint_files(checkpoint, backend="tinker"):
    """Validate the selected backend's checkpoint format."""
    if backend == "local":
        from .local_training import checkpoint_files as local_files  # noqa: ABS101

        return local_files(checkpoint)
    from .tinker_training import read_checkpoint  # noqa: ABS101

    checkpoint = Path(checkpoint).resolve(strict=True)
    for name in ("tinker_checkpoint.json",):
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
    if resolve_backend(args.backend) == "local":
        from .local_training import sft as local_sft  # noqa: ABS101

        return local_sft(args)
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
                "backend": "tinker",
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
    print(f"SFT complete: {output}. Use this as --checkpoint for rl.agent_training rl.")


def _freeze_into(data, submission, agent_dir, output, backend="tinker"):
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
    checkpoint = checkpoint_files(checkpoint, backend)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = prepare(data, output / "data", MODELS[backend])
    receipt_path = (checkpoint / "rl_submission.json").resolve(strict=True)
    if not receipt_path.is_relative_to(checkpoint):
        raise ValueError("RL receipt escaped the checkpoint directory")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("backend") != backend
        or receipt.get("base_model") != MODELS[backend]
        or receipt.get("stage") != "rl"
        or receipt.get("completed_steps", 0) < 1
    ):
        raise ValueError("Submit a checkpoint with at least one completed RL step")
    if backend == "tinker":
        record = json.loads((checkpoint / "tinker_checkpoint.json").read_text())
        if receipt.get("sampler_path") != record.get("sampler_path"):
            raise ValueError("RL receipt does not match the selected sampler")
    # Weights stay on Tinker; freeze the recorded paths and receipt, never agent code.
    frozen_checkpoint = output / "initial_checkpoint"
    frozen_checkpoint.mkdir()
    if backend == "local":
        from .local_training import freeze_checkpoint  # noqa: ABS101

        freeze_checkpoint(checkpoint, frozen_checkpoint, agent_dir)
    else:
        for name in ("tinker_checkpoint.json", "rl_submission.json"):
            shutil.copyfile(checkpoint / name, frozen_checkpoint / name)
    checkpoint_files(frozen_checkpoint, backend)
    manifest = {p.name: digest(p) for p in sorted(frozen_checkpoint.iterdir())}
    (output / "submission_manifest.json").write_text(
        json.dumps(
            {
                "submission_kind": "final_rl",
                "selected_checkpoint": str(checkpoint),
                "checkpoint_sha256": manifest,
                "custom_rm_path": "rl.rubric_rm.custom_rm",
                "base_model": MODELS[backend],
                "backend": backend,
                "thinking": True,
            },
            indent=2,
        )
        + "\n"
    )
    return prepared, frozen_checkpoint


def freeze(data, submission, agent_dir, output, backend="tinker"):
    """Publish a validated submission bundle after transactional preparation."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Final bundle already exists; use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        _freeze_into(data, submission, agent_dir, staged, backend)
        if output.exists():
            raise ValueError("Final bundle appeared during preparation")
        staged.rename(output)
    return output / "data", output / "initial_checkpoint"


def rl_command(
    data,
    checkpoint,
    output,
    tinker_ledger=None,
    probe=False,
    judge_config=None,
    backend="tinker",
    num_rollout=80,
    rollout_batch_size=32,
    n_samples_per_prompt=8,
):
    """Build a command with an explicit backend for probes or host final RL."""
    if backend == "local":
        command = [
            sys.executable,
            "-m",
            "rl.agent_training",
            "rl-run",
            "--backend",
            "local",
            "--data-dir",
            str(data),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--slime-env",
            os.environ.get("SLIME_ENV", "/usr"),
            "--slime-deps",
            os.environ.get("SLIME_DEPS", "/opt"),
        ]
        command += [
            "--num-rollout",
            str(num_rollout),
            "--rollout-batch-size",
            str(rollout_batch_size),
            "--n-samples-per-prompt",
            str(n_samples_per_prompt),
        ]
        if probe:
            command.append("--local-probe")
        if judge_config:
            for key, flag in JUDGE_FLAGS.items():
                command.extend([flag, judge_config[key]])
        return command
    command = [
        sys.executable,
        "-m",
        "rl.agent_training",
        "rl-run",
        "--backend",
        "tinker",
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
    command += [
        "--num-rollout",
        str(num_rollout),
        "--rollout-batch-size",
        str(rollout_batch_size),
        "--n-samples-per-prompt",
        str(n_samples_per_prompt),
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
    """Freeze the selected final RL checkpoint, then run only secret evaluation."""
    args.output_dir = str(Path(args.output_dir).resolve())
    backend = saved_backend(args.agent_dir, args.backend)
    data, checkpoint = freeze(
        args.data, args.submission, args.agent_dir, args.output_dir, backend
    )
    manifest_path = Path(args.output_dir) / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["agent_termination_reason"] = getattr(
        args, "agent_termination_reason", "manual_recovery"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "rl.agent_training",
        "run-final",
        "--bundle",
        args.output_dir,
    ]
    (Path(args.output_dir) / "final_command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    if not args.prepare_only:
        run_final(argparse.Namespace(bundle=args.output_dir))


def run_final(args):
    """Verify a frozen bundle and resume its recorded submission contract."""
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
    submission = json.loads((bundle / "submission_manifest.json").read_text())
    if submission.get("submission_kind") == "final_rl":
        backend = resolve_backend(submission["backend"])
        checkpoint = checkpoint_files(bundle / "initial_checkpoint", backend)
        model = str(checkpoint)
        if backend == "tinker":
            from .tinker_training import read_checkpoint  # noqa: ABS101

            model = read_checkpoint(checkpoint)["sampler_path"]
        return final_evaluation(model, bundle / ".hidden/final_eval", backend=backend)
    # Recovery for bundles frozen under the previous SFT-submission contract.
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
    backend = resolve_backend(
        json.loads((bundle / "submission_manifest.json").read_text()).get(
            "backend", "tinker"
        )
    )
    if backend == "local":
        from .local_training import execute_rl  # noqa: ABS101

        return execute_rl(
            bundle / "data",
            bundle / "initial_checkpoint",
            bundle / "rl",
            judge_config,
            slime_env=command[command.index("--slime-env") + 1],
            slime_deps=command[command.index("--slime-deps") + 1],
            num_rollout=100,
            evaluate=True,
        )
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
            num_rollout=int(command[command.index("--num-rollout") + 1])
            if "--num-rollout" in command
            else 25,
            rollout_batch_size=int(command[command.index("--rollout-batch-size") + 1])
            if "--rollout-batch-size" in command
            else 128,
        )
    )


def rl_run(args):
    """Run the selected RL recipe, then the secret test evaluation."""
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
    if resolve_backend(args.backend) == "local":
        from .local_training import execute_rl  # noqa: ABS101

        return execute_rl(
            args.data_dir,
            checkpoint_files(args.checkpoint, "local"),
            args.output_dir,
            judge_config,
            probe=args.local_probe,
            slime_env=args.slime_env,
            slime_deps=args.slime_deps,
            num_rollout=args.num_rollout,
            rollout_batch_size=args.rollout_batch_size,
            n_samples_per_prompt=args.n_samples_per_prompt,
            evaluate=not args.skip_final_eval,
        )
    if args.tinker_usage_file is None or args.tinker_usd_limit is None:
        raise ValueError("Tinker RL requires its saved usage ledger and limit")
    result = run_rl(
        args.data_dir,
        read_checkpoint(checkpoint_files(args.checkpoint)),
        args.output_dir,
        TinkerBudget(args.tinker_usage_file, args.tinker_usd_limit, name="rl"),
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
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


def final_evaluation(sampler_path, out_dir, backend="tinker"):
    """Evaluate final weights using the backend recorded by the host recipe."""
    env = {
        **os.environ,
        "TINKER_EVAL_BUDGET": "eval",
        "PDB_EVAL_BACKEND": "tinker" if backend == "tinker" else "vllm",
    }
    # eval_launcher.sh runs a bare `python`; without this it can find the login
    # shell's older interpreter (e.g. Python 3.9) instead of this environment's.
    env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), env.get("PATH", "")]
    )
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
            env["PDB_EVAL_BACKEND"],
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
    train.add_argument("--model")
    train.add_argument("--backend", choices=tuple(MODELS))
    train.add_argument(
        "--conda-env",
        default=os.environ.get("SFT_CONDA_ENV", "/home/vijayv/miniconda3/envs/trl"),
    )
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument(
        "--lr", type=float, default=1e-6, help="Full-finetune LR; LoRA uses 10x"
    )
    train.add_argument("--lora-rank", type=int)
    train.add_argument("--batch-size", type=int, default=128)
    train.set_defaults(action=sft)
    for name in ("probe", "rl", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--backend", choices=tuple(MODELS))
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
            command.add_argument("--num-rollout", type=int, default=80)
            command.add_argument("--rollout-batch-size", type=int, default=32)
            command.add_argument("--n-samples-per-prompt", type=int, default=8)
            command.set_defaults(action=probe if name == "probe" else train_rl)
    # Internal: built by rl_command from trusted code, never from agent input.
    rl = commands.add_parser("rl-run", help=argparse.SUPPRESS)
    rl.add_argument("--data-dir", required=True)
    rl.add_argument("--checkpoint", required=True)
    rl.add_argument("--output-dir", required=True)
    rl.add_argument("--backend", choices=tuple(MODELS))
    rl.add_argument("--local-probe", action="store_true")
    rl.add_argument("--slime-env", default=os.environ.get("SLIME_ENV", "/usr"))
    rl.add_argument("--slime-deps", default=os.environ.get("SLIME_DEPS", "/opt"))
    rl.add_argument("--tinker-usage-file")
    rl.add_argument("--tinker-usd-limit", type=float)
    rl.add_argument("--num-rollout", type=int, default=80)
    rl.add_argument("--rollout-batch-size", type=int, default=32)
    rl.add_argument("--n-samples-per-prompt", type=int, default=8)
    rl.add_argument("--save-interval", type=int, default=5)
    rl.add_argument("--eval-interval", type=int, default=5)
    rl.add_argument("--skip-final-eval", action="store_true")
    for key, flag in JUDGE_FLAGS.items():
        rl.add_argument(flag, dest=key.lower())
    rl.set_defaults(action=rl_run)
    args = parser.parse_args()
    try:
        if args.command == "sft":
            args.backend = saved_backend(Path(args.data).resolve().parent, args.backend)
            if args.lora_rank is None:
                args.lora_rank = 32 if args.backend == "tinker" else 256
            args.model = args.model or MODELS[args.backend]
            if args.model != MODELS[args.backend]:
                raise ValueError("Model does not match the selected backend recipe")
        args.action(args)
    except ValueError as exc:
        parser.exit(1, f"Validation error: {exc}\n")


def probe(args):
    """Run a small RL experiment during the agent session."""
    return train_rl(args, short=True)


def train_rl(args, short=False):
    """Run agent-controlled RL without secret benchmark evaluation."""
    if any(
        value < 1
        for value in (
            args.num_rollout,
            args.rollout_batch_size,
            args.n_samples_per_prompt,
        )
    ):
        raise ValueError("RL steps and batch sizes must be positive")
    args.output_dir = str(Path(args.output_dir).resolve())
    if Path(args.output_dir).exists():
        raise ValueError("Use a new output directory for each RL experiment")
    judge_config = judge_configuration(
        Path(args.data).resolve().parent / "judge_token_usage.json",
        task=getattr(args, "task", None),
    )
    backend = saved_backend(Path(args.data).resolve().parent, args.backend)
    checkpoint = checkpoint_files(args.checkpoint, backend)
    data = prepare(args.data, str(args.output_dir) + ".data", MODELS[backend])
    reward_preflight(data, judge_config)
    command = rl_command(
        data,
        checkpoint,
        args.output_dir,
        rl_ledger(Path(args.data).resolve().parent) if backend == "tinker" else None,
        backend=backend,
        probe=short,
        judge_config=judge_config,
        num_rollout=args.num_rollout,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
    )
    command.append("--skip-final-eval")
    run(command)


if __name__ == "__main__":
    main()

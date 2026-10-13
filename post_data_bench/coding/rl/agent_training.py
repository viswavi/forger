"""Agent-side SFT/probe helpers and host-side finalization for unified submissions.

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

CODING = Path(__file__).resolve().parents[1]
JUDGE_FLAGS = {
    "PDB_TASK": "--task",
    "JUDGE_MODEL": "--judge-model",
    "JUDGE_REASONING_EFFORT": "--judge-reasoning-effort",
    "JUDGE_TOKEN_UNIT_LIMIT": "--judge-token-unit-limit",
    "JUDGE_TOKEN_USAGE_FILE": "--judge-token-usage-file",
}


def run(command, cwd=CODING):
    """Print and execute the trusted training command."""
    import shlex

    print("$ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def prepare(data, directory, model):
    """Validate shared triples and compile the identical SFT and RL prompts."""
    from transformers import AutoTokenizer

    return compile_data(data, directory, AutoTokenizer.from_pretrained(model))


def checkpoint_files(checkpoint):
    """Check a full HF checkpoint, including every shard referenced by its index."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        path = checkpoint / name
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(
            checkpoint
        ):
            raise ValueError("Checkpoint metadata escaped its directory")
    for name in ("config.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"checkpoint missing {name}: {checkpoint}")
    config = json.loads((checkpoint / "config.json").read_text())
    # The supplied Megatron recipe sources scripts/models/qwen3-4B.sh.
    expected = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("the fixed RL recipe requires a Qwen3-4B checkpoint")
    weights = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = checkpoint / index_name
        if index.exists():
            weights += list(set(json.loads(index.read_text())["weight_map"].values()))
    if not weights:
        weights = [
            n
            for n in ("model.safetensors", "pytorch_model.bin")
            if (checkpoint / n).is_file()
        ]
    if not weights:
        raise ValueError("submit a full/merged HF checkpoint, not a LoRA adapter")
    for name in weights:
        path = (checkpoint / name).resolve(strict=True)
        if (
            not path.is_relative_to(checkpoint)
            or not path.is_file()
            or path.stat().st_size == 0
        ):
            raise ValueError(f"invalid checkpoint weight file: {name}")
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
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError(f"use a new output directory for each SFT trial: {output}")
    prepared = prepare(args.data, str(output) + ".data", args.model)
    run(
        [
            "bash",
            CODING / "train_model.sh",
            "--model",
            args.model,
            "--data-file",
            prepared / "sft.jsonl",
            "--output-dir",
            output,
            "--epochs",
            args.epochs,
            "--lr",
            args.lr,
            "--conda-env",
            args.conda_env,
        ]
    )
    checkpoint_files(output)
    # Record the actual prompts/targets used, separate from mutable rubrics.
    (output / "sft_submission.json").write_text(
        json.dumps(
            {
                "sft_sha256": digest(prepared / "sft.jsonl"),
                "base_model": args.model,
                "data_directory": str(prepared),
                "thinking": True,
                "epochs": args.epochs,
                "learning_rate": args.lr,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        (
            'SFT complete. To select this trial: write {"checkpoint": '
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
    from huggingface_hub import snapshot_download

    host_model = Path(snapshot_download("Qwen/Qwen3-4B", local_files_only=True))
    prepared = prepare(data, output / "data", host_model)
    receipt_path = (checkpoint / "sft_submission.json").resolve(strict=True)
    if not receipt_path.is_relative_to(agent_dir):
        raise ValueError("SFT receipt escaped the agent directory")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("sft_sha256") != digest(prepared / "sft.jsonl"):
        raise ValueError(
            (
                "submitted SFT checkpoint was trained on different "
                "prompts/targets; rerun SFT on the final dataset"
            )
        )
    # Dereference symlinks and copy only HF runtime files, never agent .py code.
    frozen_checkpoint = output / "initial_checkpoint"
    frozen_checkpoint.mkdir()
    for path in checkpoint.iterdir():
        if path.name == "config.json" or path.name.startswith(
            ("model.safetensors", "model-", "pytorch_model")
        ):
            if not path.resolve(strict=True).is_relative_to(agent_dir):
                raise ValueError("Checkpoint file escaped the agent directory")
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                shutil.copyfile(path, frozen_checkpoint / path.name)
    for path in host_model.iterdir():
        if path.name in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
            "tokenizer.model",
        }:
            shutil.copyfile(path, frozen_checkpoint / path.name)
    checkpoint_files(frozen_checkpoint)
    manifest = {p.name: digest(p) for p in sorted(frozen_checkpoint.iterdir())}
    (output / "submission_manifest.json").write_text(
        json.dumps(
            {
                "selected_checkpoint": str(checkpoint),
                "checkpoint_sha256": manifest,
                "custom_rm_path": "rl.rubric_rm.custom_rm",
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


def rl_command(
    data, checkpoint, output, slime_env, slime_deps, probe=False, judge_config=None
):
    """Build the fixed Slime command for a probe or final run."""
    command = [
        "bash",
        str(CODING / "rl/run_rl_training_and_evaluation.sh"),
        "--rl-data-dir",
        str(data),
        "--init-checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
        "--rubric-rewards",
        "--slime-env",
        slime_env,
        "--slime-deps",
        slime_deps,
    ]
    if judge_config:
        for key, flag in JUDGE_FLAGS.items():
            command.extend([flag, judge_config[key]])
    if probe:
        command += [
            "--num-rollout",
            "2",
            "--save-interval",
            "1",
            "--eval-interval",
            "1",
            "--skip-final-eval",
        ]
    return command


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
        args.slime_env,
        args.slime_deps,
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
    # Rebuild from fixed trusted code, taking only the recorded environment paths.
    run(
        rl_command(
            bundle / "data",
            bundle / "initial_checkpoint",
            bundle / "rl",
            command[command.index("--slime-env") + 1],
            command[command.index("--slime-deps") + 1],
            judge_config=judge_config,
        )
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
    train.add_argument("--model", default="Qwen/Qwen3-4B")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-6)
    train.add_argument(
        "--conda-env", default=os.environ.get("SFT_CONDA_ENV", sys.prefix)
    )
    train.set_defaults(action=sft)
    for name in ("probe", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--task", default=os.environ.get("PDB_TASK", "coding"))
        command.add_argument(
            "--slime-env",
            default=os.environ.get("SLIME_ENV", "/home/vijayv/miniconda3/envs/slime"),
        )
        command.add_argument(
            "--slime-deps",
            default=os.environ.get("SLIME_DEPS", "/data/user_data/vijayv/slime_deps"),
        )
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
    data = prepare(args.data, str(args.output_dir) + ".data", checkpoint)
    reward_preflight(data, judge_config)
    run(
        rl_command(
            data,
            checkpoint,
            args.output_dir,
            args.slime_env,
            args.slime_deps,
            probe=True,
            judge_config=judge_config,
        )
    )


if __name__ == "__main__":
    main()

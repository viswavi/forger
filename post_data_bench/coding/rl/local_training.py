"""Local GPU training and full Hugging Face checkpoint handling."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .agent_training import CODING, JUDGE_FLAGS, run  # noqa: ABS101
from .backends import RL_DEFAULTS  # noqa: ABS101

BASE_MODEL = "Qwen/Qwen3-4B"
# The local RL recipe, with 100 steps x 8 prompts x 8 samples (RL_DEFAULTS):
# lr 1e-6, 16,000-token responses, clip 0.2 on both sides, weight decay 0.01,
# Adam beta2 0.999, and a checkpoint every 10 steps. The wrapper's defaults
# supply the rest (KL 1e-3 low_var_kl, temperature 1.0, eps-clip 0.2 below).
RECIPE = [
    "--lr",
    "1e-6",
    "--max-response-len",
    "16000",
    "--eps-clip-high",
    "0.2",
    "--weight-decay",
    "0.01",
    "--adam-beta2",
    "0.999",
    "--save-interval",
    "10",
]


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


def start_checkpoint(checkpoint=None):
    """The checkpoint an RL run starts from: the base model when None.

    The base model is the host's Hugging Face cache snapshot, whose files are
    links into the cache's blobs, so it skips the agent-checkpoint checks.
    """
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(BASE_MODEL, local_files_only=True)).resolve()
    if checkpoint is None or Path(checkpoint).resolve() == base:
        return base
    return checkpoint_files(checkpoint)


def rl_command(
    data,
    checkpoint,
    output,
    slime_env,
    slime_deps,
    probe=False,
    judge_config=None,
    num_rollout=RL_DEFAULTS["local"]["num_rollout"],
    rollout_batch_size=RL_DEFAULTS["local"]["rollout_batch_size"],
    n_samples_per_prompt=RL_DEFAULTS["local"]["n_samples_per_prompt"],
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
        "--skip-final-eval",
        *RECIPE,
    ]
    command += [
        "--num-rollout",
        str(num_rollout),
        "--rollout-batch-size",
        str(rollout_batch_size),
        "--n-samples-per-prompt",
        str(n_samples_per_prompt),
        "--global-batch-size",
        str(rollout_batch_size * n_samples_per_prompt),
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
        ]
    return command


def freeze_checkpoint(checkpoint, destination, agent_dir):
    """Copy validated weights and trusted base-model tokenizer files."""
    from huggingface_hub import snapshot_download

    host_model = Path(snapshot_download(BASE_MODEL, local_files_only=True))
    for path in checkpoint.iterdir():
        if path.name == "config.json" or path.name.startswith(
            ("model.safetensors", "model-", "pytorch_model")
        ):
            if not path.resolve(strict=True).is_relative_to(agent_dir):
                raise ValueError("Checkpoint file escaped the agent directory")
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}:
                shutil.copyfile(path, destination / path.name)
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
            shutil.copyfile(path, destination / path.name)
    shutil.copyfile(
        checkpoint / "rl_submission.json", destination / "rl_submission.json"
    )
    checkpoint_files(destination)


def execute_rl(
    data,
    checkpoint,
    output,
    judge_config,
    probe=False,
    slime_env=None,
    slime_deps=None,
    num_rollout=RL_DEFAULTS["local"]["num_rollout"],
    rollout_batch_size=RL_DEFAULTS["local"]["rollout_batch_size"],
    n_samples_per_prompt=RL_DEFAULTS["local"]["n_samples_per_prompt"],
    evaluate=False,
):
    """Run Slime then evaluate the resulting checkpoint with local vLLM."""
    from .agent_training import final_evaluation  # noqa: ABS101
    from .rl_result import usable_checkpoint  # noqa: ABS101

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), env.get("PATH", "")]
    )
    run(
        rl_command(
            data,
            checkpoint,
            output,
            slime_env or os.environ.get("SLIME_ENV", "/usr"),
            slime_deps or os.environ.get("SLIME_DEPS", "/opt"),
            probe=probe,
            judge_config=judge_config,
            num_rollout=num_rollout,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
        ),
        env=env,
    )
    for candidate in Path(output).glob("hf_rollout*"):
        suffix = candidate.name.removeprefix("hf_rollout")
        if not candidate.is_dir() or not suffix.isdigit():
            continue
        if not usable_checkpoint(candidate):
            continue
        checkpoint_files(candidate)
        (candidate / "rl_submission.json").write_text(
            json.dumps(
                {
                    "stage": "rl",
                    "backend": "local",
                    "base_model": BASE_MODEL,
                    "completed_steps": int(suffix) + 1,
                    "data_directory": str(Path(data).resolve()),
                },
                indent=2,
            )
            + "\n"
        )
    if evaluate and not probe:
        result = json.loads((Path(output) / "rl_result.json").read_text())
        final_evaluation(
            result["checkpoint"], Path(output) / ".hidden/final_eval", backend="local"
        )

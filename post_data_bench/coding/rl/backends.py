"""Backend identity shared by the launcher, training and recovery."""
from __future__ import annotations

import json
import os
from pathlib import Path

MODELS = {"tinker": "Qwen/Qwen3-8B", "local": "Qwen/Qwen3-4B"}
# Agent RL schedule when --num-rollout etc. are omitted; the local optimizer
# settings are local_training.RECIPE.
RL_DEFAULTS = {
    "local": {"num_rollout": 100, "rollout_batch_size": 8, "n_samples_per_prompt": 8},
    "tinker": {"num_rollout": 80, "rollout_batch_size": 32, "n_samples_per_prompt": 8},
}


def resolve_backend(value=None):
    """Resolve an explicit selection or the sandbox's configured backend."""
    backend = value or os.environ.get("PDB_TRAIN_BACKEND", "tinker")
    if backend not in MODELS:
        raise ValueError(f"Unknown training backend: {backend}")
    return backend


def saved_backend(directory, requested=None):
    """Read the run identity; old runs without a backend are Tinker runs."""
    path = Path(directory) / "token_budget_config.json"
    if path.exists():
        config = json.loads(path.read_text())
        backend = resolve_backend(config.get("backend", "tinker"))
        if requested is not None and requested != backend:
            raise ValueError("Requested backend differs from the saved run")
        return backend
    return resolve_backend(requested)

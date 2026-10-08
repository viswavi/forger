"""Persist evaluation identity and reject incompatible/partial resumes."""
import hashlib
import json
from argparse import Namespace
from pathlib import Path


def identity(args):
    """Identify model, data and settings that predictions may safely resume."""
    fields = (
        "model_path",
        "split",
        "limit",
        "num_shards",
        "enable_thinking",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
    )
    result = {key: getattr(args, key, None) for key in fields}
    result["evaluation_contract"] = CONTRACT_VERSION
    model = Path(args.model_path) if getattr(args, "model_path", None) else None
    result["model_signature"] = (
        [
            [p.name, p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(model.iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".jinja"}
        ]
        if model and model.is_dir()
        else None
    )
    return result


def check_resume(args, rows):
    """Reject persisted predictions from incompatible evaluation settings."""
    if not args.log_dir:
        return
    directory = Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state = identity(args)
    state["shard_index"] = args.shard_index
    state["tasks_sha256"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()
    ).hexdigest()
    path = directory / "evaluation_state.json"
    if path.exists():
        if json.loads(path.read_text()) != state:
            raise SystemExit(
                "Evaluation configuration changed; choose a new --out-dir or FORCE=1"
            )
    elif (directory / "predictions.jsonl").exists():
        raise SystemExit(
            (
                "Unversioned predictions cannot be resumed; choose a new "
                "--out-dir or FORCE=1"
            )
        )
    atomic_json(path, state)
    # A failed retry must never leave a prior success marker.
    (directory / "metrics.json").unlink(missing_ok=True)


def merge_rows(args, load_tasks, read_rows, key):
    """Merge shards only when they cover the requested tasks exactly."""
    rows, states = [], []
    for name in args.merge_shards:
        path = Path(name)
        if not path.is_file() or not (path.parent / "metrics.json").is_file():
            raise SystemExit("Cannot merge an incomplete shard")
        states.append(json.loads((path.parent / "evaluation_state.json").read_text()))
        rows.extend(read_rows(path))
    configs = [
        {k: v for k, v in state.items() if k not in ("shard_index", "tasks_sha256")}
        for state in states
    ]
    if not configs or any(config != configs[0] for config in configs):
        raise SystemExit("Shard configurations differ")
    config = configs[0]
    current = Namespace(**{**vars(args), **config})
    if identity(current) != config:
        raise SystemExit("Model files changed since evaluation")
    for state in states:
        shard_args = Namespace(
            **{**vars(args), **config, "shard_index": state["shard_index"]}
        )
        current_rows = load_tasks(shard_args)
        digest = hashlib.sha256(
            json.dumps(current_rows, sort_keys=True).encode()
        ).hexdigest()
        if digest != state["tasks_sha256"]:
            raise SystemExit("Task content changed since shard evaluation")
    expected = load_tasks(
        Namespace(**{**vars(args), **config, "num_shards": 1, "shard_index": 0})
    )
    ids = [row[key] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {row[key] for row in expected}:
        raise SystemExit("Shard merge has duplicate, missing, or unexpected task ids")
    return rows


CONTRACT_VERSION = 2


def read_predictions(path):
    """Repair only a torn final JSONL record; interior corruption is an error."""
    import warnings

    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("rb+") as stream:
        end = 0
        while line := stream.readline():
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError("Corrupt completed prediction record") from None
                warnings.warn(
                    "Discarding interrupted final prediction record", RuntimeWarning
                )
                stream.truncate(end)
                break
            rows.append(row)
            end = stream.tell()
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                break
    return rows


def atomic_json(path, value):
    """Publish a JSON document atomically."""
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stamp_metrics(metrics, args, data_dir):
    """Allow consumers to refuse baseline history from a different data contract."""
    from split_contract import sha256

    tasks_file = getattr(args, "tasks_file", None) or (
        Path(data_dir)
        / (
            "validation_tasks.jsonl"
            if args.split == "val"
            else ".hidden/test_tasks.jsonl"
        )
    )
    metrics["evaluation_contract"] = CONTRACT_VERSION
    metrics["tasks_sha256"] = sha256(tasks_file)
    return metrics

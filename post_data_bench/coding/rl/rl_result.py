"""Select the last usable HF checkpoint after completion or judge quota exhaustion."""
import argparse
import json
import re
import sys
from pathlib import Path


def usable_checkpoint(path):
    """Reject missing/partial exports without loading model weights into memory."""
    try:
        for name in ("config.json", "tokenizer_config.json"):
            json.loads((path / name).read_text())
        weights = []
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if (path / name).exists():
                weights.extend(
                    json.loads((path / name).read_text())["weight_map"].values()
                )
        if not weights:
            weights = [
                name
                for name in ("model.safetensors", "pytorch_model.bin")
                if (path / name).exists()
            ]
        if not weights:
            return False
        for name in set(weights):
            file = (path / name).resolve()
            if (
                not file.is_relative_to(path.resolve())
                or not file.is_file()
                or file.stat().st_size == 0
            ):
                return False
            if file.suffix == ".safetensors":
                with file.open("rb") as stream:
                    length = int.from_bytes(stream.read(8), "little")
                    if length <= 0 or length > min(
                        file.stat().st_size - 8, 100_000_000
                    ):
                        return False
                    header = json.loads(stream.read(length))
                tensors = [
                    value for key, value in header.items() if key != "__metadata__"
                ]
                if (
                    not tensors
                    or max(value["data_offsets"][1] for value in tensors)
                    != file.stat().st_size - 8 - length
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def select_result(output, initial_checkpoint, exit_code, quota_file):
    """Select the last usable checkpoint according to the termination reason."""
    output = Path(output)
    quota = False
    if quota_file and Path(quota_file).is_file():
        quota = (
            json.loads(Path(quota_file).read_text()).get("reason")
            == "judge_quota_exhausted"
        )
    if exit_code and not quota:
        raise RuntimeError(
            f"RL training failed (exit {exit_code}); no judge-quota stop was recorded"
        )
    candidates = [
        (int(match.group(1)), path)
        for path in output.glob("hf_rollout*")
        if (match := re.fullmatch(r"hf_rollout(\d+)", path.name))
    ]
    selected = next(
        (
            path
            for _, path in sorted(candidates, reverse=True)
            if usable_checkpoint(path)
        ),
        None,
    )
    if not quota and candidates and selected != max(candidates)[1]:
        raise RuntimeError(
            (
                "Final RL export is unusable; refusing to report an older "
                "checkpoint as completed"
            )
        )
    initial_fallback = False
    if selected is None:
        if not quota or not usable_checkpoint(Path(initial_checkpoint)):
            raise RuntimeError("No usable saved RL checkpoint is available")
        selected, initial_fallback = Path(initial_checkpoint), True
    warning = None
    if quota:
        warning = (
            "Rubric LLM quota exhausted; RL ended early. Returning the "
            "latest usable saved checkpoint."
        )
        if initial_fallback:
            warning += (
                " No RL checkpoint was saved; returning the starting "
                "checkpoint (no saved RL progress)."
            )
        print("WARNING: " + warning, file=sys.stderr, flush=True)
    result = {
        "checkpoint": str(selected.resolve()),
        "termination_reason": "judge_quota_exhausted" if quota else "completed",
        "training_exit_code": exit_code,
        "initial_checkpoint_fallback": initial_fallback,
        "warning": warning,
    }
    (output / "rl_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--training-exit-code", required=True, type=int)
    parser.add_argument("--quota-file", required=True)
    args = parser.parse_args()
    try:
        result = select_result(
            args.output_dir,
            args.init_checkpoint,
            args.training_exit_code,
            args.quota_file,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(result["checkpoint"])

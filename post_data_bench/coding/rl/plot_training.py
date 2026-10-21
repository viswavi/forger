"""Plot the SFT and RL learning curves of finished runs, one line per run.

Usage: python -m rl.plot_training RUN [RUN ...] [--out training.png]

RUN is a job log (runs/pdb-forger-v4_10861894.out, named in the legend
"pdb-forger-v4"), an agent run directory (.../agent_runs/agent_run_<tag>), or a
final directory (.../Qwen__Qwen3-8B_<tag>_final). Panels:

- SFT loss of the submitted trial, per step;
- RL reward per step, with the held-out reward of each eval as dots;
- share of RL prompts whose samples all got the same reward (no signal for RL);
- share of RL prompts every sample solved, and every sample failed (from
  rollouts.jsonl, written since 2026-10-18);
- mean response length and the share of responses cut off at max_response_len.

It also prints one summary line per run. A missing log leaves its panel empty
for that run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402

TAG = r"\d{8}_\d{6}_[0-9a-f]{6}"


class Run:
    """The logs of one run: the submitted SFT trial and the final RL run."""

    def __init__(self, label: str, final_dir: Path, agent_dir: Path):
        """Locate the run's directories; nothing is read until asked."""
        self.label = label
        self.final_dir = final_dir
        self.agent_dir = agent_dir

    def sft_dir(self) -> Path | None:
        """The submitted SFT trial's directory, from the final dir's manifests."""
        manifest = _read_json(self.final_dir / "submission_manifest.json") or {}
        selected = manifest.get("selected_checkpoint")
        if selected:
            return Path(selected)
        sub = _read_json(self.final_dir / "initial_checkpoint" / "sft_submission.json")
        data = (sub or {}).get("data_directory")  # .../sft_trial_4.data
        return Path(data.removesuffix(".data")) if data else None

    def sft(self) -> list[dict]:
        """Rows of the submitted trial's train_log.jsonl."""
        trial = self.sft_dir()
        return _read_jsonl(trial / "train_log.jsonl") if trial else []

    def rl(self) -> list[dict]:
        """Rows of the final RL run's train_log.jsonl."""
        return _read_jsonl(self.final_dir / "rl" / "train_log.jsonl")

    def heldout(self) -> list[dict]:
        """Rows of the final RL run's heldout_log.jsonl."""
        return _read_jsonl(self.final_dir / "rl" / "heldout_log.jsonl")

    def rollouts_by_step(self) -> dict[int, dict[str, float]]:
        """Per step, the share of groups all solved, all failed, and without signal."""
        counts: dict[int, Counter] = {}
        for g in _read_jsonl(self.final_dir / "rl" / "rollouts.jsonl"):
            rewards = g.get("rewards") or []
            if not rewards:
                continue
            c = counts.setdefault(int(g["step"]), Counter())
            c["groups"] += 1
            c["solved"] += all(r >= 1.0 - 1e-9 for r in rewards)
            c["failed"] += all(r <= 1e-9 for r in rewards)
            c["flat"] += max(rewards) - min(rewards) <= 1e-9
        return {
            step: {k: c[k] / c["groups"] for k in ("solved", "failed", "flat")}
            for step, c in sorted(counts.items())
        }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a line cut off by a killed run
    return out


def resolve(arg: str, models_dir: Path | None) -> Run:
    """A Run from a job log, an agent run directory, or a final directory."""
    path = Path(arg)
    if path.is_file():  # a job log: its most-named run tag, and where it lives
        text = path.read_text(errors="replace")
        tags = Counter(re.findall(rf"agent_run_({TAG})", text))
        if not tags:
            raise SystemExit(f"{arg}: no agent_run_<tag> in the log")
        tag = tags.most_common(1)[0][0]
        if models_dir is None:
            roots = Counter(
                re.findall(rf"(/[\w./-]+?)/agent_runs/agent_run_{tag}", text)
            )
            if not roots:
                raise SystemExit(
                    f"{arg}: cannot tell where the run lives; pass --models-dir"
                )
            models_dir = Path(roots.most_common(1)[0][0])
        label = re.sub(
            r"_\d+$", "", path.stem
        )  # pdb-forger-v4_10861894 -> pdb-forger-v4
    else:
        match = re.search(TAG, path.name)
        if not match:
            raise SystemExit(f"{arg}: not a job log or a run directory")
        tag, label = match.group(0), path.name
        if models_dir is None:
            models_dir = (
                path.parent.parent if path.parent.name == "agent_runs" else path.parent
            )
    final = next(
        iter(sorted(models_dir.glob(f"*_{tag}_final"))), models_dir / f"_{tag}_final"
    )
    return Run(label, final, models_dir / "agent_runs" / f"agent_run_{tag}")


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None and "step" in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(runs: list[Run], out: Path) -> None:
    """Draw the six panels for ``runs`` into ``out`` (PNG, or any matplotlib format)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    (sft_ax, rew_ax, flat_ax), (easy_ax, len_ax, trunc_ax) = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        sft_ax.plot(*_series(run.sft(), "loss"), color=color, label=run.label)
        rl = run.rl()
        rew_ax.plot(*_series(rl, "reward_mean"), color=color, label=run.label)
        held = _series(run.heldout(), "heldout_reward_mean")
        rew_ax.plot(*held, "o--", color=color, markersize=5, linewidth=0.8)
        roll = run.rollouts_by_step()
        steps = list(roll)
        if steps:
            flat_ax.plot(
                steps, [roll[s]["flat"] for s in steps], color=color, label=run.label
            )
            easy_ax.plot(
                steps, [roll[s]["solved"] for s in steps], color=color, label=run.label
            )
            easy_ax.plot(steps, [roll[s]["failed"] for s in steps], "--", color=color)
        else:  # older runs: the train log's count of groups with signal
            batch = 128
            xs, ys = _series(rl, "groups_with_signal")
            flat_ax.plot(
                xs, [1 - y / batch for y in ys], ":", color=color, label=run.label
            )
        len_ax.plot(*_series(rl, "response_len_mean"), color=color, label=run.label)
        trunc_ax.plot(*_series(rl, "truncated_frac"), color=color, label=run.label)
    titles = [
        (sft_ax, "SFT loss (submitted trial)", "SFT step", "loss"),
        (
            rew_ax,
            "RL reward (line) and held-out reward (dots)",
            "RL step",
            "mean reward",
        ),
        (
            flat_ax,
            "RL prompts with no signal (all samples same reward)",
            "RL step",
            "share",
        ),
        (
            easy_ax,
            "RL prompts all solved (solid) / all failed (dashed)",
            "RL step",
            "share",
        ),
        (len_ax, "RL mean response length", "RL step", "tokens"),
        (trunc_ax, "RL responses cut off at max length", "RL step", "share"),
    ]
    for ax, title, xlabel, ylabel in titles:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in (rew_ax, flat_ax, easy_ax, trunc_ax):
        ax.set_ylim(-0.02, 1.02)
    sft_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summary(run: Run) -> str:
    """One line: rows, SFT loss, RL and held-out reward, and RL's all-solved share."""

    def ends(values: list[float], fmt: str = "{:.2f}") -> str:
        return f"{fmt.format(values[0])}→{fmt.format(values[-1])}" if values else "-"

    sft, rl, held = run.sft(), run.rl(), run.heldout()
    roll = run.rollouts_by_step()
    data = run.final_dir / "data" / "train.jsonl"
    rows = len(_read_jsonl(data)) if data.exists() else None
    return (
        f"{run.label:16} rows {rows if rows is not None else '-':>5} · "
        f"SFT loss {ends(_series(sft, 'loss')[1])} · "
        f"RL reward {ends(_series(rl, 'reward_mean')[1])} · "
        f"held-out {ends(_series(held, 'heldout_reward_mean')[1])} · "
        f"all solved {ends([v['solved'] for v in roll.values()], '{:.0%}')}"
    )


def main() -> None:
    """Parse arguments, print the summaries, and write the figure."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", help="job logs or run directories")
    p.add_argument(
        "--out", default="training.png", help="figure path (default %(default)s)"
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        help="the agent_data_models/<task> directory (default: read from each job log)",
    )
    args = p.parse_args()
    runs = [resolve(r, args.models_dir) for r in args.runs]
    for run in runs:
        print(summary(run))
    plot(runs, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

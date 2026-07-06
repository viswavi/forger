#!/usr/bin/env python3
"""Evaluate Qwen3-1.7B on GPQA Diamond via the Forger Inspect bridge.

Routes generation through :class:`pyforger.inference.VLLMBackend` (preserving
the shared-engine registry) using a Qwen3-correct chat template, and lets
Inspect AI + ``inspect_evals`` own dataset loading, scoring, and logging.

Designed to run on a single CUDA GPU; do **not** run on a login node. See
``scripts/README.md`` for Babel-specific setup, Slurm usage, and the GPQA
license / HF auth steps.

Example
-------
::

    # Smoke test (5 samples)
    python scripts/eval_qwen3_gpqa.py --limit 5

    # Full GPQA Diamond run (198 questions)
    python scripts/eval_qwen3_gpqa.py --task inspect_evals/gpqa_diamond
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI flags. Defaults are tuned for Qwen3-1.7B on a single GPU."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3 model on GPQA via the Forger Inspect bridge.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HF model id or local path (passed to VLLMBackend).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="inspect_evals/gpqa_diamond",
        help=(
            "Inspect task id (must be registered, e.g. by importing "
            "inspect_evals.gpqa). GPQA Diamond is 198 questions."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples evaluated. Use a small value (e.g. 5) for smoke tests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help=(
            "Generation budget per sample. Qwen3 thinking mode emits long "
            "<think> traces, so keep this generous."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is greedy.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode in the chat template (default).",
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode for faster / cheaper runs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="Fraction of GPU memory vLLM may use.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max context length for the vLLM engine.",
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1,
        help=(
            "Concurrent generations Inspect may dispatch. Offline vLLM is "
            "not built for concurrent .generate() calls, so keep at 1."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Where Inspect writes eval logs (defaults to ./logs).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to dump the flattened metrics dict as JSON.",
    )
    return parser.parse_args()


def build_qwen3_render(model: str, *, enable_thinking: bool):
    """Return a renderer that applies Qwen3's chat template via ``transformers``.

    The Forger bridge expects ``render(messages) -> str | list[messages]``. For
    a text-completion backend like :class:`VLLMBackend` we need a string with
    the model's chat template already applied (and the assistant-turn cue
    added), otherwise an instruct model like Qwen3 produces degenerate output
    and scores collapse.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the Qwen3 chat-template renderer. "
            "Install with: pip install 'transformers>=4.51'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model)

    def render(messages):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    return render


def main() -> int:
    """Wire CLI -> backend -> bridge -> Inspect eval, then print/dump metrics."""
    args = parse_args()

    try:
        from inspect_ai.util._display import init_display_type
    except ImportError as exc:
        print(
            "inspect_ai is not installed. Install with: "
            "pip install 'pyforger[inspect]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    init_display_type("plain")

    try:
        import inspect_evals.gpqa  # noqa: F401  # registers the gpqa tasks
    except ImportError as exc:
        print(
            "inspect_evals is not installed. Install with: "
            "pip install git+https://github.com/UKGovernmentBEIS/inspect_evals",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from pyforger.inference import (
        SamplingConfig,
        VLLMBackend,
        evaluate_with_inspect,
    )

    render = build_qwen3_render(args.model, enable_thinking=args.enable_thinking)

    backend = VLLMBackend(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling = SamplingConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    eval_kwargs: dict = {"max_connections": args.max_connections}
    if args.log_dir is not None:
        eval_kwargs["log_dir"] = args.log_dir

    result = evaluate_with_inspect(
        backend,
        args.task,
        render=render,
        default_max_tokens=args.max_tokens,
        limit=args.limit,
        sampling=sampling,
        **eval_kwargs,
    )

    print(f"task={result.task}")
    print(f"model={result.model}")
    print(f"samples={result.samples}")
    print("metrics:")
    for key, value in sorted(result.metrics.items()):
        print(f"  {key}: {value}")

    if args.json_output is not None:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(
                {
                    "task": result.task,
                    "model": result.model,
                    "samples": result.samples,
                    "metrics": result.metrics,
                },
                f,
                indent=2,
            )
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

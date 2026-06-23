#!/usr/bin/env python3
"""Estimate the quality of a Forger candidate dataset via HumanEval ICL uplift.

Wires :func:`pyforger.evaluate.estimate_quality` to a vLLM-backed Qwen3 model
and a held-out HumanEval validation slice, then prints/dumps a
:class:`pyforger.evaluate.QualityReport`. Designed for a single CUDA GPU on
Babel; never run on a login node. See ``scripts/README.md`` for setup.

Example::

    # Smoke test
    python scripts/eval_forger_humaneval.py \\
        --forger-jsonl /path/to/coding_questions_trl_format.jsonl \\
        --n-val 5 --n-subsets 2 --k 2 --allow-code-execution

    # Full run
    python scripts/eval_forger_humaneval.py \\
        --forger-jsonl /path/to/coding_questions_trl_format.jsonl \\
        --n-demos 20 --n-val 30 --n-subsets 5 --k 4 \\
        --allow-code-execution \\
        --json-output logs/forger-humaneval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The HumanEval executor forks subprocesses after the chat-template tokenizer
# has been used, which makes HuggingFace `tokenizers` spam a fork-after-parallelism
# warning. Disable tokenizer parallelism before transformers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the quality of a Forger candidate dataset via "
            "HumanEval ICL uplift."
        ),
    )
    parser.add_argument(
        "--forger-jsonl",
        type=str,
        required=True,
        help="Path to the Forger TRL-format JSONL (Slack candidate data).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HF model id or local path (passed to VLLMBackend).",
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=20,
        help="Size of the seeded candidate-set subsample.",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=30,
        help="Number of HumanEval problems to use as validation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Demos per few-shot prompt.",
    )
    parser.add_argument(
        "--n-subsets",
        type=int,
        default=5,
        help="Random demo subsets per validation problem.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed for subsample, holdout, and subset sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help=(
            "Generation budget per completion. Code-completion ICL "
            "rarely needs more; raise for thinking mode."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is greedy (the pass@1 setting).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help=(
            "Enable Qwen3 thinking mode in the chat template. Off by "
            "default for code-completion ICL since the demos contain no "
            "<think> traces."
        ),
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
        "--timeout-s",
        type=int,
        default=10,
        help="Per-program wall-clock limit in the executor subprocess.",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        default=False,
        help=(
            "Required: opt in to running model-generated code in a "
            "subprocess (hardened, not sandboxed). Refuse to score "
            "without this flag."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to dump the QualityReport as JSON.",
    )
    return parser.parse_args()


def build_qwen3_render(model: str, *, enable_thinking: bool):
    """Multi-turn Qwen3 chat-template renderer (mirrors ``eval_qwen3_gpqa.py``)."""
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
    args = parse_args()

    if not args.allow_code_execution:
        print(
            "Refusing to run: HumanEval scoring executes model-generated "
            "code in a subprocess. Re-run with --allow-code-execution to opt in.",
            file=sys.stderr,
        )
        return 2

    from pyforger.evaluate import (
        estimate_quality,
        holdout_split,
        load_forger_jsonl,
        load_humaneval,
        subsample_pool,
    )
    from pyforger.inference import SamplingConfig, VLLMBackend

    print(f"loading Forger demos from {args.forger_jsonl}")
    demos_full = load_forger_jsonl(args.forger_jsonl)
    print(f"  {len(demos_full)} demos in file; subsampling to n_demos={args.n_demos}")
    demo_pool = subsample_pool(demos_full, n=args.n_demos, seed=args.seed)

    print("loading HumanEval (openai_humaneval)")
    he = load_humaneval()
    print(f"  {len(he)} problems; holding out n_val={args.n_val}")
    eval_data = holdout_split(he, n=args.n_val, seed=args.seed)

    print(f"building Qwen3 renderer (thinking={args.thinking})")
    render = build_qwen3_render(args.model, enable_thinking=args.thinking)

    print(
        f"constructing VLLMBackend(model={args.model!r}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len})"
    )
    backend = VLLMBackend(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # Few-shot prompts share long demo prefixes within a problem; prefix
        # caching pays for itself here.
        enable_prefix_caching=True,
    )

    sampling = SamplingConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    report = estimate_quality(
        backend,
        demo_pool,
        eval_data,
        render=render,
        k=args.k,
        n_subsets=args.n_subsets,
        sampling=sampling,
        seed=args.seed,
        timeout_s=args.timeout_s,
        allow_code_execution=True,
    )

    print()
    print(f"model={args.model}")
    print(f"n_demos={len(demo_pool)}  n_val={len(eval_data)}  k={args.k}  "
          f"n_subsets={args.n_subsets}  seed={args.seed}")
    print(f"baseline_pass@1   = {report.baseline_pass_at_1:.4f}")
    print(f"conditioned_pass@1= {report.conditioned_pass_at_1:.4f}")
    print(f"uplift            = {report.metric:+.4f}  "
          f"(95% CI [{report.bootstrap_ci[0]:+.4f}, {report.bootstrap_ci[1]:+.4f}])")

    if args.json_output is not None:
        out_path = Path(args.json_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

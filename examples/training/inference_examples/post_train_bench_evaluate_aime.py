#!/usr/bin/env python3
from __future__ import annotations

import os

import argparse
import json

from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

import inspect_evals.aime2025  # noqa: F401, E402  (registers task definitions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI eval without banners.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Path to the Hugging Face model (directory or model identifier).",
    )
    # this is a good limit for this task, just keep it like that (or use less in case you want faster tests)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
    )
    # Qwen3 thinking-mode recommended sampling (Qwen3 manual): without an explicit
    # temperature the vLLM server defaults to 1.0, which we measured swinging the
    # same checkpoint between 0.267 and 0.533 on this 30-problem benchmark.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    # You can adjust --max-connections if you want faster tests and don't receive errors (or if you have issues with vllm, try lowering this value)
    parser.add_argument(
        "--max-connections",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default="templates/",
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help="Directory to write Inspect eval logs (the per-sample outputs).",
    )
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        help="Force the model's chat template into thinking mode "
             "(passes chat_template_kwargs={'enable_thinking': True} to vLLM). "
             "Only meaningful for models with a thinking mode, e.g. Qwen3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = "inspect_evals/aime2025"
    # Use each model's native chat template (do NOT pass a custom `chat_template`):
    # these off-the-shelf instruct models ship correct templates. Thinking is
    # forced on explicitly (when requested) by passing chat_template_kwargs to
    # vLLM via extra_body. NB: Inspect's vLLM path defaults Qwen3 to NON-thinking,
    # so this must be set explicitly, and it must be passed as an eval() kwarg
    # (the authoritative call-level config) -- a model-level config is overridden.
    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }

    if args.enable_thinking:
        # Force thinking via a server-side chat template that never injects an
        # empty <think></think> block. This is robust to Inspect's vLLM path
        # (which otherwise defaults Qwen3 to non-thinking). extra_body is kept as
        # belt-and-suspenders for vLLM builds that honor chat_template_kwargs.
        model_args["chat_template"] = os.path.join(
            args.templates_dir, "qwen3_think.jinja"
        )
        other_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    if args.log_dir is not None:
        other_kwargs["log_dir"] = args.log_dir

    # min_p is not a GenerateConfig field; vLLM's OpenAI server accepts it via
    # extra_body (merge: the thinking path above may already have set extra_body).
    other_kwargs.setdefault("extra_body", {})["min_p"] = args.min_p

    eval_out = inspect_eval(
        task,
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        score_display=False,
        timeout=18000000,
        attempt_timeout=18000000,
        log_realtime=False,
        log_format='json',
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_connections=args.max_connections,
        **other_kwargs,
    )
    
    if args.json_output_file is not None:
        assert len(eval_out) == 1, eval_out
        assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
        metrics = {}
        for k, v in eval_out[0].results.scores[0].metrics.items():
            metrics[k] = v.value

        with open(args.json_output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def model_type(args) -> str:
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    architecture = config['architectures'][0].lower()
    if 'gemma' in architecture:
        return 'gemma'
    if 'llama' in architecture:
        return 'llama'
    if 'qwen' in architecture:
        return 'qwen'
    if 'smollm' in architecture:
        return 'smollm'
    raise ValueError(architecture)

def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == 'qwen':
        template = 'qwen3.jinja'
    elif model_type_str == 'llama':
        template = 'llama3.jinja'
    elif model_type_str == 'gemma':
        template = 'gemma3.jinja'
    elif model_type_str == 'smollm':
        template = 'smollm.jinja'
    else:
        raise ValueError(model_type_str)
    return {
        'chat_template': os.path.join(args.templates_dir, template)
    }

if __name__ == "__main__":
    main()

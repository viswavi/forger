r"""Chat completions through Tinker, rendered exactly as a vLLM server would.

Tinker samples token ids, so we apply the HF chat template ourselves
(e.g. enable_thinking=True gives "...<|im_start|>assistant\n") and decode
with skip_special_tokens=True, like vLLM's chat endpoint without a reasoning
parser. TINKER_API_KEY is read from the environment by the SDK.

Every request is charged to a dollar ledger (rl/tinker_budget.py): the one
named by TINKER_EVAL_BUDGET (sft, rl or eval), else a record-only
<log_dir>/tinker_usage.json.
"""
from __future__ import annotations

import os
import time

from rl.tinker_budget import (
    PRICES_PER_MILLION,
    TinkerBudget,
    budget_from_env,
    cost_usd,
    sample_tokens,
)


class TinkerChat:
    """Callable returning (completion, finish_reason, usage) for one chat prompt."""

    def __init__(self, args, user_metadata: dict[str, str] | None = None):
        """Load the tokenizer and open a Tinker sampling client for args.model_path."""
        import tinker
        from transformers import AutoTokenizer, GenerationConfig

        if args.min_p:
            raise SystemExit("Tinker sampling has no min_p; use --min-p 0")
        self.types = tinker.types
        if os.environ.get("TINKER_EVAL_BUDGET"):
            self.budget = budget_from_env(os.environ["TINKER_EVAL_BUDGET"])
        else:
            self.budget = TinkerBudget(
                os.path.join(args.log_dir or ".", "tinker_usage.json"),
                None,
                name="eval",
            )
        service = tinker.ServiceClient(user_metadata=user_metadata)
        # Lets rl.tinker_budget.reconcile match the ledger to Tinker billing rows.
        self.session_id = getattr(service.holder, "_session_id", None)
        # A tinker://... path samples saved weights; anything else is a base model.
        if args.model_path.startswith("tinker://"):
            self.client = service.create_sampling_client(model_path=args.model_path)
        else:
            self.client = service.create_sampling_client(base_model=args.model_path)
        self.base_model = self.client.get_base_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        # vLLM stops on generation_config's eos ids (Qwen3: <|im_end|>, <|endoftext|>).
        eos = GenerationConfig.from_pretrained(self.base_model).eos_token_id
        self.stop = (
            eos if isinstance(eos, list) else [eos or self.tokenizer.eos_token_id]
        )
        self.max_context = next(
            (
                m.max_context_length
                for m in service.get_server_capabilities().supported_models
                if m.model_name == self.base_model and m.max_context_length
            ),
            32768,
        )
        self.args = args

    async def __call__(self, messages: list[dict]) -> tuple[str, str, dict]:
        """Sample one completion for messages."""
        args = self.args
        kwargs = {}
        if args.enable_thinking is not None:
            kwargs["enable_thinking"] = args.enable_thinking
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        prompt = self.tokenizer.encode(text, add_special_tokens=False)
        # Keep prompt + completion inside the context window (vLLM would reject it).
        max_tokens = min(args.max_tokens, self.max_context - len(prompt))
        params = self.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=self.stop,
        )
        worst_case = cost_usd(
            self.base_model, sample_tokens(len(prompt), 1, 0, [max_tokens])
        )
        # Holds the worst case until the call is recorded, waiting for other
        # requests when it does not fit; raises TinkerBudgetExceeded when none
        # are in flight. Resume the run once the budget allows.
        async with self.budget.reserve(worst_case):
            start = time.monotonic()
            resp = await self.client.sample_async(
                prompt=self.types.ModelInput.from_ints(prompt),
                num_samples=1,
                sampling_params=params,
            )
            seq = resp.sequences[0]
            self.budget.record(
                sample_tokens(
                    len(prompt), 1, resp.prompt_cache_hit_tokens, [len(seq.tokens)]
                ),
                self.session_id,
            )
        completion = self.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        usage = {
            "base_model": self.base_model,
            "prompt_tokens": len(prompt),
            "sampled_tokens": len(seq.tokens),
            "prompt_cache_hit_tokens": resp.prompt_cache_hit_tokens,
            "max_tokens_clamped": max_tokens < args.max_tokens,
            "latency_s": round(time.monotonic() - start, 3),
        }
        return completion, seq.stop_reason, usage


def usage_summary(rows: list[dict]) -> dict | None:
    """Total token usage and estimated USD over rows that recorded usage."""
    usages = [r["usage"] for r in rows if r.get("usage")]
    if not usages:
        return None
    prompt = sum(u["prompt_tokens"] for u in usages)
    sampled = sum(u["sampled_tokens"] for u in usages)
    cached = sum(u.get("prompt_cache_hit_tokens") or 0 for u in usages)
    lengths = sorted(u["sampled_tokens"] for u in usages)
    latencies = sorted(u["latency_s"] for u in usages if "latency_s" in u)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    summary = {
        "n_requests": len(usages),
        "prompt_tokens": prompt,
        "sampled_tokens": sampled,
        "prompt_cache_hit_tokens": cached,
        "mean_sampled_tokens": sampled / len(usages),
        "p50_sampled_tokens": pct(lengths, 0.5),
        "p90_sampled_tokens": pct(lengths, 0.9),
        "max_sampled_tokens": lengths[-1],
        "n_truncated": sum(1 for r in rows if r.get("truncated")),
        "n_max_tokens_clamped": sum(1 for u in usages if u.get("max_tokens_clamped")),
        "p50_latency_s": pct(latencies, 0.5),
        "p90_latency_s": pct(latencies, 0.9),
    }
    price = PRICES_PER_MILLION.get(usages[0].get("base_model"))
    if price:
        # Conservative: every prompt token at the full prefill rate.
        summary["estimated_cost_usd"] = (
            prompt * price["prefill"] + sampled * price["sample"]
        ) / 1e6
        summary["estimated_cost_usd_with_cache_hits"] = (
            (prompt - cached) * price["prefill"]
            + cached * price["cached_prefill"]
            + sampled * price["sample"]
        ) / 1e6
    return summary

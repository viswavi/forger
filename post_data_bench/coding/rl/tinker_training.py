"""Tinker-hosted LoRA SFT and rubric RL for Qwen3-8B, charged to dollar ledgers.

Training runs remotely on Tinker, so neither the agent sandbox nor the host
needs a GPU. Prompts are rendered with the model's own chat template in
thinking mode, exactly as the evaluator does. Checkpoints are recorded in a
local directory as `tinker_checkpoint.json` (tinker:// paths) plus receipts.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

# Supports both `rl` CLI imports and `coding.rl` library imports.
from .rubric_data import prompt_id  # noqa: ABS101
from .tinker_budget import (  # noqa: ABS101
    BASE_MODEL,
    TinkerBudgetExceeded,
    cost_usd,
    max_concurrent_samples,
    sample_tokens,
)

# Full-finetune LR -> LoRA LR, the rule train_sft.py and the Tinker cookbook use.
LORA_LR_MULTIPLIER = 10.0
DEFAULT_LORA_RANK = 32
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
_SAMPLE_SLOTS = {}  # event loop -> semaphore capping in-flight sample requests


def _sample_slot():
    loop = asyncio.get_running_loop()
    if loop not in _SAMPLE_SLOTS:
        _SAMPLE_SLOTS[loop] = asyncio.Semaphore(max_concurrent_samples())
    return _SAMPLE_SLOTS[loop]


def load_tokenizer(model=BASE_MODEL):
    """Load the base model's tokenizer and chat template."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def prompt_tokens(tokenizer, messages):
    """Thinking-mode generation prompt, identical to the evaluator's template."""
    return list(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
            return_dict=False,
        )
    )


def sft_tokens(tokenizer, messages):
    """Split prompt and target tokens.

    The target keeps its thinking and ends at <|im_end|>.
    """
    prompt = prompt_tokens(tokenizer, messages[:-1])
    full = list(
        tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template does not extend the generation prompt")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Drop the template's trailing newline after the final <|im_end|>.
    full = full[: len(full) - full[::-1].index(end)]
    return prompt, full[len(prompt) :]


def stop_token_ids(tokenizer):
    """Token ids that end an assistant turn."""
    return [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]


def response_text(tokenizer, tokens):
    """Decode a sampled turn the way the evaluator sees it, minus the stop token."""
    stops = set(stop_token_ids(tokenizer))
    while tokens and tokens[-1] in stops:
        tokens = tokens[:-1]
    return tokenizer.decode(tokens, skip_special_tokens=False)


def datum(tokens, **per_target):
    """Next-token datum: inputs tokens[:-1], targets tokens[1:], aligned extras."""
    import tinker

    inputs = {"target_tokens": tokens[1:], **per_target}
    for name, values in inputs.items():
        if len(values) != len(tokens) - 1:
            raise ValueError(f"{name} must align with the {len(tokens) - 1} targets")
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs
    )


def service_client(budget, run_tag=None):
    """Tinker session tagged with its ledger, so billing records can be matched."""
    import tinker

    metadata = {"pdb_budget": budget.name}
    if run_tag:
        metadata["pdb_run"] = run_tag
    return tinker.ServiceClient(user_metadata=metadata)


def session_id(client):
    """Best-effort session id for billing reconciliation."""
    return getattr(getattr(client, "holder", None), "_session_id", None)


def save_checkpoint(training_client, name):
    """Save resumable state and sampler weights; return both tinker:// paths."""
    state = training_client.save_state(name)
    sampler = training_client.save_weights_for_sampler(name)
    return {"state_path": state.result().path, "sampler_path": sampler.result().path}


def read_checkpoint(directory):
    """Load and check a recorded Tinker checkpoint."""
    record = json.loads((Path(directory) / "tinker_checkpoint.json").read_text())
    if record.get("base_model") != BASE_MODEL:
        raise ValueError(f"the fixed recipe requires a {BASE_MODEL} checkpoint")
    for key in ("state_path", "sampler_path"):
        if not str(record.get(key, "")).startswith("tinker://"):
            raise ValueError(f"checkpoint {key} must be a tinker:// path")
    return record


def cosine_lr(peak, step, total, warmup_ratio=0.03):
    """Linear warmup then cosine decay to zero, like the HF trainer default."""
    warmup = max(1, math.ceil(total * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def run_sft(
    sft_jsonl,
    output_dir,
    budget,
    *,
    epochs=4,
    lr=1e-6,
    lora_rank=DEFAULT_LORA_RANK,
    batch_size=128,
    seed=0,
    run_tag=None,
):
    """Run LoRA SFT on prompt/target pairs; loss is the mean NLL over target tokens.

    `lr` is the full-finetune learning rate; LoRA uses LORA_LR_MULTIPLIER times it.
    """
    import tinker

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer()
    examples = []
    for line in Path(sft_jsonl).read_text().splitlines():
        prompt, target = sft_tokens(tokenizer, json.loads(line)["messages"])
        examples.append((prompt + target, len(prompt)))
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total = steps_per_epoch * epochs
    peak = lr * LORA_LR_MULTIPLIER
    # Refuse a trial that cannot finish: a run cut off by the budget saves no
    # checkpoint.
    estimate = cost_usd(
        BASE_MODEL, {"train": epochs * sum(len(t) - 1 for t, _ in examples)}
    )
    spent = budget.check(estimate)["cost_usd"]
    print(
        f"SFT: {len(examples)} rows x {epochs} epochs, estimated ${estimate:.2f} "
        f"(${spent:.2f} of the {budget.name} budget already spent)",
        flush=True,
    )
    service = service_client(budget, run_tag)
    trainer = service.create_lora_training_client(
        base_model=BASE_MODEL, rank=lora_rank, seed=seed
    )
    rng = random.Random(seed)
    order = list(range(len(examples)))
    started = time.time()
    with (output / "train_log.jsonl").open("w") as log:
        for step in range(total):
            if step % steps_per_epoch == 0:
                rng.shuffle(order)
            first = (step % steps_per_epoch) * batch_size
            batch = [examples[i] for i in order[first : first + batch_size]]
            target_count = sum(len(tokens) - start for tokens, start in batch)
            train_tokens = sum(len(tokens) - 1 for tokens, _ in batch)
            budget.check(cost_usd(BASE_MODEL, {"train": train_tokens}))
            # Weights sum to 1 per batch: mean target-token NLL, as in TRL.
            data = [
                datum(
                    tokens,
                    weights=[0.0] * (start - 1)
                    + [1.0 / target_count] * (len(tokens) - start),
                )
                for tokens, start in batch
            ]
            step_lr = cosine_lr(peak, step, total)
            backward = trainer.forward_backward(data, "cross_entropy")
            update = trainer.optim_step(
                tinker.AdamParams(learning_rate=step_lr, grad_clip_norm=1.0)
            )
            result = backward.result()
            update.result()
            spent = budget.record({"train": train_tokens}, session_id(service))
            log.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "epoch": step // steps_per_epoch + 1,
                        "lr": step_lr,
                        "loss": result.metrics.get("loss:sum"),
                        "train_tokens": train_tokens,
                        "tinker_sft_cost_usd": spent["cost_usd"],
                        "elapsed_s": round(time.time() - started, 1),
                    }
                )
                + "\n"
            )
            log.flush()
    record = {
        "base_model": BASE_MODEL,
        "lora_rank": lora_rank,
        "session_id": session_id(service),
        **save_checkpoint(trainer, f"sft-{output.name}"),
    }
    (output / "tinker_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


async def sample_group(
    sampler,
    tokenizer,
    messages,
    budget,
    *,
    num_samples,
    max_tokens,
    temperature,
    top_p=1.0,
    top_k=-1,
    service=None,
):
    """Sample one prompt and charge its billed tokens.

    Returns (prompt_tokens, sequences).
    """
    import tinker

    prompt = prompt_tokens(tokenizer, messages)
    worst_case = cost_usd(
        BASE_MODEL,
        sample_tokens(len(prompt), num_samples, 0, [max_tokens] * num_samples),
    )
    # The worst case stays reserved on the ledger until the call is recorded.
    async with _sample_slot(), budget.reserve(worst_case):
        response = await sampler.sample_async(
            prompt=tinker.ModelInput.from_ints(prompt),
            num_samples=num_samples,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop_token_ids(tokenizer),
            ),
        )
        budget.record(
            sample_tokens(
                len(prompt),
                num_samples,
                response.prompt_cache_hit_tokens,
                [len(sequence.tokens) for sequence in response.sequences],
            ),
            session_id(service),
        )
    return prompt, list(response.sequences)


async def _score(rows_and_texts):
    """(rewards, blocked) for (row, response_text) pairs via the fixed reward function.

    ``blocked[i]`` is True when the judge's content filter blocked one of
    response i's prompt criteria (custom_rm's remove_sample); RL skips it.
    """
    from .rubric_rm import custom_rm  # noqa: ABS101

    samples = [
        SimpleNamespace(
            metadata=row["metadata"],
            response=text,
            label=row["label"],
            remove_sample=False,
        )
        for row, text in rows_and_texts
    ]
    rewards = await asyncio.gather(*[custom_rm(None, s) for s in samples])
    return list(rewards), [s.remove_sample for s in samples]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def run_rl(
    data_dir,
    init_checkpoint,
    output_dir,
    budget,
    *,
    num_rollout=25,
    rollout_batch_size=128,
    n_samples_per_prompt=8,
    max_response_len=16384,
    temperature=1.0,
    lr=6e-6,
    eval_interval=5,
    n_samples_per_eval_prompt=4,
    save_interval=5,
    max_steps_off_policy=1,
    seed=0,
    run_tag=None,
):
    """GRPO on rubric rewards from a recorded SFT checkpoint.

    Each step trains on rollout_batch_size prompts x n_samples_per_prompt
    completions with group-normalized advantages and one importance-sampling
    optimizer step. Tinker samples a whole batch in parallel, so wall-clock is
    roughly steps x the slowest sample (~9 min at 16K tokens and ~30 tokens/s).
    Hence fewer, larger steps than Slime's 100 x 32 prompts: the default
    25 x 128 prompts keeps the same 25,600 rollouts. With max_steps_off_policy=1
    the next batch is sampled from the pre-update weights while a step trains;
    the loss uses the sampler's logprobs, which corrects for the one-step lag.
    Held-out reward is scored in the background. `lr` is the full-finetune
    rate (LoRA uses 10x). Budget or judge-quota exhaustion stops early and
    keeps a final checkpoint.
    """
    return asyncio.run(
        _run_rl(
            data_dir,
            init_checkpoint,
            output_dir,
            budget,
            num_rollout=num_rollout,
            rollout_batch_size=rollout_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            max_response_len=max_response_len,
            temperature=temperature,
            lr=lr,
            eval_interval=eval_interval,
            n_samples_per_eval_prompt=n_samples_per_eval_prompt,
            save_interval=save_interval,
            max_steps_off_policy=max_steps_off_policy,
            seed=seed,
            run_tag=run_tag,
        )
    )


_ANSWER_CHARS = 300  # of each response's final answer kept in rollouts.jsonl


def _log_rollouts(stream, step, row, sequences, texts, rewards, blocked):
    """Append one training group to rollouts.jsonl.

    e.g. {"step": 3, "prompt_id": "9f2c...", "rewards": [1.0, 0.0, ...],
    "blocked": [false, true, ...], "truncated": [false, true, ...], "answers":
    ["This task is impossible...", ...]}. prompt_id is rubric_data.prompt_id of
    the row, so groups join back to the submitted rows; answers are the text
    after the last </think>, cut short. A blocked sample (the content filter
    blocked one of its judged criteria) is not trained on.
    """
    stream.write(
        json.dumps(
            {
                "step": step,
                "prompt_id": prompt_id(row),
                "rewards": rewards,
                "blocked": blocked,
                "truncated": [s.stop_reason == "length" for s in sequences],
                "answers": [
                    text.rsplit("</think>", 1)[-1].strip()[:_ANSWER_CHARS]
                    for _, text in texts
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    stream.flush()


def _group_datums(groups):
    """Importance-sampling data from (prompt, sequences, rewards, blocked) groups.

    Advantages are group-normalized, then divided by the batch's sampled
    tokens so the loss is a token mean. Groups with identical rewards carry
    no signal and are skipped. Blocked samples get no datum and stay out of
    their group's mean and std, e.g. rewards [1, 0, 0.9 (blocked)] train on
    [1, 0] alone.
    """
    sampled_total = sum(
        len(s.tokens) for _, sequences, _, _ in groups for s in sequences
    )
    data, kept = [], 0
    for prompt, sequences, rewards, blocked in groups:
        trusted = [r for r, skip in zip(rewards, blocked) if not skip]
        spread = statistics.pstdev(trusted) if trusted else 0
        if spread == 0:
            continue
        kept += 1
        mean = statistics.fmean(trusted)
        for sequence, reward, skip in zip(sequences, rewards, blocked):
            if skip:
                continue
            advantage = (reward - mean) / (spread + 1e-6)
            pad = [0.0] * (len(prompt) - 1)
            data.append(
                datum(
                    prompt + list(sequence.tokens),
                    logprobs=pad + list(sequence.logprobs),
                    advantages=pad + [advantage / sampled_total] * len(sequence.tokens),
                )
            )
    return data, kept


async def _run_rl(data_dir, init_checkpoint, output_dir, budget, **config):
    import tinker

    from .rubric_rm import JudgeQuotaExhausted  # noqa: ABS101

    # One process scores whole batches; Slime's per-worker limit of 16 would
    # serialize them.
    os.environ.setdefault("RUBRIC_SCORE_CONCURRENCY", "128")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps({**config, "init_checkpoint": init_checkpoint}, indent=2) + "\n"
    )
    train = _read_jsonl(Path(data_dir) / "train.jsonl")
    held_out = _read_jsonl(Path(data_dir) / "val.jsonl")
    tokenizer = load_tokenizer()
    service = service_client(budget, config["run_tag"])
    trainer = await service.create_training_client_from_state_async(
        init_checkpoint["state_path"]
    )
    rng = random.Random(config["seed"])
    order, cursor = [], 0
    checkpoints = output / "checkpoints.jsonl"
    sampling = dict(
        max_tokens=config["max_response_len"],
        temperature=config["temperature"],
        service=service,
    )

    def next_batch():
        nonlocal order, cursor
        batch = []
        while len(batch) < config["rollout_batch_size"]:
            if cursor == len(order):
                order, cursor = list(range(len(train))), 0
                rng.shuffle(order)
            batch.append(train[order[cursor]])
            cursor += 1
        return batch

    rollout_log = (output / "rollouts.jsonl").open("a")

    async def rollout(sampler, row, n, step=None):
        # Score each group as soon as it arrives, overlapping the sampling tail.
        prompt, sequences = await sample_group(
            sampler, tokenizer, row["prompt"], budget, num_samples=n, **sampling
        )
        texts = [(row, response_text(tokenizer, s.tokens)) for s in sequences]
        rewards, blocked = await _score(texts)
        if step is not None:
            _log_rollouts(rollout_log, step, row, sequences, texts, rewards, blocked)
        return prompt, sequences, rewards, blocked

    async def collect(rows, n, step=None):
        sampler = await trainer.save_weights_and_get_sampling_client_async()
        started = time.time()
        groups = await asyncio.gather(*[rollout(sampler, row, n, step) for row in rows])
        return groups, time.time() - started

    async def heldout(step):
        groups, seconds = await collect(held_out, config["n_samples_per_eval_prompt"])
        # A blocked criterion is ungraded; its sample stays in the mean.
        record = {
            "step": step,
            "heldout_reward_mean": statistics.fmean(
                r for _, _, rewards, _ in groups for r in rewards
            ),
            "blocked_samples": sum(b for *_, blocked in groups for b in blocked),
            "sample_s": round(seconds, 1),
        }
        with (output / "heldout_log.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return record

    def save(name, step):
        record = {"step": step, **save_checkpoint(trainer, name)}
        with checkpoints.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return record

    log = (output / "train_log.jsonl").open("a")
    started = time.time()
    stop_reason, completed = "completed", 0
    n, total = config["n_samples_per_prompt"], config["num_rollout"]
    # The step argument labels each batch's rollouts with the step it trains.
    pending = asyncio.create_task(collect(next_batch(), n, 1))
    background = []
    try:
        for step in range(1, total + 1):
            groups, sample_s = await pending
            pending = None
            if step < total and config["max_steps_off_policy"] >= 1:
                # Sample the next batch from these pre-update weights while this
                # step trains.
                pending = asyncio.create_task(collect(next_batch(), n, step + 1))
            data, kept = _group_datums(groups)
            train_started = time.time()
            if data:
                train_tokens = sum(d.model_input.length for d in data)
                # Counts the next batch's in-flight samples too.
                async with budget.reserve(
                    cost_usd(BASE_MODEL, {"train": train_tokens})
                ):
                    backward = await trainer.forward_backward_async(
                        data, "importance_sampling"
                    )
                    update = await trainer.optim_step_async(
                        tinker.AdamParams(
                            learning_rate=config["lr"] * LORA_LR_MULTIPLIER,
                            grad_clip_norm=1.0,
                        )
                    )
                    await backward.result_async()
                    await update.result_async()
                    budget.record({"train": train_tokens}, session_id(service))
            completed = step
            if step < total and pending is None:
                pending = asyncio.create_task(collect(next_batch(), n, step + 1))
            if step % config["eval_interval"] == 0 or step == total:
                background.append(asyncio.create_task(heldout(step)))
            sequences = [s for _, seqs, _, _ in groups for s in seqs]
            rewards = [r for _, _, group, _ in groups for r in group]
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "reward_mean": statistics.fmean(rewards),
                        "groups_with_signal": kept,
                        "blocked_samples": sum(
                            b for *_, blocked in groups for b in blocked
                        ),
                        "response_len_mean": statistics.fmean(
                            len(s.tokens) for s in sequences
                        ),
                        "truncated_frac": statistics.fmean(
                            s.stop_reason == "length" for s in sequences
                        ),
                        "sample_s": round(sample_s, 1),
                        "train_s": round(time.time() - train_started, 1),
                        "tinker_rl_cost_usd": budget.snapshot()["cost_usd"],
                        "elapsed_s": round(time.time() - started, 1),
                    }
                )
                + "\n"
            )
            log.flush()
            if step % config["save_interval"] == 0 and step < total:
                save(f"rl-step-{step:04d}", step)
        heldout_results = await asyncio.gather(*background)
    except TinkerBudgetExceeded as exc:
        stop_reason, heldout_results = f"tinker_budget_exhausted: {exc}", []
    except JudgeQuotaExhausted:
        stop_reason, heldout_results = "judge_quota_exhausted", []
    finally:
        log.close()
        rollout_log.close()
        # Stop sampling that would only spend budget, then let the tasks unwind.
        unfinished = [
            t for t in (pending, *background) if t is not None and not t.done()
        ]
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)
    final = save("rl-final", completed)
    result = {
        "stop_reason": stop_reason,
        "completed_rollouts": completed,
        "final_checkpoint": final,
        "heldout": heldout_results,
        "base_model": BASE_MODEL,
        "tinker_rl_cost_usd": budget.snapshot()["cost_usd"],
    }
    (output / "rl_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

#!/usr/bin/env python3
"""Supervised fine-tuning of a base LLM on a TRL-format (conversational) dataset.

Launched under `accelerate launch ... fsdp.yaml` (see train_and_eval.sh) so the
model is sharded across all available GPUs with FSDP.

This stays close to TRL's stock SFTTrainer flow: load data, set parameters, train.
Loss masking is left at SFTTrainer's default (no custom prompt masking).

The one piece of setup that is NOT optional is the chat template. Two of the four
target base models (SmolLM3-3B-Base, gemma-3-4b-pt) ship no chat template at all,
so SFTTrainer cannot format conversational data for them; and the official eval
*forces* its own per-family template at inference. We therefore load that same eval
template into the tokenizer, which both makes training possible and keeps the train
and eval formats identical. (Qwen3-Base does ship a template, but it differs from
the eval's, so we override it for consistency.)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# model-family substring -> eval chat template filename. Mirrors the detection in
# PostTrainBench/src/eval/tasks/*/evaluate.py so train and eval stay in lockstep.
TEMPLATE_FOR_FAMILY = {
    "qwen": "qwen3.jinja",
    "gemma": "gemma3.jinja",
    "smollm": "smollm.jinja",
    "llama": "llama3.jinja",
}

# The token each eval template uses to END an assistant turn. We point the saved
# model's generation eos_token_id at this so vLLM stops at the end of the answer.
# Base models default eos to e.g. <|endoftext|> (Qwen: 151643), which the chat
# template never emits -- so without this the model rambles to max_tokens at eval
# (huge slowdown + degraded extraction).
EOS_FOR_FAMILY = {
    "qwen": "<|im_end|>",
    "gemma": "<end_of_turn>",
    "smollm": "<|im_end|>",
    "llama": "<|eot_id|>",
}


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_main_process():
        print(f"[train_sft] {msg}", flush=True)


def infer_family(model_id: str) -> str | None:
    """Map a model id to a known family substring (for eos handling / template lookup),
    or None if it matches no known family. None is fine when --use-default-chat-template
    is set: we then leave the model's own template and eos untouched."""
    name = model_id.lower()
    for family in TEMPLATE_FOR_FAMILY:
        if family in name:
            return family
    return None


def resolve_template(model_id: str, templates_dir: str) -> tuple[Path, str]:
    family = infer_family(model_id)
    if family is None:
        raise ValueError(
            f"Could not infer a chat-template family from model id '{model_id}'. "
            f"Known families: {sorted(TEMPLATE_FOR_FAMILY)}"
        )
    path = Path(templates_dir) / TEMPLATE_FOR_FAMILY[family]
    if not path.is_file():
        raise FileNotFoundError(f"Expected chat template not found: {path}")
    return path, family


def set_generation_eos(model, tokenizer, family: str) -> None:
    """Point the model's in-memory generation eos_token_id at the chat template's
    turn-end token (keeping the original eos too), so vLLM stops at the end of the
    assistant turn instead of running to max_tokens. Setting it on the in-memory
    generation_config (rather than editing generation_config.json after the fact) makes
    EVERY artifact Trainer writes correct natively -- all per-epoch checkpoints and the
    final model, including any checkpoints written after a --resume. Inert for training:
    SFT never calls model.generate, so eos only affects what gets serialized."""
    if family is None:
        log("unknown model family; leaving generation eos as the model's own default")
        return
    turn_end_id = tokenizer.convert_tokens_to_ids(EOS_FOR_FAMILY[family])
    if turn_end_id is None or turn_end_id == tokenizer.unk_token_id:
        log(f"WARNING: turn-end token {EOS_FOR_FAMILY[family]!r} not in vocab; leaving eos as-is")
        return
    gc = model.generation_config
    orig = gc.eos_token_id
    eos_ids = [turn_end_id]
    if isinstance(orig, int) and orig != turn_end_id:
        eos_ids.append(orig)
    elif isinstance(orig, list):
        eos_ids += [t for t in orig if t != turn_end_id]
    gc.eos_token_id = eos_ids
    log(f"generation_config eos_token_id -> {eos_ids} (stop at {EOS_FOR_FAMILY[family]})")


def load_base_model(model_id: str):
    """Load the BASE model as an in-memory object (bf16, on CPU until the trainer
    places it). Used (a) for Gemma-3, which must be loaded as text-only, and (b) to
    rebuild the base for merging a LoRA adapter at save time.

    Gemma-3 4B/12B/27B ship as multimodal checkpoints, so AutoModelForCausalLM loads
    Gemma3ForConditionalGeneration (language model + a SigLIP vision tower). We only do
    text SFT, and the vision classes in the model's _no_split_modules break FSDP's
    transformer wrap. So load the text-only Gemma3ForCausalLM (its missing_keys is empty
    against the multimodal checkpoint; the vision weights are just ignored) and pin
    _no_split_modules to the text decoder layer."""
    if "gemma-3" in model_id.lower():
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(model_id, dtype="bfloat16")
        model._no_split_modules = ["Gemma3DecoderLayer"]
        return model
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True,
                   help="HF model id of the BASE model, e.g. Qwen/Qwen3-1.7B-Base")
    p.add_argument("--data-file", required=True, help="Path to the *_trl_format.jsonl dataset")
    p.add_argument("--output-dir", required=True, help="Where to save the fine-tuned model")
    p.add_argument("--templates-dir", default=None,
                   help="PostTrainBench eval templates dir (used so train format == eval format). "
                        "Required unless --use-default-chat-template is set.")
    p.add_argument("--use-default-chat-template", action="store_true",
                   help="Keep the model's own built-in chat template instead of forcing the "
                        "PostTrainBench eval template. Use for INSTRUCT/chat models (e.g. "
                        "Qwen/Qwen3-4B) that ship a template the eval also uses by default -- then "
                        "train and eval formats already match. Skips --templates-dir and the "
                        "<think> reasoning-loop rewrite, and leaves the tokenizer's chat_template "
                        "and the model's eos untouched. Errors if the model ships no template.")

    # Training hyperparameters.
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=3000)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", type=str, default="cosine")
    p.add_argument("--use-lora", action="store_true",
                   help="Fine-tune with LoRA adapters (peft) instead of full fine-tuning. "
                        "Far less memory/disk, and the learning rate is automatically scaled up "
                        "10x (LoRA's small adapter matrices need a higher LR than full FT). "
                        "The saved <output_dir> is the merged full model, so eval is unchanged.")
    p.add_argument("--lora-r", type=int, default=8,
                   help="LoRA rank (only used with --use-lora).")
    p.add_argument("--lora-alpha", type=int, default=8,
                   help="LoRA alpha scaling (only used with --use-lora).")
    p.add_argument("--packing", action="store_true",
                   help="Pack multiple samples per sequence (faster; off by default for clarity)")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every-epoch", action="store_true",
                   help="Save a full checkpoint at the end of every epoch into "
                        "<output_dir>/checkpoint-* so you can (a) sweep epochs at debug time and "
                        "(b) resume with --resume. The final consolidated model is still written "
                        "to <output_dir> for eval.")
    p.add_argument("--resume", action="store_true",
                   help="If <output_dir> already holds checkpoint-* dirs (e.g. a cancelled job), "
                        "resume from the latest one. HF Trainer continues toward the SAME --epochs "
                        "total (no extra epochs) and skips already-seen batches (no data re-iter). "
                        "No-op when no checkpoint is present. Keep --epochs identical across relaunches.")

    # Smoke-test knobs: keep them out of the way for real runs.
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap optimizer steps (>0). Used for cheap end-to-end smoke tests.")
    p.add_argument("--max-train-samples", type=int, default=-1,
                   help="Truncate the dataset to N rows (>0). Used for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # LoRA trains only small low-rank adapter matrices, which both tolerate and need a
    # markedly higher LR than full fine-tuning, so scale it up 10x whenever --use-lora is
    # set. Done before SFTConfig is built so it flows through to the scheduler/warmup.
    if args.use_lora:
        args.learning_rate *= 10
        log(f"--use-lora set: learning rate scaled 10x -> {args.learning_rate:g}")
        # The LoRA save path (below) merges from the last in-loop adapter checkpoint
        # rather than the standalone Trainer.save_model() -- that call deadlocks the
        # FSDP2 adapter gather. So we require per-epoch checkpoints to exist; enable
        # them automatically if the caller forgot.
        if not args.save_every_epoch:
            args.save_every_epoch = True
            log("--use-lora: auto-enabling --save-every-epoch (the merge reads the last "
                "in-loop adapter checkpoint)")

    if args.use_default_chat_template:
        template_path = None
        family = infer_family(args.model_id)  # for eos only; None => leave eos as-is
    else:
        if not args.templates_dir:
            raise SystemExit(
                "--templates-dir is required unless --use-default-chat-template is set")
        template_path, family = resolve_template(args.model_id, args.templates_dir)
    log(f"model           = {args.model_id}")
    log(f"data            = {args.data_file}")
    log(f"chat template   = {template_path or '(model built-in)'}")
    log(f"output dir      = {args.output_dir}")

    # --- dataset -----------------------------------------------------------
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    if args.max_train_samples and args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    log(f"loaded {len(dataset)} training rows")

    # --- tokenizer + eval-matched chat template ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if args.use_default_chat_template:
        # Instruct/chat model: the eval also uses the model's built-in template, so leave
        # it byte-for-byte untouched (no override, no <think> rewrite) to keep train == eval.
        if tokenizer.chat_template is None:
            raise ValueError(
                f"--use-default-chat-template set but {args.model_id} ships no chat template; "
                f"omit the flag and pass --templates-dir to use an eval template instead.")
        log("using the model's built-in chat template unchanged")
    else:
        chat_template = template_path.read_text()
        # The Qwen3 template unconditionally wraps the FINAL assistant turn in an empty
        # `<think>\n\n</think>\n\n` block (its `loop.last` branch), so every SFT target gets a
        # think prefix the model never sees at inference (add_generation_prompt doesn't add it,
        # and `enable_thinking` only guards that branch -- it can't remove this one). Make the
        # think wrapper fire only when there's actual reasoning content, so train matches infer.
        # No-op for templates without this construct (smollm/gemma/llama).
        chat_template = chat_template.replace(
            "loop.last or (not loop.last and reasoning_content)", "reasoning_content")
        tokenizer.chat_template = chat_template
    # Base models often have no pad token; reuse eos so batching/padding works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model -------------------------------------------------------------
    # Non-Gemma models are passed to SFTTrainer as a string so accelerate/FSDP can
    # handle low-memory sharded loading via model_init_kwargs. Gemma-3 must be
    # pre-loaded text-only (see load_base_model), so for it we pass the object and
    # drop model_init_kwargs.
    model_arg = args.model_id
    model_init_kwargs = {"dtype": "bfloat16"}
    if "gemma-3" in args.model_id.lower():
        log("Gemma-3 detected: loading text-only Gemma3ForCausalLM (no vision tower)")
        model_arg = load_base_model(args.model_id)
        model_init_kwargs = None  # model is now a pre-loaded object, not a string

    # --- LoRA (peft) -------------------------------------------------------
    # When --use-lora is set, hand SFTTrainer a peft LoraConfig; it wraps the base
    # model with adapters and trains only those. We merge them back into a standalone
    # full model at save time (below) so the eval pipeline loads output_dir unchanged.
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # peft's per-architecture default mapping lacks some model types
            # (e.g. Olmo-3); pin the modules it would pick for qwen3/llama so
            # every architecture gets the same adapter placement.
            target_modules=["q_proj", "v_proj"],
        )
        log(f"LoRA enabled (r={args.lora_r}, alpha={args.lora_alpha}): training peft "
            f"adapters (merged into a full model on save)")

    # --- SFT config --------------------------------------------------------
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        packing=args.packing,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        # Per-epoch checkpoints are opt-in (disk-heavy x #epochs). We save FULL
        # checkpoints (optimizer + scheduler + RNG, not just weights) so a cancelled
        # run can --resume exactly where it left off; the consolidated weights inside
        # each checkpoint-* are still directly loadable for epoch sweeps. The final
        # model is also written to output_dir below via save_model, which the eval uses.
        save_strategy=("epoch" if args.save_every_epoch else "no"),
        save_only_model=False,
        report_to="none",
        seed=args.seed,
        dataset_num_proc=4,
        # bf16 weights (string-model path only; None when we pre-loaded the model above).
        model_init_kwargs=model_init_kwargs,
    )

    trainer = SFTTrainer(
        model=model_arg,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  # None => full fine-tuning
    )

    # Set the generation eos BEFORE training so every checkpoint Trainer saves (and the
    # final model) natively stops at the chat template's turn-end token under vLLM.
    # Trainer serializes generation_config.json from this in-memory config on each save.
    set_generation_eos(trainer.model, tokenizer, family)

    # Resume from the latest checkpoint-* in output_dir if asked and one exists.
    # get_last_checkpoint returns None for a missing/empty dir, so this is a safe
    # no-op on a fresh run. Trainer restores global_step/epoch and trains only the
    # epochs remaining toward num_train_epochs (no extra epochs, no re-iterated data).
    resume_from = None
    if args.resume and os.path.isdir(args.output_dir):
        resume_from = get_last_checkpoint(args.output_dir)
    log(f"resuming from {resume_from}" if resume_from else "starting training (fresh)")
    trainer.train(resume_from_checkpoint=resume_from)

    log("saving model")
    if args.use_lora:
        # PEFT + FSDP2: the standalone Trainer.save_model() gather of the (tiny) adapter
        # DEADLOCKS under FULL_STATE_DICT -- ranks disagree on the collective and rank 0
        # hangs until the 30-min NCCL watchdog SIGABRTs the whole job (observed in the
        # wild: a COALESCED gather of NumelOut=2560, i.e. the LoRA params). The per-epoch
        # in-loop checkpoint adapters, by contrast, are written reliably during training.
        # So -- exactly like the full-FT branch distrusts its corrupt standalone save and
        # promotes the last in-loop checkpoint -- we never call save_model here and merge
        # from the last in-loop adapter checkpoint instead.
        #
        # First tear the process group down cleanly on ALL ranks (the barrier + destroy
        # are symmetric collectives, unlike the save_model gather that hung). This lets
        # the rank-0-only CPU merge below -- which issues no collectives and can run for
        # minutes -- proceed with no live NCCL work and no risk of a peer exiting out from
        # under it and tripping the watchdog.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if is_main_process():
            from peft import PeftModel

            last_ckpt = get_last_checkpoint(args.output_dir)
            if not last_ckpt or not os.path.isfile(os.path.join(last_ckpt, "adapter_config.json")):
                raise RuntimeError(
                    "LoRA run produced no in-loop adapter checkpoint to merge from "
                    f"(looked in {args.output_dir}). Use --save-every-epoch (auto-enabled "
                    "for --use-lora) and run at least one full epoch.")

            def merge_adapter(adapter_path: str, out_dir: str) -> None:
                # Rebuild the base on CPU and fold the adapter in, so out_dir is a plain
                # full model the eval loads exactly like a full-FT artifact.
                base = load_base_model(args.model_id)
                merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
                # The freshly reloaded base carries the base generation_config, not the
                # turn-end eos we set on the training model, so reapply it here.
                set_generation_eos(merged, tokenizer, family)
                # Some bases (e.g. Olmo-3-Instruct) ship a generation_config with
                # temperature/top_p set but do_sample unset, which strict save_pretrained
                # validation rejects. Keep the shipped sampling values and just mark
                # do_sample=True so they're valid.
                gc = merged.generation_config
                if gc is not None and not getattr(gc, "do_sample", False) and (
                        getattr(gc, "temperature", None) is not None
                        or getattr(gc, "top_p", None) is not None):
                    gc.do_sample = True
                merged.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
                del base, merged

            # The eval loads output_dir directly: merge the FINAL adapter into a full
            # model there.
            log(f"merging final adapter ({os.path.basename(last_ckpt)}) into base -> {args.output_dir}")
            merge_adapter(last_ckpt, args.output_dir)

            # Also merge every per-epoch adapter into checkpoint-N/merged/ so an epoch
            # sweep can load them directly, leaving the raw adapter + optimizer state in
            # checkpoint-N itself untouched for --resume.
            for ckpt in sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                               key=lambda p: int(p.rsplit("-", 1)[-1])):
                if not os.path.isfile(os.path.join(ckpt, "adapter_config.json")):
                    continue
                merged_dir = os.path.join(ckpt, "merged")
                merge_adapter(ckpt, merged_dir)
                log(f"merged adapter checkpoint -> {merged_dir}")
    else:
        # FSDP FULL_STATE_DICT (set in fsdp.yaml) makes this a consolidated HF model
        # that vLLM can load directly for eval. Trainer guards rank-0-only writing.
        trainer.save_model(args.output_dir)
        if is_main_process():
            # The standalone final save_model() under FSDP FULL_STATE_DICT was observed
            # to corrupt a subset of consolidated tensors (e.g. ~68/327, layers ~4-8),
            # producing a model that generates first-token garbage even though the
            # in-loop checkpoint-N saved during training is byte-correct. So when a
            # per-epoch checkpoint exists (final epoch == final step), promote ITS
            # consolidated weights over the root. config/tokenizer/generation_config
            # written by save_model are fine (eos was set pre-train); only weights move.
            last_ckpt = get_last_checkpoint(args.output_dir)
            if last_ckpt:
                weight_files = sorted(glob.glob(os.path.join(last_ckpt, "model*.safetensors")))
                index = os.path.join(last_ckpt, "model.safetensors.index.json")
                if weight_files:
                    # Clear root's (corrupt) weight files first so a differing shard
                    # layout from save_model can't leave stale tensors alongside the index.
                    for stale in glob.glob(os.path.join(args.output_dir, "model*.safetensors")) + \
                            glob.glob(os.path.join(args.output_dir, "model.safetensors.index.json")):
                        os.remove(stale)
                    for f in weight_files + ([index] if os.path.exists(index) else []):
                        shutil.copy2(f, args.output_dir)
                    log(f"promoted intact weights from {os.path.basename(last_ckpt)} over root "
                        f"(FSDP save_model corruption workaround)")
            else:
                log("WARNING: no checkpoint-N to promote; root relies on save_model() "
                    "(pass --save-every-epoch so an intact in-loop checkpoint exists)")
            tokenizer.save_pretrained(args.output_dir)
    log("done")


if __name__ == "__main__":
    main()

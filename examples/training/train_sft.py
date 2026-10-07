#!/usr/bin/env python3
"""Fine-tune conversational TRL data with Accelerate/FSDP.

Use the evaluation chat template for consistent train/eval formatting, including
base models without a template. Loss masking defaults to SFTTrainer’s behavior."""
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

# Use the chat template’s turn-end token as EOS so evaluation stops after answers.
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
    """Save the template’s turn-end token alongside the original EOS for evaluation.

    Updating generation_config covers final and resumed checkpoints; SFT does not
    call generate, so this only affects saved artifacts."""
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
    """Load a bf16 base on CPU for training or LoRA merging.

    Gemma-3 loads text-only to avoid vision-layer FSDP wrapping; vision weights
    are ignored and wrapping targets the text decoder."""
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
    p.add_argument("--lora-target-modules", type=str, default="q_proj,v_proj",
                   help="Comma-separated module names to attach LoRA adapters to (only used "
                        "with --use-lora). Default matches peft's attention-only mapping for "
                        "qwen3/llama, pinned here since peft's per-architecture default lacks "
                        "some model types (e.g. Olmo-3). Per "
                        "https://thinkingmachines.ai/blog/lora/, attention-only LoRA "
                        "underperforms LoRA that also covers the MLP projections regardless of "
                        "rank -- pass e.g. "
                        "'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' for full "
                        "coverage.")
    p.add_argument("--assistant-only-loss", action="store_true",
                   help="Compute the loss on assistant tokens only, instead of SFTTrainer's "
                        "default of the whole rendered sequence (system and user turns "
                        "included). Opt-in, so existing callers keep their current behaviour. "
                        "TRL derives the mask from a {% generation %}-annotated copy of the "
                        "chat template; that copy stays internal to dataset prep and is not "
                        "the template written into the checkpoint, so eval/inference is "
                        "unaffected (verified: saved chat_template.jinja is byte-identical to "
                        "an all-token run's). Matters when targets are short relative to the "
                        "prompt -- on the impossible-LCB no-thinking data the prompt carries "
                        "75.7% of the tokens, so the default spends three quarters of the "
                        "gradient modelling problem statements.")
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
    p.add_argument("--merge-all-checkpoints", action="store_true",
                   help="With --use-lora, also merge every per-epoch adapter checkpoint into "
                        "checkpoint-N/merged/ (for sweeping epochs). Each merge reloads and merges "
                        "the full base model on CPU, so this is several extra full-model passes; "
                        "off by default since eval only needs the final merged <output_dir>.")

    # Smoke-test knobs: keep them out of the way for real runs.
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap optimizer steps (>0). Used for cheap end-to-end smoke tests.")
    p.add_argument("--max-train-samples", type=int, default=-1,
                   help="Truncate the dataset to N rows (>0). Used for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.resume and os.path.isdir(args.output_dir) and get_last_checkpoint(args.output_dir):
        raise ValueError("Output contains old checkpoints; use a new output directory or --resume")
    args.save_every_epoch = True

    # LoRA trains only small low-rank adapter matrices, which both tolerate and need a
    # markedly higher LR than full fine-tuning, so scale it up 10x whenever --use-lora is
    # set. Done before SFTConfig is built so it flows through to the scheduler/warmup.
    if args.use_lora:
        args.learning_rate *= 10
        log(f"--use-lora set: learning rate scaled 10x -> {args.learning_rate:g}")
        # FSDP2 standalone adapter saves can deadlock; require in-loop epoch checkpoints.
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
    log(f"loss over {'assistant tokens only' if args.assistant_only_loss else 'the full sequence (prompt included)'}")

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
        # Only wrap actual reasoning in <think> so training matches inference.
        # Other templates are unaffected.
        if family == "qwen" and "loop.last or (not loop.last and reasoning_content)" not in chat_template:
            log("WARNING: Qwen template rewrite pattern absent; inspect assistant reasoning formatting")
        chat_template = chat_template.replace(
            "loop.last or (not loop.last and reasoning_content)", "reasoning_content")
        tokenizer.chat_template = chat_template
    # Base models often have no pad token; reuse eos so batching/padding works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Let FSDP load non-Gemma models by ID; Gemma needs a preloaded text-only model.
    model_arg = args.model_id
    model_init_kwargs = {"dtype": "bfloat16"}
    if "gemma-3" in args.model_id.lower():
        log("Gemma-3 detected: loading text-only Gemma3ForCausalLM (no vision tower)")
        model_arg = load_base_model(args.model_id)
        model_init_kwargs = None  # model is now a pre-loaded object, not a string

    # Train LoRA adapters, then merge them into a standalone evaluation model.
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
        )
        log(f"LoRA enabled (r={args.lora_r}, alpha={args.lora_alpha}, "
            f"target_modules={target_modules}): training peft adapters (merged into a full "
            f"model on save)")

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
        assistant_only_loss=args.assistant_only_loss,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        # Save optimizer/scheduler/RNG state for exact resume and weights for epoch sweeps.
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

    # Resume the latest checkpoint if present, training only the remaining epochs.
    resume_from = None
    if args.resume and os.path.isdir(args.output_dir):
        resume_from = get_last_checkpoint(args.output_dir)
    log(f"resuming from {resume_from}" if resume_from else "starting training (fresh)")
    trainer.train(resume_from_checkpoint=resume_from)

    final_checkpoint = os.path.join(args.output_dir, f"checkpoint-{trainer.state.global_step}")
    if not os.path.isdir(final_checkpoint):
        raise RuntimeError("Training did not save its final global-step checkpoint")
    log("saving model")
    if args.use_lora:
        # Standalone FSDP2 adapter gathers can deadlock; merge an in-loop checkpoint.
        # Tear down distributed state on all ranks before rank 0’s CPU-only merge.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if is_main_process():
            from peft import PeftModel

            last_ckpt = final_checkpoint
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
                # Validate shipped sampling parameters by setting do_sample=True.
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

            # Optionally merge epoch adapters for evaluation; retain raw checkpoints for resume.
            if args.merge_all_checkpoints:
                for ckpt in sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                                   key=lambda p: int(p.rsplit("-", 1)[-1])):
                    if not os.path.isfile(os.path.join(ckpt, "adapter_config.json")):
                        continue
                    merged_dir = os.path.join(ckpt, "merged")
                    merge_adapter(ckpt, merged_dir)
                    log(f"merged adapter checkpoint -> {merged_dir}")
    else:
        # Standalone FSDP full-model saves can hang or corrupt tensors.
        # Tear down distributed state on all ranks, then promote the last intact
        # in-loop checkpoint on rank 0.
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if is_main_process():
            last_ckpt = final_checkpoint
            if not last_ckpt:
                raise RuntimeError(
                    "Full-finetune run produced no in-loop checkpoint to promote from "
                    f"(looked in {args.output_dir}). Use --save-every-epoch (already "
                    "the default here) and run at least one full epoch.")
            # Skip resume-only bookkeeping (optimizer/scheduler/rng/trainer state) --
            # everything else (weights, possibly sharded, plus config/generation_config/
            # tokenizer/chat_template) is exactly what an inference-loadable model needs.
            skip = {"optimizer.bin", "pytorch_model_fsdp.bin", "scheduler.pt",
                    "trainer_state.json", "training_args.bin"}
            for f in glob.glob(os.path.join(last_ckpt, "*")):
                name = os.path.basename(f)
                if name in skip or name.startswith("rng_state"):
                    continue
                shutil.copy2(f, os.path.join(args.output_dir, name))
            log(f"promoted {os.path.basename(last_ckpt)} (in-loop checkpoint, no "
                f"save_model() gather) -> {args.output_dir}")
    log("done")


if __name__ == "__main__":
    main()

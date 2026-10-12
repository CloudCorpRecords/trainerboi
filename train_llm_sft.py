"""
train_llm_sft.py - Supervised fine-tuning (SFT) for LLMs on real datasets.

Backends:
  unsloth : (default) ~2x faster, much lower VRAM. Best on 16 GB cards.
  hf      : plain Transformers + PEFT + TRL. Most compatible with exotic models.

Modes:
  qlora   : 4-bit NF4 base + LoRA adapters  (7-8B models fit on 16 GB)
  lora    : bf16 base + LoRA adapters        (~3B and below)
  full    : full-parameter fine-tune         (~0.5B and below)

Examples:
  python train_llm_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset yahma/alpaca-cleaned --max_steps 200
  python train_llm_sft.py --model unsloth/Llama-3.2-3B-Instruct --dataset my_chats.jsonl --epochs 2 --packing
  python train_llm_sft.py --model HuggingFaceTB/SmolLM2-360M --dataset roneneldan/TinyStories --text_field text --mode full
  python train_llm_sft.py --resume --output_dir runs/my-run        # continue from the last checkpoint
"""

import os
import time
import argparse

import trainer_utils
import llm_data


def build_model(args):
    """Returns (model, tokenizer) with adapters attached when mode != full."""
    import torch
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            load_in_4bit=args.mode == "qlora",
            full_finetuning=args.mode == "full",
            dtype=None,
        )
        if args.mode != "full":
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                target_modules=lora_targets,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
        return model, tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = None
    if args.mode == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=args.attn,
    )
    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.mode != "full":
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
        ))
    return model, tokenizer


def train_sft(args):
    # Unsloth must be imported before transformers/trl so it can patch them.
    if args.backend == "unsloth":
        import unsloth  # noqa: F401
    from live_monitor import LiveMonitor

    live = LiveMonitor(args.output_dir, task="sft", config=vars(args), sample_prompt=args.sample_prompt,
                       sample_every=args.sample_every)
    with live.guard():
        return _train_sft(args, live)


def _train_sft(args, live):
    import torch
    from trl import SFTTrainer, SFTConfig
    from transformers.trainer_utils import get_last_checkpoint

    trainer_utils.enable_maximum_gpu_performance()
    trainer_utils.clear_gpu_memory()

    print("=" * 65)
    print(f"LLM SFT  |  model={args.model}  backend={args.backend}  mode={args.mode}")
    print("=" * 65)

    live.phase("Loading model")
    model, tokenizer = build_model(args)
    tokenizer = llm_data.ensure_chat_template(tokenizer)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    live.tokenizer = tokenizer
    if args.backend == "unsloth":
        from unsloth import FastLanguageModel
        live.before_generate = FastLanguageModel.for_inference
        live.after_generate = FastLanguageModel.for_training

    live.phase("Loading dataset", trainable_params=trainable, total_params=total)
    raw = llm_data.load_raw(args.dataset, split=args.split, subset=args.subset)
    ds = llm_data.normalize_sft(raw, text_field=args.text_field)
    if args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
    if args.backend == "unsloth":
        # Unsloth's trainer wants pre-rendered text, so apply the chat template up front.
        ds = llm_data.render_text(ds, tokenizer)
    train_ds, eval_ds = llm_data.split_train_eval(ds, args.eval_ratio, args.seed)
    print(f"--> Dataset: {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval  columns={train_ds.column_names}")
    live.phase("Tokenizing", train_rows=len(train_ds), eval_rows=len(eval_ds) if eval_ds else 0)

    report_to = [r for r in args.report_to.split(",") if r and r != "none"]

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # < 1 means a fraction of total steps
        weight_decay=0.01,
        optim="paged_adamw_8bit" if args.mode != "full" else "adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.backend == "hf",
        max_length=args.max_seq_len,
        packing=args.packing,
        use_liger_kernel=args.liger and args.backend == "hf",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=report_to or "none",
        seed=args.seed,
        dataset_num_proc=1,  # multiprocessing on Windows is slow/fragile
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[live.callback()],
    )

    resume = get_last_checkpoint(args.output_dir) if args.resume and os.path.isdir(args.output_dir) else None
    if resume:
        print(f"--> Resuming from {resume}")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    print("-" * 65)
    print(f"--> Done in {time.time() - t0:.1f}s | train loss {result.training_loss:.4f} "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    if eval_ds is not None:
        live.phase("Final evaluation")
        metrics = trainer.evaluate()
        print(f"--> Eval loss {metrics['eval_loss']:.4f}")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"--> Saved {'model' if args.mode == 'full' else 'LoRA adapter'} to {final_dir}")

    if args.merge and args.mode != "full":
        merged_dir = os.path.join(args.output_dir, "merged")
        if args.backend == "unsloth":
            live.phase("Merging adapter")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        else:
            print("--> For hf backend, merge with: python merge_lora.py --adapter", final_dir)
            merged_dir = None
        if merged_dir:
            print(f"--> Merged 16-bit model saved to {merged_dir}")
    live.finish(final_dir, train_loss=result.training_loss,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
    return final_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM supervised fine-tuning (SFT)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="demo", help="HF Hub id, local .jsonl/.json/.csv/.parquet/.txt, or 'demo'")
    p.add_argument("--subset", default=None, help="HF dataset config name")
    p.add_argument("--split", default="train")
    p.add_argument("--text_field", default=None, help="Train on raw text in this column (continued pretraining)")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.02)
    p.add_argument("--backend", choices=["unsloth", "hf"], default="unsloth")
    p.add_argument("--mode", choices=["qlora", "lora", "full"], default="qlora")
    p.add_argument("--output_dir", default="runs/sft")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="Overrides epochs when > 0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4, help="Use ~1e-5 for --mode full")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--packing", action="store_true", help="Pack short samples into full sequences (faster)")
    p.add_argument("--liger", action="store_true", help="Liger fused kernels (hf backend)")
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa", help="Attention kernels (hf backend)")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--sample_prompt", default="Explain what a neural network is in two sentences.",
                   help="Generated periodically so you can watch the model change (dashboard)")
    p.add_argument("--sample_every", type=int, default=50, help="0 disables sample generations")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge", action="store_true", help="Also save a merged 16-bit model (unsloth backend)")
    p.add_argument("--report_to", default="tensorboard", help="tensorboard, wandb, none (comma-separated)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train_sft(parse_args())

"""
train_llm_dpo.py - Preference tuning (DPO) on top of a base or SFT'd model.

Data needs 'chosen' and 'rejected' (and usually 'prompt') columns, e.g. trl-lib/ultrafeedback_binarized.

Examples:
  python train_llm_dpo.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/ultrafeedback_binarized --max_steps 200
  python train_llm_dpo.py --model runs/sft/merged --dataset prefs.jsonl
"""

import os
import time
import argparse

import trainer_utils
import llm_data


def train_dpo(args):
    from live_monitor import LiveMonitor
    live = LiveMonitor(args.output_dir, task="dpo", config=vars(args), sample_prompt=args.sample_prompt,
                       sample_every=args.sample_every)
    with live.guard():
        return _train_dpo(args, live)


def _train_dpo(args, live):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

    trainer_utils.enable_maximum_gpu_performance()
    trainer_utils.clear_gpu_memory()
    print("=" * 65)
    print(f"LLM DPO  |  model={args.model}  4bit={args.load_4bit}")
    print("=" * 65)

    live.phase("Loading model")
    tokenizer = llm_data.ensure_chat_template(AutoTokenizer.from_pretrained(args.model))
    live.tokenizer = tokenizer
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.load_4bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    live.phase("Loading dataset")
    ds = llm_data.normalize_dpo(llm_data.load_raw(args.dataset, split=args.split))
    if args.max_samples:
        ds = ds.shuffle(seed=42).select(range(min(args.max_samples, len(ds))))
    train_ds, eval_ds = llm_data.split_train_eval(ds, args.eval_ratio)
    print(f"--> Dataset: {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval")
    live.phase("Tokenizing", train_rows=len(train_ds), eval_rows=len(eval_ds) if eval_ds else 0)

    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_len,
        logging_steps=5,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=args.report_to,
        dataset_num_proc=1,
    )
    # With a LoRA adapter, the frozen base doubles as the reference model -> no second copy in VRAM.
    trainer = DPOTrainer(
        model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                               task_type="CAUSAL_LM", target_modules="all-linear"),
        callbacks=[live.callback()],
    )
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint
    resume = get_last_checkpoint(args.output_dir) if args.resume and os.path.isdir(args.output_dir) else None
    if resume:
        print(f"--> Resuming from {resume}")
    result = trainer.train(resume_from_checkpoint=resume)
    print(f"--> Done in {time.time() - t0:.1f}s | loss {result.training_loss:.4f} "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"--> Saved DPO adapter to {final_dir}")
    live.finish(final_dir, train_loss=result.training_loss, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
    return final_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM preference tuning (DPO)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--split", default="train")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.01)
    p.add_argument("--output_dir", default="runs/dpo")
    p.add_argument("--load_4bit", action="store_true", help="QLoRA-style 4-bit base (for 3B+ models)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint in --output_dir")
    p.add_argument("--sample_prompt", default="Explain what a neural network is in two sentences.")
    p.add_argument("--sample_every", type=int, default=50)
    train_dpo(p.parse_args())

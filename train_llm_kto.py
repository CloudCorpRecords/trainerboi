"""
train_llm_kto.py - KTO: learn from simple thumbs-up / thumbs-down feedback (no paired answers needed).

Data: rows with prompt, completion, label (true = good, false = bad). Preference pairs (prompt/chosen/rejected)
are converted automatically (each pair becomes one good and one bad example).

Examples:
  python train_llm_kto.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/ultrafeedback_binarized --max_steps 100
  python train_llm_kto.py --dataset feedback.jsonl        # columns: prompt, completion, label
"""

import os
import time
import argparse

import trainer_utils
import llm_data


def to_kto(ds):
    cols = set(ds.column_names)
    if {"prompt", "completion", "label"} <= cols:
        return ds.select_columns(["prompt", "completion", "label"])
    if not {"chosen", "rejected"} <= cols:
        raise ValueError(f"KTO needs prompt/completion/label columns or chosen/rejected pairs; got {sorted(cols)}")
    from datasets import Dataset

    def split(x):
        if isinstance(x, list):                       # conversational: [..., {"role": "assistant", ...}]
            return x[:-1], x[-1:]
        return None, [{"role": "assistant", "content": str(x)}]

    rows = {"prompt": [], "completion": [], "label": []}
    for r in ds:
        for key, label in (("chosen", True), ("rejected", False)):
            p, c = split(r[key])
            if p is None:
                p = r.get("prompt", "")
                p = p if isinstance(p, list) else [{"role": "user", "content": str(p)}]
            rows["prompt"].append(p)
            rows["completion"].append(c)
            rows["label"].append(label)
    return Dataset.from_dict(rows)


def train_kto(args):
    from live_monitor import LiveMonitor
    live = LiveMonitor(args.output_dir, task="kto", config=vars(args), sample_prompt=args.sample_prompt,
                       sample_every=args.sample_every)
    with live.guard():
        return _train_kto(args, live)


def _train_kto(args, live):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import KTOTrainer, KTOConfig

    trainer_utils.enable_maximum_gpu_performance()
    trainer_utils.clear_gpu_memory()
    print("=" * 65)
    print(f"LLM KTO  |  model={args.model}  4bit={args.load_4bit}")
    print("=" * 65)

    live.phase("Loading model")
    tokenizer = llm_data.ensure_chat_template(AutoTokenizer.from_pretrained(args.model))
    live.tokenizer = tokenizer
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.load_4bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    if args.load_4bit:
        model.enable_input_require_grads()

    live.phase("Loading dataset")
    raw = llm_data.load_raw(args.dataset, split=args.split)
    if args.max_samples:
        raw = raw.shuffle(seed=42).select(range(min(args.max_samples, len(raw))))
    ds = to_kto(raw)
    train_ds, eval_ds = llm_data.split_train_eval(ds, args.eval_ratio)
    print(f"--> Dataset: {len(train_ds)} train / {len(eval_ds) if eval_ds else 0} eval")
    live.phase("Tokenizing", train_rows=len(train_ds), eval_rows=len(eval_ds) if eval_ds else 0)

    config = KTOConfig(
        output_dir=args.output_dir, beta=args.beta,
        num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
        max_length=args.max_len, logging_steps=5,
        eval_strategy="steps" if eval_ds is not None else "no", eval_steps=args.save_steps,
        save_steps=args.save_steps, save_total_limit=2, report_to=args.report_to, dataset_num_proc=1,
    )
    trainer = KTOTrainer(
        model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                               task_type="CAUSAL_LM", target_modules="all-linear"),
        callbacks=[live.callback()],
    )
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint
    resume = get_last_checkpoint(args.output_dir) if args.resume and os.path.isdir(args.output_dir) else None
    result = trainer.train(resume_from_checkpoint=resume)
    print(f"--> Done in {time.time() - t0:.1f}s | loss {result.training_loss:.4f} "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"--> Saved KTO adapter to {final_dir}")
    live.finish(final_dir, train_loss=result.training_loss, peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
    return final_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM KTO (thumbs-up / thumbs-down feedback)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized")
    p.add_argument("--split", default="train")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.01)
    p.add_argument("--output_dir", default="runs/kto")
    p.add_argument("--load_4bit", action="store_true")
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_prompt", default="Explain what a neural network is in two sentences.")
    p.add_argument("--sample_every", type=int, default=50)
    train_kto(p.parse_args())

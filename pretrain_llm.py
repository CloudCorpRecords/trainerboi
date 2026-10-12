"""
pretrain_llm.py - Train a small Llama-architecture language model FROM SCRATCH.

Good for learning how LLMs train end-to-end. On a 16 GB card, ~20M-150M params is the sweet spot.

Size presets (params approx, with a 32k-49k vocab):
  tiny  : ~20M   (d=256,  8 layers)   - minutes, TinyStories-level fluency
  small : ~60M   (d=512,  8 layers)
  base  : ~150M  (d=768, 12 layers)   - hours

Examples:
  python pretrain_llm.py --size tiny --dataset roneneldan/TinyStories --max_steps 3000
  python pretrain_llm.py --size small --dataset my_corpus.txt --tokenizer HuggingFaceTB/SmolLM2-135M
"""

import os
import math
import time
import argparse

import trainer_utils
import llm_data

PRESETS = {
    "tiny": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4),
    "small": dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4),
    "base": dict(hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=4),
}


def pretrain(args):
    from live_monitor import LiveMonitor
    live = LiveMonitor(args.output_dir, task="pretrain", config=vars(args), sample_prompt=args.sample_prompt,
                       chat=False, sample_every=args.sample_every)
    with live.guard():
        return _pretrain(args, live)


def _pretrain(args, live):
    import torch
    from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments,
                              DataCollatorForLanguageModeling)

    trainer_utils.enable_maximum_gpu_performance()
    trainer_utils.clear_gpu_memory()

    live.phase("Building model")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    live.tokenizer = tokenizer

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        **PRESETS[args.size],
    )
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16 if args.pure_bf16 else torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 65)
    print(f"PRETRAIN FROM SCRATCH | {args.size} | {n_params / 1e6:.1f}M params | seq_len {args.seq_len}")
    print("=" * 65)

    live.phase("Loading dataset", total_params=n_params, trainable_params=n_params)
    raw = llm_data.load_raw(args.dataset, split=args.split)
    text_col = args.text_field or ("text" if "text" in raw.column_names else raw.column_names[0])
    if args.max_samples:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    # Tokenize, then concatenate everything and cut into fixed-length blocks (standard LM pretraining).
    def tok(batch):
        return tokenizer([t + tokenizer.eos_token for t in batch[text_col]], add_special_tokens=False)

    def group(batch):
        ids = sum(batch["input_ids"], [])
        n = (len(ids) // args.seq_len) * args.seq_len
        return {"input_ids": [ids[i:i + args.seq_len] for i in range(0, n, args.seq_len)]}

    live.phase("Tokenizing")
    ds = raw.map(tok, batched=True, remove_columns=raw.column_names, desc="Tokenizing")
    ds = ds.map(group, batched=True, batch_size=1000, remove_columns=ds.column_names, desc="Packing")
    splits = ds.train_test_split(test_size=0.005, seed=42)
    print(f"--> {len(splits['train'])} training blocks = {len(splits['train']) * args.seq_len / 1e6:.1f}M tokens")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(500, args.max_steps // 20),
        weight_decay=0.1,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        bf16=not args.pure_bf16,
        tf32=True,
        torch_compile=args.compile,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=args.report_to,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=splits["train"], eval_dataset=splits["test"],
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                      callbacks=[live.callback()])
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint
    resume = get_last_checkpoint(args.output_dir) if args.resume and os.path.isdir(args.output_dir) else None
    if resume:
        print(f"--> Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    ev = trainer.evaluate()
    print(f"--> Done in {(time.time() - t0) / 60:.1f} min | eval loss {ev['eval_loss']:.3f} "
          f"| perplexity {math.exp(ev['eval_loss']):.1f}")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Quick sample so you can see what it learned.
    model.eval()
    prompt = tokenizer(args.sample_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**prompt, max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.95)
    print("--> Sample:", tokenizer.decode(out[0], skip_special_tokens=True))
    print(f"--> Saved to {final_dir}")
    live.finish(final_dir, eval_loss=ev["eval_loss"], perplexity=math.exp(ev["eval_loss"]),
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
    return final_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pretrain a small LLM from scratch")
    p.add_argument("--size", choices=list(PRESETS), default="tiny")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--split", default="train")
    p.add_argument("--text_field", default=None)
    p.add_argument("--max_samples", type=int, default=200_000)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M", help="Reuse an existing tokenizer")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--compile", action="store_true", help="torch.compile (faster after warmup)")
    p.add_argument("--pure_bf16", action="store_true", help="Store weights in bf16 (less VRAM, less stable)")
    p.add_argument("--output_dir", default="runs/pretrain")
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--resume", action="store_true", help="Continue from the last checkpoint in --output_dir")
    p.add_argument("--sample_prompt", default="Once upon a time")
    p.add_argument("--sample_every", type=int, default=250)
    pretrain(p.parse_args())

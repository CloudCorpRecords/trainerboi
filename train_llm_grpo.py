"""
train_llm_grpo.py - GRPO: reinforcement learning from a reward function (the method behind "reasoning" models).

The model writes several answers per prompt, a reward function scores them, and the model learns to produce the
higher-scoring ones. No answer key per example is required for style rewards; math/exact-answer rewards need an
'answer' column.

Reward presets (--reward):
  math      correct final answer (needs 'answer' column; works with openai/gsm8k) + a bonus for showing reasoning
  concise   short, to-the-point answers
  format    answers that use <think>...</think> then a final answer
  keywords  answers that mention words from --keywords (comma separated)

Examples:
  python train_llm_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --reward math --dataset openai/gsm8k --subset main --max_steps 100
  python train_llm_grpo.py --reward concise --dataset prompts.jsonl
"""

import os
import re
import time
import argparse

import device_utils
import trainer_utils
import llm_data


# ----------------------------------------------------------------------------- rewards
def _text(c):
    return c[0]["content"] if isinstance(c, list) else str(c)


def reward_math(prompts, completions, answer=None, **kw):
    """1.0 for a verified-correct final answer (handles \boxed{}, ####, fractions, decimals, %, commas) plus a small bonus for
    putting the answer in \boxed{} and a penalty for rambling that never finishes."""
    import math_gym
    out = []
    for c, a in zip(completions, answer or [None] * len(completions)):
        t = _text(c)
        gold = str(a).split("####")[-1].strip() if a is not None else None
        r = 1.0 if (gold is not None and math_gym.is_correct(t, gold)) else 0.0
        if math_gym.has_boxed(t):
            r += 0.1
        elif r == 0.0 and len(t) > 900:
            r -= 0.1
        out.append(r)
    return out


def reward_reasoning_shown(prompts, completions, **kw):
    return [0.2 if len(_text(c).split()) > 25 else 0.0 for c in completions]


def reward_concise(prompts, completions, **kw):
    return [max(0.0, 1.0 - abs(len(_text(c).split()) - 40) / 120.0) for c in completions]


def reward_format(prompts, completions, **kw):
    return [1.0 if re.search(r"<think>.+?</think>\s*\S+", _text(c), re.S) else 0.0 for c in completions]


def make_keyword_reward(words):
    words = [w.strip().lower() for w in words.split(",") if w.strip()]

    def reward(prompts, completions, **kw):
        return [sum(w in _text(c).lower() for w in words) / max(len(words), 1) for c in completions]
    return reward


def custom_reward(path):
    """Your own checker: a .py file defining reward(prompts, completions, **columns) -> list[float]  (see docs in README)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reward


def get_rewards(args):
    if args.reward == "custom":
        return [custom_reward(args.reward_file)]
    if args.reward == "math":
        return [reward_math, reward_reasoning_shown]
    if args.reward == "concise":
        return [reward_concise]
    if args.reward == "format":
        return [reward_format]
    if args.reward == "keywords":
        return [make_keyword_reward(args.keywords)]
    raise ValueError(f"Unknown reward '{args.reward}'")


def to_prompts(ds, system=None):
    cols = set(ds.column_names)
    col = next((c for c in ("prompt", "question", "instruction", "input") if c in cols), None)
    if col is None and "messages" in cols:
        return ds.map(lambda r: {"prompt": [m for m in r["messages"] if m["role"] != "assistant"][:2]},
                      remove_columns=[c for c in cols if c != "answer"])
    if col is None:
        raise ValueError(f"GRPO needs a prompt/question/instruction column; got {sorted(cols)}")

    def conv(r):
        p = r[col] if isinstance(r[col], str) else str(r[col])
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
        return {"prompt": msgs}
    drop = [c for c in cols if c not in ("answer",)]
    return ds.map(conv, remove_columns=drop)


# ----------------------------------------------------------------------------- training
def train_grpo(args):
    from live_monitor import LiveMonitor
    live = LiveMonitor(args.output_dir, task="grpo", config=vars(args), sample_prompt=args.sample_prompt,
                       sample_every=args.sample_every)
    with live.guard():
        return _train_grpo(args, live)


def _train_grpo(args, live):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import GRPOTrainer, GRPOConfig

    trainer_utils.enable_maximum_gpu_performance()
    trainer_utils.clear_gpu_memory()
    print("=" * 65)
    print(f"LLM GRPO  |  model={args.model}  reward={args.reward}  4bit={args.load_4bit}")
    print("=" * 65)

    live.phase("Loading model")
    tokenizer = llm_data.ensure_chat_template(AutoTokenizer.from_pretrained(args.model))
    live.tokenizer = tokenizer
    quant, note = device_utils.quant_config_4bit(args.load_4bit, "GRPO")
    if note:
        print(note)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, attn_implementation="sdpa",
                                                 **device_utils.model_load_kwargs())
    model = device_utils.to_device(model)
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    live.phase("Loading dataset")
    if args.dataset == "math_gym":
        import math_gym
        from datasets import Dataset
        lo, _, hi = args.levels.partition("-")
        probs = math_gym.make_set(args.n_problems, levels=range(int(lo), int(hi or lo) + 1), seed=args.seed)
        ds = Dataset.from_dict({"prompt": [math_gym.prompt_text(p) for p in probs], "answer": [p["answer"] for p in probs]})
        print(f"--> math_gym: generated {len(ds)} verified problems (levels {args.levels})")
    else:
        subset = args.subset or ("main" if args.dataset == "openai/gsm8k" else None)
        ds = llm_data.load_raw(args.dataset, split=args.split, subset=subset)
    system = (None if (args.reward == "math" and args.dataset == "math_gym") else
              "Think step by step, then put the final answer in \boxed{}." if args.reward == "math" else
              "First think inside <think></think> tags, then give the final answer." if args.reward == "format" else None)
    ds = to_prompts(ds, system)
    if args.max_samples:
        ds = ds.shuffle(seed=42).select(range(min(args.max_samples, len(ds))))
    print(f"--> {len(ds)} prompts")

    n_gen = args.num_generations
    batch = max(args.batch_size, 1) * n_gen  # per-device batch must be a multiple of num_generations
    mp = device_utils.mixed_precision()
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=n_gen,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion,
        max_steps=args.max_steps or -1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        optim=device_utils.training_optim(),
        bf16=mp["bf16"],
        fp16=mp["fp16"],
        gradient_checkpointing=True,
        logging_steps=2,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=args.report_to,
        temperature=0.9,
        dataloader_pin_memory=device_utils.accelerator() == "cuda",
        **device_utils.extra_training_args(),
    )
    trainer = GRPOTrainer(
        model=model, args=config, train_dataset=ds, reward_funcs=get_rewards(args), processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                               task_type="CAUSAL_LM", target_modules="all-linear"),
        callbacks=[live.callback()],
    )
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint
    resume = get_last_checkpoint(args.output_dir) if args.resume and os.path.isdir(args.output_dir) else None
    result = trainer.train(resume_from_checkpoint=resume)
    print(f"--> Done in {time.time() - t0:.1f}s | peak memory {device_utils.peak_gb_text()}")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"--> Saved GRPO adapter to {final_dir}")
    live.finish(final_dir, train_loss=result.training_loss, peak_vram_gb=device_utils.peak_gb())
    return final_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLM GRPO (reinforcement learning with reward functions)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--subset", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--reward", default="math", choices=["math", "concise", "format", "keywords", "custom"])
    p.add_argument("--reward_file", default=None, help="for --reward custom: a .py file with reward(prompts, completions, **cols)")
    p.add_argument("--levels", default="1-3", help="for --dataset math_gym: difficulty levels, e.g. 1-4")
    p.add_argument("--n_problems", type=int, default=2000, help="for --dataset math_gym: how many problems to generate")
    p.add_argument("--keywords", default="")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir", default="runs/grpo")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--num_generations", type=int, default=4, help="Answers sampled per prompt")
    p.add_argument("--max_completion", type=int, default=256)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1, help="Prompts per step (multiplied by --num_generations)")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample_prompt", default="If a train travels 60 miles in 1.5 hours, what is its speed?")
    p.add_argument("--sample_every", type=int, default=25)
    train_grpo(p.parse_args())

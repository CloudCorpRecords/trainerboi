"""
autopilot.py - The "expert in a box" behind Quick Start.

Turns a beginner's choices (a goal, some data, how much time they have) plus the GPU into a complete, safe
training plan: which model, which method, every hyperparameter - each with a plain-English reason - and
rough time / VRAM estimates. Used by the Quick Start tab and by autopilot_run.py.
"""

import math

# ----------------------------------------------------------------------------- goals
GOALS = {
    "docs": {
        "title": "📄 Teach it about my documents",
        "blurb": "Upload manuals, notes, PDFs, a website… The model learns to answer questions about them.",
        "needs": "files",
        "data_hint": "Add your documents (PDF, Word, text, web pages…). We'll turn them into question-and-answer practice automatically.",
        "task": "sft", "synthetic": True, "format": "sft",
        "suggested": [],
        "system_prompt": "You are a helpful assistant who answers accurately based on what you have learned.",
        "test_prompts": ["Summarize the most important ideas you know about this topic.",
                         "What is one practical tip you can give about this subject?"],
    },
    "persona": {
        "title": "💬 Make a chat assistant with a personality",
        "blurb": "Give it a name, a tone and a role (support agent, tutor, pirate…). Uses your example chats or a general chat set.",
        "needs": "optional",
        "data_hint": "Optional: upload example conversations (CSV/JSON/JSONL with question & answer columns). "
                     "Otherwise we use a high-quality general chat dataset and teach it your persona.",
        "task": "sft", "synthetic": False, "format": "sft",
        "suggested": ["HuggingFaceH4/no_robots||train"],
        "system_prompt": "",
        "test_prompts": ["Introduce yourself in two sentences.", "Can you help me plan my week?",
                         "Explain how a rainbow forms."],
    },
    "instruct": {
        "title": "🧠 Make it better at following instructions",
        "blurb": "General upgrade: clearer, more helpful answers. Uses a well-known instruction dataset — no data needed.",
        "needs": "optional",
        "data_hint": "Nothing needed — we'll use a proven public instruction dataset. You can add your own instruction data too.",
        "task": "sft", "synthetic": False, "format": "sft",
        "suggested": ["yahma/alpaca-cleaned"],
        "system_prompt": "",
        "test_prompts": ["Give me three tips for better sleep.", "Write a haiku about autumn.",
                         "Explain what an API is to a 10-year-old."],
    },
    "code": {
        "title": "💻 Teach it my codebase",
        "blurb": "Point it at a GitHub repo or a code folder. It learns to explain and answer questions about that code.",
        "needs": "files",
        "data_hint": "Add a GitHub repo (owner/repo) or a local code folder.",
        "task": "sft", "synthetic": True, "format": "sft",
        "suggested": [],
        "system_prompt": "You are an expert software engineer who knows this codebase well.",
        "test_prompts": ["What does this project do?", "Explain the main components of this codebase."],
    },
    "prefer": {
        "title": "⚖️ Make it prefer better answers",
        "blurb": "Preference tuning (DPO): shows the model pairs of good vs. bad answers so it leans toward the good ones.",
        "needs": "optional",
        "data_hint": "Optional: upload data with prompt / chosen / rejected columns. Otherwise we use a public preference dataset.",
        "task": "dpo", "synthetic": False, "format": "dpo",
        "suggested": ["trl-lib/ultrafeedback_binarized"],
        "system_prompt": "",
        "test_prompts": ["How do I politely decline a meeting invitation?", "What should I consider when buying a used car?"],
    },
    "scratch": {
        "title": "🏗️ Build a tiny AI from scratch",
        "blurb": "Learn how language models work: train a brand-new mini model that learns to write short stories.",
        "needs": "optional",
        "data_hint": "Nothing needed — we'll use TinyStories (simple children's stories). Or add your own text files.",
        "task": "pretrain", "synthetic": False, "format": "pretrain",
        "suggested": ["roneneldan/TinyStories"],
        "system_prompt": "",
        "test_prompts": ["Once upon a time", "The little dog", "One day, a girl named Lily"],
    },
}

EFFORT = {
    "quick": {"title": "⚡ Quick test", "minutes": 8, "blurb": "~5-10 min. Small model — see that everything works."},
    "balanced": {"title": "⚖️ Balanced", "minutes": 35, "blurb": "~30-45 min. Good results for most goals."},
    "best": {"title": "🏆 Best quality", "minutes": 150, "blurb": "~2-3 h. Bigger model, more training."},
}

# Base models per effort (instruction-tuned, permissive licences, Unsloth-friendly).
SFT_MODELS = {"quick": "Qwen/Qwen2.5-0.5B-Instruct", "balanced": "Qwen/Qwen2.5-1.5B-Instruct",
              "best": "Qwen/Qwen2.5-3B-Instruct", "best_big": "Qwen/Qwen2.5-7B-Instruct"}
GEN_MODELS = {"quick": "Qwen/Qwen2.5-1.5B-Instruct", "balanced": "Qwen/Qwen2.5-3B-Instruct", "best": "Qwen/Qwen2.5-7B-Instruct"}
SYNTH_CHUNKS = {"quick": 60, "balanced": 300, "best": 1200}
SYNTH_SEC_PER_CHUNK = {"quick": 1.3, "balanced": 2.6, "best": 6.0}
HF_ROWS = {"quick": 3000, "balanced": 20000, "best": 100000}
SCRATCH_SIZE = {"quick": "tiny", "balanced": "small", "best": "base"}

# Very rough training throughput on a 16 GB Ampere laptop GPU (tokens/sec, QLoRA via Unsloth).
TOKENS_PER_SEC = {"0.5B": 9000, "1.5B": 4500, "3B": 2500, "7B": 1000, "tiny": 120000, "small": 60000, "base": 25000}


def model_size(model_id):
    for tag in ("0.5B", "1.5B", "3B", "7B", "8B", "1B", "360M", "135M", "1.7B"):
        if tag.lower() in model_id.lower():
            return {"8B": "7B", "1B": "1.5B", "1.7B": "1.5B", "360M": "0.5B", "135M": "0.5B"}.get(tag, tag)
    return "1.5B"


def gpu_info():
    try:
        import device_utils
        return {"name": device_utils.device_name(), "vram_gb": device_utils.usable_memory_gb(),
                "bf16": device_utils.is_bf16_supported(), "accelerator": device_utils.accelerator()}
    except Exception:
        pass
    return {"name": "no GPU", "vram_gb": 0, "bf16": False, "accelerator": "cpu"}


def pick_model(goal, effort, vram_gb, override=None):
    if override:
        return override
    if effort == "best" and vram_gb >= 14 and GOALS[goal]["task"] == "sft":
        import device_utils
        # 7B in 4-bit fits a 16 GB NVIDIA card, but in full precision (Mac/CPU) it needs ~20 GB.
        if device_utils.supports_4bit() or (vram_gb or 0) >= 20:
            return SFT_MODELS["best_big"]
        return SFT_MODELS["best"]
    if vram_gb and vram_gb < 7:
        return SFT_MODELS["quick"]
    return SFT_MODELS[effort]


def _fmt_min(sec):
    return f"{max(1, round(sec / 60))} min" if sec < 5400 else f"{sec / 3600:.1f} h"


# ----------------------------------------------------------------------------- data plan
def data_plan(goal, effort, has_user_data):
    """How the dataset will be prepared (before we know its size)."""
    import device_utils
    g = GOALS[goal]
    gen_model = GEN_MODELS[effort]
    if ("7B" in gen_model and not device_utils.supports_4bit()
            and device_utils.usable_memory_gb() < 20):
        # 7B in full precision needs ~15 GB just to load; use the 3B helper on smaller machines.
        gen_model = GEN_MODELS["balanced"]
    plan = {"format": g["format"], "synthetic": g["synthetic"] and has_user_data,
            "use_suggested": not has_user_data and bool(g["suggested"]),
            "hf": [] if has_user_data else list(g["suggested"]),  # your data only, else a proven public set
            "max_rows": HF_ROWS[effort], "gen_model": gen_model, "max_synth_chunks": SYNTH_CHUNKS[effort]}
    if goal == "scratch":
        plan["synthetic"] = False
    return plan


# ----------------------------------------------------------------------------- training plan
def train_plan(goal, effort, stats=None, vram_gb=16.0, persona=None, model_override=None):
    """Full plan with reasons. `stats` = dataset stats.json (None before the data is built -> estimates)."""
    import device_utils
    g = GOALS[goal]
    task = g["task"]
    speed = device_utils.speed_factor()  # MPS/CPU train slower than the NVIDIA baseline
    budget = EFFORT[effort]["minutes"] * 60
    n = (stats or {}).get("train_examples") or {"quick": 2000, "balanced": 15000, "best": 60000}[effort]
    p95 = (stats or {}).get("tokens_p95") or 400
    mean_tok = (stats or {}).get("tokens_mean") or 200
    reasons, plan = [], {"task": task, "goal": goal, "effort": effort}

    if task == "pretrain":
        size = SCRATCH_SIZE[effort]
        steps = {"quick": 600, "balanced": 3000, "best": 12000}[effort]
        batch, accum, seq = (32, 2, 512) if size != "base" else (16, 4, 512)
        lr = {"tiny": 1e-3, "small": 8e-4, "base": 6e-4}[size]
        tok_per_step = batch * accum * seq
        est = steps * tok_per_step / (TOKENS_PER_SEC[size] * speed) + 60
        plan.update(size=size, max_steps=steps, batch_size=batch, grad_accum=accum, seq_len=seq, lr=lr,
                    sample_prompt=g["test_prompts"][0], sample_every=max(100, steps // 10))
        reasons += [
            f"**Model size: {size}** ({ {'tiny': '~20M', 'small': '~60M', 'base': '~150M'}[size] } parameters) — "
            "a brand-new network starts knowing nothing, so we keep it small enough to learn in your time budget.",
            f"**{steps:,} training steps** × {batch * accum} sequences × {seq} tokens ≈ {steps * tok_per_step / 1e6:.0f}M tokens of reading.",
            f"**Learning rate {lr:g}** — small fresh models learn fastest with a relatively high rate, which then decays smoothly.",
        ]
        plan["estimate"] = {"train_sec": est, "vram_gb": {"tiny": 3, "small": 5, "base": 9}[size]}
        plan["reasons"] = reasons
        return plan

    model = pick_model(goal, effort, vram_gb, model_override)
    size = model_size(model)
    big = size == "7B"
    # Sequence length: fit 95% of examples, never above what the GPU handles comfortably.
    seq = int(min(2048 if not big else 1536, max(512, math.ceil(p95 * 1.1 / 256) * 256)))
    if task == "dpo":
        batch, accum, lr = 2, 8, 5e-6
        seq = min(seq, 1024)
    else:
        batch, accum = (2, 8) if big or seq > 1536 else (4, 4)
        lr = 2e-4 if size in ("0.5B", "1.5B", "3B") else 1e-4
    eff = batch * accum
    epochs = 3 if n < 1000 else 2 if n < 10000 else 1
    full_steps = math.ceil(epochs * n / eff)
    sec_per_step = eff * min(mean_tok * 1.3, seq) / (TOKENS_PER_SEC[size] * speed) * (2.2 if task == "dpo" else 1.0) + 0.35
    budget_steps = int(budget / sec_per_step)
    steps = max(30, min(full_steps, budget_steps))
    lora_r = 32 if effort == "best" else 16
    plan.update(model=model, mode="qlora", max_steps=steps, batch_size=batch, grad_accum=accum, max_seq_len=seq,
                lr=lr, lora_r=lora_r, epochs_equiv=round(steps * eff / max(n, 1), 2),
                packing=task == "sft" and mean_tok < seq / 4,
                save_steps=max(50, steps // 4), sample_every=max(25, steps // 6),
                sample_prompt=g["test_prompts"][0], system_prompt=persona or g["system_prompt"] or None)

    qlora = device_utils.supports_4bit()
    reasons += [
        f"**Base model: {model}** — {'a strong 7B model' + (' (loaded in 4-bit so it fits your GPU)' if qlora else ' (full precision)') if big else f'a {size} instruction-tuned model'}: "
        f"it already speaks well, so we only teach it *your* goal. Bigger = smarter but slower.",
        ("**Method: QLoRA** — the base model is frozen in compressed 4-bit form and we train small \"adapter\" layers on top. "
         f"Uses far less memory and trains only ~{1 if lora_r == 16 else 2}% of the weights, with almost the same quality as full training."
         if qlora else
         "**Method: LoRA** — the base model is frozen and we train small \"adapter\" layers on top "
         f"(only ~{1 if lora_r == 16 else 2}% of the weights), with almost the same quality as full training. "
         "(4-bit QLoRA needs an NVIDIA GPU, so this machine uses full-precision weights.)"),
        f"**{steps:,} steps** of {eff} examples each ≈ **{plan['epochs_equiv']} pass(es)** over your {n:,} examples"
        + (" (capped to fit your time budget)." if steps < full_steps else "."),
        f"**Learning rate {lr:g}** — {'the standard, safe rate for LoRA adapters' if task == 'sft' else 'preference tuning needs a tiny rate so the model does not forget what it knows'}; "
        "it warms up briefly then decays to zero.",
        f"**Max length {seq} tokens** — long enough for 95% of your examples ({p95} tokens) without wasting memory.",
    ]
    if plan["packing"]:
        reasons.append("**Packing on** — your examples are short, so several are packed into one sequence to train faster.")
    if task == "dpo":
        reasons.append("**DPO** — for each prompt the model sees a better and a worse answer and shifts toward the better one. "
                       "Its own frozen copy acts as the reference, so no second model is needed in memory.")
    if plan["system_prompt"]:
        reasons.append(f"**Persona / system prompt** added to every example: _{plan['system_prompt'][:120]}_")
    vram = {"0.5B": 3, "1.5B": 5, "3B": 7, "7B": 11}[size] + (seq / 2048) * (2 if big else 1.2)
    plan["estimate"] = {"train_sec": steps * sec_per_step + 120, "vram_gb": round(vram, 1)}
    if vram > vram_gb * 0.92 and vram_gb:
        reasons.append(f"⚠️ Estimated memory ({vram:.0f} GB) is close to your GPU's {vram_gb:.0f} GB — if it runs out, choose a lower effort.")
    plan["reasons"] = reasons
    return plan


def data_estimate_sec(goal, effort, has_user_data):
    import device_utils
    dp = data_plan(goal, effort, has_user_data)
    sec = 60
    if dp["synthetic"]:
        sec += dp["max_synth_chunks"] * SYNTH_SEC_PER_CHUNK[effort] / device_utils.speed_factor() + 90
    elif dp["hf"]:
        sec += 60
    return sec


def summary_text(goal, effort, has_user_data, vram_gb, persona=None, model_override=None):
    """Plain-English plan preview shown before starting."""
    g, e = GOALS[goal], EFFORT[effort]
    dp = data_plan(goal, effort, has_user_data)
    # Before the data exists, guess its size: synthetic Q&A is capped by the chunk budget (~3 pairs/chunk, some filtered).
    guess = {"train_examples": int(dp["max_synth_chunks"] * 3 * 0.85)} if dp["synthetic"] else None
    tp = train_plan(goal, effort, guess, vram_gb, persona, model_override)
    steps = ["**1. Prepare data** — "]
    if dp["synthetic"]:
        steps[0] += (f"read your files, clean them, then a helper AI (*{dp['gen_model'].split('/')[-1]}*) writes up to "
                     f"{dp['max_synth_chunks'] * 3:,} question/answer pairs from them.")
    elif has_user_data:
        steps[0] += "read and clean your data, remove duplicates and junk, and convert it to the right format."
    else:
        steps[0] += f"download up to {dp['max_rows']:,} examples from **{dp['hf'][0].split('|')[0]}**, then clean and check them."
    if tp["task"] == "pretrain":
        steps.append(f"**2. Train** — build a new *{tp['size']}* model and train it for {tp['max_steps']:,} steps.")
    else:
        import device_utils
        method = "QLoRA" if device_utils.supports_4bit() else "LoRA"
        steps.append(f"**2. Train** — fine-tune **{tp['model'].split('/')[-1]}** with {method} for about {tp['max_steps']:,} steps "
                     "(adjusted once we know your data size).")
    steps.append("**3. Test** — ask the model the same questions before and after training so you can compare.")
    total = data_estimate_sec(goal, effort, has_user_data) + tp["estimate"]["train_sec"] + 90
    first_run = " · first run also downloads the model (a few GB)" if tp["task"] != "pretrain" else ""
    head = (f"### Your plan: {g['title']} · {e['title']}\n"
            f"⏱ **About {_fmt_min(total)}** · 💾 ~{tp['estimate']['vram_gb']} GB GPU memory{first_run}\n\n")
    return head + "\n\n".join(steps), tp["reasons"]

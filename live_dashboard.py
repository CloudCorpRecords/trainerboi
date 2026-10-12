"""
live_dashboard.py - Real-time LLM training dashboard tab for the TrainerBoi Studio.

Training runs in a separate process (so the UI never freezes and a crash can't take it down).
The training scripts write telemetry via live_monitor.py; this tab polls it every 2 seconds.
"""

import os
import re
import sys
import json
import time
import glob
import threading
import subprocess
from collections import deque

import pandas as pd
import gradio as gr

from live_monitor import nvidia_smi

RUNS_DIR = "runs"
ROOT = os.path.dirname(os.path.abspath(__file__))
PROCS = {}  # run_dir -> Popen launched from this Studio session

# ---------------------------------------------------------------- GPU history (background sampler)
GPU_HISTORY = deque(maxlen=300)  # ~5 minutes at 1 Hz


def _gpu_sampler():
    while True:
        s = nvidia_smi()
        if s:
            s["t"] = time.time()
            GPU_HISTORY.append(s)
        time.sleep(1)


threading.Thread(target=_gpu_sampler, daemon=True).start()

# ---------------------------------------------------------------- presets
TASKS = {
    "SFT fine-tune": dict(script="train_llm_sft.py", model="Qwen/Qwen2.5-0.5B-Instruct", dataset="yahma/alpaca-cleaned", lr=2e-4),
    "DPO preference tuning": dict(script="train_llm_dpo.py", model="Qwen/Qwen2.5-0.5B-Instruct", dataset="trl-lib/ultrafeedback_binarized", lr=5e-6),
    "ORPO (SFT + preference in one step)": dict(script="train_llm_orpo.py", model="Qwen/Qwen2.5-0.5B-Instruct", dataset="trl-lib/ultrafeedback_binarized", lr=8e-6),
    "KTO (thumbs up / down feedback)": dict(script="train_llm_kto.py", model="Qwen/Qwen2.5-0.5B-Instruct", dataset="trl-lib/ultrafeedback_binarized", lr=5e-6),
    "GRPO (reinforcement learning)": dict(script="train_llm_grpo.py", model="Qwen/Qwen2.5-0.5B-Instruct", dataset="openai/gsm8k", lr=5e-6),
    "Pretrain from scratch": dict(script="pretrain_llm.py", model="(built from --size)", dataset="roneneldan/TinyStories", lr=1e-3),
}
MODEL_CHOICES = [
    "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
]
DATASET_CHOICES = ["demo", "yahma/alpaca-cleaned", "trl-lib/ultrafeedback_binarized", "roneneldan/TinyStories"]

CSS = """
/* Dropdown option lists: cap height and scroll inside the list (wheel won't scroll the page away). */
ul.option-list {max-height: min(320px, 45vh) !important; overflow-y: auto !important; overscroll-behavior: contain;}
.tb-cards {display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px;}
.tb-card {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px;
          background:var(--block-background-fill);}
.tb-card .lbl {font-size:12px; opacity:.7; display:flex; justify-content:space-between; gap:4px;}
.tb-card .val {font-size:22px; font-weight:700; margin-top:2px; font-variant-numeric:tabular-nums;}
.tb-card .sub {font-size:11px; opacity:.6; margin-top:2px;}
.tb-tip {cursor:help; opacity:.6;}
.tb-phase {display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:8px;}
.tb-pill {padding:3px 10px; border-radius:999px; font-weight:600; font-size:13px; color:#fff;}
.tb-running {background:#2563eb;} .tb-done {background:#16a34a;} .tb-error {background:#dc2626;}
.tb-stopped {background:#6b7280;} .tb-dead {background:#d97706;}
.tb-bar {height:14px; border-radius:7px; background:var(--border-color-primary); overflow:hidden; margin:6px 0 2px;}
.tb-bar > div {height:100%; background:linear-gradient(90deg,#2563eb,#7c3aed); transition:width .8s;}
.tb-insights {margin-top:10px; padding:10px 12px; border-radius:10px; border:1px solid var(--border-color-primary);}
.tb-insights li {margin:3px 0;}
.tb-gauge {height:6px; border-radius:3px; background:var(--border-color-primary); margin-top:6px; overflow:hidden;}
.tb-gauge > div {height:100%;}
"""


# ---------------------------------------------------------------- run discovery / IO
def list_runs():
    paths = glob.glob(os.path.join(ROOT, RUNS_DIR, "**", "live"), recursive=True)
    paths.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(os.path.dirname(p), ROOT) for p in paths]


def _read_json(path, default=None):
    for _ in range(3):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
    return default


def _read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partially-written last line
    except FileNotFoundError:
        pass
    return rows


def _tail(path, max_bytes=60_000, max_lines=250):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return "(no console log - this run was started outside the Studio; its metrics still show above)"
    lines = []
    for line in data.split("\n"):
        # Progress bars rewrite the same line with \r - keep only the final state.
        parts = [p for p in line.split("\r") if p.strip()]
        if parts:
            lines.append(parts[-1])
    return "\n".join(lines[-max_lines:])


def _pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


# ---------------------------------------------------------------- formatting helpers
def _fmt_dur(s):
    if s is None:
        return "—"
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{digits}g}"
    return str(v)


def _card(label, value, sub="", tip=""):
    tip_html = f'<span class="tb-tip" title="{tip}">ⓘ</span>' if tip else ""
    return (f'<div class="tb-card"><div class="lbl"><span>{label}</span>{tip_html}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def _gauge_card(label, value, pct, unit_txt, tip=""):
    pct = max(0, min(100, pct or 0))
    color = "#16a34a" if pct < 70 else "#d97706" if pct < 90 else "#dc2626"
    tip_html = f'<span class="tb-tip" title="{tip}">ⓘ</span>' if tip else ""
    return (f'<div class="tb-card"><div class="lbl"><span>{label}</span>{tip_html}</div>'
            f'<div class="val">{value}</div><div class="sub">{unit_txt}</div>'
            f'<div class="tb-gauge"><div style="width:{pct}%;background:{color}"></div></div></div>')


# ---------------------------------------------------------------- insights ("what's happening")
def _insights(status, metrics, gpu):
    out = []
    train = [r["loss"] for r in metrics if "loss" in r]
    evals = [r["eval_loss"] for r in metrics if "eval_loss" in r]
    grads = [r["grad_norm"] for r in metrics if "grad_norm" in r]
    phase = status.get("phase", "")

    if phase in ("Loading model", "Building model"):
        out.append("⏳ Downloading / loading model weights. First run of a model downloads it from Hugging Face — can take a while.")
    elif phase in ("Loading dataset", "Tokenizing"):
        out.append("⏳ Preparing data: downloading, formatting with the chat template, and converting text to tokens.")
    elif phase.startswith("Generating"):
        out.append("✍️ Pausing briefly to generate a sample with the current weights (see *Sample generations*).")

    if status.get("cooling"):
        out.append(f"🧊 **Paused to let the GPU cool** — it reached {status.get('max_gpu_temp', 82):.0f} °C. Very hot GPUs can "
                   "compute wrong results; training continues by itself. (Set TB_MAX_GPU_TEMP=0 to disable.)")
    if status.get("cool_pauses"):
        out.append(f"🌡️ Cooled down {status['cool_pauses']}× ({status.get('cool_paused_s', 0) / 60:.1f} min total). "
                   "Better laptop cooling or a Balanced power mode makes training faster — see 🩺 GPU check in Quick Start.")
    if any(x != x or x in (float("inf"), float("-inf")) for x in train):
        out.append("🛑 **Loss became NaN/inf** — training diverged. Lower the learning rate.")
    if len(train) >= 6:
        k = max(3, len(train) // 5)
        prev, last = sum(train[-2 * k:-k]) / k, sum(train[-k:]) / k
        delta = (last - prev) / max(abs(prev), 1e-8)
        if delta < -0.02:
            out.append(f"📉 Train loss is **falling** ({prev:.3f} → {last:.3f}) — the model is learning.")
        elif delta > 0.05:
            out.append(f"📈 Train loss is **rising** ({prev:.3f} → {last:.3f}). Some noise is normal; sustained rises usually mean LR is too high.")
        else:
            out.append(f"➖ Train loss is **flat** (~{last:.3f}). Normal late in training; early on, try a higher LR.")
    if len(evals) >= 2:
        if evals[-1] > min(evals) * 1.03 and len(train) >= 2 and train[-1] < train[0]:
            out.append("⚠️ **Eval loss rising while train loss falls → overfitting.** Use the best checkpoint, fewer epochs, or more data.")
        elif evals[-1] <= min(evals):
            out.append(f"✅ Eval loss at its best so far ({evals[-1]:.4f}) — generalizing well.")
    if len(grads) >= 5:
        med = sorted(grads)[len(grads) // 2]
        if grads[-1] > 10 * max(med, 1e-6):
            out.append(f"⚡ Gradient norm spiked ({grads[-1]:.2f} vs typical {med:.2f}). One-off spikes are OK; frequent ones mean instability.")
    if gpu and status.get("state") == "running" and phase == "Training":
        if gpu.get("temp_c") and gpu["temp_c"] >= 87:
            out.append(f"🌡️ GPU is hot ({gpu['temp_c']:.0f}°C) — laptop may throttle. Raise it off the desk / use a cooling pad.")
        if gpu.get("vram_used_gb") and gpu["vram_used_gb"] / gpu["vram_total_gb"] > 0.95:
            out.append("💾 VRAM almost full — if you hit out-of-memory, lower batch size or sequence length.")
        recent = [g.get("gpu_util") or 0 for g in list(GPU_HISTORY)[-10:]]
        if recent and sum(recent) / len(recent) < 50:
            out.append("🐢 GPU utilization is low — the GPU is waiting (small model/batch or data loading). Try a larger batch size.")
    if (status.get("task") == "pretrain" or cfg.get("size")) and (latest.get("loss") or 10) > 3.0:
        out.append("💡 **Pretraining from scratch:** you trained an 18M model from random noise (loss ~9.9). "
                   "A model from scratch needs millions of tokens and loss < 3.0 to learn English. "
                   "To teach an existing smart model (like Qwen or Llama) with your data, switch 'What to train' to **SFT fine-tune**!")
    if status.get("state") == "done":
        out.append(f"🎉 Finished. Model saved to `{status.get('final_dir')}` — chat with it: `python chat_llm.py --model {status.get('final_dir')}`")
    return out


# ---------------------------------------------------------------- main refresh
EMPTY = pd.DataFrame({"step": [], "value": [], "series": []})


def _long(metrics, keys):
    rows = [{"step": r["step"], "value": r[k], "series": label}
            for r in metrics for k, label in keys if isinstance(r.get(k), (int, float))]
    return pd.DataFrame(rows) if rows else EMPTY


def refresh(run):
    gpu = GPU_HISTORY[-1] if GPU_HISTORY else nvidia_smi()
    gpu_df = (pd.DataFrame([{"t": round(g["t"] - time.time()), "value": v, "series": name}
                            for g in GPU_HISTORY for name, v in (
                                ("GPU util %", g.get("gpu_util")),
                                ("VRAM %", 100 * g["vram_used_gb"] / g["vram_total_gb"] if g.get("vram_total_gb") else None),
                                ("Temp °C", g.get("temp_c"))) if v is not None])
              if GPU_HISTORY else pd.DataFrame({"t": [], "value": [], "series": []}))

    gpu_html = '<div class="tb-cards">' + "".join([
        _gauge_card("GPU utilization", f"{_fmt(gpu.get('gpu_util'))}%", gpu.get("gpu_util"), "compute busy",
                    "How busy the GPU cores are. 90-100% while training = good."),
        _gauge_card("VRAM", f"{gpu.get('vram_used_gb', 0):.1f} GB",
                    100 * gpu.get("vram_used_gb", 0) / max(gpu.get("vram_total_gb", 1), 1),
                    f"of {gpu.get('vram_total_gb', 0):.0f} GB (all processes)", "GPU memory in use by everything on the GPU."),
        _gauge_card("Temperature", f"{_fmt(gpu.get('temp_c'))}°C", ((gpu.get("temp_c") or 0) - 30) / 0.6, "throttles ~87°C+"),
        _gauge_card("Power", f"{_fmt(gpu.get('power_w'), 3)} W", (gpu.get("power_w") or 0) / 1.5, "of ~150 W"),
        _card("SM clock", f"{_fmt(gpu.get('sm_clock_mhz'))}", "MHz", "Core clock. Drops when hot or idle."),
    ]) + "</div>"

    if not run:
        return ("<i>No run selected. Start one on the left, or pick an existing run.</i>", gpu_html,
                _plot(EMPTY), _plot(EMPTY), _plot(EMPTY), _plot(EMPTY), _plot(gpu_df), "", "", "")

    run_dir = os.path.join(ROOT, run)
    live = os.path.join(run_dir, "live")
    status = _read_json(os.path.join(live, "status.json"), {}) or {}
    metrics = _read_jsonl(os.path.join(live, "metrics.jsonl"))
    samples = _read_jsonl(os.path.join(live, "samples.jsonl"))
    latest = status.get("latest", {})

    state = status.get("state", "unknown")
    if state == "running" and not _pid_alive(status.get("pid")):
        state = "dead"
    pill = {"running": "tb-running", "done": "tb-done", "error": "tb-error", "stopped": "tb-stopped", "dead": "tb-dead"}.get(state, "tb-stopped")
    state_txt = {"dead": "process exited (see console)"}.get(state, state)

    step, max_steps = status.get("step", 0), status.get("max_steps") or 0
    pct = 100 * step / max_steps if max_steps else 0
    elapsed = status.get("elapsed_s") if state != "running" else time.time() - status.get("started_at", time.time())
    cfg = status.get("config", {})

    header = (f'<div class="tb-phase"><span class="tb-pill {pill}">{state_txt.upper()}</span>'
              f'<b style="font-size:18px">{status.get("phase", "")}</b>'
              f'<span style="opacity:.7">{status.get("task", "").upper()} · {cfg.get("model", cfg.get("size", ""))} · {cfg.get("dataset", "")}</span></div>'
              f'<div class="tb-bar"><div style="width:{pct:.1f}%"></div></div>'
              f'<div style="display:flex;justify-content:space-between;font-size:13px;opacity:.8">'
              f'<span>step {step:,} / {max_steps:,} ({pct:.1f}%) · epoch {status.get("epoch", 0):.2f}</span>'
              f'<span>elapsed {_fmt_dur(elapsed)} · ETA {_fmt_dur(status.get("eta_s")) if state == "running" else "—"}</span></div>')

    sps = status.get("steps_per_s")
    cards = [
        _card("Train loss", _fmt(latest.get("loss")), "lower = better predictions",
              "Cross-entropy of next-token predictions on the training batch. Should trend downward."),
        _card("Eval loss", _fmt(latest.get("eval_loss")), f"best {_fmt(status.get('best_eval_loss'))}",
              "Loss on held-out data the model never trains on. The honest measure of learning."),
        _card("Learning rate", _fmt(latest.get("learning_rate")), "warmup → cosine decay",
              "Step size for weight updates. Ramps up briefly, then decays to 0."),
        _card("Grad norm", _fmt(latest.get("grad_norm")), "size of the update",
              "Magnitude of the gradient. Big spikes = instability."),
        _card("Speed", f"{sps:.2f}" if sps else "—", f"steps/s · {1 / sps:.1f}s per step" if sps else "steps/s"),
        _card("Tokens / sec", _fmt(latest.get("tokens_per_s"), 5), f"{_fmt(latest.get('num_tokens'), 4)} tokens seen",
              "How many tokens the model trains on per second.")
        if latest.get("tokens_per_s") else
        _card("Samples / sec", f"{sps * status.get('effective_batch', 1):.1f}" if sps else "—",
              f"effective batch {status.get('effective_batch', '—')}",
              "Training examples processed per second (steps/s × batch × grad accumulation)."),
        _card("Peak VRAM (trainer)", f"{latest.get('torch_peak_vram_gb', 0):.2f} GB", "PyTorch allocations",
              "Highest memory PyTorch allocated in the training process."),
        _card("Params", _fmt((status.get("trainable_params") or 0) / 1e6, 4) + "M",
              f"trainable of {_fmt((status.get('total_params') or 0) / 1e6, 4)}M",
              "LoRA trains only a small fraction of weights; full fine-tuning trains all of them."),
    ]
    if "mean_token_accuracy" in latest:
        cards.append(_card("Token accuracy", f"{100 * latest['mean_token_accuracy']:.1f}%", "next-token top-1",
                           "How often the model's top guess for the next token is exactly right."))
    if "rewards/accuracies" in latest:
        cards.append(_card("Reward accuracy", f"{100 * latest['rewards/accuracies']:.0f}%", "chosen > rejected",
                           "How often the model prefers the chosen answer over the rejected one. Should rise toward 100%."))
        cards.append(_card("Reward margin", _fmt(latest.get("rewards/margins")), "chosen − rejected",
                           "Gap between chosen and rejected rewards. Growing = learning the preference."))
    cards.append(_card("Checkpoints", str(len(status.get("checkpoints", []))),
                       f"last @ step {status['checkpoints'][-1]['step']}" if status.get("checkpoints") else "none yet"))
    if status.get("train_rows"):
        cards.append(_card("Dataset", f"{status['train_rows']:,}", f"train rows · {status.get('eval_rows', 0):,} eval"))

    tips = [re.sub(r"`(.+?)`", r"<code>\1</code>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t))
            for t in _insights(status, metrics, gpu)]
    insights = ('<div class="tb-insights"><b>What\'s happening</b><ul>' + "".join(f"<li>{t}</li>" for t in tips) + "</ul></div>") if tips else ""
    if status.get("error"):
        import friendly_errors
        insights += friendly_errors.html(status["error"])
        insights += f'<details><summary>Technical details</summary><pre style="color:#dc2626;white-space:pre-wrap;max-height:240px;overflow:auto">{status["error"][-3000:]}</pre></details>'
    summary = header + '<div class="tb-cards" style="margin-top:10px">' + "".join(cards) + "</div>" + insights

    # Loss chart: raw + EMA-smoothed train loss, plus eval.
    loss_rows, ema = [], None
    for r in metrics:
        if isinstance(r.get("loss"), (int, float)):
            ema = r["loss"] if ema is None else 0.8 * ema + 0.2 * r["loss"]
            loss_rows += [{"step": r["step"], "value": r["loss"], "series": "train"},
                          {"step": r["step"], "value": ema, "series": "train (smoothed)"}]
        if isinstance(r.get("eval_loss"), (int, float)):
            loss_rows.append({"step": r["step"], "value": r["eval_loss"], "series": "eval"})
    loss_df = pd.DataFrame(loss_rows) if loss_rows else EMPTY

    lr_df = _long(metrics, [("learning_rate", "learning rate")])
    grad_df = _long(metrics, [("grad_norm", "grad norm")])
    if any("rewards/margins" in r for r in metrics):
        extra_df = _long(metrics, [("rewards/accuracies", "reward accuracy"), ("rewards/margins", "reward margin")])
    elif any("mean_token_accuracy" in r for r in metrics):
        extra_df = _long(metrics, [("mean_token_accuracy", "train token acc"), ("eval_mean_token_accuracy", "eval token acc")])
    else:
        extra_df = _long(metrics, [("torch_vram_gb", "VRAM GB (trainer)")])

    samples_md = "\n\n".join(
        f"**Step {s['step']}** — _{s['prompt']}_\n> " + ((s["output"] or "").strip() or "*(empty output — model predicted end-of-text or newlines; typical for early-stage pretraining)*").replace("\n", "\n> ")
        for s in reversed(samples[-8:])) or "_Samples appear at step 0 and every few dozen steps so you can see the model's answers change as it learns._"

    cfg_md = "  ".join(f"`{k}={v}`" for k, v in cfg.items() if v not in (None, False, ""))
    if not status:
        summary = '<div class="tb-phase"><span class="tb-pill tb-running">STARTING</span><b>Launching Python & loading libraries…</b></div>'
    return (summary, gpu_html, _plot(loss_df), _plot(lr_df), _plot(grad_df), _plot(extra_df), _plot(gpu_df),
            samples_md, _tail(os.path.join(live, "console.log")), cfg_md)


def _plot(df):
    """Hide charts until they have data (empty Vega charts render '-Infinity' legends)."""
    return gr.update(value=df, visible=not df.empty)


# ---------------------------------------------------------------- auto settings + pre-flight
_AUTO_GOAL = {"SFT fine-tune": "instruct", "DPO preference tuning": "prefer", "ORPO (SFT + preference in one step)": "prefer",
              "KTO (thumbs up / down feedback)": "prefer", "Pretrain from scratch": "scratch"}
_EFFORT_KEYS = {"⚡ Quick (~8 min)": "quick", "⚖️ Balanced (~35 min)": "balanced", "🏆 Best (~2.5 h)": "best"}


def _stats_for(dataset):
    for p in glob.glob(os.path.join(ROOT, "data", "*", "stats.json")):
        s = _read_json(p, {}) or {}
        if s.get("train_path") and os.path.normpath(s["train_path"]) == os.path.normpath(dataset or ""):
            return s
    return None


def auto_settings(task, dataset, upload, effort):
    """Fill every setting with sensible values for this data + GPU + time budget, and explain why."""
    import autopilot
    if upload is not None:
        dataset = upload if isinstance(upload, str) else upload.name
    key = _EFFORT_KEYS.get(effort, "balanced")
    if task == "GRPO (reinforcement learning)":
        steps = {"quick": 60, "balanced": 200, "best": 600}[key]
        md = ("**Reinforcement learning needs few steps but each one is slow** (the model writes several answers per question). "
              "Using a small model, 4 answers per question and a low learning rate.")
        return (gr.skip(), steps, 5e-6, 1, 4, 512, 16, "lora", gr.skip(), md)
    goal = _AUTO_GOAL.get(task, "instruct")
    gpu = autopilot.gpu_info()
    plan = autopilot.train_plan(goal, key, _stats_for(dataset), gpu["vram_gb"] or 16.0)
    why = "\n".join(f"- {r}" for r in plan["reasons"])
    est = autopilot._fmt_min(plan["estimate"]["train_sec"])
    note = "" if _stats_for(dataset) else "\n\n<small>This dataset wasn't built here, so sizes are estimated.</small>"
    md = f"### ✨ Recommended settings (about {est})\n{why}{note}"
    if plan["task"] == "pretrain":
        return (gr.skip(), plan["max_steps"], plan["lr"], plan["batch_size"], plan["grad_accum"], plan["seq_len"], 16,
                "qlora", plan["size"], md)
    return (plan["model"], plan["max_steps"], plan["lr"], plan["batch_size"], plan["grad_accum"], plan["max_seq_len"],
            plan["lora_r"], "qlora", gr.skip(), md)


def preflight(task, model, dataset):
    """Plain-English checks before spawning a run; raises gr.Error with the fix."""
    import shutil
    if not dataset:
        raise gr.Error("Choose a dataset first (or press 'Use recommended settings' to see suggestions).")
    if os.path.exists(dataset) is False and dataset != "demo" and "/" not in dataset:
        raise gr.Error(f"I can't find the dataset '{dataset}'. Pick one from the list, upload a file, or use a Hugging Face id like owner/name.")
    if task != "Pretrain from scratch" and not (model or "").strip():
        raise gr.Error("Choose a base model.")
    for name, proc in list(PROCS.items()):
        if proc.poll() is None:
            raise gr.Error(f"'{os.path.basename(name)}' is still training and using the GPU. Stop it first, or wait for it to finish.")
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 8:
        raise gr.Error(f"Only {free:.1f} GB of disk space is free - training needs at least 8 GB. Delete old runs or datasets first.")
    try:
        import gpu_guard
        t = gpu_guard.gpu_temp()
        if t and t >= gpu_guard.max_temp_setting() > 0:
            gr.Warning(f"The GPU is already at {t:.0f} °C. Training will pause itself to cool down when needed.")
    except Exception:
        pass

# ---------------------------------------------------------------- launch / stop
def launch(task, model, dataset, upload, mode, steps, lr, batch, grad_accum, seq_len, lora_r, size,
           sample_prompt, sample_every, run_name):
    spec = TASKS[task]
    if upload is not None:
        dataset = upload if isinstance(upload, str) else upload.name
    preflight(task, model, dataset)
    name = (run_name or "").strip() or f"{task.split()[0].lower()}-{time.strftime('%m%d-%H%M%S')}"
    out_dir = os.path.join(RUNS_DIR, name)
    os.makedirs(os.path.join(ROOT, out_dir, "live"), exist_ok=True)

    cmd = [sys.executable, "-u", spec["script"], "--dataset", dataset, "--output_dir", out_dir,
           "--max_steps", str(int(steps)), "--lr", str(lr), "--batch_size", str(int(batch)),
           "--sample_every", str(int(sample_every))]
    if sample_prompt:
        cmd += ["--sample_prompt", sample_prompt]
    if task == "SFT fine-tune":
        cmd += ["--model", model, "--mode", mode, "--grad_accum", str(int(grad_accum)),
                "--max_seq_len", str(int(seq_len)), "--lora_r", str(int(lora_r))]
        if mode == "full":
            cmd += ["--backend", "hf"]  # most reliable for full fine-tunes
    elif task in ("DPO preference tuning", "ORPO (SFT + preference in one step)", "KTO (thumbs up / down feedback)"):
        cmd += ["--model", model, "--grad_accum", str(int(grad_accum)), "--max_len", str(int(seq_len)),
                "--lora_r", str(int(lora_r))]
        if mode == "qlora":
            cmd.append("--load_4bit")
    elif task == "GRPO (reinforcement learning)":
        cmd += ["--model", model, "--grad_accum", str(int(grad_accum)), "--lora_r", str(int(lora_r)),
                "--reward", "math" if "gsm8k" in dataset else "concise"]
        if mode == "qlora":
            cmd.append("--load_4bit")
    else:
        cmd += ["--size", size, "--seq_len", str(int(min(seq_len, 1024))), "--grad_accum", str(int(grad_accum)),
                "--save_steps", str(max(100, int(steps) // 6))]

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    log = open(os.path.join(ROOT, out_dir, "live", "console.log"), "w", encoding="utf-8")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    PROCS[out_dir] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gr.Info(f"Started run '{name}'")
    return gr.update(choices=list_runs(), value=out_dir)


def stop(run):
    if not run:
        return
    status_path = os.path.join(ROOT, run, "live", "status.json")
    status = _read_json(status_path, {}) or {}
    pids = {status.get("pid")}
    if run in PROCS:
        pids.add(PROCS[run].pid)
    for pid in filter(None, pids):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    if status:
        status.update(state="stopped", phase="Stopped by user")
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f)
    gr.Warning(f"Stopped {run}. Checkpoints saved so far are kept; resume SFT with --resume.")


def on_task_change(task):
    spec = TASKS[task]
    is_pre = task == "Pretrain from scratch"
    return (gr.update(value=spec["model"], interactive=not is_pre), gr.update(value=spec["dataset"]),
            gr.update(value=spec["lr"]), gr.update(visible=is_pre),
            gr.update(value="Once upon a time" if is_pre else "Explain what a neural network is in two sentences."))


# ---------------------------------------------------------------- UI
def dataset_choices():
    """Presets + datasets built with the Dataset Builder (newest first)."""
    built = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "*", "stats.json")), key=os.path.getmtime, reverse=True):
        path = (_read_json(p, {}) or {}).get("train_path")  # train.jsonl, or a folder of parquet shards
        if path:
            built.append(path)
    return built + DATASET_CHOICES


def build_tab():
    with gr.TabItem("📡 Live training", id="live"):
        with gr.Row():
            # ---------------- left: launcher
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 🚀 Start a run")
                task = gr.Radio(list(TASKS), value="SFT fine-tune", label="What to train")
                model = gr.Dropdown(MODEL_CHOICES, value=TASKS["SFT fine-tune"]["model"], allow_custom_value=True,
                                    label="Base model (any Hugging Face id)")
                dataset = gr.Dropdown(dataset_choices(), value=TASKS["SFT fine-tune"]["dataset"], allow_custom_value=True,
                                      label="Dataset (built in 📚 Dataset Builder, HF id, local path, or 'demo')")
                upload = gr.File(label="…or upload a dataset (.jsonl/.json/.csv/.parquet/.txt)",
                                 file_types=[".jsonl", ".json", ".csv", ".parquet", ".txt"], type="filepath")
                mode = gr.Radio(["qlora", "lora", "full"], value="qlora", label="Mode",
                                info="qlora = 4-bit base + adapters (least VRAM) · lora = bf16 + adapters · full = all weights")
                size = gr.Radio(["tiny", "small", "base"], value="tiny", label="Model size (from scratch)",
                                info="~20M / ~60M / ~150M params", visible=False)
                with gr.Row():
                    effort_pick = gr.Dropdown(list(_EFFORT_KEYS), value="⚖️ Balanced (~35 min)", label="Time budget", scale=2)
                    auto_btn = gr.Button("✨ Use recommended settings", scale=2)
                auto_md = gr.Markdown()
                steps = gr.Slider(10, 5000, value=300, step=10, label="Training steps")
                with gr.Accordion("Advanced", open=False):
                    lr = gr.Number(value=2e-4, label="Learning rate")
                    batch = gr.Slider(1, 64, value=4, step=1, label="Batch size per step")
                    grad_accum = gr.Slider(1, 32, value=4, step=1, label="Gradient accumulation",
                                           info="effective batch = batch × accumulation")
                    seq_len = gr.Slider(128, 4096, value=1024, step=128, label="Max sequence length (tokens)")
                    lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA rank")
                    sample_prompt = gr.Textbox(value="Explain what a neural network is in two sentences.",
                                               label="Sample prompt (generated during training)")
                    sample_every = gr.Slider(0, 500, value=50, step=10, label="Generate a sample every N steps (0 = off)")
                    run_name = gr.Textbox(label="Run name (optional)", placeholder="auto: sft-1009-2130")
                with gr.Row():
                    start_btn = gr.Button("▶ Start training", variant="primary")
                    stop_btn = gr.Button("■ Stop", variant="stop")

            # ---------------- right: live view
            with gr.Column(scale=3):
                with gr.Row():
                    run = gr.Dropdown(list_runs(), value=(list_runs() or [None])[0], label="Watching run", scale=4,
                                      allow_custom_value=True)
                    live_toggle = gr.Checkbox(value=True, label="Live updates (2s)", scale=1)
                summary = gr.HTML()
                with gr.Row():
                    loss_plot = gr.LinePlot(EMPTY, x="step", y="value", color="series", title="Loss",
                                            y_title="loss", height=260)
                    extra_plot = gr.LinePlot(EMPTY, x="step", y="value", color="series", title="Accuracy / rewards",
                                             height=260)
                with gr.Row():
                    lr_plot = gr.LinePlot(EMPTY, x="step", y="value", color="series", title="Learning rate schedule", height=200)
                    grad_plot = gr.LinePlot(EMPTY, x="step", y="value", color="series", title="Gradient norm", height=200)
                gr.Markdown("#### 🖥️ GPU (live)")
                gpu_html = gr.HTML()
                gpu_plot = gr.LinePlot(pd.DataFrame({"t": [], "value": [], "series": []}), x="t", y="value",
                                       color="series", title="GPU — last 5 minutes (seconds ago)", height=200)
                with gr.Tabs():
                    with gr.TabItem("✍️ Sample generations"):
                        samples = gr.Markdown()
                    with gr.TabItem("🖨️ Console"):
                        console = gr.Textbox(lines=18, max_lines=18, autoscroll=True, show_label=False)
                    with gr.TabItem("⚙️ Run config"):
                        cfg = gr.Markdown()

        outputs = [summary, gpu_html, loss_plot, lr_plot, grad_plot, extra_plot, gpu_plot, samples, console, cfg]
        timer = gr.Timer(2)
        timer.tick(refresh, inputs=run, outputs=outputs, show_progress="hidden")
        live_toggle.change(lambda on: gr.Timer(active=on), live_toggle, timer)
        run.change(refresh, inputs=run, outputs=outputs, show_progress="hidden")
        run.focus(lambda r: gr.update(choices=list_runs(), value=r), run, run)

        # .input (user clicks only) so programmatic task changes from the Dataset Builder keep their dataset.
        task.input(on_task_change, task, [model, dataset, lr, size, sample_prompt])
        task.change(lambda t: (gr.update(visible=t == "Pretrain from scratch"),
                               gr.update(interactive=t != "Pretrain from scratch")), task, [size, model])
        dataset.focus(lambda v: gr.update(choices=dataset_choices(), value=v), dataset, dataset)
        start_btn.click(launch, [task, model, dataset, upload, mode, steps, lr, batch, grad_accum, seq_len,
                                 lora_r, size, sample_prompt, sample_every, run_name], run)
        stop_btn.click(stop, run, None)
        auto_btn.click(auto_settings, [task, dataset, upload, effort_pick],
                       [model, steps, lr, batch, grad_accum, seq_len, lora_r, mode, size, auto_md])
    return {"task": task, "dataset": dataset}

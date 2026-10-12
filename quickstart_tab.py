"""
quickstart_tab.py - "✨ Quick Start": train an AI model in 4 guided steps, no ML knowledge needed.

  1. Goal      pick what you want in plain words
  2. Data      add files / a repo / a website - or let us pick a proven public dataset
  3. Time      quick / balanced / best -> see the full plan and *why* before starting
  4. Watch     a simple timeline, live loss chart, then before/after answers and a chat box

Everything technical is decided by autopilot.py and executed by autopilot_run.py in the background.
"""

import os
import re
import sys
import json
import glob
import time
import threading
import subprocess

import pandas as pd
import gradio as gr

import autopilot
import data_intake
from dataset_sources import POPULAR_HF_DATASETS, WIKIPEDIA_PRESETS, preview_hf_dataset
from live_dashboard import ROOT, _read_json, _pid_alive, _fmt_dur, _read_jsonl

GOAL_KEYS = list(autopilot.GOALS)
GOAL_TITLES = [autopilot.GOALS[k]["title"] for k in GOAL_KEYS]
EFFORT_KEYS = list(autopilot.EFFORT)
EFFORT_TITLES = [f"{autopilot.EFFORT[k]['title']} — {autopilot.EFFORT[k]['blurb']}" for k in EFFORT_KEYS]
MODEL_CHOICES = ["(automatic — recommended)", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                 "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
                 "unsloth/Llama-3.2-3B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"]

CSS = """
.qs-goal label {padding:10px 14px !important;}
.qs-tl {list-style:none; padding:0; margin:0;}
.qs-tl li {display:flex; gap:12px; align-items:flex-start; padding:10px 12px; margin:6px 0; border-radius:10px;
           border:1px solid var(--border-color-primary); background:var(--block-background-fill);}
.qs-tl .ic {font-size:20px; width:28px; text-align:center;}
.qs-tl .t {font-weight:700;} .qs-tl .d {font-size:13px; opacity:.75; margin-top:2px;}
.qs-tl li.running {border-color:#2563eb; box-shadow:0 0 0 1px #2563eb inset;}
.qs-tl li.error {border-color:#dc2626;}
.qs-cmp {display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:6px 0 16px;}
.qs-cmp > div {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px; white-space:pre-wrap;
               font-size:14px; background:var(--block-background-fill);}
.qs-cmp .h {font-size:12px; font-weight:700; opacity:.7; margin-bottom:4px;}
.qs-q {font-weight:700; margin-top:10px;}
.qs-ref {font-size:13px; opacity:.7; margin:2px 0 4px;}
@media (max-width: 700px) {.qs-cmp {grid-template-columns:1fr;}}
"""

_GPU = autopilot.gpu_info()


# ---------------------------------------------------------------- helpers
def _goal(title):
    return GOAL_KEYS[GOAL_TITLES.index(title)] if title in GOAL_TITLES else "instruct"


def _effort(title):
    for k, t in zip(EFFORT_KEYS, EFFORT_TITLES):
        if title == t:
            return k
    return "balanced"


def _lines(t):
    return [l.strip() for l in (t or "").splitlines() if l.strip()]


def library_choices(goal_title):
    fmt = autopilot.GOALS[_goal(goal_title)]["format"]
    out = ["(none)"]
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "*", "stats.json")), key=os.path.getmtime, reverse=True):
        s = _read_json(p, {}) or {}
        if s.get("format") == fmt:
            out.append(f"{s['name']}  ·  {s['train_examples']:,} examples")
    return out


def previous_runs():
    paths = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "autopilot.json")), key=os.path.getmtime, reverse=True)
    return [os.path.basename(os.path.dirname(p)) for p in paths]


# ---------------------------------------------------------------- step 1/2/3 logic
def on_goal(goal_title):
    g = autopilot.GOALS[_goal(goal_title)]
    return (f"**{g['title']}** — {g['blurb']}",
            f"💡 {g['data_hint']}",
            gr.update(visible=_goal(goal_title) == "persona"),
            gr.update(choices=library_choices(goal_title), value="(none)"))


def on_hf_preset_selected(choice, current_text):
    if not choice or choice == "(custom)":
        return current_text, 20000
    for d in POPULAR_HF_DATASETS:
        if choice.startswith(d["name"]):
            lines = _lines(current_text)
            if d["id"] not in lines:
                lines.append(d["id"])
            return "\n".join(lines), d.get("default_rows", 20000)
    return current_text, 20000


def on_hf_preview_click(hf_text):
    lines = _lines(hf_text)
    if not lines:
        raise gr.Error("Please enter or select a Hugging Face dataset ID first.")
    spec = lines[0]
    try:
        rows = preview_hf_dataset(spec, max_rows=3)
        if not rows:
            raise gr.Error("No rows returned from dataset.")
        df = pd.DataFrame([{k: str(v)[:150] for k, v in r.items()} for r in rows])
        return gr.update(value=df, visible=True)
    except Exception as e:
        raise gr.Error(f"Failed to preview {spec}: {e}")


def add_wiki_preset(topic_key, current_text):
    presets = WIKIPEDIA_PRESETS.get(topic_key, [])
    lines = _lines(current_text)
    for p in presets:
        if p not in lines:
            lines.append(p)
    return "\n".join(lines)


def preview_plan(goal_title, effort_title, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap,
                 hf_user, hf_max_rows, wikipedia, wiki_per_query, arxiv, github, youtube, max_videos,
                 transcribe, library, persona, model):
    goal, effort = _goal(goal_title), _effort(effort_title)
    has_data = bool(
        uploads or _lines(folders) or _lines(urls) or _lines(hf_user) or
        _lines(wikipedia) or _lines(arxiv) or _lines(github) or _lines(youtube)
    ) or (library and library != "(none)")
    g = autopilot.GOALS[goal]
    warn = ""
    if g["needs"] == "files" and not has_data:
        warn = "⚠️ **This goal needs your data** — go back to step 2 and add files, a Hugging Face dataset, a website, Wikipedia, or a GitHub repo.\n\n"
    override = None if (not model or model.startswith("(automatic")) else model
    text, reasons = autopilot.summary_text(goal, effort, has_data, _GPU["vram_gb"], (persona or "").strip() or None, override)

    src_list = []
    if uploads or _lines(folders):
        src_list.append(f"📁 Local files/folders ({len(uploads or [])} upload(s), {len(_lines(folders))} folder(s))")
    if _lines(hf_user):
        limit_str = f" (max {int(hf_max_rows):,} rows)" if hf_max_rows else ""
        src_list.append(f"🤗 Hugging Face Hub: {', '.join(_lines(hf_user)[:2])}{limit_str}")
    if _lines(urls):
        crawl_str = f" [depth {int(crawl_depth)}, max {int(max_pages)} pages{' + sitemap' if sitemap else ''}]" if (crawl_depth or sitemap) else ""
        src_list.append(f"🌐 Web / Documentation: {len(_lines(urls))} root URL(s){crawl_str}")
    if _lines(wikipedia):
        src_list.append(f"📖 Wikipedia: {len(_lines(wikipedia))} query/queries ({int(wiki_per_query)} articles each)")
    if _lines(arxiv):
        src_list.append(f"🔬 Research Papers: {len(_lines(arxiv))} query/queries")
    if _lines(github):
        src_list.append(f"💻 GitHub: {', '.join(_lines(github)[:2])}")
    if _lines(youtube):
        src_list.append(f"▶ YouTube: {len(_lines(youtube))} URL(s) (max {int(max_videos)} videos)")
    if library and library != "(none)":
        src_list.append(f"🗂️ Library: {library}")

    src_section = ""
    if src_list:
        src_section = "**Sourcing Plan:**\n" + "\n".join(f"- {s}" for s in src_list) + "\n\n"

    why = src_section + "\n".join(f"- {r}" for r in reasons) + \
          "\n\n<small>These are starting estimates — the plan is finalized automatically once your data is ready.</small>"
    return warn + text, why


def start(goal_title, effort_title, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap,
          hf_user, hf_max_rows, wikipedia, wiki_per_query, arxiv, github, youtube, max_videos,
          transcribe, library, persona, model, name):
    goal, effort = _goal(goal_title), _effort(effort_title)
    g = autopilot.GOALS[goal]
    sources = list(uploads or []) + _lines(folders)
    has_data = bool(
        sources or _lines(urls) or _lines(hf_user) or _lines(wikipedia) or
        _lines(arxiv) or _lines(github) or _lines(youtube)
    ) or (library and library != "(none)")
    if g["needs"] == "files" and not has_data:
        raise gr.Error("This goal needs your data — add files, a Hugging Face dataset, a website, Wikipedia, or a GitHub repo in step 2.")
    for r in previous_runs():
        st = _read_json(os.path.join(ROOT, "runs", r, "autopilot.json"), {}) or {}
        if st.get("state") == "running" and _pid_alive(st.get("pid")):
            raise gr.Error(f"Another Quick Start run ('{r}') is still going. Stop it first or wait for it to finish.")
    name = re.sub(r"[^\w.-]+", "-", (name or "").strip()) or f"qs-{goal}-{time.strftime('%m%d-%H%M')}"
    run_dir = os.path.join(ROOT, "runs", name)
    if os.path.exists(os.path.join(run_dir, "autopilot.json")):
        raise gr.Error(f"A run named '{name}' already exists — pick another name.")
    os.makedirs(run_dir, exist_ok=True)
    cfg = {
        "name": name, "goal": goal, "effort": effort, "sources": sources,
        "ocr": bool(ocr), "urls": _lines(urls), "crawl_depth": int(crawl_depth),
        "max_pages": int(max_pages), "sitemap": bool(sitemap),
        "hf_user": _lines(hf_user), "max_rows": int(hf_max_rows) if hf_max_rows else None,
        "wikipedia": _lines(wikipedia), "wiki_per_query": int(wiki_per_query),
        "arxiv": _lines(arxiv), "github": _lines(github),
        "youtube": _lines(youtube), "max_videos": int(max_videos),
        "transcribe": bool(transcribe),
        "persona": (persona or "").strip() or None,
        "dataset_ref": library.split("  ·  ")[0] if library and library != "(none)" else None,
        "model_override": None if (not model or model.startswith("(automatic")) else model
    }
    cfg_path = os.path.join(run_dir, "autopilot_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    log = open(os.path.join(run_dir, "autopilot_console.log"), "w", encoding="utf-8")
    subprocess.Popen([sys.executable, "-u", "autopilot_run.py", "--config", cfg_path], cwd=ROOT, stdout=log,
                     stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gr.Info("Started! You can leave this page open — or come back later, it keeps running.")
    return name, gr.Walkthrough(selected=3), gr.update(choices=previous_runs(), value=name)


def stop(name):
    if not name:
        return
    path = os.path.join(ROOT, "runs", name, "autopilot.json")
    st = _read_json(path, {}) or {}
    if st.get("pid"):
        subprocess.run(["taskkill", "/PID", str(st["pid"]), "/T", "/F"], capture_output=True)
    if st:
        st["state"] = "stopped"
        for s in st["steps"]:
            if s["state"] == "running":
                s["state"], s["detail"] = "error", "Stopped by you."
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st, f)
    gr.Warning("Stopped.")


# ---------------------------------------------------------------- step 4: live view
ICON = {"pending": "⚪", "running": "🔵", "done": "✅", "error": "❌"}


def _progress(st, name):
    """Sub-progress of the current step from the child process' own status file."""
    cur = st.get("current")
    if cur == "data" and st.get("dataset"):
        b = _read_json(os.path.join(ROOT, "data", st["dataset"], "build", "status.json"), {}) or {}
        p = b.get("progress", {})
        frac = p["done"] / p["total"] if p.get("total") else None
        msg = b.get("phase", "Preparing…")
        if b.get("counters", {}).get("examples kept"):
            msg += f" · {b['counters']['examples kept']:,} examples so far"
        return frac, msg, p.get("eta_s")
    if cur == "train":
        live = _read_json(os.path.join(ROOT, "runs", name, "live", "status.json"), {}) or {}
        if live.get("max_steps"):
            loss = live.get("latest", {}).get("loss")
            msg = f"Step {live['step']:,} of {live['max_steps']:,}"
            if loss is not None:
                msg += f" · loss {loss:.3f} (lower = learning)"
            if live.get("cooling"):
                msg = f"🧊 {live['phase']} — pausing keeps results correct, it continues by itself · " + msg
            elif live.get("phase") and live["phase"] != "Training":
                msg = live["phase"] + " · " + msg
            if live.get("cool_pauses"):
                msg += f" · cooled {live['cool_pauses']}× so far"
            return live["step"] / live["max_steps"], msg, live.get("eta_s")
        return None, live.get("phase", "Loading the model (first time downloads it)…"), None
    if cur == "test":
        return None, "Loading both versions of the model and asking them questions…", None
    return None, "", None


def _timeline(st, name):
    frac, msg, eta = _progress(st, name)
    items = []
    for s in st.get("steps", []):
        extra = ""
        if s["state"] == "running":
            bar = (f'<div class="tb-bar" style="margin-top:6px"><div style="width:{100 * frac:.1f}%"></div></div>'
                   if frac is not None else "")
            extra = f'<div class="d">{msg}{" · about " + _fmt_dur(eta) + " left" if eta else ""}</div>{bar}'
        items.append(f'<li class="{s["state"]}"><div class="ic">{ICON[s["state"]]}</div><div style="flex:1">'
                     f'<div class="t">{s["title"]}</div><div class="d">{s.get("detail", "")}</div>{extra}</div></li>')
    state = st.get("state")
    if state == "running" and not _pid_alive(st.get("pid")):
        state = "crashed"
    head = {"running": "⏳ Working on it — this page updates by itself.",
            "done": "🎉 **Done!** Scroll down to compare answers and chat with your model.",
            "error": "❌ Something went wrong — see the message below.",
            "stopped": "⏹ Stopped.", "crashed": "⚠️ The background process ended unexpectedly — see the log below."}.get(state, "")
    g = autopilot.GOALS.get(st.get("goal"), {})
    html = (f"<h3 style='margin:0 0 4px'>{g.get('title', '')} · {autopilot.EFFORT.get(st.get('effort'), {}).get('title', '')}</h3>"
            f"<p style='margin:0 0 8px;opacity:.8'>{head} &nbsp;·&nbsp; elapsed {_fmt_dur(time.time() - st.get('started_at', time.time()) if state == 'running' else st.get('updated_at', 0) - st.get('started_at', 0))}</p>"
            f"<ul class='qs-tl'>{''.join(items)}</ul>")
    if st.get("error") and state in ("error", "crashed"):
        first = st["error"].split("\n\n")[0]
        import friendly_errors
        html += friendly_errors.html(st["error"])
        html += (f"<div class='tb-insights' style='border-color:#dc2626'><b>Technical details:</b><pre style='white-space:pre-wrap'>{first[:1500]}</pre>"
                 "<b>Try:</b> a lower effort level (less memory), checking your files open normally, or the Console below.</div>")
    return html


def _results_html(st):
    res = st.get("results")
    if not res:
        return ""
    out = ["<h3>🔍 Before vs. after training</h3>",
           "<p style='opacity:.8'>Same questions, same settings. <b>Before</b> = the original model, <b>After</b> = your trained model."
           + (" Questions marked 📘 come from data the model never trained on." if any(p.get("reference") for p in res["pairs"]) else "")
           + "</p>"]
    for p in res["pairs"]:
        out.append(f"<div class='qs-q'>{'📘 ' if p.get('reference') else '💬 '}{_esc(p['prompt'][:500])}</div>")
        if p.get("reference"):
            out.append(f"<div class='qs-ref'>Expected (from your data): {_esc(p['reference'][:300])}</div>")
        out.append(f"<div class='qs-cmp'><div><div class='h'>BEFORE</div>{_esc(p.get('before', ''))}</div>"
                   f"<div><div class='h'>AFTER ✨</div>{_esc(p.get('after', ''))}</div></div>")
    return "".join(out)


def _esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def refresh(name):
    if not name:
        return ("<i>Start a run in step 3, or open a previous run above.</i>", gr.update(visible=False), "",
                gr.update(visible=False), "", "")
    run_dir = os.path.join(ROOT, "runs", name)
    st = _read_json(os.path.join(run_dir, "autopilot.json"), {}) or {}
    if not st:
        return ("<p>⏳ Starting up…</p>", gr.update(visible=False), "", gr.update(visible=False), "",
                _tail(os.path.join(run_dir, "autopilot_console.log")))
    metrics = _read_jsonl(os.path.join(run_dir, "live", "metrics.jsonl"))
    rows = [{"step": r["step"], "loss": r[k], "series": lbl} for r in metrics
            for k, lbl in (("loss", "training"), ("eval_loss", "test")) if isinstance(r.get(k), (int, float))]
    loss_df = pd.DataFrame(rows) if rows else None
    samples = _read_jsonl(os.path.join(run_dir, "live", "samples.jsonl"))
    sample_md = ""
    if samples and st.get("current") == "train":
        s = samples[-1]
        sample_md = f"**✍️ Live peek — step {s['step']}:** _{s['prompt']}_\n> " + (s["output"][:500] or "…").replace("\n", "\n> ")
    done = st.get("state") == "done"
    plan = st.get("plan") or {}
    why = "\n".join(f"- {r}" for r in plan.get("reasons", []))
    log = _tail(os.path.join(run_dir, "autopilot.log")) or _tail(os.path.join(run_dir, "autopilot_console.log"))
    return (_timeline(st, name),
            gr.update(value=loss_df, visible=loss_df is not None),
            sample_md,
            gr.update(visible=done),
            _results_html(st) + (f"\n\n<details><summary><b>Why these settings?</b></summary>\n\n{why}\n</details>" if why else ""),
            log)


def _tail(path, n=60):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 30000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")
        return "\n".join(l for l in lines if l.strip() and "Enum subclass" not in l)[-6000:]
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------- GPU health check
def gpu_check(seconds=150):
    """Streams progress of `gpu_guard.py --check` (runs in its own process), then a plain-English verdict."""
    import gpu_guard
    for r in previous_runs():
        st = _read_json(os.path.join(ROOT, "runs", r, "autopilot.json"), {}) or {}
        if st.get("state") == "running" and _pid_alive(st.get("pid")):
            raise gr.Error("A training run is using the GPU right now — run the check when it's finished.")
    unload()
    yield f"🩺 Testing your GPU under full load for {seconds}s — it will get hot, that's the point…"
    p = subprocess.Popen([sys.executable, "-u", "gpu_guard.py", "--check", str(seconds)], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env=dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in p.stdout:
        m = re.match(r"\s*(\d+)/(\d+)s\s+([\d.]+|None) °C\s+errors (\d+)", line)
        if m:
            done, total, temp, errs = m.groups()
            bar = "█" * int(20 * int(done) / int(total)) + "░" * (20 - int(20 * int(done) / int(total)))
            yield f"🩺 Testing… `{bar}` {done}/{total}s · GPU {temp} °C · calculation errors so far: **{errs}**"
    p.wait()
    yield gpu_guard.explain(_read_json(gpu_guard.HEALTH_FILE, None))


def last_health():
    import gpu_guard
    r = _read_json(gpu_guard.HEALTH_FILE, None)
    return (gpu_guard.explain(r) + f"\n\n<small>Last checked {r['checked_at']}</small>") if r else \
        "Not checked yet. Recommended once — especially on laptops."


# ---------------------------------------------------------------- chat with the result
_CHAT = {"name": None, "model": None, "tok": None, "raw": False, "system": None}
_LOCK = threading.Lock()


def _load_for_chat(name):
    st = _read_json(os.path.join(ROOT, "runs", name, "autopilot.json"), {}) or {}
    res, plan = st.get("results") or {}, st.get("plan") or {}
    if not res.get("model_dir"):
        raise gr.Error("This run hasn't finished yet.")
    if _CHAT["name"] == name:
        return
    unload()
    import torch
    from chat_llm import load
    model, tok = load(res["model_dir"], load_4bit="7B" in plan.get("model", "") or "8B" in plan.get("model", ""))
    _CHAT.update(name=name, model=model, tok=tok, raw=plan.get("task") == "pretrain", system=plan.get("system_prompt"))


def chat(message, history, name):
    if not message.strip():
        return history, ""
    import torch
    with _LOCK:
        _load_for_chat(name)
        model, tok = _CHAT["model"], _CHAT["tok"]
        if _CHAT["raw"]:
            enc = tok(message, return_tensors="pt")
        else:
            msgs = ([{"role": "system", "content": _CHAT["system"]}] if _CHAT["system"] else []) + \
                   [{"role": m["role"], "content": m["content"]} for m in history] + [{"role": "user", "content": message}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9,
                                 repetition_penalty=1.1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}], ""


def unload():
    if _CHAT["model"] is not None:
        _CHAT.update(name=None, model=None, tok=None)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


def export_model(name):
    st = _read_json(os.path.join(ROOT, "runs", name or "", "autopilot.json"), {}) or {}
    res, plan = st.get("results") or {}, st.get("plan") or {}
    if not res.get("model_dir"):
        raise gr.Error("Finish a run first.")
    if plan.get("task") == "pretrain":
        return f"✅ Your model is already a standalone folder: `{res['model_dir']}`"
    unload()
    out = os.path.join("runs", name, "merged")
    p = subprocess.run([sys.executable, "merge_lora.py", "--adapter", res["model_dir"], "--out", out], cwd=ROOT,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"))
    if p.returncode:
        raise gr.Error(p.stderr[-500:])
    return (f"✅ Saved a standalone model to `{out}` — load it with any Hugging Face tool, chat with "
            f"`python chat_llm.py --model {out}`, or convert it to GGUF for Ollama / LM Studio.")


# ---------------------------------------------------------------- UI
def build_tab():
    with gr.TabItem("✨ Quick Start", id="quick"):
        job = gr.State(previous_runs()[0] if previous_runs() else None)
        gr.Markdown("## ✨ Train your own AI in 4 steps\nNo machine-learning knowledge needed — we pick the model and "
                    "all the settings for you, explain why, and show you the result.")
        with gr.Walkthrough(selected=0) as wt:
            # ------------------------------------------------ 1. goal
            with gr.Step("1 · Your goal", id=0):
                goal = gr.Radio(GOAL_TITLES, value=GOAL_TITLES[2], label="What do you want your AI to do?",
                                elem_classes="qs-goal")
                goal_md = gr.Markdown(on_goal(GOAL_TITLES[2])[0])
                with gr.Row():
                    gr.Markdown("")
                    next1 = gr.Button("Next: your data →", variant="primary", scale=0, min_width=200)
                with gr.Accordion("🩺 Check my GPU (recommended once — about 3 minutes)", open=False):
                    gr.Markdown("Runs the same calculation thousands of times under full load and checks the answers never "
                                "change. Overheating laptop GPUs can silently compute wrong results, which ruins training.")
                    health_md = gr.Markdown(last_health())
                    health_btn = gr.Button("🩺 Run GPU check")
                with gr.Accordion("📂 Open a previous Quick Start run", open=False):
                    with gr.Row():
                        prev = gr.Dropdown(previous_runs(), label="Previous runs", allow_custom_value=True, scale=3)
                        open_prev = gr.Button("Open", scale=1)
            # ------------------------------------------------ 2. data
            with gr.Step("2 · Your data", id=1):
                data_hint = gr.Markdown(on_goal(GOAL_TITLES[2])[1])
                persona = gr.Textbox(label="🎭 Describe your assistant's personality and role", lines=3, visible=False,
                                     placeholder="You are Max, a cheerful support agent for AcmeBank. You answer briefly, "
                                                 "warmly, and always suggest a next step.")
                with gr.Group():
                    gr.Markdown("**➕ Add anything.** Paste links, a YouTube video, a GitHub repo, a Hugging Face dataset, a file or "
                                "folder path - or just **name a topic** if you have no data. We sort it out for you.")
                    smart = gr.Textbox(show_label=False, lines=3, container=False,
                                       placeholder="beekeeping basics\nhttps://docs.python.org/3/tutorial/\nhttps://github.com/owner/repo\n"
                                                   "https://youtu.be/...\nyahma/alpaca-cleaned\nC:\\Users\\you\\Documents\\manuals")
                    smart_btn = gr.Button("➕ Add these", variant="secondary")
                    smart_note = gr.Markdown()
                    advice_box = gr.Markdown()
                with gr.Tabs() as data_tabs:
                    with gr.TabItem("📁 Files & Folders", id="tab_files"):
                        uploads = gr.File(label="Drop files or a .zip here", file_count="multiple", type="filepath")
                        folders = gr.Textbox(label="Folder on this PC (one per line)", placeholder="C:\\Users\\you\\Documents\\manuals", lines=2)
                        ocr = gr.Checkbox(label="My PDFs are scanned images (read them with OCR — slower)")

                    with gr.TabItem("🤗 Hugging Face Hub (Internet)", id="tab_hf"):
                        gr.Markdown("#### Stream millions of rows directly from Hugging Face Hub\n"
                                    "Pick a popular dataset preset or enter any public Hugging Face dataset ID.")
                        hf_preset_dropdown = gr.Dropdown(
                            choices=["(custom)"] + [f"{d['name']} — {d['desc']}" for d in POPULAR_HF_DATASETS],
                            value="(custom)",
                            label="⭐ Popular Curated Datasets (1-Click)",
                        )
                        hf_user = gr.Textbox(
                            label="Hugging Face Dataset ID(s) (one per line: id or id|config|split)",
                            placeholder="yahma/alpaca-cleaned\nHuggingFaceH4/no_robots||train\nHuggingFaceFW/fineweb-edu|sample-10BT|train",
                            lines=2
                        )
                        with gr.Row():
                            hf_max_rows = gr.Slider(1000, 100000, value=20000, step=1000, label="Max rows to stream (limit download time)")
                            hf_preview_btn = gr.Button("👀 Preview First 3 Rows", scale=0, min_width=180)
                        hf_preview_df = gr.Dataframe(interactive=False, wrap=True, max_height=200, visible=False, label="Dataset Preview")

                    with gr.TabItem("🌐 Web Crawler & Docs", id="tab_web"):
                        gr.Markdown("#### Scrape documentation sites or crawl internet pages\n"
                                    "TrainerBoi automatically strips navbars, footers, and cookie banners.")
                        urls = gr.Textbox(label="Website pages / Documentation root URLs (one per line)",
                                          placeholder="https://docs.python.org/3/tutorial/\nhttps://fastapi.tiangolo.com/", lines=2)
                        with gr.Row():
                            crawl_depth = gr.Slider(0, 3, value=1, step=1, label="Crawl Depth", info="0 = only these pages, 1 = follow links 1 hop, 2 = 2 hops")
                            max_pages = gr.Number(value=50, precision=0, label="Max Pages to Crawl")
                            sitemap = gr.Checkbox(value=True, label="🗺️ Auto-discover sitemap.xml", info="Rapidly extracts full docs sites")

                    with gr.TabItem("📖 Wikipedia & ArXiv", id="tab_wiki"):
                        gr.Markdown("#### Bulk encyclopedic and academic research data")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Wikipedia Articles & Categories**")
                                wikipedia = gr.Textbox(label="Topics, searches, or categories (one per line)",
                                                       placeholder="Quantum Computing\nsearch: Large Language Models\ncategory: Machine learning", lines=3)
                                wiki_per_query = gr.Slider(5, 100, value=20, step=5, label="Articles per search/category")
                                with gr.Row():
                                    wiki_preset_ai = gr.Button("Add AI Topics", size="sm")
                                    wiki_preset_cs = gr.Button("Add Python/CS", size="sm")
                                    wiki_preset_sci = gr.Button("Add Physics", size="sm")
                            with gr.Column():
                                gr.Markdown("**ArXiv & Academic Papers**")
                                arxiv = gr.Textbox(label="Research paper queries (one per line)",
                                                   placeholder="deep reinforcement learning\ntransformer attention mechanism", lines=3)

                    with gr.TabItem("💻 GitHub Repos", id="tab_gh"):
                        gr.Markdown("#### Ingest codebases and project documentation directly from GitHub")
                        github = gr.Textbox(label="GitHub repo (owner/repo or URL, one per line)",
                                            placeholder="pallets/flask\nhttps://github.com/fastapi/fastapi/tree/main", lines=3)
                        gr.Markdown("<small>Downloads a snapshot zip directly — no git required. Automatically indexes code files and markdown docs.</small>")

                    with gr.TabItem("▶ YouTube Transcripts", id="tab_yt"):
                        gr.Markdown("#### Ingest subtitles from YouTube videos, playlists, or channels")
                        youtube = gr.Textbox(label="YouTube URLs (one per line)",
                                             placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...", lines=3)
                        with gr.Row():
                            max_videos = gr.Slider(1, 100, value=25, step=5, label="Max videos per playlist")
                            transcribe = gr.Checkbox(label="🎙️ Transcribe with Whisper on GPU (for videos without subtitles)")

                    with gr.TabItem("🗂️ Dataset Library", id="tab_lib"):
                        library = gr.Dropdown(library_choices(GOAL_TITLES[2]), value="(none)",
                                              label="Reuse a dataset previously built in your Library")

                with gr.Row():
                    back2 = gr.Button("← Back", scale=0, min_width=120)
                    gr.Markdown("")
                    next2 = gr.Button("Next: time & plan →", variant="primary", scale=0, min_width=200)
            # ------------------------------------------------ 3. time + plan
            with gr.Step("3 · Time & plan", id=2):
                effort = gr.Radio(EFFORT_TITLES, value=EFFORT_TITLES[1], label="How much time do you have?")
                plan_md = gr.Markdown()
                with gr.Accordion("🤔 Why these settings? (the expert's reasoning)", open=False):
                    why_md = gr.Markdown()
                with gr.Accordion("Optional: choose the base model yourself", open=False):
                    model = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], allow_custom_value=True,
                                        label="Base model", info="Leave on automatic unless you know you want a specific one.")
                    name = gr.Textbox(label="Name this run (optional)")
                with gr.Row():
                    back3 = gr.Button("← Back", scale=0, min_width=120)
                    gr.Markdown("")
                    go = gr.Button("🚀 Start training", variant="primary", scale=0, min_width=220)
            # ------------------------------------------------ 4. watch + results
            with gr.Step("4 · Watch & try it", id=3):
                timeline = gr.HTML()
                loss = gr.LinePlot(x="step", y="loss", color="series", title="Loss — how wrong the model is (should go down)",
                                   height=220, visible=False)
                peek = gr.Markdown()
                with gr.Row():
                    stop_btn = gr.Button("■ Stop", variant="stop", scale=0, min_width=120)
                    new_btn = gr.Button("➕ Start another", scale=0, min_width=160)
                results = gr.HTML()
                with gr.Group(visible=False) as done_box:
                    gr.Markdown("### 💬 Chat with your trained model")
                    chatbot = gr.Chatbot(height=360, label="Your model")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Ask your model anything…", show_label=False, scale=5)
                        send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        clear = gr.Button("Clear chat")
                        unload_btn = gr.Button("Free GPU memory")
                        export_btn = gr.Button("💾 Save as a standalone model")
                    export_md = gr.Markdown()
                with gr.Accordion("🖨️ Technical log", open=False):
                    console = gr.Textbox(lines=14, max_lines=14, autoscroll=True, show_label=False)
                gr.Markdown("<small>Want every detail? The 📡 Live LLM Training tab shows this run's full metrics, "
                            "and 🗂️ Dataset Library lets you inspect the data it trained on.</small>")

        # ---------------- wiring
        plan_inputs = [
            goal, effort, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap,
            hf_user, hf_max_rows, wikipedia, wiki_per_query, arxiv, github, youtube, max_videos,
            transcribe, library, persona, model
        ]
        goal.change(on_goal, goal, [goal_md, data_hint, persona, library])
        next1.click(lambda: gr.Walkthrough(selected=1), None, wt)
        health_btn.click(gpu_check, None, health_md)
        back2.click(lambda: gr.Walkthrough(selected=0), None, wt)
        next2.click(preview_plan, plan_inputs, [plan_md, why_md]).then(lambda: gr.Walkthrough(selected=2), None, wt)
        back3.click(lambda: gr.Walkthrough(selected=1), None, wt)
        for comp in (effort, model):
            comp.change(preview_plan, plan_inputs, [plan_md, why_md])

        # Preset & preview callbacks
        hf_preset_dropdown.change(on_hf_preset_selected, [hf_preset_dropdown, hf_user], [hf_user, hf_max_rows])
        hf_preview_btn.click(on_hf_preview_click, hf_user, hf_preview_df)
        wiki_preset_ai.click(lambda cur: add_wiki_preset("AI & Machine Learning", cur), wikipedia, wikipedia)
        wiki_preset_cs.click(lambda cur: add_wiki_preset("Computer Science & Python", cur), wikipedia, wikipedia)
        wiki_preset_sci.click(lambda cur: add_wiki_preset("Science & Physics", cur), wikipedia, wikipedia)

        def _goal_update(key):
            return gr.update(value=autopilot.GOALS[key]["title"]) if key else gr.skip()

        def smart_add(text, f_folders, f_urls, f_hf, f_wiki, f_arxiv, f_gh, f_yt):
            fields, note = data_intake.route(text)
            if not any(fields.values()):
                raise gr.Error("Paste at least one link, path or topic first.")
            merge = lambda cur, new: "\n".join(dict.fromkeys(_lines(cur) + new))
            key, advice = None, ""
            if fields["folders"]:
                key, _, advice = data_intake.inspect(fields["folders"])
            if not key:
                if fields["github"]:
                    key = "code"
                elif fields["urls"] or fields["youtube"] or fields["wikipedia"] or fields["arxiv"]:
                    key = "docs"
            if key:
                note += f"\n\n➡️ Goal set to **{autopilot.GOALS[key]['title']}** (change it in step 1 if you prefer)."
            return (merge(f_folders, fields["folders"]), merge(f_urls, fields["urls"]), merge(f_hf, fields["hf"]),
                    merge(f_wiki, fields["wikipedia"]), merge(f_arxiv, fields["arxiv"]), merge(f_gh, fields["github"]),
                    merge(f_yt, fields["youtube"]), note, advice, "", _goal_update(key))

        smart_ins = [smart, folders, urls, hf_user, wikipedia, arxiv, github, youtube]
        smart_outs = [folders, urls, hf_user, wikipedia, arxiv, github, youtube, smart_note, advice_box, smart, goal]
        smart_btn.click(smart_add, smart_ins, smart_outs)

        def inspect_uploads(files):
            if not files:
                return "", gr.skip()
            key, _, md = data_intake.inspect(list(files))
            return md, _goal_update(key)

        uploads.change(inspect_uploads, uploads, [advice_box, goal])
        start_inputs = [
            goal, effort, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap,
            hf_user, hf_max_rows, wikipedia, wiki_per_query, arxiv, github, youtube, max_videos,
            transcribe, library, persona, model, name
        ]
        go.click(start, start_inputs, [job, wt, prev])
        open_prev.click(lambda r: (r, gr.Walkthrough(selected=3)), prev, [job, wt])
        prev.focus(lambda v: gr.update(choices=previous_runs(), value=v), prev, prev)

        outs = [timeline, loss, peek, done_box, results, console]
        gr.Timer(2).tick(refresh, job, outs, show_progress="hidden")
        job.change(refresh, job, outs, show_progress="hidden")
        stop_btn.click(stop, job, None)
        new_btn.click(lambda: (gr.Walkthrough(selected=0), []), None, [wt, chatbot])
        send.click(chat, [msg, chatbot, job], [chatbot, msg])
        msg.submit(chat, [msg, chatbot, job], [chatbot, msg])
        clear.click(lambda: [], None, chatbot)
        unload_btn.click(lambda: (unload(), gr.Info("GPU memory freed"))[0], None, None)
        export_btn.click(export_model, job, export_md)
    return {"job": job}

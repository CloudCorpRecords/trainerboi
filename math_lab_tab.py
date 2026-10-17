"""
math_lab_tab.py - UI for Math Lab: verified self-improvement for maths with an unlimited self-checking problem generator.
"""

import os
import re
import sys
import json
import glob
import time
import subprocess
import pandas as pd
import gradio as gr

import math_gym

ROOT = os.path.dirname(os.path.abspath(__file__))
PROCS = {}
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
EMPTY_BAR = pd.DataFrame({"skill": [], "accuracy": [], "when": []})
EMPTY_LINE = pd.DataFrame({"round": [], "accuracy": []})


def _status(name):
    try:
        with open(os.path.join(ROOT, "runs", name, "math", "status.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_runs():
    return [os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "math", "status.json")), key=os.path.getmtime, reverse=True)]


def _alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return str(pid) in out
    except Exception:
        return False


def start(model, rounds, problems, samples, max_level, name):
    for r in list_runs():
        st = _status(r) or {}
        if st.get("state") == "running" and _alive(st.get("pid")):
            raise gr.Error(f"Math Lab run '{r}' is still going. Stop it first.")
    name = re.sub(r"[^\w.-]+", "-", (name or "").strip()) or f"math-{time.strftime('%m%d-%H%M')}"
    if os.path.exists(os.path.join(ROOT, "runs", name, "math", "status.json")):
        raise gr.Error(f"A run named '{name}' exists - pick another name.")
    cmd = [sys.executable, "-u", "math_lab.py", "--name", name, "--model", model, "--rounds", str(int(rounds)),
           "--problems", str(int(problems)), "--samples", str(int(samples)), "--max_level", str(int(max_level))]
    os.makedirs(os.path.join(ROOT, "runs", name, "math"), exist_ok=True)
    log = open(os.path.join(ROOT, "runs", name, "math", "console.log"), "w", encoding="utf-8")
    PROCS[name] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore"),
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gr.Info(f"Math Lab '{name}' started.")
    return gr.update(choices=list_runs(), value=name)


def stop(name):
    st = _status(name) if name else None
    if not st:
        return
    import device_utils
    pids = {st.get("pid")} | ({PROCS[name].pid} if name in PROCS else set())
    for pid in sorted(filter(None, pids), key=lambda x: x != (PROCS[name].pid if name in PROCS else None)):  # supervisor first
        device_utils.kill_pid(pid)
    st.update(state="stopped", phase="Stopped by you")
    with open(os.path.join(ROOT, "runs", name, "math", "status.json"), "w", encoding="utf-8") as f:
        json.dump(st, f)
    gr.Warning("Stopped. Finished rounds are kept in the run folder.")


def _pct(x):
    return f"{100 * x:.0f}%"


def refresh(name):
    st = _status(name) if name else None
    if not st:
        return "<i>No run selected. Start one on the left.</i>", EMPTY_BAR, EMPTY_LINE, "", ""
    state = st.get("state")
    if state == "running" and not _alive(st.get("pid")):
        state = "crashed"
    rounds = st.get("rounds", [])
    base = rounds[0]["overall"] if rounds else None
    cur = rounds[-1]["overall"] if rounds else None
    delta = f" ({'+' if cur >= base else ''}{100 * (cur - base):.0f} points)" if rounds and len(rounds) > 1 else ""
    color = {"running": "#2563eb", "done": "#16a34a", "error": "#dc2626", "crashed": "#dc2626", "stopped": "#6b7280"}.get(state, "#6b7280")
    head = (f"<div style='display:flex;gap:24px;flex-wrap:wrap;align-items:center'>"
            f"<span class='tb-pill' style='background:{color}'>{state}</span><b>{st.get('phase', '')}</b></div>"
            f"<div class='tb-cards' style='margin-top:10px'>"
            f"<div class='tb-card'><div class='lbl'>Exam score</div><div class='val'>{_pct(cur) if cur is not None else '-'}</div>"
            f"<div class='sub'>{'started at ' + _pct(base) + delta if base is not None else ''}</div></div>"
            f"<div class='tb-card'><div class='lbl'>Rounds done</div><div class='val'>{max(0, len(rounds) - 1)}/{st.get('rounds_total', '?')}</div></div>"
            f"<div class='tb-card'><div class='lbl'>Verified solutions learned</div><div class='val'>{st.get('kept', 0):,}</div>"
            f"<div class='sub'>only attempts a checker proved correct</div></div>"
            f"<div class='tb-card'><div class='lbl'>Answers in \\boxed{{}}</div><div class='val'>{_pct(rounds[-1]['boxed_pct']) if rounds else '-'}</div></div></div>")
    if st.get("error"):
        import friendly_errors
        head += friendly_errors.html(st["error"]) + f"<details><summary>Technical details</summary><pre>{st['error'][-1800:]}</pre></details>"
    rows = []
    if rounds:
        for s, a in rounds[0]["skills"].items():
            rows.append({"skill": s, "accuracy": 100 * a, "when": "before"})
        for s, a in rounds[-1]["skills"].items():
            if len(rounds) > 1:
                rows.append({"skill": s, "accuracy": 100 * a, "when": "now"})
    line = pd.DataFrame([{"round": r["round"], "accuracy": 100 * r["overall"]} for r in rounds]) if rounds else EMPTY_LINE
    samples = ""
    lat = st.get("latest") or st.get("baseline")
    if lat:
        samples = "\n\n".join(f"{'✅' if s['ok'] else '❌'} **{s['q']}**  \nexpected `{s['gold']}`  \n<small>{s['out'][-260:]}</small>"
                              for s in lat.get("samples", []))
    final = st.get("final_model") or st.get("current_model")
    info = (f"**Model so far:** `{final}`  \nUse it in 💬 Chat & test (paste the folder path), or 📦 Export & share." if final else "")
    return head, (pd.DataFrame(rows) if rows else EMPTY_BAR), line, samples, info


def build_tab():
    with gr.TabItem("🧮 Math Lab", id="math"):
        gr.Markdown(
            "## Teach a small model to do maths - with no dataset at all\n"
            "Every problem is **generated with its exact answer**, so the model can practise endlessly and a checker knows instantly whether it was right. "
            "Each round it attempts new problems aimed at **its weakest skills**, we keep only the attempts **proven correct**, "
            "and it learns from those. It improves from its own verified successes - on one laptop GPU, no teacher model, no human labels.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                model = gr.Dropdown(MODELS, value=MODELS[0], allow_custom_value=True, label="Model to improve",
                                    info="Bigger starts better but each round takes longer.")
                rounds = gr.Slider(1, 8, value=3, step=1, label="Rounds of practice")
                problems = gr.Slider(100, 3000, value=400, step=100, label="New problems per round")
                samples = gr.Slider(2, 8, value=4, step=1, label="Attempts per problem",
                                    info="More attempts find more correct solutions on hard problems.")
                max_level = gr.Slider(1, 5, value=3, step=1, label="Hardest level", info="1 = easy arithmetic, 5 = multi-step, big numbers")
                name = gr.Textbox(label="Run name (optional)")
                with gr.Row():
                    go = gr.Button("▶ Start practising", variant="primary")
                    stop_btn = gr.Button("■ Stop", variant="stop")
                with gr.Accordion("What does it practise?", open=False):
                    gr.Markdown(", ".join(math_gym.SKILLS) + ".  \nAll answers are checked robustly (\\boxed{}, fractions, decimals, commas, %).")
                gr.Markdown("**Prefer reinforcement learning?** Use *Live training → GRPO* with dataset `math_gym` and reward `math` "
                            "(run `python train_llm_grpo.py --dataset math_gym --reward math`). For your own subject write a "
                            "checker file and use `--reward custom --reward_file my_checker.py`.")
            with gr.Column(scale=3):
                run = gr.Dropdown(list_runs(), value=(list_runs() or [None])[0], label="Watching run", allow_custom_value=True)
                head = gr.HTML()
                with gr.Row():
                    bar = gr.BarPlot(EMPTY_BAR, x="skill", y="accuracy", color="when", title="Accuracy by skill (%)", height=330,
                                     x_label_angle=-40, y_lim=[0, 100], x_title=" ")
                    line = gr.LinePlot(EMPTY_LINE, x="round", y="accuracy", title="Exam score after each round (%)", height=330,
                                       y_lim=[0, 100])
                info = gr.Markdown()
                with gr.Accordion("Example answers from the latest exam", open=False):
                    samples_md = gr.Markdown()
        go.click(start, [model, rounds, problems, samples, max_level, name], run)
        stop_btn.click(stop, run, None)
        outs = [head, bar, line, samples_md, info]
        gr.Timer(3).tick(refresh, run, outs, show_progress="hidden")
        run.change(refresh, run, outs, show_progress="hidden")
        run.focus(lambda v: gr.update(choices=list_runs(), value=v), run, run)

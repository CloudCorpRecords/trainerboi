"""
home_tab.py - The landing screen: what do you want to do? Big, friendly entry points.
"""

import os
import json
import glob
import gradio as gr

ROOT = os.path.dirname(os.path.abspath(__file__))


def recent_runs_html():
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "live", "status.json")), key=os.path.getmtime, reverse=True)[:5]:
        try:
            s = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        name = p.split(os.sep)[-3]
        st = s.get("state", "?")
        loss = (s.get("latest") or {}).get("loss")
        rows.append(f"<tr><td><b>{name}</b></td><td>{s.get('task','')}</td><td>{st}</td>"
                    f"<td>{'%.3f' % loss if isinstance(loss, (int, float)) else ''}</td></tr>")
    if not rows:
        return "<p class='tb-muted'>No training runs yet - your first one takes about 10 minutes.</p>"
    return "<table class='tb-table'><tr><th>Run</th><th>Type</th><th>State</th><th>Loss</th></tr>" + "".join(rows) + "</table>"


def build_tab():
    with gr.TabItem("🏠 Home", id="home"):
        gr.HTML("<div class='tb-hero'><h1>Train your own AI.</h1>"
                "<p>No code, no cloud, no PhD. Pick a goal, bring any data - or none - and TrainerBoi handles the rest on your own computer.</p></div>")
        with gr.Row(elem_classes="tb-home-cards"):
            with gr.Column():
                b_guided = gr.Button("✨  Guided training\nPick a goal. We choose the model and settings.", variant="primary", elem_classes="tb-big")
            with gr.Column():
                b_math = gr.Button("🧮  Math Lab\nTeach a model maths with no data at all.", elem_classes="tb-big")
            with gr.Column():
                b_data = gr.Button("📚  Get data\nUpload anything, pull from the web, or start with none.", elem_classes="tb-big")
            with gr.Column():
                b_chat = gr.Button("🏆  My models\nSee everything you trained, chat and share.", elem_classes="tb-big")
        with gr.Row():
            b_methods = gr.Button("🧭 Not sure which method? Open the guide", size="sm", scale=0)
            b_manual = gr.Button("🛠️ Manual training (all settings)", size="sm", scale=0)
        with gr.Accordion("🗺️ Where is everything?", open=False):
            gr.Markdown(
                "| Sidebar | What's inside |\n|---|---|\n"
                "| ✨ **Guided training** | Pick a goal → add data → choose time → watch, test and chat. The easiest path. |\n"
                "| 🧮 **Math Lab** | Teach a model maths with no data: unlimited self-checking problems, learns from its verified successes. |\n"
                "| 📚 **Data** | *Build* (any files, web, YouTube, GitHub, Hugging Face) · *Library* (explore, edit, **Data doctor**, mix, export) · *AI generator* |\n"
                "| 🛠️ **Manual training** | *Live training* (every setting + **recommended settings**) · *Methods* (SFT, LoRA, DPO, ORPO, KTO, RL…) · *Auto-tune* |\n"
                "| 🏆 **My models** · 💬 **Chat & share** | *My models* (every model you trained, chat, zip, publish, public link) · *Chat* · *Compare* · *Export* (GGUF, Ollama) |\n"
                "| 🧰 **More tools** | *AutoTrain* (tables, images, text) · *Classic trainers* · *RAG* · *Audio* |\n"
                "| 📊 **Hardware** · 📖 **Guide** | GPU stats, health check, benchmarks · plain-English ML guide |")
        gr.Markdown("### Recent runs")
        recent = gr.HTML(recent_runs_html())
        refresh = gr.Button("Refresh", size="sm")
        refresh.click(recent_runs_html, outputs=recent)
    return {"guided": b_guided, "math": b_math, "data": b_data, "methods": b_methods, "manual": b_manual, "chat": b_chat,
            "recent": recent}

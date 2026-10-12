"""
methods_tab.py - "Training Methods" explorer: every way to train a model explained in plain English,
with a quick "help me choose" and a one-click jump into the Live Training launcher.
"""

import gradio as gr
import methods as M
import data_intake


def _card(m):
    return (f"<div class='tb-method'><div class='tb-method-h'><span class='ic'>{m['icon']}</span>"
            f"<div><b>{m['name']}</b><span class='lvl'>{m['level']}</span></div></div>"
            f"<p class='tg'>{m['tagline']}</p><p class='an'><i>{m['analogy']}</i></p>"
            f"<table><tr><td>Needs</td><td>{m['data']}</td></tr><tr><td>GPU</td><td>{m['gpu']}</td></tr>"
            f"<tr><td>Time</td><td>{m['time']}</td></tr><tr><td>Use when</td><td>{m['use_when']}</td></tr>"
            f"<tr><td>Skip when</td><td>{m['avoid_when']}</td></tr></table></div>")


def gallery_html():
    return "<div class='tb-methods'>" + "".join(_card(m) for m in M.METHODS) + "</div>"


def advise(have, want):
    if not have or not want:
        return "Pick both answers and I'll suggest a method.", None
    mid, why, alts = M.recommend(M.HAVE[have], M.WANT[want])
    m = M.BY_ID[mid]
    alt = ", ".join(M.BY_ID[a]["name"] for a in alts)
    md = (f"### {m['icon']} Recommended: {m['name']}\n{why}\n\n"
          f"**Needs:** {m['data']}  \n**GPU:** {m['gpu']} · **Time:** {m['time']}  \n**Also consider:** {alt}")
    return md, mid


def look_at(files):
    if not files:
        return "", None
    goal, method, md = data_intake.inspect(list(files))
    if method:
        m = M.BY_ID[method]
        md += f"\n\n**Recommended method:** {m['icon']} {m['name']} - {m['tagline']}"
    return md, method


def build_tab():
    picked = gr.State(None)
    with gr.TabItem("🧭 Methods", id="methods"):
        gr.Markdown("## Every way to train a model, in plain English\n"
                    "Not sure which method fits? Answer two questions. Everything below runs on your own GPU.")
        with gr.Row():
            have = gr.Dropdown(list(M.HAVE), label="What do you have?", scale=1)
            want = gr.Dropdown(list(M.WANT), label="What do you want the model to do?", scale=1)
        advice = gr.Markdown("Pick both answers and I'll suggest a method.")
        with gr.Row():
            try_guided = gr.Button("✨ Do it for me (Guided Training)", variant="primary")
            try_manual = gr.Button("🛠️ Open it in the training launcher")
        with gr.Accordion("📂 Or drop your data and let me look at it", open=False):
            drop = gr.File(file_count="multiple", type="filepath", label="Your files (any type)")
            look = gr.Markdown()
            drop.change(look_at, drop, [look, picked])
        have.change(advise, [have, want], [advice, picked])
        want.change(advise, [have, want], [advice, picked])
        gr.HTML(gallery_html())
    return {"picked": picked, "guided_btn": try_guided, "manual_btn": try_manual}

"""
models_tab.py - "My models": every model you've trained in one place, with chat, one-click sharing and a model card.
"""

import os
import re
import sys
import time
import json
import shutil
import subprocess
from threading import Thread

import pandas as pd
import gradio as gr

import models_registry as R

ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["Model", "Type", "Result", "Size", "Trained", "Status"]
_CHAT = {"id": None, "model": None, "tok": None}
_SHARE = {"proc": None, "id": None, "log": None, "port": 7890}


def _ago(ts):
    s = time.time() - ts
    return ("just now" if s < 90 else f"{s / 60:.0f} min ago" if s < 5400 else f"{s / 3600:.0f} h ago" if s < 172800
            else time.strftime("%b %d", time.localtime(ts)))


def _size(mb):
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def table():
    recs = R.scan()
    rows = [[r["title"], R.TASK_LABEL.get(r["kind"], r["kind"]) + (" · adapter" if r["adapter"] else ""), r.get("metric") or "-",
             _size(r["size_mb"]), _ago(r["created"]), {"done": "✅ finished", "stopped": "⏹ stopped early", "error": "❌ failed"}.get(r.get("status"), r.get("status") or "")]
            for r in recs]
    return pd.DataFrame(rows or [["(no trained models yet - try Guided training)", "", "", "", "", ""]], columns=COLS), [r["id"] for r in recs]


def card_html(rec):
    if not rec:
        return "<i>Pick a model on the left.</i>"
    chips = [R.TASK_LABEL.get(rec["kind"], rec["kind"]) + (" (LoRA adapter)" if rec["adapter"] else ""), _size(rec["size_mb"])]
    if rec.get("base"):
        chips.append(f"base: {rec['base'].split('/')[-1]}")
    if rec.get("metric"):
        chips.append(rec["metric"])
    warn = ""
    if rec.get("status") in ("stopped", "error"):
        warn = "<div class='tb-insights' style='border-color:#f59e0b'>⚠️ This run didn't finish, so the model may be poor.</div>"
    return (f"<h3 style='margin:0 0 6px'>{rec['title']}</h3>"
            + "".join(f"<span class='tb-pill' style='background:#475569;margin-right:6px'>{c}</span>" for c in chips)
            + f"<p style='opacity:.7;margin:8px 0 0'><small>{rec['path']}</small></p>{warn}")


def _rec(mid):
    r = R.by_id(mid) if mid else None
    if not r:
        raise gr.Error("Pick a model in the table first.")
    return r


# ------------------------------------------------------------------------------------------------ select
def select(mid):
    r = R.by_id(mid) if mid else None
    if not r:
        return card_html(None), "", "", "", gr.update(visible=False), [], ""
    return (card_html(r), r["title"], r.get("notes", ""), R.model_card(r, with_license_lookup=False),
            gr.update(visible=r["adapter"]), [], "")


def save_meta(mid, title, notes):
    r = _rec(mid)
    R.save_meta(r, title, notes)
    gr.Info("Saved.")
    return table()[0]


# ------------------------------------------------------------------------------------------------ chat
def _load(rec):
    if _CHAT["id"] == rec["id"] and _CHAT["model"] is not None:
        return
    unload()
    import chat_llm
    import llm_data
    model, tok = chat_llm.load(rec["path"])
    llm_data.ensure_chat_template(tok) if rec["kind"] != "pretrain" else None
    _CHAT.update(id=rec["id"], model=model, tok=tok)


def unload():
    import device_utils
    _CHAT.update(id=None, model=None, tok=None)
    device_utils.empty_cache()
    return "Model unloaded - GPU memory is free."


def chat(msg, history, mid):
    rec = _rec(mid)
    if not rec.get("chat_ok", True):
        raise gr.Error("This is an embedding model - it finds similar text instead of chatting. Use it in More tools → RAG.")
    msg = (msg or "").strip()
    if not msg:
        yield history, ""
        return
    history = list(history or [])
    history += [{"role": "user", "content": msg}, {"role": "assistant", "content": "…loading the model (first message only)"}]
    yield history, ""
    from transformers import TextIteratorStreamer
    _load(rec)
    model, tok = _CHAT["model"], _CHAT["tok"]
    raw = rec["kind"] == "pretrain" or not getattr(tok, "chat_template", None)
    if raw:
        enc = tok(msg, return_tensors="pt").to(model.device)
    else:
        msgs = [{"role": m["role"], "content": m["content"]} for m in history[:-1]]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    stream = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(**enc, streamer=stream, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9,
                                              pad_token_id=tok.pad_token_id or tok.eos_token_id)).start()
    out = ""
    for chunk in stream:
        out += chunk
        history[-1]["content"] = out
        yield history, ""


# ------------------------------------------------------------------------------------------------ share
def make_zip(mid, kind):
    rec = _rec(mid)
    which = "adapter" if (rec["adapter"] and str(kind).startswith("Small")) else "full"
    yield gr.update(), "⏳ Preparing the package…"
    try:
        pkg = R.build_package(rec, which)
        zp = R.make_zip(pkg)
        shutil.rmtree(pkg, ignore_errors=True)  # keep only the zip
    except Exception as e:
        raise gr.Error(str(e))
    yield zp, (f"✅ **{os.path.basename(zp)}** ready ({os.path.getsize(zp) / 1024 ** 2:.0f} MB). Send this file to anyone. "
               f"It includes the model, a model card (README.md) and `use_this_model.py` to chat with it.")


def publish(mid, kind, repo, private, token):
    rec = _rec(mid)
    repo = (repo or "").strip() or re.sub(r"[^\w.-]+", "-", rec["title"])
    which = "adapter" if (rec["adapter"] and str(kind).startswith("Small")) else "full"
    yield "⏳ Preparing and uploading… (large models can take several minutes)"
    try:
        pkg = R.build_package(rec, which)
        url = R.publish_to_hub(pkg, repo, private, (token or "").strip() or None)
    except Exception as e:
        yield f"❌ {e}"
        return
    yield (f"✅ Published: [{url}]({url}) — {'private (only you can see it; change it on the Hub to share)' if private else 'public'}.")


def start_share(mid):
    rec = _rec(mid)
    if _SHARE["proc"] and _SHARE["proc"].poll() is None:
        raise gr.Error("A public link is already running. Stop it first.")
    if not rec.get("chat_ok", True):
        raise gr.Error("Only chat models can be shared as a live page.")
    unload()
    log_path = os.path.join(ROOT, "exports", "share_chat.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    cmd = [sys.executable, "-u", "share_chat.py", "--model", rec["path"], "--public", "--title", rec["title"], "--port", str(_SHARE["port"])]
    _SHARE.update(proc=subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore"),
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)), id=rec["id"], log=log_path)
    gr.Info("Starting the shared chat page…")


def stop_share():
    import device_utils
    device_utils.kill_tree(_SHARE["proc"])
    _SHARE.update(proc=None)
    return share_status()


def share_status():
    p = _SHARE["proc"]
    if not p:
        return "No public link is running."
    if p.poll() is not None:
        tail = open(_SHARE["log"], encoding="utf-8", errors="replace").read()[-600:] if _SHARE["log"] else ""
        return f"⏹ The shared page stopped.\n\n```\n{tail}\n```" if tail.strip() else "⏹ The shared page stopped."
    txt = open(_SHARE["log"], encoding="utf-8", errors="replace").read() if _SHARE["log"] else ""
    m = re.search(r"https://[a-z0-9-]+\.gradio\.live", txt)
    if m:
        return (f"🟢 **Live:** [{m.group(0)}]({m.group(0)})  \nAnyone with this link can chat with the model while your computer is on. "
                f"It goes through Gradio's relay servers and expires in about a week. Press **Stop** when you're done.")
    return "⏳ Loading the model and creating the link (about a minute)…"


def delete_model(mid, confirm):
    rec = _rec(mid)
    if not confirm:
        raise gr.Error("Tick the confirmation box first.")
    if _CHAT["id"] == rec["id"]:
        unload()
    run_dir = os.path.join(R.RUNS, rec["run"].split("/")[0]) if not rec["run"].startswith("exported/") else rec["path"]
    shutil.rmtree(run_dir, ignore_errors=True)
    gr.Info(f"Deleted {rec['title']}.")
    return table()


# ------------------------------------------------------------------------------------------------ UI
def build_tab():
    with gr.TabItem("🏆 My models", id="models"):
        sel = gr.State(None)
        ids = gr.State(table()[1])
        gr.Markdown("### Every model you've trained\nClick a model to chat with it, send it to someone, or publish it.")
        with gr.Row():
            with gr.Column(scale=2, min_width=360):
                refresh = gr.Button("🔄 Refresh", size="sm")
                tbl = gr.Dataframe(table()[0], interactive=False, wrap=True, max_height=560, label="Your models — click a row")
            with gr.Column(scale=3):
                card = gr.HTML(card_html(None))
                with gr.Tabs():
                    with gr.TabItem("💬 Try it"):
                        bot = gr.Chatbot(height=330, show_label=False)
                        with gr.Row():
                            msg = gr.Textbox(placeholder="Ask your model anything…", show_label=False, scale=5, container=False)
                            send = gr.Button("Send", variant="primary", scale=1)
                        with gr.Row():
                            clear = gr.Button("Clear chat", size="sm")
                            free = gr.Button("Free GPU memory", size="sm")
                        free_md = gr.Markdown()
                    with gr.TabItem("🚀 Share"):
                        pack_kind = gr.Radio(["Small adapter (tiny file; the other person also needs the base model)",
                                              "Full model (bigger, works on its own)"], value="Full model (bigger, works on its own)",
                                             label="What to send", visible=False)
                        with gr.Accordion("📦 Send as a file (zip)", open=True):
                            gr.Markdown("Creates one `.zip` with the model, a model card and a chat script. Works offline; share it any way you like.")
                            zip_btn = gr.Button("📦 Create zip", variant="primary")
                            zip_file = gr.File(label="Your zip")
                            zip_md = gr.Markdown()
                        with gr.Accordion("🤗 Publish on Hugging Face", open=False):
                            _hf_login = "hf auth login" if os.name != "nt" else ".venv\\Scripts\\hf.exe auth login"
                            gr.Markdown("Free hosting with a shareable link. Needs a Hugging Face account and an access token with **write** permission "
                                        f"(or run `{_hf_login}` once and leave the token empty). "
                                        "Check the base model's licence first - the model card shows it.")
                            with gr.Row():
                                repo = gr.Textbox(label="Name on the Hub", placeholder="my-model  (or  username/my-model)")
                                private = gr.Checkbox(value=True, label="Private (only you)")
                            token = gr.Textbox(label="Access token (optional)", type="password")
                            pub_btn = gr.Button("🤗 Publish")
                            pub_md = gr.Markdown()
                        with gr.Accordion("🔗 Temporary public chat link", open=False):
                            gr.Markdown("Let friends chat with the model in their browser - no install. It runs on **your** computer through "
                                        "Gradio's relay, so it only works while the app is open, and anyone with the link can use it. Stop it when done.")
                            with gr.Row():
                                share_btn = gr.Button("🔗 Create link", variant="primary")
                                stop_btn = gr.Button("■ Stop sharing")
                            share_md = gr.Markdown(share_status())
                        with gr.Accordion("🦙 Use it in Ollama / LM Studio", open=False):
                            gr.Markdown("The *Full model* zip contains a ready `Modelfile`. For one-click Ollama import use **Use your model → Export**.")
                    with gr.TabItem("📝 Name, notes & card"):
                        title = gr.Textbox(label="Name")
                        notes = gr.Textbox(label="Notes (shown on the model card)", lines=2)
                        save = gr.Button("Save", size="sm")
                        card_md = gr.Code(label="Model card (README.md that ships with the model)", language="markdown", lines=14)
                    with gr.TabItem("🗑️ Delete"):
                        gr.Markdown("Deletes this run's folder (model, checkpoints and logs). **This can't be undone.**")
                        confirm = gr.Checkbox(label="Yes, delete this model permanently")
                        del_btn = gr.Button("Delete", variant="stop")

        def on_row(evt: gr.SelectData, id_list):
            i = evt.index[0]
            return id_list[i] if id_list and i < len(id_list) else None

        def refresh_all():
            df, id_list = table()
            return df, id_list

        outs = [card, title, notes, card_md, pack_kind, bot, free_md]
        tbl.select(on_row, ids, sel).then(select, sel, outs)
        refresh.click(refresh_all, None, [tbl, ids])
        send.click(chat, [msg, bot, sel], [bot, msg])
        msg.submit(chat, [msg, bot, sel], [bot, msg])
        clear.click(lambda: [], None, bot)
        free.click(unload, None, free_md)
        zip_btn.click(make_zip, [sel, pack_kind], [zip_file, zip_md])
        pub_btn.click(publish, [sel, pack_kind, repo, private, token], pub_md)
        share_btn.click(start_share, sel, None)
        stop_btn.click(stop_share, None, share_md)
        gr.Timer(3).tick(share_status, None, share_md, show_progress="hidden")
        save.click(save_meta, [sel, title, notes], tbl)
        del_btn.click(delete_model, [sel, confirm], [tbl, ids]).then(lambda: (card_html(None), None), None, [card, sel])
    return {"table": tbl, "ids": ids, "refresh": refresh, "selected": sel}

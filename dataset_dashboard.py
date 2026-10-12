"""
dataset_dashboard.py - "Dataset Builder" tab for the TrainerBoi Studio.

Point it at files, folders, zips, web pages, Hugging Face datasets (with a built-in browser), GitHub repos,
Wikipedia, YouTube and audio/video. It runs dataset_builder.py in a background process and shows live progress,
what was kept / dropped and why, token statistics and a preview. Builds can be stopped and resumed.
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

from live_dashboard import ROOT, _card, _fmt_dur, _tail, _read_json, _pid_alive
from dataset_sources import POPULAR_HF_DATASETS, WIKIPEDIA_PRESETS

DATA_DIR = "data"
PROCS = {}

GEN_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
              "unsloth/Llama-3.2-3B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]
WHISPER_MODELS = ["openai/whisper-large-v3-turbo", "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"]

FORMAT_HELP = {
    "auto": "Decide from the data: chat/instruction rows → SFT, chosen/rejected → DPO, raw text → pretraining.",
    "sft": "Chat examples ({messages}). Raw documents need **Synthetic Q&A** turned on to become chat data.",
    "pretrain": "Plain text chunks ({text}) for pretraining / continued pretraining. Everything becomes text.",
    "dpo": "Preference pairs ({prompt, chosen, rejected}). Only rows that already have chosen/rejected are used.",
}

DROP_TIPS = {
    "raw text (enable synthetic Q&A to use)": "Your documents are raw text. Turn on **Synthetic Q&A** to convert them into chat examples, or choose format **pretrain**.",
    "no text layer (needs OCR)": "Some PDFs are scanned images with no text layer. Tick **OCR** (Files tab) and rebuild to read them.",
    "near duplicate": "Near-identical content was removed (≥85% overlap) so the model doesn't over-learn repeated text.",
    "exact duplicate": "Identical content appeared more than once — only one copy is kept.",
    "too short": "Very short items carry little signal. Lower *Min characters* in Advanced if they matter.",
    "synthetic: unparseable output": "The generator sometimes doesn't return valid JSON. A bigger generator model (3B+) helps.",
    "refusal / AI boilerplate": "Responses like 'As an AI language model…' were removed — they teach the model to refuse.",
    "YouTube: no transcript (enable transcription)": "Some videos have no captions. Tick **Transcribe** to run Whisper on their audio.",
}


def list_datasets():
    paths = glob.glob(os.path.join(ROOT, DATA_DIR, "*", "build"))
    paths.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(os.path.dirname(p), ROOT) for p in paths]


def list_trainable():
    """Finished datasets (jsonl file or parquet folder) the training tab can use, newest first."""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, DATA_DIR, "*", "stats.json")), key=os.path.getmtime, reverse=True):
        s = _read_json(p, {}) or {}
        if s.get("train_path"):
            out.append(s["train_path"])
    return out


def _lines(text):
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def on_hf_preset_add(choice, current_text):
    if not choice or choice == "(custom)":
        return current_text
    for d in POPULAR_HF_DATASETS:
        if choice.startswith(d["name"]):
            lines = _lines(current_text)
            if d["id"] not in lines:
                lines.append(d["id"])
            return "\n".join(lines)
    return current_text


def add_wiki_preset_dash(topic_key, current_text):
    presets = WIKIPEDIA_PRESETS.get(topic_key, [])
    lines = _lines(current_text)
    for p in presets:
        if p not in lines:
            lines.append(p)
    return "\n".join(lines)


# ---------------------------------------------------------------- launch / stop / resume
def _launch(name, cmd):
    out = os.path.join(DATA_DIR, name)
    os.makedirs(os.path.join(ROOT, out, "build"), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    log = open(os.path.join(ROOT, out, "build", "console.log"), "a", encoding="utf-8")
    log.write("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n\n")
    log.flush()
    PROCS[out] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out


def build(name, if_exists, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap, hf, max_rows, github,
          wikipedia, wiki_per_query, arxiv, youtube, max_videos, transcribe, whisper_model, fmt,
          synthetic, gen_model, qa_per_chunk, max_synth_chunks, qa_with_context,
          tokenizer, chunk_tokens, min_chars, near_dedup, redact_pii, system_prompt, eval_ratio, max_examples,
          output_format, shard_rows, workers, min_free_gb):
    name = re.sub(r"[^\w.-]+", "-", (name or "").strip()) or f"ds-{time.strftime('%m%d-%H%M%S')}"
    sources = list(uploads or []) + _lines(folders)
    if not (sources or _lines(urls) or _lines(hf) or _lines(github) or _lines(wikipedia) or _lines(arxiv) or _lines(youtube)):
        raise gr.Error("Add at least one source in any of the source tabs.")
    st = _read_json(os.path.join(ROOT, DATA_DIR, name, "build", "status.json"), {}) or {}
    if st.get("state") == "running" and _pid_alive(st.get("pid")):
        raise gr.Error(f"'{name}' is already building.")
    finished = os.path.exists(os.path.join(ROOT, DATA_DIR, name, "stats.json"))
    if finished and if_exists != "overwrite":
        raise gr.Error(f"'{name}' already exists and is finished. Choose another name, or select 'overwrite'.")
    exists = finished or bool(st)

    cmd = [sys.executable, "-u", "dataset_builder.py", "--name", name, "--format", fmt,
           "--tokenizer", tokenizer, "--chunk_tokens", str(int(chunk_tokens)), "--min_chars", str(int(min_chars)),
           "--eval_ratio", str(eval_ratio), "--crawl_depth", str(int(crawl_depth)), "--max_pages", str(int(max_pages)),
           "--max_rows", str(int(max_rows)), "--max_examples", str(int(max_examples)),
           "--wiki_per_query", str(int(wiki_per_query)), "--max_videos", str(int(max_videos)),
           "--output_format", output_format, "--shard_rows", str(int(shard_rows)), "--workers", str(int(workers)),
           "--min_free_gb", str(min_free_gb)]
    if exists:
        cmd.append("--resume" if if_exists == "resume" else "--overwrite")
    if sitemap:
        cmd.append("--sitemap")
    for flag, values in (("--source", sources), ("--url", _lines(urls)), ("--hf", _lines(hf)),
                         ("--github", _lines(github)), ("--wikipedia", _lines(wikipedia)),
                         ("--arxiv", _lines(arxiv)), ("--youtube", _lines(youtube))):
        for v in values:
            cmd += [flag, v]
    if ocr:
        cmd.append("--ocr")
    if transcribe:
        cmd += ["--transcribe", "--whisper_model", whisper_model]
    if synthetic:
        cmd += ["--synthetic", "--gen_model", gen_model, "--qa_per_chunk", str(int(qa_per_chunk)),
                "--max_synth_chunks", str(int(max_synth_chunks))]
        if any(s in gen_model for s in ("7B", "8B", "9B")):
            cmd.append("--gen_4bit")
        if qa_with_context:
            cmd.append("--qa_with_context")
    if not near_dedup:
        cmd.append("--no_near_dedup")
    if redact_pii:
        cmd.append("--redact_pii")
    if system_prompt and system_prompt.strip():
        cmd += ["--system_prompt", system_prompt.strip()]

    out = _launch(name, cmd)
    gr.Info(f"{'Resuming' if exists and if_exists == 'resume' else 'Building'} dataset '{name}'")
    return gr.update(choices=list_datasets(), value=out)


def resume(ds):
    if not ds:
        raise gr.Error("Pick a build in 'Viewing build' first.")
    st = _read_json(os.path.join(ROOT, ds, "build", "status.json"), {}) or {}
    if st.get("state") == "running" and _pid_alive(st.get("pid")):
        raise gr.Error("It's still running.")
    if st.get("state") == "done":
        raise gr.Error("That build already finished.")
    name = os.path.basename(ds)
    if not os.path.exists(os.path.join(ROOT, DATA_DIR, name, "build", "status.json")):
        raise gr.Error("Nothing to resume.")
    _launch(name, [sys.executable, "-u", "dataset_builder.py", "--name", name, "--resume"])
    gr.Info(f"Resuming '{name}' — finished sources and phases are skipped.")


def stop(ds):
    if not ds:
        return
    path = os.path.join(ROOT, ds, "build", "status.json")
    status = _read_json(path, {}) or {}
    for pid in filter(None, {status.get("pid"), PROCS[ds].pid if ds in PROCS else None}):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    if status:
        status.update(state="stopped", phase="Stopped by user")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(status, f)
    gr.Warning(f"Stopped {ds}. Click ↻ Resume to continue later — finished work is kept.")


# ---------------------------------------------------------------- Hugging Face browser
def hf_search(q, sort):
    from huggingface_hub import HfApi
    try:
        res = list(HfApi().list_datasets(search=(q or "").strip() or None, sort=sort, limit=40))
    except Exception as e:
        raise gr.Error(f"Hugging Face search failed: {e}")
    rows, ids = [], []
    for d in res:
        tags = d.tags or []
        tag = lambda p: ", ".join(t.split(":", 1)[1] for t in tags if t.startswith(p))[:40]
        ids.append(d.id)
        rows.append({"dataset": d.id, "downloads": f"{d.downloads or 0:,}", "likes": d.likes or 0,
                     "size": tag("size_categories:"), "task": tag("task_categories:"),
                     "updated": str(d.last_modified or "")[:10]})
    df = pd.DataFrame(rows) if rows else pd.DataFrame({"dataset": ["(no results)"]})
    return df, ids


def hf_pick(evt: gr.SelectData, ids):
    if not ids or evt.index[0] >= len(ids):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    ds_id = ids[evt.index[0]]
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(ds_id)
    except Exception as e:
        return ds_id, gr.update(choices=[], value=None), gr.update(choices=[], value=None), f"⚠️ Couldn't read configs: {e}"
    cfg = configs[0] if configs else None
    splits = _splits(ds_id, cfg)
    return (ds_id, gr.update(choices=configs, value=cfg), gr.update(choices=splits, value=splits[0] if splits else None),
            f"**[{ds_id}](https://huggingface.co/datasets/{ds_id})** — {len(configs)} config(s). Click **Preview** to peek.")


def _splits(ds_id, cfg):
    try:
        from datasets import get_dataset_split_names
        return get_dataset_split_names(ds_id, cfg)
    except Exception:
        return ["train"]


def hf_config_change(ds_id, cfg):
    splits = _splits(ds_id, cfg) if ds_id else []
    return gr.update(choices=splits, value=splits[0] if splits else None)


def hf_preview(ds_id, cfg, split):
    if not ds_id:
        raise gr.Error("Pick a dataset from the results first.")
    from datasets import load_dataset
    from dataset_builder import classify_record
    try:
        rows = list(load_dataset(ds_id, cfg, split=split or "train", streaming=True).take(5))
    except Exception as e:
        raise gr.Error(f"Preview failed (gated or script-based datasets may need login/approval): {e}")
    kind = Counter_kind([classify_record(r)["kind"] for r in rows])
    df = pd.DataFrame([{k: (str(v)[:200]) for k, v in r.items()} for r in rows])
    label = {"sft": "chat / instruction → SFT", "dpo": "preference pairs → DPO", "text": "raw text → pretraining (or Synthetic Q&A)"}[kind]
    return df, f"Detected as **{label}** · columns: `{', '.join(df.columns)}`"


def Counter_kind(kinds):
    from collections import Counter
    return Counter(kinds).most_common(1)[0][0] if kinds else "text"


def hf_add(ds_id, cfg, split, current):
    if not ds_id:
        raise gr.Error("Pick a dataset first.")
    line = ds_id + (f"|{cfg or ''}|{split or ''}" if (cfg and cfg != "default") or (split and split != "train") else "")
    lines = _lines(current)
    if line not in lines:
        lines.append(line)
    gr.Info(f"Added {line}")
    return "\n".join(lines)


# ---------------------------------------------------------------- live refresh
def _bar_df(d, key, val="count"):
    rows = sorted(d.items(), key=lambda x: -x[1])
    return pd.DataFrame({key: [k for k, _ in rows], val: [v for _, v in rows]})


def _preview(ds_dir, status):
    rows = []
    examples = _read_json(os.path.join(ds_dir, "preview.json"), None) if status.get("state") == "done" else None
    if examples is None:  # while building, peek at what has been kept so far
        examples = []
        try:
            with open(os.path.join(ds_dir, "build", "kept.jsonl"), encoding="utf-8") as f:
                for _, line in zip(range(25), f):
                    examples.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    for ex in examples[:25]:
        if "messages" in ex:
            user = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
            asst = next((m["content"] for m in reversed(ex["messages"]) if m["role"] == "assistant"), "")
            rows.append({"type": "chat", "prompt / text": user[:400], "response": asst[:400]})
        elif "chosen" in ex:
            rows.append({"type": "pref", "prompt / text": ex["prompt"][:300], "response": f"✅ {ex['chosen'][:200]}\n❌ {ex['rejected'][:200]}"})
        else:
            rows.append({"type": "text", "prompt / text": ex.get("text", "")[:600], "response": ""})
    return pd.DataFrame(rows or [{"type": "", "prompt / text": "(examples appear here as they're qualified)", "response": ""}])


def _insights(status):
    tips = []
    dropped = status.get("dropped", {})
    for reason, tip in DROP_TIPS.items():
        if dropped.get(reason) or dropped.get(f"chunk: {reason}"):
            tips.append(tip)
    unsup = [k.split("(", 1)[1].rstrip(")") for k in dropped if k.startswith("unsupported type")]
    if unsup:
        tips.append(f"Skipped files with no extractable text: {', '.join(unsup[:8])}.")
    tips += [f"⚠️ {w}" for w in status.get("warnings", [])[:6]]
    if status.get("state") in ("stopped", "error") or (status.get("state") == "running" and not _pid_alive(status.get("pid"))):
        tips.append("↻ Click **Resume** to continue this build — finished sources and phases are skipped.")
    stats = status.get("stats")
    if stats:
        n, tok, fmt = stats["train_examples"], stats["total_tokens"], stats["format"]
        if fmt == "sft":
            tips.append("✅ Good SFT size." if n >= 1000 else
                        f"ℹ️ {n:,} chat examples. Fine for style/format tuning; 1k–50k is typical for teaching new skills or knowledge.")
        elif fmt == "pretrain":
            tips.append(f"ℹ️ {tok / 1e6:.2f}M tokens. Pretraining from scratch wants ≥ 50M tokens for fluent text; "
                        "for continued pretraining of an existing model, a few million is already useful.")
        if stats["tokens_p95"] > 2048 and fmt != "pretrain":
            tips.append(f"⚠️ 5% of examples exceed {stats['tokens_p95']:,} tokens — raise *Max sequence length* when training or they'll be truncated.")
        trainer = {"sft": "SFT fine-tune", "dpo": "DPO preference tuning", "pretrain": "Pretrain from scratch"}[fmt]
        tips.append(f"🎉 Ready. Browse/edit it in **🗂️ Dataset Library**, or click **Use for training →** (*{trainer}*).")
    return tips


def refresh(ds):
    empty_bar = pd.DataFrame({"reason": [], "count": []})
    if not ds:
        return ("<i>No build selected. Add sources on the left and click <b>Build dataset</b>.</i>",
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), _preview("", {}), "", "")
    ds_dir = os.path.join(ROOT, ds)
    status = _read_json(os.path.join(ds_dir, "build", "status.json"), {}) or {}
    if not status:
        html = '<div class="tb-phase"><span class="tb-pill tb-running">STARTING</span><b>Launching builder…</b></div>'
        return (html, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), _preview(ds_dir, {}),
                _tail(os.path.join(ds_dir, "build", "console.log")), "")

    state = status.get("state", "?")
    if state == "running" and not _pid_alive(status.get("pid")):
        state = "dead"
    pill = {"running": "tb-running", "done": "tb-done", "error": "tb-error", "stopped": "tb-stopped", "dead": "tb-dead"}.get(state, "tb-stopped")
    prog = status.get("progress", {})
    done, total = prog.get("done", 0), prog.get("total", 0)
    pct = min(100, 100 * done / total) if total else (100 if state == "done" else 0)
    elapsed = status.get("elapsed_s") if state != "running" else time.time() - status.get("started_at", time.time())
    counters = status.get("counters", {})
    dropped_total = sum(status.get("dropped", {}).values())
    stats = status.get("stats")
    fmt_num = lambda x: f"{x:,.1f}" if isinstance(x, float) and x != int(x) else f"{int(x):,}"

    html = (f'<div class="tb-phase"><span class="tb-pill {pill}">{("process exited" if state == "dead" else state).upper()}</span>'
            f'<b style="font-size:18px">{status.get("phase", "")}</b>'
            f'<span style="opacity:.7">format: {status.get("format", status.get("config", {}).get("format", "auto"))}'
            f'{" · resumed" if status.get("resumed") else ""}</span></div>'
            f'<div class="tb-bar"><div style="width:{pct:.1f}%"></div></div>'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;opacity:.8">'
            f'<span>{fmt_num(done)} / {fmt_num(total)} {prog.get("unit", "")}</span>'
            f'<span>elapsed {_fmt_dur(elapsed)} · phase ETA {_fmt_dur(prog.get("eta_s")) if state == "running" else "—"}</span></div>')

    extracted = {k.split(": ")[1]: v for k, v in counters.items() if k.startswith("extracted: ")}
    cards = [
        _card("Files found", f"{counters.get('files found', 0):,}", "after skipping folders like .git"),
        _card("Extracted", f"{sum(extracted.values()):,}", " · ".join(f"{v:,} {k}" for k, v in extracted.items()) or "docs / rows",
              "text = raw documents · sft = chat/instruction rows · dpo = preference rows"),
        _card("Examples kept", f"{counters.get('examples kept', 0):,}", "qualified for training"),
        _card("Dropped", f"{dropped_total:,}", "see breakdown chart", "Everything removed by cleaning, quality filters and dedup."),
    ]
    if counters.get("synthetic pairs"):
        cards.append(_card("Synthetic Q&A", f"{counters['synthetic pairs']:,}", "generated by local LLM"))
    if stats:
        est = " (est.)" if stats.get("tokens_estimated") else ""
        cards += [
            _card("Train / eval", f"{stats['train_examples']:,}", f"+ {stats['eval_examples']:,} eval"),
            _card("Total tokens", (f"{stats['total_tokens'] / 1e6:.2f}M" if stats["total_tokens"] > 1e5 else f"{stats['total_tokens']:,}") + est,
                  f"tokenizer: {stats['tokenizer'].split('/')[-1]}"),
            _card("Tokens / example", f"{stats['tokens_p50']:,}", f"median · p95 {stats['tokens_p95']:,} · max {stats['tokens_max']:,}"),
            _card("Storage", stats.get("output_format", "jsonl"), f"{len(stats.get('files', []))} file(s)"),
        ]
    html += '<div class="tb-cards" style="margin-top:10px">' + "".join(cards) + "</div>"
    tips = [re.sub(r"\*(.+?)\*", r"<i>\1</i>", re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)) for t in _insights(status)]
    if tips:
        html += '<div class="tb-insights"><b>What this means</b><ul>' + "".join(f"<li>{t}</li>" for t in tips) + "</ul></div>"
    if status.get("error"):
        html += f'<pre style="color:#dc2626;white-space:pre-wrap;max-height:240px;overflow:auto">{status["error"][-3000:]}</pre>'
    if stats:
        html += f'<p style="margin-top:8px">📁 <code>{stats["train_path"]}</code></p>'

    dropped_df = _bar_df(status.get("dropped", {}), "reason") if status.get("dropped") else empty_bar
    sources_df = _bar_df(status.get("sources", {}), "source") if status.get("sources") else None
    hist_df = pd.DataFrame(stats["length_histogram"]) if stats else None
    return (html,
            gr.update(value=dropped_df, visible=not dropped_df.empty),
            gr.update(value=sources_df, visible=sources_df is not None),
            gr.update(value=hist_df, visible=hist_df is not None,
                      sort=list(hist_df["tokens"]) if hist_df is not None else None),  # keep bins in numeric order
            _preview(ds_dir, status),
            _tail(os.path.join(ds_dir, "build", "console.log")),
            "  ".join(f"`{k}={v}`" for k, v in status.get("config", {}).items() if v not in (None, False, "", [], 0)))


def use_for_training(ds):
    stats = _read_json(os.path.join(ROOT, ds or "", "stats.json"), None)
    if not stats:
        raise gr.Error("This dataset hasn't finished building yet.")
    task = {"sft": "SFT fine-tune", "dpo": "DPO preference tuning", "pretrain": "Pretrain from scratch"}[stats["format"]]
    gr.Info(f"Loaded {stats['train_path']} into the training launcher")
    return task, gr.update(choices=list_trainable(), value=stats["train_path"]), gr.Tabs(selected="live")


# ---------------------------------------------------------------- UI
def build_tab():
    with gr.TabItem("📚 Build", id="data"):
        hf_ids = gr.State([])
        hf_sel = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("### 📥 Sources — mix as many as you like")
                with gr.Accordion("📁 Files, folders & zips", open=True):
                    uploads = gr.File(label="Upload files / zips", file_count="multiple", type="filepath")
                    folders = gr.Textbox(label="Folder / file paths or globs (one per line)", lines=3,
                                         placeholder="C:\\Users\\you\\Documents\\notes\nD:\\books\\**\\*.pdf")
                    ocr = gr.Checkbox(label="🔍 OCR images & scanned PDFs (png, jpg, tiff… — CPU, slower)")
                    gr.Markdown("<small>Reads txt, md, pdf, docx, pptx, epub, html, rtf, csv, tsv, xlsx, json, jsonl, "
                                "parquet, ipynb, 40+ code languages, zip. Audio/video → see ▶ tab.</small>")
                with gr.Accordion("🌐 Web pages", open=False):
                    urls = gr.Textbox(label="URLs (one per line)", lines=3, placeholder="https://docs.python.org/3/tutorial/")
                    with gr.Row():
                        crawl_depth = gr.Slider(0, 3, value=0, step=1, label="Crawl depth", info="follow links N hops (same site)")
                        max_pages = gr.Number(value=100, label="Max pages", precision=0)
                        sitemap = gr.Checkbox(value=True, label="🗺️ Auto-discover sitemap.xml", info="Rapidly extracts full docs sites")
                with gr.Accordion("🤗 Hugging Face datasets", open=False):
                    hf_preset_dropdown = gr.Dropdown(
                        choices=["(custom)"] + [f"{d['name']} — {d['desc']}" for d in POPULAR_HF_DATASETS],
                        value="(custom)",
                        label="⭐ Popular Curated Datasets (1-Click)",
                    )
                    hf = gr.Textbox(label="Datasets to include (one per line: id or id|config|split)", lines=3,
                                    placeholder="yahma/alpaca-cleaned\nHuggingFaceH4/no_robots||train")
                    max_rows = gr.Number(value=0, precision=0, label="Max rows per HF dataset / table file (0 = all)")
                    with gr.Accordion("🔎 Browse the Hub", open=True):
                        with gr.Row():
                            hf_q = gr.Textbox(placeholder="search e.g. instruct, math, code, medical…", show_label=False, scale=3)
                            hf_sort = gr.Dropdown(["downloads", "likes", "trending_score", "last_modified"], value="downloads",
                                                  show_label=False, scale=1, min_width=120)
                            hf_go = gr.Button("Search", scale=0, min_width=80)
                        hf_results = gr.Dataframe(interactive=False, wrap=True, max_height=260, label="Click a dataset")
                        hf_info = gr.Markdown()
                        with gr.Row():
                            hf_cfg = gr.Dropdown(label="Config", choices=[], allow_custom_value=True)
                            hf_split = gr.Dropdown(label="Split", choices=[], allow_custom_value=True)
                        with gr.Row():
                            hf_prev_btn = gr.Button("👀 Preview")
                            hf_add_btn = gr.Button("➕ Add to sources", variant="primary")
                        hf_kind = gr.Markdown()
                        hf_prev = gr.Dataframe(interactive=False, wrap=True, max_height=240, visible=True)
                with gr.Accordion("💻 GitHub repos", open=False):
                    github = gr.Textbox(label="Repos (one per line)", lines=3,
                                        placeholder="huggingface/transformers\nhttps://github.com/pallets/flask/tree/main")
                    gr.Markdown("<small>Downloads a snapshot zip (no git needed). Set a GITHUB_TOKEN env var for higher rate limits.</small>")
                with gr.Accordion("📖 Wikipedia & Academic Research", open=False):
                    wikipedia = gr.Textbox(label="Wikipedia topics / searches (one per line)", lines=3,
                                           placeholder="Photosynthesis\nsearch: quantum computing\ncategory: Machine learning\nhttps://de.wikipedia.org/wiki/Physik")
                    wiki_per_query = gr.Slider(1, 500, value=20, step=1, label="Articles per search / category")
                    with gr.Row():
                        wiki_btn_ai = gr.Button("Add AI Topics", size="sm")
                        wiki_btn_cs = gr.Button("Add Python/CS", size="sm")
                        wiki_btn_sci = gr.Button("Add Physics", size="sm")
                    arxiv = gr.Textbox(label="ArXiv & Research Paper queries (one per line)", lines=2,
                                       placeholder="deep reinforcement learning\ntransformer attention mechanism")
                with gr.Accordion("▶ YouTube, audio & video", open=False):
                    youtube = gr.Textbox(label="YouTube videos / playlists / channels (one per line)", lines=3,
                                         placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/playlist?list=...")
                    max_videos = gr.Slider(1, 1000, value=50, step=1, label="Max videos per playlist / channel")
                    transcribe = gr.Checkbox(label="🎙️ Transcribe audio/video files (+ videos without captions) with Whisper on the GPU")
                    whisper_model = gr.Dropdown(WHISPER_MODELS, value=WHISPER_MODELS[0], label="Whisper model",
                                                info="turbo = fast & accurate · small = quickest")
                    gr.Markdown("<small>Add audio/video files (mp3, wav, m4a, mp4, mkv…) via 📁 Files and tick Transcribe.</small>")

                gr.Markdown("### 🎯 Make it trainable")
                fmt = gr.Radio(list(FORMAT_HELP), value="auto", label="Target format")
                fmt_help = gr.Markdown(FORMAT_HELP["auto"])
                synthetic = gr.Checkbox(label="🤖 Synthetic Q&A: turn raw documents into chat examples with a local LLM")
                with gr.Group(visible=False) as synth_box:
                    gen_model = gr.Dropdown(GEN_MODELS, value="Qwen/Qwen2.5-3B-Instruct", allow_custom_value=True,
                                            label="Generator model", info="3B = good quality/speed balance · 7B = best (loads in 4-bit)")
                    with gr.Row():
                        qa_per_chunk = gr.Slider(1, 8, value=3, step=1, label="Q&A pairs per chunk")
                        max_synth_chunks = gr.Number(value=2000, precision=0, label="Max chunks (0 = all)")
                    qa_with_context = gr.Checkbox(label="Include the source passage in the prompt (RAG-style)")
                with gr.Accordion("Advanced cleaning & chunking", open=False):
                    tokenizer = gr.Textbox(value="Qwen/Qwen2.5-0.5B-Instruct", label="Tokenizer (use your target model's)")
                    chunk_tokens = gr.Slider(128, 4096, value=1024, step=64, label="Pretrain chunk size (tokens)")
                    min_chars = gr.Slider(0, 1000, value=80, step=10, label="Min characters per item")
                    near_dedup = gr.Checkbox(value=True, label="Near-duplicate removal (MinHash)")
                    redact_pii = gr.Checkbox(label="Redact PII (emails, phones, card numbers, IPs, API keys)")
                    system_prompt = gr.Textbox(label="System prompt added to every chat example (optional)")
                    eval_ratio = gr.Slider(0, 0.2, value=0.02, step=0.01, label="Eval split")
                    max_examples = gr.Number(value=0, precision=0, label="Stop after N examples (0 = all)")
                with gr.Accordion("🏋️ Scale & output (big datasets)", open=False):
                    output_format = gr.Radio(["jsonl", "parquet"], value="jsonl", label="Output storage",
                                             info="parquet = zstd-compressed shards, ~4-6× smaller, best for millions of rows")
                    shard_rows = gr.Number(value=500_000, precision=0, label="Rows per parquet shard")
                    workers = gr.Slider(1, max(2, os.cpu_count() or 4), value=max(1, (os.cpu_count() or 4) - 2), step=1,
                                        label="CPU worker processes")
                    min_free_gb = gr.Slider(1, 50, value=2, step=1, label="Stop safely if free disk drops below (GB)")
                with gr.Row():
                    name = gr.Textbox(label="Dataset name", placeholder="auto: ds-1009-2130", scale=2)
                    if_exists = gr.Radio(["resume", "overwrite"], value="resume", label="If the name exists", scale=1,
                                         info="resume = continue an unfinished build")
                with gr.Row():
                    build_btn = gr.Button("🔨 Build dataset", variant="primary")
                    stop_btn = gr.Button("■ Stop", variant="stop")
                    resume_btn = gr.Button("↻ Resume")

            with gr.Column(scale=2):
                with gr.Row():
                    ds = gr.Dropdown(list_datasets(), value=(list_datasets() or [None])[0], label="Viewing build",
                                     allow_custom_value=True, scale=4)
                    use_btn = gr.Button("Use for training →", variant="primary", scale=1)
                summary = gr.HTML()
                dropped_plot = gr.BarPlot(x="reason", y="count", title="Dropped — and why", height=300, sort="-y",
                                          x_label_angle=-30, x_title=" ")
                with gr.Row():
                    sources_plot = gr.BarPlot(x="source", y="count", title="Sources", height=260, sort="-y",
                                              x_label_angle=-30, x_title=" ")
                    hist_plot = gr.BarPlot(x="tokens", y="examples", title="Example length (tokens)", height=260,
                                           x_label_angle=-45)
                with gr.Tabs():
                    with gr.TabItem("👀 Preview"):
                        preview = gr.Dataframe(wrap=True, interactive=False, column_widths=["8%", "50%", "42%"])
                    with gr.TabItem("🖨️ Console"):
                        console = gr.Textbox(lines=16, max_lines=16, autoscroll=True, show_label=False)
                    with gr.TabItem("⚙️ Build config"):
                        cfg = gr.Markdown()

        outputs = [summary, dropped_plot, sources_plot, hist_plot, preview, console, cfg]
        timer = gr.Timer(2)
        timer.tick(refresh, ds, outputs, show_progress="hidden")
        ds.change(refresh, ds, outputs, show_progress="hidden")
        fmt.change(lambda f: FORMAT_HELP[f], fmt, fmt_help)
        synthetic.change(lambda on: gr.Group(visible=on), synthetic, synth_box)
        hf_preset_dropdown.change(on_hf_preset_add, [hf_preset_dropdown, hf], hf)
        wiki_btn_ai.click(lambda cur: add_wiki_preset_dash("AI & Machine Learning", cur), wikipedia, wikipedia)
        wiki_btn_cs.click(lambda cur: add_wiki_preset_dash("Computer Science & Python", cur), wikipedia, wikipedia)
        wiki_btn_sci.click(lambda cur: add_wiki_preset_dash("Science & Physics", cur), wikipedia, wikipedia)
        build_btn.click(build, [name, if_exists, uploads, folders, ocr, urls, crawl_depth, max_pages, sitemap, hf, max_rows, github,
                                wikipedia, wiki_per_query, arxiv, youtube, max_videos, transcribe, whisper_model, fmt, synthetic,
                                gen_model, qa_per_chunk, max_synth_chunks, qa_with_context, tokenizer, chunk_tokens,
                                min_chars, near_dedup, redact_pii, system_prompt, eval_ratio, max_examples,
                                output_format, shard_rows, workers, min_free_gb], ds)
        stop_btn.click(stop, ds, None)
        resume_btn.click(resume, ds, None)
        # Refresh the build list when opening the dropdown, without touching the open list afterwards.
        ds.focus(lambda v: gr.update(choices=list_datasets(), value=v), ds, ds)

        hf_go.click(hf_search, [hf_q, hf_sort], [hf_results, hf_ids])
        hf_q.submit(hf_search, [hf_q, hf_sort], [hf_results, hf_ids])
        hf_results.select(hf_pick, hf_ids, [hf_sel, hf_cfg, hf_split, hf_info])
        hf_cfg.input(hf_config_change, [hf_sel, hf_cfg], hf_split)
        hf_prev_btn.click(hf_preview, [hf_sel, hf_cfg, hf_split], [hf_prev, hf_kind])
        hf_add_btn.click(hf_add, [hf_sel, hf_cfg, hf_split, hf], hf)
    return {"ds": ds, "use_btn": use_btn}

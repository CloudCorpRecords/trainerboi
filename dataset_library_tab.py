"""
dataset_library_tab.py - "Dataset Library" tab: every dataset in one place.

  * searchable table of all datasets (click a row to open it - no fiddly dropdowns)
  * Explorer: page through any split, full-text / regex search, inspect, edit or delete individual examples
    (edits are staged, then committed in one pass - safe on huge files)
  * Stats: token-length histogram, sources, what was dropped, conversation structure
  * Mix & merge: combine datasets with per-dataset row targets (subsample or oversample), optional dedup
  * Manage: rename, convert jsonl <-> parquet shards, download as zip, push to Hugging Face Hub, delete
"""

import os
import sys
import json
import time
import subprocess

import pandas as pd
import gradio as gr

import dataset_store as store
from live_dashboard import ROOT, _card, _fmt_dur

EMPTY_ROWS = pd.DataFrame({"#": [], "type": [], "prompt / text": [], "response": [], "status": []})


# ---------------------------------------------------------------- library table
def _human(n):
    if n is None:
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def library_table(query=""):
    rows, names = [], []
    q = (query or "").lower().strip()
    for d in store.list_datasets():
        if q and q not in d["name"].lower() and q not in d["format"]:
            continue
        names.append(d["name"])
        state_icon = {"done": "✅", "running": "⏳", "error": "❌", "stopped": "⏹", "interrupted": "⚠️"}.get(d["state"], "·")
        rows.append({"dataset": d["name"], "format": d["format"], "train": _human(d["train"]), "eval": _human(d["eval"]),
                     "tokens": _human(d["tokens"]), "storage": d["storage"], "size": f"{d['size_mb']:,.1f} MB",
                     "built": d["built"][:16], "status": f"{state_icon} {d['state']}"})
    df = pd.DataFrame(rows) if rows else pd.DataFrame({"dataset": ["(no datasets yet — build one in 📚 Dataset Builder)"]})
    return df, names


def refresh_library(query, sig):
    """Timer: only re-render the table if something actually changed (keeps scroll/selection stable)."""
    df, names = library_table(query)
    new_sig = df.to_json()
    if new_sig == sig:
        return gr.skip(), gr.skip(), sig
    return df, names, new_sig


# ---------------------------------------------------------------- header / stats
def header_html(name):
    if not name:
        return "<i>Select a dataset in the table to open it.</i>"
    stats = store.read_json(os.path.join(store.ds_dir(name), "stats.json"), {}) or {}
    status = store.read_json(os.path.join(store.ds_dir(name), "build", "status.json"), {}) or {}
    if not stats:
        return (f"<h3 style='margin:0'>{name}</h3><p>Not finished yet (state: <b>{status.get('state', '?')}</b>, "
                f"phase: {status.get('phase', '?')}). Watch it in <b>📚 Dataset Builder</b>.</p>")
    pend = store.pending_count(name)
    est = " (est.)" if stats.get("tokens_estimated") else ""
    cards = [
        _card("Format", stats["format"], stats.get("output_format", "jsonl")),
        _card("Train", f"{stats['train_examples']:,}", f"+ {stats['eval_examples']:,} eval"),
        _card("Tokens", _human(stats["total_tokens"]) + est, stats.get("tokenizer", "").split("/")[-1]),
        _card("Tokens / example", f"{stats.get('tokens_p50', 0):,}", f"median · p95 {stats.get('tokens_p95', 0):,}"),
        _card("On disk", f"{store.dir_size(store.ds_dir(name)) / 1e6:,.1f} MB", stats.get("built_at", "")),
    ]
    if stats.get("mixed_from"):
        cards.append(_card("Mixed from", str(len(stats["mixed_from"])), ", ".join(stats["mixed_from"])[:60]))
    html = f"<h3 style='margin:0 0 8px'>{name}</h3><div class='tb-cards'>" + "".join(cards) + "</div>"
    if pend:
        html += f"<p style='margin-top:8px'>✏️ <b>{pend} pending change(s)</b> — commit them in the Explorer.</p>"
    return html


def stats_view(name):
    stats = store.read_json(os.path.join(store.ds_dir(name or ""), "stats.json"), None)
    if not stats:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), ""
    hist = pd.DataFrame(stats.get("length_histogram", []))
    src = pd.DataFrame(sorted(stats.get("sources", {}).items(), key=lambda x: -x[1]), columns=["source", "count"])
    drop = pd.DataFrame(sorted(stats.get("dropped", {}).items(), key=lambda x: -x[1]), columns=["reason", "count"])
    det = store.detailed_stats(name)
    lines = [f"**Sampled {det.get('sampled', 0):,} examples**"]
    if det.get("multi_turn_pct") is not None:
        lines += [f"- Multi-turn conversations: **{det['multi_turn_pct']:.1f}%**",
                  f"- Has a system prompt: **{det['system_prompt_pct']:.1f}%**",
                  f"- Median prompt: **{det['median_prompt_chars']:,} chars** · median response: **{det['median_response_chars']:,} chars**"]
    if det.get("median_text_chars"):
        lines.append(f"- Median document chunk: **{det['median_text_chars']:,} chars**")
    if stats.get("warnings"):
        lines += ["", "**Build warnings**"] + [f"- {w}" for w in stats["warnings"][:10]]
    return (gr.update(value=hist, visible=not hist.empty, sort=list(hist["tokens"]) if not hist.empty else None),
            gr.update(value=src, visible=not src.empty), gr.update(value=drop, visible=not drop.empty), "\n".join(lines))


# ---------------------------------------------------------------- explorer
def _summarize(ex):
    if "messages" in ex:
        user = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
        asst = next((m["content"] for m in reversed(ex["messages"]) if m["role"] == "assistant"), "")
        turns = sum(m["role"] == "user" for m in ex["messages"])
        return ("chat" + (f" ×{turns}" if turns > 1 else ""), user[:300], asst[:300])
    if "chosen" in ex:
        return ("pref", ex.get("prompt", "")[:250], f"✅ {ex['chosen'][:140]}\n❌ {ex['rejected'][:140]}")
    if "text" in ex:
        return ("text", ex["text"][:500], "")
    return ("?", json.dumps(ex)[:300], "")


def explore(name, split, page, page_size, hits):
    """Returns (rows_df, page_info, shown_indices, page)."""
    if not name:
        return EMPTY_ROWS, "", [], 0
    r = store.SplitReader(name, split)
    try:
        total = len(hits) if hits is not None else len(r)
        pages = max(1, -(-total // page_size))
        page = max(0, min(page, pages - 1))
        if hits is not None:
            idx = hits[page * page_size:(page + 1) * page_size]
            items = [row for i in idx for row in r.rows(i, 1)]
        else:
            items = r.rows(page * page_size, page_size)
    finally:
        r.close()
    edits = store.load_edits(name).get(split, {"deleted": [], "replaced": {}})
    rows = []
    for i, ex in items:
        status = ""
        if str(i) in edits["replaced"]:
            ex, status = edits["replaced"][str(i)], "✏️ edited"
        if i in edits["deleted"]:
            status = "🗑 deleted"
        t, a, b = _summarize(ex)
        rows.append({"#": i, "type": t, "prompt / text": a, "response": b, "status": status})
    what = f"{total:,} matches" if hits is not None else f"{total:,} examples"
    info = f"**Page {page + 1:,} / {pages:,}** · {what} in *{split}*"
    return (pd.DataFrame(rows) if rows else EMPTY_ROWS), info, [i for i, _ in items], page


def load_example(name, split, idx):
    if not name or idx is None:
        return gr.update(value="", label="Example"), None
    edits = store.load_edits(name).get(split, {"deleted": [], "replaced": {}})
    if str(idx) in edits["replaced"]:
        ex = edits["replaced"][str(idx)]
    else:
        r = store.SplitReader(name, split)
        try:
            ex = r.rows(idx, 1)[0][1]
        finally:
            r.close()
    label = f"Example #{idx:,}" + (" — 🗑 marked for deletion" if idx in edits["deleted"] else "")
    return gr.update(value=json.dumps(ex, indent=2, ensure_ascii=False), label=label), idx


def _validate(ex, fmt):
    if fmt == "sft":
        assert isinstance(ex.get("messages"), list) and ex["messages"], "needs a non-empty 'messages' list"
        for m in ex["messages"]:
            assert m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str), \
                "each message needs role (system/user/assistant) and string content"
    elif fmt == "pretrain":
        assert isinstance(ex.get("text"), str) and ex["text"].strip(), "needs a non-empty 'text' string"
    elif fmt == "dpo":
        assert all(isinstance(ex.get(k), str) for k in ("prompt", "chosen", "rejected")), "needs prompt/chosen/rejected strings"


def _require_jsonl(name, split):
    if not os.path.exists(os.path.join(store.ds_dir(name), f"{split}.jsonl")):
        raise gr.Error("This dataset is stored as Parquet shards. Convert it to JSONL in ⚙️ Manage to edit examples.")


def save_example(name, split, idx, text):
    if not name or idx is None:
        raise gr.Error("Select an example first.")
    _require_jsonl(name, split)
    try:
        ex = json.loads(text)
        fmt = (store.read_json(os.path.join(store.ds_dir(name), "stats.json"), {}) or {}).get("format")
        _validate(ex, fmt)
    except (json.JSONDecodeError, AssertionError) as e:
        raise gr.Error(f"Not saved — {e}")
    store.edit_row(name, split, idx, ex)
    gr.Info(f"Example #{idx} edited (pending — commit to write it)")


def delete_example(name, split, idx):
    if not name or idx is None:
        raise gr.Error("Select an example first.")
    _require_jsonl(name, split)
    store.delete_row(name, split, idx)
    gr.Info(f"Example #{idx} marked for deletion (pending)")


def restore_example(name, split, idx):
    if name and idx is not None:
        store.restore_row(name, split, idx)


def pending_md(name):
    n = store.pending_count(name) if name else 0
    return f"✏️ **{n} pending change(s)** — not written yet." if n else ""


def commit(name):
    if not name or not store.pending_count(name):
        raise gr.Error("No pending changes.")
    stats = store.commit_edits(name)
    gr.Info(f"Committed. Train now has {stats['train_examples']:,} examples.")


def discard(name):
    store.discard_edits(name)
    gr.Info("Pending changes discarded.")


def search(name, split, query, regex):
    if not name:
        raise gr.Error("Select a dataset first.")
    if not (query or "").strip():
        return None, ""
    r = store.SplitReader(name, split)
    try:
        hits, scanned, complete = r.search(query.strip(), regex=regex)
    except Exception as e:
        raise gr.Error(f"Search failed: {e}")
    finally:
        r.close()
    note = f"🔎 **{len(hits):,}** matches for `{query}`" + ("" if complete else f" (stopped after scanning {scanned:,} rows — refine the query)")
    return hits, note


# ---------------------------------------------------------------- mix
def mix_table():
    rows = [{"use": False, "dataset": d["name"], "format": d["format"], "examples": (d["train"] or 0) + (d["eval"] or 0),
             "take": 0} for d in store.list_datasets() if d["state"] == "done"]
    return pd.DataFrame(rows or [{"use": False, "dataset": "", "format": "", "examples": 0, "take": 0}])


def start_mix(df, name, dedup, out_fmt):
    name = (name or "").strip() or f"mix-{time.strftime('%m%d-%H%M%S')}"
    parts = [(r["dataset"], int(r["take"] or 0)) for _, r in df.iterrows() if r["use"] and r["dataset"]]
    if len(parts) < 1:
        raise gr.Error("Tick 'use' for at least one dataset.")
    fmts = {r["format"] for _, r in df.iterrows() if r["use"]}
    if len(fmts) > 1:
        raise gr.Error(f"All parts must have the same format (got {', '.join(sorted(fmts))}).")
    if os.path.exists(store.ds_dir(name)):
        raise gr.Error(f"'{name}' already exists.")
    cmd = [sys.executable, "-u", "dataset_store.py", "mix", "--name", name, "--output_format", out_fmt]
    for p, t in parts:
        cmd += ["--part", f"{p}:{t}"]
    if dedup:
        cmd.append("--dedup")
    os.makedirs(os.path.join(store.ds_dir(name), "build"), exist_ok=True)
    log = open(os.path.join(store.ds_dir(name), "build", "console.log"), "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gr.Info(f"Mixing into '{name}'…")
    return name


def mix_status(name):
    if not name:
        return ""
    st = store.read_json(os.path.join(store.ds_dir(name), "build", "status.json"), {}) or {}
    if not st:
        return f"⏳ Starting mix **{name}**…"
    p = st.get("progress", {})
    if st.get("state") == "done":
        s = st.get("stats", {})
        return f"✅ **{name}** ready — {s.get('train_examples', 0):,} train / {s.get('eval_examples', 0):,} eval. Select it in the table."
    if st.get("state") == "error":
        return f"❌ Mix failed: {st.get('error', '')[:500]}"
    return f"⏳ **{name}** — {st.get('phase')} · {p.get('done', 0):,}/{p.get('total', 0):,} {p.get('unit', '')}"


# ---------------------------------------------------------------- data doctor
GRADE_COLOR = {"A": "#16a34a", "B": "#65a30d", "C": "#d97706", "D": "#ea580c", "F": "#dc2626"}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _esc_html(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def doctor_html(rep):
    g = rep["grade"]
    recs = "".join(f"<li>{SEV_ICON[r['severity']]} {r['text']}</li>" for r in rep["recommendations"])
    rows = "".join(
        f"<tr><td>{v['label']}</td><td>{v['pct']}%</td><td><small>{_esc_html((v['examples'][0]['response'] or v['examples'][0]['prompt'])[:140]) if v['examples'] else ''}</small></td></tr>"
        for v in rep["flags"].values())
    flags_tbl = f"<table class='tb-table'><tr><th>Problem</th><th>Share</th><th>Example</th></tr>{rows}</table>" if rows else ""
    rc = rep["response_chars"]
    return (f"<div style='display:flex;gap:18px;align-items:center;margin-bottom:8px'>"
            f"<div style='font-size:54px;font-weight:800;color:{GRADE_COLOR[g]};line-height:1'>{g}</div>"
            f"<div><b>Data health {rep['health']:.0f}/100</b><br><small>{rep['sampled']:,} of {rep['train_examples']:,} examples checked · "
            f"answers typically {rc['p50']} characters · duplicate questions {rep['duplicate_prompt_pct']}% · "
            f"test/train overlap {rep['leaked_eval_prompts']}</small></div></div>"
            f"<b>What to fix</b><ul style='margin-top:4px'>{recs}</ul>{flags_tbl}")


def run_doctor(name):
    if not name:
        raise gr.Error("Open a dataset first (click a row in the table).")
    import data_analyzer
    try:
        rep = data_analyzer.analyze(name)
    except Exception as e:
        raise gr.Error(f"Couldn't analyse '{name}': {e}")
    choices = [(f"{v['label']} ({v['pct']}%)", f) for f, v in rep["flags"].items()]
    default = [f for f, v in rep["flags"].items() if v["default_drop"] and v["pct"] >= 0.5]
    dedup = any((r.get("action") or {}).get("dedup_prompts") for r in rep["recommendations"])
    return (doctor_html(rep), pd.DataFrame(rep["score_hist"]), gr.update(choices=choices, value=default), dedup,
            f"{name}-clean")


def start_clean(name, drop, min_score, dedup, new_name):
    if not name:
        raise gr.Error("Open a dataset first.")
    new_name = (new_name or "").strip() or f"{name}-clean"
    if os.path.exists(store.ds_dir(new_name)):
        raise gr.Error(f"'{new_name}' already exists - choose another name.")
    if not drop and not min_score and not dedup:
        raise gr.Error("Tick at least one thing to remove.")
    cmd = [sys.executable, "-u", "data_analyzer.py", "clean", "--name", name, "--out", new_name,
           "--drop", ",".join(drop or []), "--min_score", str(int(min_score or 0))]
    if dedup:
        cmd.append("--dedup_prompts")
    os.makedirs(os.path.join(store.ds_dir(new_name), "build"), exist_ok=True)
    log = open(os.path.join(store.ds_dir(new_name), "build", "console.log"), "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"),
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    gr.Info(f"Creating the cleaned copy '{new_name}'… your original is untouched.")
    return new_name

# ---------------------------------------------------------------- manage
def do_rename(name, new):
    if not name:
        raise gr.Error("Select a dataset first.")
    try:
        new = store.rename(name, new)
    except (ValueError, OSError) as e:
        raise gr.Error(str(e))
    gr.Info(f"Renamed to {new}")
    return new


def do_convert(name, to):
    if not name:
        raise gr.Error("Select a dataset first.")
    if store.pending_count(name):
        raise gr.Error("Commit or discard pending edits first.")
    out = subprocess.run([sys.executable, "dataset_store.py", "convert", "--name", name, "--to", to], cwd=ROOT,
                         capture_output=True, text=True, env=dict(os.environ, PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8"))
    if out.returncode:
        raise gr.Error(out.stderr[-600:])
    gr.Info(f"Converted to {to}")


def do_zip(name):
    if not name:
        raise gr.Error("Select a dataset first.")
    if store.dir_size(store.ds_dir(name)) > 4e9:
        raise gr.Error("Over 4 GB — push to the Hub or copy the folder instead of downloading through the browser.")
    return store.export_zip(name)


def do_push(name, repo, private):
    if not name or not (repo or "").strip():
        raise gr.Error("Select a dataset and enter a repo name (e.g. my-dataset or user/my-dataset).")
    try:
        url = store.push_to_hub(name, repo.strip(), private)
    except Exception as e:
        raise gr.Error(str(e))
    return f"✅ Uploaded: [{url}]({url}) ({'private' if private else 'public'})"


def do_delete(name, confirm):
    if not name:
        raise gr.Error("Select a dataset first.")
    if not confirm:
        raise gr.Error("Tick the confirmation box first.")
    st = store.read_json(os.path.join(store.ds_dir(name), "build", "status.json"), {}) or {}
    if st.get("state") == "running" and store._pid_alive(st.get("pid")):
        raise gr.Error("This dataset is still building — stop it first.")
    store.delete(name)
    gr.Warning(f"Deleted {name}")


# ---------------------------------------------------------------- UI
def build_tab():
    with gr.TabItem("🗂️ Library", id="library"):
        init_df, init_names = library_table()
        selected = gr.State(None)       # dataset name
        names = gr.State(init_names)    # names in current table order
        sig = gr.State("")              # table signature (skip no-op refreshes)
        shown = gr.State([])            # row indices on current explorer page
        hits = gr.State(None)           # search hits (None = browse all)
        page = gr.State(0)
        sel_row = gr.State(None)
        mix_name = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2, min_width=340):
                with gr.Row():
                    q = gr.Textbox(placeholder="🔍 filter by name or format…", show_label=False, scale=4)
                    refresh_btn = gr.Button("🔄", scale=0, min_width=50)
                table = gr.Dataframe(init_df, interactive=False, wrap=True, max_height=620, label="Your datasets — click a row to open")
                gr.Markdown("Tip: datasets stay here until you delete them. Each lives in `data/<name>/`.")
            with gr.Column(scale=3):
                header = gr.HTML("<i>Select a dataset in the table to open it.</i>")
                train_btn = gr.Button("🚀 Train with this →", variant="primary")
                with gr.Tabs():
                    with gr.TabItem("🔎 Explore & edit"):
                        with gr.Row():
                            split = gr.Radio(["train", "eval"], value="train", show_label=False, scale=1, min_width=160)
                            query = gr.Textbox(placeholder="Search text…", show_label=False, scale=3)
                            regex = gr.Checkbox(label="regex", scale=0, min_width=80)
                            search_btn = gr.Button("Search", scale=0, min_width=80)
                            clear_btn = gr.Button("Clear", scale=0, min_width=70)
                        search_note = gr.Markdown()
                        with gr.Row():
                            prev_btn = gr.Button("◀ Prev", scale=0, min_width=90)
                            page_info = gr.Markdown()
                            next_btn = gr.Button("Next ▶", scale=0, min_width=90)
                            page_size = gr.Dropdown([10, 25, 50, 100], value=25, show_label=False, scale=0, min_width=90)
                            jump = gr.Number(label="Go to #", precision=0, scale=0, min_width=110)
                        rows = gr.Dataframe(EMPTY_ROWS, interactive=False, wrap=True, max_height=520,
                                            column_widths=["7%", "8%", "42%", "33%", "10%"])
                        editor = gr.Code(language="json", label="Example", lines=14, interactive=True)
                        with gr.Row():
                            save_btn = gr.Button("💾 Save edit")
                            del_btn = gr.Button("🗑 Delete example", variant="stop")
                            restore_btn = gr.Button("↩ Restore")
                        pending = gr.Markdown()
                        with gr.Row():
                            commit_btn = gr.Button("✅ Commit changes to disk", variant="primary")
                            discard_btn = gr.Button("✖ Discard pending changes")
                    with gr.TabItem("📊 Stats"):
                        details = gr.Markdown()
                        hist = gr.BarPlot(x="tokens", y="examples", title="Example length (tokens)", height=240, x_label_angle=-45)
                        with gr.Row():
                            src_plot = gr.BarPlot(x="source", y="count", title="Sources", height=260, sort="-y", x_label_angle=-30, x_title=" ")
                            drop_plot = gr.BarPlot(x="reason", y="count", title="Dropped — and why", height=260, sort="-y", x_label_angle=-30, x_title=" ")
                    with gr.TabItem("🩺 Data doctor"):
                        gr.Markdown("Checks quality, duplicates, test-set leakage and variety, then makes a **cleaned copy** "
                                    "(your original is never changed).")
                        doc_btn = gr.Button("🩺 Analyze this dataset", variant="primary")
                        doc_html = gr.HTML()
                        doc_hist = gr.BarPlot(x="score", y="examples", title="Example quality scores", height=200, x_title="score")
                        with gr.Accordion("🧹 Fix it", open=True):
                            doc_drop = gr.CheckboxGroup(label="Remove examples with these problems")
                            with gr.Row():
                                doc_min = gr.Slider(0, 90, value=0, step=5, label="Also remove anything scoring below")
                                doc_dedup = gr.Checkbox(label="Remove repeated questions")
                            doc_new = gr.Textbox(label="Name of the cleaned copy")
                            doc_fix = gr.Button("🧹 Create cleaned copy", variant="primary")
                            doc_md = gr.Markdown()
                    with gr.TabItem("🧪 Mix & merge"):
                        gr.Markdown("Tick datasets to combine. **take** = rows to use: `0` = all, fewer = random subsample, "
                                    "more than available = oversample (repeat). All parts must share a format.")
                        mix_df = gr.Dataframe(mix_table(), interactive=True, datatype=["bool", "str", "str", "number", "number"],
                                              static_columns=[1, 2, 3])
                        with gr.Row():
                            mix_new = gr.Textbox(label="New dataset name", placeholder="auto: mix-1009-2200")
                            mix_fmt = gr.Radio(["jsonl", "parquet"], value="jsonl", label="Storage")
                            mix_dedup = gr.Checkbox(label="Remove exact duplicates across parts")
                        with gr.Row():
                            mix_btn = gr.Button("🧪 Create mix", variant="primary")
                            mix_reload = gr.Button("🔄 Reload list")
                        mix_md = gr.Markdown()
                    with gr.TabItem("⚙️ Manage & export"):
                        with gr.Row():
                            new_name = gr.Textbox(label="Rename to", scale=3)
                            rename_btn = gr.Button("Rename", scale=1)
                        with gr.Row():
                            conv_to = gr.Radio(["jsonl", "parquet"], value="parquet", label="Convert storage to",
                                               info="parquet = compressed shards (big data) · jsonl = editable, readable")
                            conv_btn = gr.Button("Convert")
                        with gr.Row():
                            zip_btn = gr.Button("📦 Download as .zip")
                            zip_file = gr.File(label="Download", interactive=False)
                        gr.Markdown("**Publish to Hugging Face Hub** (needs `hf auth login` once in a terminal)")
                        with gr.Row():
                            repo = gr.Textbox(label="Repo (name or user/name)", scale=3)
                            private = gr.Checkbox(value=True, label="Private", scale=1)
                            push_btn = gr.Button("⬆ Push", scale=1)
                        push_md = gr.Markdown()
                        gr.Markdown("**Danger zone**")
                        with gr.Row():
                            confirm = gr.Checkbox(label="Yes, permanently delete this dataset from disk")
                            delete_btn = gr.Button("Delete dataset", variant="stop")

        # ---------------- wiring
        def open_dataset(nm):
            df, info, idx, pg = explore(nm, "train", 0, 25, None)
            h, s, d, det = stats_view(nm)
            return (nm, header_html(nm), "train", df, info, idx, 0, None, "", "",
                    gr.update(value="", label="Example"), None, pending_md(nm), h, s, d, det, nm or "")

        open_outputs = [selected, header, split, rows, page_info, shown, page, hits, query, search_note,
                        editor, sel_row, pending, hist, src_plot, drop_plot, details, new_name]

        def on_table_select(evt: gr.SelectData, nms):
            nm = nms[evt.index[0]] if nms and evt.index[0] < len(nms) else None
            return open_dataset(nm)

        table.select(on_table_select, names, open_outputs)

        lib_timer = gr.Timer(4)
        lib_timer.tick(refresh_library, [q, sig], [table, names, sig], show_progress="hidden")
        q.change(lambda qq: (*library_table(qq), ""), q, [table, names, sig])
        refresh_btn.click(lambda qq: (*library_table(qq), ""), q, [table, names, sig])

        page_outputs = [rows, page_info, shown, page]
        split.change(lambda nm, sp, ps: explore(nm, sp, 0, ps, None) + (None, ""), [selected, split, page_size],
                     page_outputs + [hits, search_note])
        prev_btn.click(lambda nm, sp, pg, ps, h: explore(nm, sp, pg - 1, ps, h), [selected, split, page, page_size, hits], page_outputs)
        next_btn.click(lambda nm, sp, pg, ps, h: explore(nm, sp, pg + 1, ps, h), [selected, split, page, page_size, hits], page_outputs)
        page_size.change(lambda nm, sp, ps, h: explore(nm, sp, 0, ps, h), [selected, split, page_size, hits], page_outputs)
        jump.submit(lambda nm, sp, j, ps: explore(nm, sp, int(j or 0) // ps, ps, None) + (None,),
                    [selected, split, jump, page_size], page_outputs + [hits])

        def do_search(nm, sp, qq, rx, ps):
            h, note = search(nm, sp, qq, rx)
            return explore(nm, sp, 0, ps, h) + (h, note)

        search_btn.click(do_search, [selected, split, query, regex, page_size], page_outputs + [hits, search_note])
        query.submit(do_search, [selected, split, query, regex, page_size], page_outputs + [hits, search_note])
        clear_btn.click(lambda nm, sp, ps: explore(nm, sp, 0, ps, None) + (None, "", ""), [selected, split, page_size],
                        page_outputs + [hits, search_note, query])

        def on_row_select(evt: gr.SelectData, nm, sp, idxs):
            i = idxs[evt.index[0]] if idxs and evt.index[0] < len(idxs) else None
            return load_example(nm, sp, i)

        rows.select(on_row_select, [selected, split, shown], [editor, sel_row])

        def after_edit(nm, sp, pg, ps, h, i):
            return explore(nm, sp, pg, ps, h) + (pending_md(nm), header_html(nm)) + load_example(nm, sp, i)

        edit_outputs = page_outputs + [pending, header, editor, sel_row]
        edit_inputs = [selected, split, page, page_size, hits, sel_row]
        save_btn.click(save_example, [selected, split, sel_row, editor]).success(after_edit, edit_inputs, edit_outputs)
        del_btn.click(delete_example, [selected, split, sel_row]).success(after_edit, edit_inputs, edit_outputs)
        restore_btn.click(restore_example, [selected, split, sel_row]).success(after_edit, edit_inputs, edit_outputs)
        commit_btn.click(commit, selected).success(lambda nm: open_dataset(nm), selected, open_outputs)
        discard_btn.click(discard, selected).success(after_edit, edit_inputs, edit_outputs)

        mix_btn.click(start_mix, [mix_df, mix_new, mix_dedup, mix_fmt], mix_name)
        mix_reload.click(mix_table, None, mix_df)
        lib_timer.tick(mix_status, mix_name, mix_md, show_progress="hidden")

        doc_name = gr.State(None)
        doc_btn.click(run_doctor, selected, [doc_html, doc_hist, doc_drop, doc_dedup, doc_new])
        doc_fix.click(start_clean, [selected, doc_drop, doc_min, doc_dedup, doc_new], doc_name)
        lib_timer.tick(mix_status, doc_name, doc_md, show_progress="hidden")
        rename_btn.click(do_rename, [selected, new_name], selected).success(
            lambda nm, qq: (*library_table(qq), "", *open_dataset(nm)), [selected, q], [table, names, sig] + open_outputs)
        conv_btn.click(do_convert, [selected, conv_to]).success(lambda nm: open_dataset(nm), selected, open_outputs)
        zip_btn.click(do_zip, selected, zip_file)
        push_btn.click(do_push, [selected, repo, private], push_md)
        delete_btn.click(do_delete, [selected, confirm]).success(
            lambda qq: (*library_table(qq), "", *open_dataset(None), False), q, [table, names, sig] + open_outputs + [confirm])

    return {"selected": selected, "train_btn": train_btn, "table": table, "names": names, "sig": sig, "q": q}

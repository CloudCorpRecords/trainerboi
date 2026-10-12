"""
data_intake.py - "Paste anything" data collection + "what should I train?" detection.

classify(line)  -> which kind of source a pasted line is (youtube, github, arxiv, wikipedia, web, hf, path, topic)
route(text)     -> {field: [lines]} for the Quick Start source boxes, plus a human summary
inspect(paths)  -> looks at your files and says what kind of data they are and which goal/method fits
"""

import os
import re
import json

CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".sql", ".sh"}
DOC_EXT = {".pdf", ".docx", ".doc", ".pptx", ".txt", ".md", ".html", ".htm", ".epub", ".rtf", ".xlsx"}
TABLE_EXT = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
MEDIA_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".png", ".jpg", ".jpeg"}

KIND_LABEL = {"youtube": "▶ YouTube", "github": "💻 GitHub repo", "arxiv": "🔬 Research papers", "wikipedia": "📖 Wikipedia",
              "web": "🌐 Website", "hf": "🤗 Hugging Face dataset", "path": "📁 Local file/folder", "topic": "🪄 Topic"}


def classify(line):
    s = line.strip().strip('"').strip("'")
    low = s.lower()
    if re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/", low):
        return "youtube"
    if re.match(r"https?://(www\.)?github\.com/[\w.-]+/[\w.-]+", low):
        return "github"
    if "arxiv.org" in low or low.startswith("arxiv:") or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", low):
        return "arxiv"
    if re.match(r"https?://[a-z-]+\.wikipedia\.org/", low):
        return "wikipedia"
    if re.match(r"https?://huggingface\.co/datasets/", low):
        return "hf"
    if re.match(r"https?://", low):
        return "web"
    if os.path.exists(s):
        return "path"
    if re.fullmatch(r"[\w.-]+/[\w.-]+(\|[\w.-]*\|[\w.-]*)?", s) and " " not in s and not re.match(r"^[a-zA-Z]:[\\/]", s):
        return "hf"          # owner/name  (GitHub repos are recognised by URL; use a github.com link for those)
    if low.startswith("github:"):
        return "github"
    return "topic"


def _clean(kind, s):
    s = s.strip().strip('"').strip("'")
    if kind == "hf":
        return re.sub(r"^https?://huggingface\.co/datasets/", "", s).strip("/")
    if kind == "github":
        return re.sub(r"^github:", "", re.sub(r"^https?://(www\.)?github\.com/", "", s)).strip("/").removesuffix(".git")
    if kind == "wikipedia":
        return s
    return s


def route(text):
    """Split pasted text into lines, classify each, and return ({field: [lines]}, summary_markdown)."""
    fields = {"folders": [], "urls": [], "hf": [], "wikipedia": [], "arxiv": [], "github": [], "youtube": []}
    counts = {}
    topics = []
    for raw in re.split(r"[\r\n]+", text or ""):
        if not raw.strip():
            continue
        k = classify(raw)
        counts[k] = counts.get(k, 0) + 1
        v = _clean(k, raw)
        if k == "path":
            fields["folders"].append(v)
        elif k == "web":
            fields["urls"].append(v)
        elif k == "hf":
            fields["hf"].append(v)
        elif k == "topic":
            topics.append(v)
            fields["wikipedia"].append(f"search: {v}")
            fields["arxiv"].append(v)
        else:
            fields[k].append(v)
    if not counts:
        return fields, "Paste links, file paths, a Hugging Face dataset or just a topic - one per line."
    lines = [f"- {KIND_LABEL[k]} × {n}" for k, n in counts.items()]
    note = "**Added:**\n" + "\n".join(lines)
    if topics:
        note += "\n\nTopics become Wikipedia + research-paper searches."
    return fields, note


# ------------------------------------------------------------------------------------------------ inspect
def _sample_columns(path):
    """Column names of the first rows of a table file, or None."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".jsonl", ".json"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                head = f.read(200000)
            rows = []
            if ext == ".jsonl" or head.lstrip().startswith("{"):
                for ln in head.splitlines()[:50]:
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
            if not rows:
                data = json.load(open(path, encoding="utf-8"))
                rows = data if isinstance(data, list) else []
            cols = set()
            for r in rows[:50]:
                if isinstance(r, dict):
                    cols |= set(r)
            return cols
        if ext in (".csv", ".tsv"):
            import pandas as pd
            df = pd.read_csv(path, nrows=50, sep="\t" if ext == ".tsv" else ",")
            return set(df.columns), df
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None
    return None


def _cols_of(res):
    return res[0] if isinstance(res, tuple) else res


def kind_of_columns(cols):
    c = {x.lower() for x in cols}
    if {"chosen", "rejected"} <= c:
        return "pairs"
    if {"completion", "label"} <= c or {"response", "label"} <= c:
        return "ratings"
    if c & {"messages", "conversations"} or {"instruction", "output"} <= c or {"question", "answer"} <= c or {"prompt", "completion"} <= c \
            or {"prompt", "response"} <= c or {"input", "output"} <= c:
        return "qa"
    if c & {"text", "content", "body"} and len(c) <= 3:
        return "raw"
    return "table"


# what each kind of data means for the user
ADVICE = {
    "pairs": ("prefer", "orpo", "These look like **good vs. bad answer pairs**. They're ideal for preference tuning."),
    "ratings": ("prefer", "kto", "These look like **answers rated good or bad**. KTO learns directly from thumbs-up/down labels."),
    "qa": ("persona", "lora", "These look like **question → answer examples**. Fine-tuning with LoRA is the proven fit."),
    "raw": ("docs", "distill", "This is **plain text**. We'll turn it into question-and-answer practice, then fine-tune."),
    "docs": ("docs", "distill", "These are **documents**. We'll read them and write practice questions from them automatically."),
    "code": ("code", "distill", "This is **source code**. We'll write questions and answers about it, then fine-tune."),
    "table": (None, None, "This is a **table of numbers/columns**, not text for a language model. Use *More tools → AutoTrain* "
                          "for tabular prediction, or tell me which columns hold the question and answer."),
    "media": ("docs", "distill", "Audio/video can be **transcribed to text** automatically (tick 'transcribe' in the options)."),
}


def inspect(paths):
    """paths: list of file/folder paths. Returns (goal_key|None, method_id|None, markdown)."""
    files = []
    for p in paths or []:
        if os.path.isdir(p):
            n = 0
            for root, _, fs in os.walk(p):
                for f in fs:
                    files.append(os.path.join(root, f))
                    n += 1
                    if n > 2000:
                        break
                if n > 2000:
                    break
        elif os.path.exists(p):
            files.append(p)
    if not files:
        return None, None, "No files found yet."
    exts = {}
    for f in files:
        e = os.path.splitext(f)[1].lower()
        exts[e] = exts.get(e, 0) + 1
    total = len(files)
    kinds = {"code": sum(v for e, v in exts.items() if e in CODE_EXT), "docs": sum(v for e, v in exts.items() if e in DOC_EXT),
             "media": sum(v for e, v in exts.items() if e in MEDIA_EXT)}
    tables = [f for f in files if os.path.splitext(f)[1].lower() in TABLE_EXT]
    kind = None
    detail = ""
    if tables:
        cols = _cols_of(_sample_columns(tables[0]))
        if cols:
            kind = kind_of_columns(cols)
            detail = f" Columns: `{', '.join(sorted(cols)[:8])}`."
    if kind is None:
        kind = max(kinds, key=kinds.get) if max(kinds.values()) else "docs"
    goal, method, msg = ADVICE[kind]
    mix = ", ".join(f"{v}× {e or 'no extension'}" for e, v in sorted(exts.items(), key=lambda x: -x[1])[:5])
    md = f"### 🔍 What I found\n{total} file(s): {mix}.\n\n{msg}{detail}"
    if goal:
        import autopilot
        md += f"\n\n**Suggested goal:** {autopilot.GOALS[goal]['title']} _(choose it in step 1)_"
    return goal, method, md

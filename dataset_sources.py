"""
dataset_sources.py - Extra source connectors for dataset_builder.py.

  github     "owner/repo", "https://github.com/owner/repo", ".../tree/<branch>"  -> repo zip (no git needed)
  wikipedia  "Article Title", "search: quantum computing", "category: Machine learning", or a wikipedia URL
  youtube    video / playlist / channel URLs (transcripts; optional Whisper fallback when none exist)
  ocr        scanned PDFs and images (.png .jpg .tiff ...) via RapidOCR (CPU, no system installs)
  audio      .mp3 .wav .m4a .flac .ogg .mp4 .mkv .mov .webm -> text with Whisper on the GPU
"""

import os
import re
import json
import time
import hashlib
from urllib.parse import urlparse, unquote

import requests

UA = {"User-Agent": "TrainerBoi-DatasetBuilder/1.0 (local dataset building)"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
MEDIA_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


# =============================================================================== GitHub
def github_zip(spec, dest_dir, max_mb=2000):
    """Download a repo snapshot as a zip. Returns (zip_path, label)."""
    spec = spec.strip().rstrip("/")
    branch = None
    if "github.com" in spec:
        parts = urlparse(spec).path.strip("/").split("/")
        owner, repo = parts[0], parts[1].removesuffix(".git")
        if len(parts) >= 4 and parts[2] == "tree":
            branch = "/".join(parts[3:])
    else:
        owner, repo = spec.split("/")[:2]
    headers = dict(UA)
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    if not branch:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=20)
        r.raise_for_status()
        branch = r.json().get("default_branch", "main")
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{owner}__{repo}.zip")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        size = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                size += len(chunk)
                if size > max_mb * (1 << 20):
                    raise ValueError(f"repo archive larger than {max_mb} MB - raise --github_max_mb")
                f.write(chunk)
    return path, f"{owner}/{repo}@{branch}"


# =============================================================================== Wikipedia
def wikipedia_docs(line, per_query=20, lang="en"):
    """Yields {"title", "text", "url"} for a title, 'search: q', 'category: c', or a wikipedia URL."""
    line = line.strip()
    if "wikipedia.org/wiki/" in line:
        u = urlparse(line)
        lang = u.netloc.split(".")[0]
        titles = [unquote(u.path.split("/wiki/", 1)[1]).replace("_", " ")]
    else:
        titles = None
    api = f"https://{lang}.wikipedia.org/w/api.php"
    s = requests.Session()
    s.headers.update(UA)

    if titles is None:
        low = line.lower()
        if low.startswith("search:"):
            r = s.get(api, params={"action": "query", "list": "search", "srsearch": line[7:].strip(),
                                   "srlimit": per_query, "format": "json"}, timeout=20).json()
            titles = [h["title"] for h in r["query"]["search"]]
        elif low.startswith("category:"):
            cat = line.split(":", 1)[1].strip()
            r = s.get(api, params={"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                                   "cmlimit": per_query, "cmtype": "page", "format": "json"}, timeout=20).json()
            titles = [m["title"] for m in r["query"]["categorymembers"]]
        else:
            titles = [line]

    for title in titles:
        r = s.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                               "titles": title, "format": "json"}, timeout=30).json()
        for page in r.get("query", {}).get("pages", {}).values():
            text = page.get("extract")
            if text:
                # Drop trailing reference-only sections.
                text = re.split(r"\n== (See also|References|External links|Notes|Further reading) ==", text)[0]
                yield {"title": page["title"], "text": f"{page['title']}\n\n{text}",
                       "url": f"https://{lang}.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"}
        time.sleep(0.1)


# =============================================================================== YouTube
def youtube_videos(line, max_videos=50):
    """Expand a video / playlist / channel URL into [(video_id, title)]."""
    import yt_dlp
    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True, "playlistend": max_videos,
            "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(line.strip(), download=False)
    entries = info.get("entries")
    if entries is None:
        return [(info["id"], info.get("title", info["id"]))]
    out = []
    for e in entries:
        if e and e.get("entries"):  # channel -> tabs -> videos
            out += [(x["id"], x.get("title", x["id"])) for x in e["entries"] if x and x.get("id")]
        elif e and e.get("id"):
            out.append((e["id"], e.get("title", e["id"])))
    return out[:max_videos]


def youtube_transcript(video_id, languages=("en", "en-US", "en-GB")):
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except Exception:
        # Fall back to whatever language exists (manual first, then auto-generated).
        tl = api.list(video_id)
        fetched = next(iter(tl)).fetch()
    words = " ".join(s.text.replace("\n", " ") for s in fetched.snippets)
    return _paragraphs(words)


def youtube_audio(video_id, dest_dir):
    """Download best audio stream (no ffmpeg needed) for Whisper fallback."""
    import yt_dlp
    os.makedirs(dest_dir, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": os.path.join(dest_dir, "%(id)s.%(ext)s")}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return ydl.prepare_filename(info)


def _paragraphs(text, sentences_per_para=6):
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return "\n\n".join(" ".join(sents[i:i + sentences_per_para]) for i in range(0, len(sents), sentences_per_para))


# =============================================================================== OCR
_OCR = None


def _ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _ocr_array(img):
    result, _ = _ocr_engine()(img)
    if not result:
        return ""
    # Sort boxes top-to-bottom, then left-to-right, and group into lines.
    boxes = sorted(result, key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
    lines, cur_y, cur = [], None, []
    for box, text, score in boxes:
        y = round(box[0][1] / 12)
        if cur_y is not None and y != cur_y:
            lines.append(" ".join(cur))
            cur = []
        cur_y = y
        if float(score) >= 0.5:
            cur.append(text)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(l for l in lines if l.strip())


def ocr_image(path):
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return _ocr_array(np.array(img))


def ocr_pdf(path, max_pages=300, scale=3.0):
    import numpy as np
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    pages = []
    for i in range(min(len(pdf), max_pages)):
        bitmap = pdf[i].render(scale=scale).to_pil().convert("RGB")
        pages.append(_ocr_array(np.array(bitmap)))
    return "\n\n".join(pages)


# =============================================================================== audio / video -> text (Whisper)
class Transcriber:
    def __init__(self, model_id="openai/whisper-large-v3-turbo"):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=torch.float16, device=0)

    @staticmethod
    def decode(path):
        """Decode any audio/video file to 16 kHz mono float32 using PyAV (bundled ffmpeg)."""
        import av
        import numpy as np
        out = []
        with av.open(path) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
            for frame in container.decode(stream):
                for f in resampler.resample(frame):
                    out.append(f.to_ndarray().reshape(-1))
            for f in resampler.resample(None):
                out.append(f.to_ndarray().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0, dtype="float32")

    def __call__(self, path):
        audio = self.decode(path)
        if audio.size < 16000:  # < 1 second
            return ""
        res = self.pipe({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=8,
                        return_timestamps=False, generate_kwargs={"task": "transcribe"})
        return _paragraphs(res["text"])


def stable_id(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


# =============================================================================== Sitemap Discovery & Crawler
def sitemap_urls(url, max_urls=200):
    """Auto-discover sitemap.xml on a domain or parse an explicit sitemap URL."""
    import xml.etree.ElementTree as ET
    from urllib.parse import urlparse

    headers = dict(UA)
    url = url.strip()
    if url.endswith(".xml"):
        candidates = [url]
    else:
        parsed = urlparse(url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or parsed.path.split("/")[0]
        base = f"{scheme}://{netloc}"
        path_root = url.rstrip("/")
        candidates = [
            f"{path_root}/sitemap.xml",
            f"{path_root}/sitemap_index.xml",
            f"{base}/sitemap.xml",
            f"{base}/sitemap_index.xml",
            url
        ]

    seen_candidates = set()
    found = []

    for cand in candidates:
        if cand in seen_candidates:
            continue
        seen_candidates.add(cand)
        try:
            r = requests.get(cand, headers=headers, timeout=12)
            if r.status_code == 200 and ("xml" in r.headers.get("content-type", "") or
                                         r.text.strip().startswith("<?xml") or
                                         "<urlset" in r.text or "<sitemapindex" in r.text):
                root = ET.fromstring(r.content)
                # Check for regular sitemap urls
                for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                    if loc.text and loc.text.startswith("http"):
                        u = loc.text.strip()
                        # If child is a sub-sitemap and we still need URLs, recurse once
                        if u.endswith(".xml") and len(found) < max_urls:
                            sub_urls = sitemap_urls(u, max_urls - len(found))
                            found.extend(sub_urls)
                        elif not u.endswith(".xml") and u not in found:
                            found.append(u)
                        if len(found) >= max_urls:
                            break
                if found:
                    return found[:max_urls]
        except Exception:
            continue

    return found[:max_urls]


# =============================================================================== Academic & Research Papers (ArXiv & Crossref)
def arxiv_docs(line, max_papers=20, timeout=12):
    """Search ArXiv and Crossref for research papers. Yields {"title", "text", "url"}."""
    import urllib.parse
    import xml.etree.ElementTree as ET

    query = line.strip()
    if query.lower().startswith("arxiv:"):
        query = query[6:].strip()

    # Strategy 1: ArXiv XML API
    api_url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote_plus(query)}&start=0&max_results={max_papers}"
    found_any = False
    try:
        r = requests.get(api_url, headers=UA, timeout=timeout)
        if r.status_code == 200 and "<entry>" in r.text:
            root = ET.fromstring(r.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            for e in entries:
                title = (e.find("atom:title", ns).text or "").strip().replace("\n", " ")
                summary = (e.find("atom:summary", ns).text or "").strip()
                id_val = (e.find("atom:id", ns).text or "").strip()
                authors = [a.find("atom:name", ns).text for a in e.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
                author_str = ", ".join(authors) if authors else "Unknown"
                if title and summary:
                    text = f"Title: {title}\nAuthors: {author_str}\nArXiv ID: {id_val}\n\nAbstract:\n{summary}"
                    found_any = True
                    yield {"title": title, "text": text, "url": id_val}
    except Exception:
        pass

    # Strategy 2: Crossref Open Academic API fallback if ArXiv was slow or rate limited
    if not found_any:
        try:
            cr_url = f"https://api.crossref.org/works?query={urllib.parse.quote_plus(query)}&rows={max_papers}"
            cr_headers = {"User-Agent": "TrainerBoi/1.0 (academic research crawler; mailto:trainerboi@local)"}
            r = requests.get(cr_url, headers=cr_headers, timeout=timeout)
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                for it in items:
                    titles = it.get("title", [])
                    title = titles[0] if titles else "Untitled Paper"
                    abstract = it.get("abstract", "")
                    if abstract:
                        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
                    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]
                    author_str = ", ".join(authors) if authors else "Unknown"
                    url = it.get("URL", f"https://doi.org/{it.get('DOI', '')}")
                    body = abstract if abstract else f"Academic paper on: {query}. Published by {it.get('publisher', 'Open Access')}."
                    text = f"Title: {title}\nAuthors: {author_str}\nSource: {url}\n\nAbstract:\n{body}"
                    yield {"title": title, "text": text, "url": url}
        except Exception:
            pass


# =============================================================================== Curated Hugging Face Datasets
POPULAR_HF_DATASETS = [
    {
        "id": "yahma/alpaca-cleaned",
        "name": "Alpaca Cleaned (52k)",
        "category": "Instruction & Chat",
        "format": "sft",
        "desc": "52,000 high-quality general instruction examples, curated & cleaned from Stanford Alpaca.",
        "default_rows": 20000,
    },
    {
        "id": "HuggingFaceH4/no_robots||train",
        "name": "No Robots (10k)",
        "category": "Instruction & Chat",
        "format": "sft",
        "desc": "10,000 100% human-written conversational and instruction prompts across diverse domains.",
        "default_rows": 10000,
    },
    {
        "id": "databricks/databricks-dolly-15k",
        "name": "Databricks Dolly (15k)",
        "category": "Instruction & Chat",
        "format": "sft",
        "desc": "15,000 human-authored instruction examples across brainstorming, QA, summarization.",
        "default_rows": 15000,
    },
    {
        "id": "iamtarun/python_code_instructions_18k_alpaca",
        "name": "Python Code Instructions (18k)",
        "category": "Code & Technical",
        "format": "sft",
        "desc": "18,000 Python programming instruction-solution pairs for teaching code generation.",
        "default_rows": 18000,
    },
    {
        "id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "name": "CodeFeedback Filtered (25k)",
        "category": "Code & Technical",
        "format": "sft",
        "desc": "Multi-turn code feedback, refactoring, and repair conversations across 50+ languages.",
        "default_rows": 25000,
    },
    {
        "id": "HuggingFaceFW/fineweb-edu|sample-10BT|train",
        "name": "FineWeb-Edu (10BT Sample)",
        "category": "Knowledge & Pretraining",
        "format": "pretrain",
        "desc": "The highest-quality educational web crawl text dataset, filtered by educational score.",
        "default_rows": 50000,
    },
    {
        "id": "roneneldan/TinyStories",
        "name": "TinyStories (Children's Stories)",
        "category": "Knowledge & Pretraining",
        "format": "pretrain",
        "desc": "Simple synthetic vocabulary stories ideal for small models and scratch pretraining.",
        "default_rows": 20000,
    },
    {
        "id": "wikitext|wikitext-103-v1|train",
        "name": "WikiText-103",
        "category": "Knowledge & Pretraining",
        "format": "pretrain",
        "desc": "High quality, verified Wikipedia articles for language model pretraining.",
        "default_rows": 10000,
    },
    {
        "id": "trl-lib/ultrafeedback_binarized",
        "name": "UltraFeedback Binarized (DPO)",
        "category": "Preference & Alignment (DPO)",
        "format": "dpo",
        "desc": "Chosen vs. rejected responses rated on helpfulness, honesty, and truthfulness.",
        "default_rows": 10000,
    },
    {
        "id": "Anthropic/hh-rlhf",
        "name": "Anthropic Helpful & Harmless (DPO)",
        "category": "Preference & Alignment (DPO)",
        "format": "dpo",
        "desc": "Human preference comparison pairs focused on safety, helpfulness, and tone.",
        "default_rows": 15000,
    },
]


# =============================================================================== Curated Wikipedia Preset Packs
WIKIPEDIA_PRESETS = {
    "AI & Machine Learning": [
        "search: Artificial intelligence",
        "search: Machine learning",
        "search: Deep learning",
        "search: Natural language processing",
        "search: Large language model",
    ],
    "Computer Science & Python": [
        "search: Python programming language",
        "search: Software design pattern",
        "search: Data structure",
        "search: Relational database",
    ],
    "Science & Physics": [
        "search: Quantum mechanics",
        "search: General relativity",
        "search: Astrophysics",
        "search: Thermodynamics",
    ],
    "Medicine & Healthcare": [
        "search: Molecular biology",
        "search: Immunology",
        "search: Pharmacology",
        "search: Human anatomy",
    ],
    "Finance & Economics": [
        "search: Macroeconomics",
        "search: Financial market",
        "search: Corporate finance",
        "search: Monetary policy",
    ],
}


def preview_hf_dataset(spec, max_rows=3):
    """Safely stream the first few rows of any Hugging Face dataset."""
    from datasets import load_dataset
    parts = spec.split("|")
    name = parts[0].strip()
    cfg = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    split = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "train"
    ds = load_dataset(name, cfg, split=split, streaming=True)
    return list(ds.take(max_rows))


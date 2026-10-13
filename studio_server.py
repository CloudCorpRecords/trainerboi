"""
studio_server.py - the TrainerBoi backend + web UI server (what the desktop app talks to).

  python studio_server.py --port 7860

Serves the React UI from ui/dist, a JSON API (/api/...) over the existing Python modules, and mounts the classic Gradio studio at
/classic for the advanced tools that don't have a native screen yet. Starts instantly: heavy libraries load lazily.
"""

import os
import re
import io
import sys
import json
import time
import glob
import shutil
import argparse
import subprocess
import threading
import traceback
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import hw_info
import models_registry as MR

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
UI_DIST = os.path.join(ROOT, "ui", "dist")
UPLOADS = os.path.join(ROOT, "uploads")
SETTINGS_PATH = os.path.join(ROOT, "settings.json")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHILD_ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")

app = FastAPI(title="TrainerBoi", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------- security
# The server only listens on 127.0.0.1, but any web page you visit could still try to talk to a local port (cross-site requests,
# or DNS rebinding). These checks make sure only this app's own pages can use the API.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _host_of(value):
    from urllib.parse import urlparse
    value = (value or "").strip()
    if not value:
        return None
    if "//" not in value:
        value = "//" + value
    return (urlparse(value).hostname or "").lower()


@app.middleware("http")
async def local_only_guard(request: Request, call_next):
    if _host_of(request.headers.get("host")) not in LOCAL_HOSTS:
        return JSONResponse({"detail": "Forbidden host"}, status_code=403)            # DNS-rebinding defence
    origin = request.headers.get("origin")
    if origin and origin != "null" and _host_of(origin) not in LOCAL_HOSTS:
        return JSONResponse({"detail": "Forbidden origin"}, status_code=403)          # cross-site requests
    if request.url.path.startswith("/api/") and request.headers.get("sec-fetch-site") == "cross-site":
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


_NAME_OK = re.compile(r"^[\w][\w .\-]{0,119}$")


def nm(name):
    """Validate a run / dataset name taken from a URL: no path separators, no '..', no hidden names."""
    if not name or not _NAME_OK.match(name) or ".." in name:
        raise HTTPException(400, "Invalid name")
    return name


def jread(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def clean_title(s):
    """Drop the leading emoji from the legacy titles ('📄 Teach it…' -> 'Teach it…')."""
    return re.sub(r"^[^\w(]+", "", s or "").strip()


def err(msg, code=400):
    raise HTTPException(code, msg)


# ================================================================================================ settings
def load_settings():
    s = jread(SETTINGS_PATH, {}) or {}
    if "gpu_max_temp" in s:
        os.environ["TB_MAX_GPU_TEMP"] = str(s["gpu_max_temp"])
    CHILD_ENV.update({k: v for k, v in os.environ.items() if k.startswith("TB_")})
    return s


@app.get("/api/settings")
def get_settings():
    s = load_settings()
    return {"gpu_max_temp": float(s.get("gpu_max_temp", os.environ.get("TB_MAX_GPU_TEMP", 82))), "root": ROOT,
            "version": "1.0"}


@app.post("/api/settings")
async def set_settings(req: Request):
    body = await req.json()
    s = jread(SETTINGS_PATH, {}) or {}
    if "gpu_max_temp" in body:
        s["gpu_max_temp"] = max(0, min(95, float(body["gpu_max_temp"])))
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f)
    load_settings()
    return get_settings()


# ================================================================================================ hardware
@app.get("/api/hardware")
def hardware():
    s = hw_info.snapshot()
    s["jobs"] = hw_info.active_jobs()
    return s


@app.get("/api/gpu/health")
def gpu_health():
    return jread(os.path.join(ROOT, "gpu_health.json"), None)


# ================================================================================================ background jobs (zip, publish, gpu check)
JOBS = {}


def start_job(fn, *a):
    jid = f"{int(time.time() * 1000)}"
    JOBS[jid] = {"state": "running", "log": [], "result": None, "error": None}

    def run():
        try:
            JOBS[jid]["result"] = fn(JOBS[jid], *a)
            JOBS[jid]["state"] = "done"
        except Exception as e:
            JOBS[jid].update(state="error", error=str(e))
    threading.Thread(target=run, daemon=True).start()
    return jid


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    return JOBS.get(jid) or err("Unknown job", 404)


@app.post("/api/gpu/check")
def gpu_check_start():
    if hw_info.active_jobs():
        err("Training is using the GPU right now. Run the check when it has finished.")

    def run(job):
        p = subprocess.Popen([sys.executable, "-u", "gpu_guard.py", "--check", "150"], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=CHILD_ENV, creationflags=NOWIN)
        for line in p.stdout:
            m = re.match(r"\s*(\d+)/(\d+)s\s+([\d.]+|None) °C\s+errors (\d+)", line)
            if m:
                job["progress"] = {"done": int(m.group(1)), "total": int(m.group(2)), "temp": m.group(3), "errors": int(m.group(4))}
            job["log"] = (job["log"] + [line.rstrip()])[-30:]
        p.wait()
        return jread(os.path.join(ROOT, "gpu_health.json"))
    return {"job": start_job(run)}


# ================================================================================================ uploads + data intake
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    folder = os.path.join(UPLOADS, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    out = []
    for f in files:
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", os.path.basename(f.filename or "file")).strip(". ") or "file"
        path = os.path.join(folder, name)
        with open(path, "wb") as dst:
            shutil.copyfileobj(f.file, dst)
        out.append({"name": name, "path": path, "size": os.path.getsize(path)})
    return out


@app.post("/api/intake")
async def intake(req: Request):
    """Paste anything -> which sources it is, plus a suggestion for goal and method."""
    import data_intake
    body = await req.json()
    fields, note = data_intake.route(body.get("text", ""))
    paths = list(fields["folders"]) + list(body.get("files", []))
    goal = method = None
    advice = ""
    if paths:
        goal, method, advice = data_intake.inspect(paths)
    elif fields["github"]:
        goal = "code"
    elif fields["urls"] or fields["youtube"] or fields["wikipedia"] or fields["arxiv"]:
        goal = "docs"
    return {"fields": fields, "note": note, "goal": goal, "method": method, "advice": advice}


# ================================================================================================ guided training
MODEL_CHOICES = [
    {"id": "", "label": "Automatic (recommended)"},
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "label": "Qwen2.5 0.5B - fastest"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5 1.5B"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5 3B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B - best, slowest"},
    {"id": "unsloth/Llama-3.2-1B-Instruct", "label": "Llama 3.2 1B"},
    {"id": "unsloth/Llama-3.2-3B-Instruct", "label": "Llama 3.2 3B"},
    {"id": "HuggingFaceTB/SmolLM2-360M-Instruct", "label": "SmolLM2 360M - tiny"},
]


@app.get("/api/guided/meta")
def guided_meta():
    import autopilot
    return {"goals": [{"key": k, "title": clean_title(g["title"]), "blurb": g["blurb"], "needs": g["needs"], "data_hint": g["data_hint"],
                       "task": g["task"]} for k, g in autopilot.GOALS.items()],
            "efforts": [{"key": k, "title": clean_title(e["title"]), "minutes": e["minutes"], "blurb": e["blurb"]}
                        for k, e in autopilot.EFFORT.items()],
            "models": MODEL_CHOICES, "vram_gb": hw_info.hw_identity().get("vram_gb") or 0}


@app.post("/api/guided/plan")
async def guided_plan(req: Request):
    import autopilot
    b = await req.json()
    goal, effort, has_data = b["goal"], b["effort"], bool(b.get("has_data"))
    vram = hw_info.hw_identity().get("vram_gb") or 0
    persona = (b.get("persona") or "").strip() or None
    override = b.get("model") or None
    dp = autopilot.data_plan(goal, effort, has_data)
    guess = {"train_examples": int(dp["max_synth_chunks"] * 3 * 0.85)} if dp["synthetic"] else None
    tp = autopilot.train_plan(goal, effort, guess, vram, persona, override)
    total = autopilot.data_estimate_sec(goal, effort, has_data) + tp["estimate"]["train_sec"] + 90
    g = autopilot.GOALS[goal]
    if dp["synthetic"]:
        data = (f"Read your files, clean them, then a helper AI ({dp['gen_model'].split('/')[-1]}) writes up to "
                f"{dp['max_synth_chunks'] * 3:,} question/answer pairs from them.")
    elif has_data:
        data = "Read and clean your data, remove duplicates and junk, and convert it to the right format."
    else:
        data = f"Download up to {dp['max_rows']:,} examples from {dp['hf'][0].split('|')[0]}, then clean and check them."
    if tp["task"] == "pretrain":
        train = f"Build a new {tp['size']} model and train it for {tp['max_steps']:,} steps."
    else:
        train = f"Fine-tune {tp['model'].split('/')[-1]} with QLoRA for about {tp['max_steps']:,} steps (adjusted once we know your data size)."
    return {"minutes": round(total / 60), "vram_gb": tp["estimate"]["vram_gb"], "needs_data": g["needs"] == "files" and not has_data,
            "downloads_model": tp["task"] != "pretrain", "steps": [{"title": "Prepare data", "text": data}, {"title": "Train", "text": train},
                                                                     {"title": "Test", "text": "Ask the model the same questions before and after training so you can compare."}],
            "reasons": tp["reasons"], "model": tp.get("model"), "task": tp["task"]}


def _lines(x):
    return [s.strip() for s in (x or []) if str(s).strip()]


@app.post("/api/guided/start")
async def guided_start(req: Request):
    import autopilot
    b = await req.json()
    goal, effort = b["goal"], b["effort"]
    g = autopilot.GOALS[goal]
    sources = _lines(b.get("files")) + _lines(b.get("folders"))
    has_data = bool(sources or _lines(b.get("urls")) or _lines(b.get("hf")) or _lines(b.get("wikipedia")) or _lines(b.get("arxiv"))
                    or _lines(b.get("github")) or _lines(b.get("youtube")) or b.get("dataset"))
    if g["needs"] == "files" and not has_data:
        err("This goal needs your data - add files, a link, a repo or a topic first.")
    for r in glob.glob(os.path.join(RUNS, "*", "autopilot.json")):
        st = jread(r, {}) or {}
        if st.get("state") == "running" and hw_info._alive(st.get("pid")):
            err(f"Another guided run ('{os.path.basename(os.path.dirname(r))}') is still going. Stop it first or wait for it to finish.")
    name = nm(re.sub(r"[^\w.-]+", "-", (b.get("name") or "").strip()).lstrip(".") or f"{goal}-{time.strftime('%m%d-%H%M')}")
    run_dir = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(run_dir, "autopilot.json")):
        err(f"A run named '{name}' already exists - choose another name.")
    os.makedirs(run_dir, exist_ok=True)
    cfg = {"name": name, "goal": goal, "effort": effort, "sources": sources, "ocr": bool(b.get("ocr")), "urls": _lines(b.get("urls")),
           "crawl_depth": 1, "max_pages": 50, "sitemap": True, "hf_user": _lines(b.get("hf")), "max_rows": None,
           "wikipedia": _lines(b.get("wikipedia")), "wiki_per_query": 15, "arxiv": _lines(b.get("arxiv")), "github": _lines(b.get("github")),
           "youtube": _lines(b.get("youtube")), "max_videos": 20, "transcribe": bool(b.get("transcribe")),
           "persona": (b.get("persona") or "").strip() or None, "dataset_ref": b.get("dataset") or None,
           "model_override": b.get("model") or None}
    cfg_path = os.path.join(run_dir, "autopilot_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    log = open(os.path.join(run_dir, "autopilot_console.log"), "w", encoding="utf-8")
    subprocess.Popen([sys.executable, "-u", "autopilot_run.py", "--config", cfg_path], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     env=CHILD_ENV, creationflags=NOWIN)
    return {"name": name}


def _tail_jsonl(path, n):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out
    except Exception:
        return []


@app.get("/api/run/{name}")
def run_detail(name: str):
    nm(name)
    d = os.path.join(RUNS, name)
    if not os.path.isdir(d):
        err("Run not found", 404)
    ap = jread(os.path.join(d, "autopilot.json"))
    live = jread(os.path.join(d, "live", "status.json"))
    state = (ap or {}).get("state") or (live or {}).get("state") or "unknown"
    pid = (ap or {}).get("pid") or (live or {}).get("pid")
    if state == "running" and not hw_info._alive(pid):
        state = "crashed"
    metrics = _tail_jsonl(os.path.join(d, "live", "metrics.jsonl"), 600)
    series = [{"step": m.get("step"), "loss": m.get("loss"), "eval_loss": m.get("eval_loss"), "lr": m.get("learning_rate"),
               "grad": m.get("grad_norm"), "tps": m.get("tokens_per_s")} for m in metrics]
    sub = None
    if ap and ap.get("current") == "data" and ap.get("dataset"):
        b = jread(os.path.join(ROOT, "data", ap["dataset"], "build", "status.json"), {}) or {}
        p = b.get("progress", {})
        sub = {"phase": b.get("phase"), "done": p.get("done"), "total": p.get("total"), "eta_s": p.get("eta_s"),
               "kept": (b.get("counters") or {}).get("examples kept")}
    goal = None
    if ap:
        import autopilot
        goal = clean_title(autopilot.GOALS.get(ap.get("goal"), {}).get("title"))
    log_path = os.path.join(d, "autopilot_console.log")
    if not os.path.exists(log_path):
        log_path = os.path.join(d, "live", "console.log")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log = f.read()[-6000:]
    except Exception:
        log = ""
    import friendly_errors
    help_ = friendly_errors.explain((ap or {}).get("error") or (live or {}).get("error") or "")
    return {"name": name, "state": state, "help": help_, "autopilot": ap, "goal_title": goal, "live": live, "series": series,
            "samples": _tail_jsonl(os.path.join(d, "live", "samples.jsonl"), 8), "sub": sub, "log": log,
            "model_id": f"{name}::{(live or {}).get('task') or ((ap or {}).get('plan') or {}).get('task') or 'sft'}"}


@app.post("/api/run/{name}/stop")
def run_stop(name: str):
    nm(name)
    d = os.path.join(RUNS, name)
    ap_path = os.path.join(d, "autopilot.json")
    ap = jread(ap_path)
    pids = set()
    if ap:
        pids |= {ap.get("pid"), ap.get("child_pid")}
    live = jread(os.path.join(d, "live", "status.json"))
    if live:
        pids.add(live.get("pid"))
    for pid in filter(None, pids):
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NOWIN)
    if ap:
        ap["state"] = "stopped"
        for s in ap.get("steps", []):
            if s["state"] == "running":
                s["state"], s["detail"] = "error", "Stopped by you."
        with open(ap_path, "w", encoding="utf-8") as f:
            json.dump(ap, f)
    if live:
        live.update(state="stopped", phase="Stopped by user")
        with open(os.path.join(d, "live", "status.json"), "w", encoding="utf-8") as f:
            json.dump(live, f)
    return {"ok": True}


@app.post("/api/run/{name}/retest")
def run_retest(name: str):
    nm(name)
    """Re-run only the before/after test of a finished training (e.g. after a GPU hiccup)."""
    ap = jread(os.path.join(RUNS, name, "autopilot.json"))
    if not ap or not any(s["id"] == "train" and s["state"] == "done" for s in ap.get("steps", [])):
        err("This run has no finished training to test.")
    if hw_info._alive(ap.get("pid")) and ap.get("state") == "running":
        err("This run is still working.")
    cfg_path = os.path.join(RUNS, name, "autopilot_config.json")
    log = open(os.path.join(RUNS, name, "autopilot_console.log"), "a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-u", "autopilot_run.py", "--config", cfg_path, "--test-only"], cwd=ROOT, stdout=log,
                     stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=NOWIN)
    return {"ok": True}


@app.get("/api/runs")
def runs_list():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*")), key=os.path.getmtime, reverse=True):
        if not os.path.isdir(d) or os.path.basename(d) == "exported":
            continue
        name = os.path.basename(d)
        ap = jread(os.path.join(d, "autopilot.json"))
        live = jread(os.path.join(d, "live", "status.json"))
        ms = jread(os.path.join(d, "math", "status.json"))
        if not (ap or live or ms):
            continue
        kind = "math" if ms else "guided" if ap else (live or {}).get("task", "run")
        state = (ms or ap or live or {}).get("state", "unknown")
        pid = (ms or ap or live or {}).get("pid")
        if state == "running" and not hw_info._alive(pid):
            state = "crashed"
        title = name
        if ap:
            import autopilot
            title = clean_title(autopilot.GOALS.get(ap.get("goal"), {}).get("title")) or name
        out.append({"name": name, "kind": kind, "state": state, "title": title, "updated": os.path.getmtime(d),
                    "loss": ((live or {}).get("latest") or {}).get("loss"), "best_eval": (live or {}).get("best_eval_loss")})
    return out[:30]


# ================================================================================================ models
@app.get("/api/models")
def models_list():
    return [{k: v for k, v in r.items() if k not in ("results", "plan")} for r in MR.scan()]


def _rec(mid):
    r = MR.by_id(mid)
    if not r:
        err("Model not found", 404)
    return r


@app.get("/api/models/{mid:path}/card")
def model_card(mid: str):
    r = _rec(mid)
    return {"markdown": MR.model_card(r, with_license_lookup=False), "results": (r.get("results") or {}).get("pairs", [])[:4]}


@app.post("/api/models/{mid:path}/meta")
async def model_meta(mid: str, req: Request):
    b = await req.json()
    MR.save_meta(_rec(mid), b.get("title"), b.get("notes"))
    return {"ok": True}


@app.post("/api/models/{mid:path}/zip")
async def model_zip(mid: str, req: Request):
    b = await req.json()
    r = _rec(mid)
    kind = "adapter" if (r["adapter"] and b.get("kind") == "adapter") else "full"

    def run(job):
        job["log"].append("Preparing the package…")
        pkg = MR.build_package(r, kind, log=lambda m: job["log"].append(m))
        zp = MR.make_zip(pkg)
        shutil.rmtree(pkg, ignore_errors=True)
        return {"path": zp, "name": os.path.basename(zp), "size_mb": round(os.path.getsize(zp) / 1024 ** 2, 1)}
    return {"job": start_job(run)}


@app.post("/api/models/{mid:path}/publish")
async def model_publish(mid: str, req: Request):
    b = await req.json()
    r = _rec(mid)
    kind = "adapter" if (r["adapter"] and b.get("kind") == "adapter") else "full"
    repo = (b.get("repo") or "").strip() or re.sub(r"[^\w.-]+", "-", r["title"])

    def run(job):
        pkg = MR.build_package(r, kind, log=lambda m: job["log"].append(m))
        job["log"].append("Uploading…")
        return {"url": MR.publish_to_hub(pkg, repo, bool(b.get("private", True)), (b.get("token") or "").strip() or None)}
    return {"job": start_job(run)}


@app.post("/api/models/{mid:path}/ollama")
def model_ollama(mid: str):
    """A folder with the full model + Modelfile + a double-click launcher, ready to import into Ollama / LM Studio."""
    r = _rec(mid)
    if r.get("kind") == "pretrain":
        err("Models built from scratch use a tiny custom layout that Ollama can't read. Share it as a zip instead.")
    if r.get("chat_ok") is False:
        err("This is an embedding model - Ollama can't chat with it.")

    def run(job):
        pkg = MR.build_package(r, "full", log=lambda m: job["log"].append(m))
        MR.write_ollama_files(pkg, r["title"])
        return {"path": pkg, "name": os.path.basename(pkg), "size_mb": round(MR.dir_mb(pkg), 1)}
    return {"job": start_job(run)}


@app.post("/api/lessons")
async def save_lesson(req: Request):
    """Teach-back: keep a chat answer you liked (or your own better one) as a training example in 'lessons-<model>'."""
    import dataset_store as ds
    b = await req.json()
    r = _rec(b.get("model_id", ""))
    prompt, answer = (b.get("prompt") or "").strip(), (b.get("answer") or "").strip()
    if not prompt or not answer:
        err("A lesson needs both a question and an answer.")
    if len(prompt) > 8000 or len(answer) > 8000:
        err("That's a bit long for one lesson - keep it under 8,000 characters.")
    name = nm("lessons-" + (re.sub(r"[^\w-]+", "-", r["title"].lower()).strip("-")[:40] or "model"))
    d = ds.ds_dir(name)
    os.makedirs(d, exist_ok=True)
    train = os.path.join(d, "train.jsonl")
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    if b.get("system"):
        messages.insert(0, {"role": "system", "content": str(b["system"])[:2000]})
    with open(train, "a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    with open(train, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    chars = os.path.getsize(train)
    stats = {"name": name, "format": "sft", "train_examples": n, "eval_examples": 0, "total_tokens": chars // 4, "tokens_estimated": True,
             "output_format": "jsonl", "files": ["train.jsonl"], "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "train_path": train.replace("\\", "/"), "sources": [f"Chats with {r['title']}"], "dropped": {}}
    with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return {"name": name, "count": n}


@app.post("/api/models/{mid:path}/delete")
def model_delete(mid: str):
    r = _rec(mid)
    CHAT.update(id=None, model=None, tok=None)
    path = r["path"] if r["run"].startswith("exported/") else os.path.join(RUNS, r["run"].split("/")[0])
    shutil.rmtree(path, ignore_errors=True)
    return {"ok": True}


@app.get("/api/download")
def download(path: str):
    p = os.path.realpath(path)
    base = os.path.realpath(MR.EXPORTS) + os.sep
    if not p.startswith(base) or not os.path.isfile(p):
        err("Not found", 404)
    return FileResponse(p, filename=os.path.basename(p))


@app.post("/api/reveal")
async def reveal(req: Request):
    b = await req.json()
    p = os.path.realpath(b.get("path", ""))
    if os.path.exists(p) and (p + os.sep).startswith(os.path.realpath(ROOT) + os.sep):
        subprocess.Popen(["explorer", "/select,", p])
    return {"ok": True}


# ---------------------------------------------------------------- chat (streaming)
CHAT = {"id": None, "model": None, "tok": None}
CHAT_LOCK = threading.Lock()


def _chat_load(rec):
    if CHAT["id"] == rec["id"] and CHAT["model"] is not None:
        return
    import gc
    CHAT.update(id=None, model=None, tok=None)
    gc.collect()
    import chat_llm
    import llm_data
    model, tok = chat_llm.load(rec["path"])
    if rec["kind"] != "pretrain":
        llm_data.ensure_chat_template(tok)
    CHAT.update(id=rec["id"], model=model, tok=tok)


@app.post("/api/chat")
async def chat(req: Request):
    b = await req.json()
    rec = _rec(b["model_id"])
    if not rec.get("chat_ok", True):
        err("This is an embedding model - it can't chat.")
    messages = b.get("messages", [])

    def gen():
        from threading import Thread
        from transformers import TextIteratorStreamer
        with CHAT_LOCK:
            try:
                yield f"data: {json.dumps({'status': 'loading'})}\n\n"
                _chat_load(rec)
                model, tok = CHAT["model"], CHAT["tok"]
                raw = rec["kind"] == "pretrain" or not getattr(tok, "chat_template", None)
                if raw:
                    enc = tok(messages[-1]["content"], return_tensors="pt").to(model.device)
                else:
                    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
                stream = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                Thread(target=model.generate, kwargs=dict(**enc, streamer=stream, max_new_tokens=int(b.get("max_new_tokens", 320)), do_sample=True,
                                                          temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id or tok.eos_token_id)).start()
                for chunk in stream:
                    yield f"data: {json.dumps({'t': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/chat/unload")
def chat_unload():
    import gc
    CHAT.update(id=None, model=None, tok=None)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"ok": True}


# ---------------------------------------------------------------- temporary public link
SHARE = {"proc": None, "id": None, "log": None, "port": 7890}


@app.post("/api/models/{mid:path}/share/start")
def share_start(mid: str):
    r = _rec(mid)
    if SHARE["proc"] and SHARE["proc"].poll() is None:
        err("A public link is already running. Stop it first.")
    chat_unload()
    log_path = os.path.join(MR.EXPORTS, "share_chat.log")
    os.makedirs(MR.EXPORTS, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    SHARE.update(proc=subprocess.Popen([sys.executable, "-u", "share_chat.py", "--model", r["path"], "--public", "--title", r["title"],
                                        "--port", str(SHARE["port"])], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV,
                                       creationflags=NOWIN), id=r["id"], log=log_path)
    return {"ok": True}


@app.post("/api/share/stop")
def share_stop():
    p = SHARE["proc"]
    if p and p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, creationflags=NOWIN)
    SHARE["proc"] = None
    return {"state": "stopped"}


@app.get("/api/share")
def share_status():
    p = SHARE["proc"]
    if not p:
        return {"state": "off"}
    txt = ""
    try:
        txt = open(SHARE["log"], encoding="utf-8", errors="replace").read()
    except Exception:
        pass
    if p.poll() is not None:
        return {"state": "stopped", "log": txt[-500:], "model_id": SHARE["id"]}
    m = re.search(r"https://[a-z0-9-]+\.gradio\.live", txt)
    return {"state": "live" if m else "starting", "url": m.group(0) if m else None, "model_id": SHARE["id"]}


# ================================================================================================ math lab
@app.post("/api/math/start")
async def math_start(req: Request):
    b = await req.json()
    for p in glob.glob(os.path.join(RUNS, "*", "math", "status.json")):
        st = jread(p, {}) or {}
        if st.get("state") == "running" and hw_info._alive(st.get("pid")):
            err("A Math Lab run is still going. Stop it first.")
    name = nm(re.sub(r"[^\w.-]+", "-", (b.get("name") or "").strip()).lstrip(".") or f"math-{time.strftime('%m%d-%H%M')}")
    if os.path.exists(os.path.join(RUNS, name, "math", "status.json")):
        err(f"A run named '{name}' exists - choose another name.")
    d = os.path.join(RUNS, name, "math")
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, "-u", "math_lab.py", "--name", name, "--model", b.get("model") or "Qwen/Qwen2.5-0.5B-Instruct",
           "--rounds", str(int(b.get("rounds", 3))), "--problems", str(int(b.get("problems", 400))),
           "--samples", str(int(b.get("samples", 4))), "--max_level", str(int(b.get("max_level", 3)))]
    log = open(os.path.join(d, "console.log"), "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=NOWIN)
    return {"name": name}


@app.get("/api/math/runs")
def math_runs():
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "math", "status.json")), key=os.path.getmtime, reverse=True):
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        st = jread(p, {}) or {}
        state = st.get("state")
        if state == "running" and not hw_info._alive(st.get("pid")):
            state = "crashed"
        out.append({"name": name, "state": state, "model": st.get("model")})
    return out


@app.get("/api/math/{name}")
def math_detail(name: str):
    nm(name)
    st = jread(os.path.join(RUNS, name, "math", "status.json"))
    if not st:
        err("Not found", 404)
    if st.get("state") == "running" and not hw_info._alive(st.get("pid")):
        st["state"] = "crashed"
    import math_gym
    st["skill_list"] = list(math_gym.SKILLS)
    st.pop("config", None)
    import friendly_errors
    st["help"] = friendly_errors.explain(st.get("error") or "")
    return st


@app.post("/api/math/{name}/stop")
def math_stop(name: str):
    nm(name)
    p = os.path.join(RUNS, name, "math", "status.json")
    st = jread(p)
    if st:
        subprocess.run(["taskkill", "/PID", str(st.get("pid")), "/T", "/F"], capture_output=True, creationflags=NOWIN)
        st.update(state="stopped", phase="Stopped by you")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st, f)
    return {"ok": True}


# ================================================================================================ datasets
@app.get("/api/datasets")
def datasets():
    import dataset_store as ds
    out = []
    for d in ds.list_datasets():
        a = jread(os.path.join(ds.ds_dir(d["name"]), "analysis.json"))
        st = jread(os.path.join(ds.ds_dir(d["name"]), "stats.json"), {}) or {}
        out.append({**{k: d.get(k) for k in ("name", "format", "train", "eval", "state", "size_mb", "tokens", "modified")},
                    "grade": (a or {}).get("grade"), "health": (a or {}).get("health"), "train_path": st.get("train_path")})
    return out


@app.post("/api/datasets/{name}/analyze")
def dataset_analyze(name: str):
    nm(name)
    import data_analyzer

    def run(job):
        return data_analyzer.analyze(name)
    return {"job": start_job(run)}


@app.get("/api/datasets/{name}/analysis")
def dataset_analysis(name: str):
    nm(name)
    import dataset_store as ds
    return jread(os.path.join(ds.ds_dir(name), "analysis.json"), None)


@app.post("/api/datasets/{name}/clean")
async def dataset_clean(name: str, req: Request):
    import dataset_store as ds
    nm(name)
    b = await req.json()
    out = nm((b.get("out") or f"{name}-clean").strip())
    if os.path.exists(ds.ds_dir(out)):
        err(f"'{out}' already exists.")
    cmd = [sys.executable, "-u", "data_analyzer.py", "clean", "--name", name, "--out", out, "--drop", ",".join(b.get("drop", [])),
           "--min_score", str(int(b.get("min_score", 0)))]
    if b.get("dedup"):
        cmd.append("--dedup_prompts")
    os.makedirs(os.path.join(ds.ds_dir(out), "build"), exist_ok=True)
    log = open(os.path.join(ds.ds_dir(out), "build", "console.log"), "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=NOWIN)
    return {"name": out}



# ================================================================================================ workshop (manual training with every setting)
TASKS = {
    "sft": dict(script="train_llm_sft.py", lr=2e-4, goal="instruct"), "dpo": dict(script="train_llm_dpo.py", lr=5e-6, goal="prefer"),
    "orpo": dict(script="train_llm_orpo.py", lr=8e-6, goal="prefer"), "kto": dict(script="train_llm_kto.py", lr=5e-6, goal="prefer"),
    "grpo": dict(script="train_llm_grpo.py", lr=5e-6, goal="instruct"), "pretrain": dict(script="pretrain_llm.py", lr=1e-3, goal="scratch"),
}
# the friendly method cards -> what actually runs
METHOD_RUN = {"sft": ("sft", "qlora"), "lora": ("sft", "qlora"), "full": ("sft", "full"), "dpo": ("dpo", "qlora"), "orpo": ("orpo", "qlora"),
              "kto": ("kto", "qlora"), "grpo": ("grpo", "lora"), "pretrain": ("pretrain", None)}
REDIRECT = {"mathlab": "/math", "distill": "/gather"}


@app.get("/api/methods")
def methods_list():
    import methods as M
    out = []
    for m in M.METHODS:
        run = METHOD_RUN.get(m["id"])
        out.append({k: m.get(k) for k in ("id", "name", "icon", "level", "tagline", "analogy", "data", "gpu", "time", "use_when", "avoid_when")}
                   | {"run": run[0] if run else None, "mode": run[1] if run else None, "redirect": REDIRECT.get(m["id"])})
    return {"methods": out, "have": list(M.HAVE), "want": list(M.WANT)}


@app.post("/api/methods/recommend")
async def methods_recommend(req: Request):
    import methods as M
    b = await req.json()
    if b.get("have") not in M.HAVE or b.get("want") not in M.WANT:
        err("Pick both answers.")
    mid, why, alts = M.recommend(M.HAVE[b["have"]], M.WANT[b["want"]])
    return {"id": mid, "why": why, "alternatives": alts}


def _stats_for(dataset):
    import dataset_store as ds
    for d in ds.list_datasets():
        st = jread(os.path.join(ds.ds_dir(d["name"]), "stats.json"), {}) or {}
        if st.get("train_path") and dataset and os.path.normpath(st["train_path"]) == os.path.normpath(dataset):
            return st
    return None


@app.post("/api/train/recommend")
async def train_recommend(req: Request):
    import autopilot
    b = await req.json()
    method, effort = b.get("method", "sft"), b.get("effort", "balanced")
    vram = hw_info.hw_identity().get("vram_gb") or 16.0
    if method == "grpo":
        steps = {"quick": 60, "balanced": 200, "best": 600}[effort]
        return {"model": "Qwen/Qwen2.5-0.5B-Instruct", "steps": steps, "lr": 5e-6, "batch": 1, "grad_accum": 4, "seq_len": 512, "lora_r": 16,
                "mode": "lora", "minutes": round(steps * 0.45), "reasons": ["**Reinforcement learning is slow per step** (the model writes several answers to every question), so it needs few steps. "
                                                                           "A small model, 4 answers per question and a low learning rate keep it stable."]}
    plan = autopilot.train_plan(TASKS[method]["goal"], effort, _stats_for(b.get("dataset")), vram)
    if plan["task"] == "pretrain":
        return {"size": plan["size"], "steps": plan["max_steps"], "lr": plan["lr"], "batch": plan["batch_size"], "grad_accum": plan["grad_accum"],
                "seq_len": plan["seq_len"], "lora_r": 16, "mode": None, "minutes": round(plan["estimate"]["train_sec"] / 60), "reasons": plan["reasons"]}
    return {"model": plan["model"], "steps": plan["max_steps"], "lr": plan["lr"], "batch": plan["batch_size"], "grad_accum": plan["grad_accum"],
            "seq_len": plan["max_seq_len"], "lora_r": plan["lora_r"], "mode": "qlora", "minutes": round(plan["estimate"]["train_sec"] / 60), "reasons": plan["reasons"]}


@app.post("/api/train/start")
async def train_start(req: Request):
    b = await req.json()
    method = b.get("method")
    if method not in TASKS:
        err("Unknown method.")
    spec = TASKS[method]
    dataset = (b.get("dataset") or "").strip()
    if not dataset:
        err("Choose a dataset first.")
    if not (os.path.exists(dataset) or os.path.exists(os.path.join(ROOT, dataset)) or dataset in ("demo", "math_gym") or "/" in dataset):
        err(f"I can't find the dataset '{dataset}'. Pick one from the list or use a Hugging Face id like owner/name.")
    if hw_info.active_jobs():
        err("Something is already using the GPU. Wait for it to finish, or stop it first.")
    import shutil as _sh
    if _sh.disk_usage(ROOT).free / 1024 ** 3 < 8:
        err("Less than 8 GB of disk space is free - training needs room. Free some space first.")
    name = nm(re.sub(r"[^\w.-]+", "-", (b.get("name") or "").strip()).lstrip(".") or f"{method}-{time.strftime('%m%d-%H%M%S')}")
    if os.path.exists(os.path.join(RUNS, name)):
        err(f"A run called '{name}' already exists - choose another name.")
    out_dir = f"runs/{name}"
    os.makedirs(os.path.join(RUNS, name, "live"), exist_ok=True)
    cmd = [sys.executable, "-u", spec["script"], "--dataset", dataset, "--output_dir", out_dir, "--max_steps", str(int(b.get("steps", 300))),
           "--lr", str(float(b.get("lr", spec["lr"]))), "--batch_size", str(int(b.get("batch", 4))), "--sample_every", str(int(b.get("sample_every", 50)))]
    if b.get("sample_prompt"):
        cmd += ["--sample_prompt", b["sample_prompt"]]
    model, mode = b.get("model") or "Qwen/Qwen2.5-0.5B-Instruct", b.get("mode") or "qlora"
    ga, seq, r = str(int(b.get("grad_accum", 4))), str(int(b.get("seq_len", 1024))), str(int(b.get("lora_r", 16)))
    if method == "sft":
        cmd += ["--model", model, "--mode", mode, "--grad_accum", ga, "--max_seq_len", seq, "--lora_r", r]
        if mode == "full":
            cmd += ["--backend", "hf"]
    elif method in ("dpo", "orpo", "kto"):
        cmd += ["--model", model, "--grad_accum", ga, "--max_len", seq, "--lora_r", r]
        if mode == "qlora":
            cmd.append("--load_4bit")
    elif method == "grpo":
        cmd += ["--model", model, "--grad_accum", ga, "--lora_r", r, "--reward", "math" if ("gsm8k" in dataset or dataset == "math_gym") else (b.get("reward") or "concise")]
        if mode == "qlora":
            cmd.append("--load_4bit")
    else:
        cmd += ["--size", b.get("size") or "tiny", "--seq_len", str(min(int(seq), 1024)), "--grad_accum", ga, "--save_steps", str(max(100, int(b.get("steps", 600)) // 6))]
    log = open(os.path.join(RUNS, name, "live", "console.log"), "w", encoding="utf-8")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=NOWIN)
    return {"name": name}


# ================================================================================================ gather data (native dataset builder)
@app.post("/api/builder/start")
async def builder_start(req: Request):
    b = await req.json()
    name = nm(re.sub(r"[^\w.-]+", "-", (b.get("name") or "").strip()).lstrip(".") or f"data-{time.strftime('%m%d-%H%M')}")
    out = os.path.join(ROOT, "data", name)
    if os.path.exists(os.path.join(out, "stats.json")):
        err(f"A dataset called '{name}' already exists - choose another name.")
    for p in glob.glob(os.path.join(ROOT, "data", "*", "build", "status.json")):
        st = jread(p, {}) or {}
        if st.get("state") == "running" and hw_info._alive(st.get("pid")):
            err("Another dataset is still being built. Wait for it to finish.")
    s = b.get("sources") or {}
    fmt = b.get("format", "sft")
    if not any(_lines(s.get(k)) for k in ("files", "folders", "urls", "hf", "wikipedia", "arxiv", "github", "youtube")):
        err("Add at least one source first - a file, a link, a topic…")
    cmd = [sys.executable, "-u", "dataset_builder.py", "--name", name, "--overwrite", "--format", fmt, "--max_rows", str(int(b.get("max_rows", 20000))),
           "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct", "--eval_ratio", "0.03"]
    if b.get("system_prompt") and fmt == "sft":
        cmd += ["--system_prompt", b["system_prompt"]]
    for flag, key in (("--source", "files"), ("--source", "folders"), ("--url", "urls"), ("--hf", "hf"), ("--github", "github"), ("--wikipedia", "wikipedia"),
                      ("--arxiv", "arxiv"), ("--youtube", "youtube")):
        for v in _lines(s.get(key)):
            cmd += [flag, v]
    if _lines(s.get("urls")):
        cmd += ["--crawl_depth", str(int(b.get("crawl_depth", 1))), "--max_pages", str(int(b.get("max_pages", 50))), "--sitemap"]
    if _lines(s.get("wikipedia")):
        cmd += ["--wiki_per_query", "15"]
    if b.get("ocr"):
        cmd.append("--ocr")
    if b.get("transcribe"):
        cmd.append("--transcribe")
    if b.get("redact_pii"):
        cmd.append("--redact_pii")
    if b.get("synthetic"):
        gen = b.get("gen_model") or "Qwen/Qwen2.5-3B-Instruct"
        cmd += ["--synthetic", "--gen_model", gen, "--max_synth_chunks", str(int(b.get("max_synth_chunks", 300)))]
        if "7B" in gen:
            cmd.append("--gen_4bit")
    os.makedirs(os.path.join(out, "build"), exist_ok=True)
    log = open(os.path.join(out, "build", "console.log"), "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV, creationflags=NOWIN)
    return {"name": name}


@app.get("/api/builder/{name}")
def builder_status(name: str):
    nm(name)
    d = os.path.join(ROOT, "data", name)
    st = jread(os.path.join(d, "build", "status.json"))
    stats = jread(os.path.join(d, "stats.json"))
    if not st and not stats:
        return {"state": "starting"}
    state = (st or {}).get("state", "done" if stats else "unknown")
    if state == "running" and not hw_info._alive((st or {}).get("pid")):
        state = "crashed"
    try:
        with open(os.path.join(d, "build", "console.log"), encoding="utf-8", errors="replace") as f:
            tail = f.read()[-2500:]
    except Exception:
        tail = ""
    import friendly_errors
    return {"state": state, "phase": (st or {}).get("phase"), "progress": (st or {}).get("progress"), "counters": (st or {}).get("counters"),
            "dropped": (st or {}).get("dropped"), "sources": (st or {}).get("sources"), "warnings": (st or {}).get("warnings", [])[-4:],
            "error": (st or {}).get("error"), "help": friendly_errors.explain((st or {}).get("error") or tail if state in ("error", "crashed") else ""),
            "stats": stats, "log": tail}


# ================================================================================================ taste test (compare two models side by side)
@app.get("/api/compare/options")
def compare_options():
    out = []
    for r in MR.scan():
        if not r.get("chat_ok", True):
            continue
        out.append({"id": r["id"], "label": r["title"], "group": "Your models"})
        if r.get("base"):
            out.append({"id": "base:" + r["id"], "label": f"{r['title']} - before training ({r['base'].split('/')[-1]})", "group": "Before training"})
    return out


def _load_spec(spec):
    """-> (model, tok, raw). Keeps one model on the GPU at a time."""
    import gc
    if CHAT["id"] == spec and CHAT["model"] is not None:
        rec = _rec(spec[5:] if spec.startswith("base:") else spec)
        return CHAT["model"], CHAT["tok"], rec["kind"] == "pretrain" and not spec.startswith("base:")
    CHAT.update(id=None, model=None, tok=None)
    gc.collect()
    import chat_llm
    import llm_data
    from model_ids import resolve_base
    if spec.startswith("base:"):
        rec = _rec(spec[5:])
        model, tok = chat_llm.load(resolve_base(rec["base"]))
        raw = False
    else:
        rec = _rec(spec)
        model, tok = chat_llm.load(rec["path"])
        raw = rec["kind"] == "pretrain"
    if not raw:
        llm_data.ensure_chat_template(tok)
    CHAT.update(id=spec, model=model, tok=tok)
    return model, tok, raw


@app.post("/api/compare")
async def compare(req: Request):
    b = await req.json()
    prompt = (b.get("prompt") or "").strip()
    if not prompt:
        err("Type a question first.")
    sides = [("a", b["a"]), ("b", b["b"])]

    def gen():
        from threading import Thread
        from transformers import TextIteratorStreamer
        with CHAT_LOCK:
            for side, spec in sides:
                try:
                    yield f"data: {json.dumps({'side': side, 'status': 'loading'})}\n\n"
                    model, tok, raw = _load_spec(spec)
                    if raw or not getattr(tok, "chat_template", None):
                        enc = tok(prompt, return_tensors="pt").to(model.device)
                    else:
                        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
                    stream = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                    t0 = time.time()
                    Thread(target=model.generate, kwargs=dict(**enc, streamer=stream, max_new_tokens=int(b.get("max_new_tokens", 260)), do_sample=True,
                                                              temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id or tok.eos_token_id)).start()
                    n = 0
                    for chunk in stream:
                        n += 1
                        yield f"data: {json.dumps({'side': side, 't': chunk})}\n\n"
                    yield f"data: {json.dumps({'side': side, 'done': True, 'seconds': round(time.time() - t0, 1), 'chunks': n})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'side': side, 'error': str(e)[:300]})}\n\n"
            yield f"data: {json.dumps({'finished': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ================================================================================================ classic Gradio studio (advanced tools)
CLASSIC = {"state": "loading", "error": None}


@app.get("/api/classic/status")
def classic_status():
    return CLASSIC


def _mount_classic():
    try:
        import gradio as gr
        os.environ["TB_EMBED"] = "1"   # the legacy studio hides entries that now have native screens
        import app as classic  # builds the legacy studio UI (slow: loads torch etc.)
        here = ROOT

        def rd(n):
            p = os.path.join(here, n)
            return open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        css = "\n".join([rd("studio_theme.css"), classic.live_dashboard.CSS, classic.quickstart_tab.CSS, rd("redesign.css"), rd("classic_cute.css")])
        before = len(app.router.routes)
        gr.mount_gradio_app(app, classic.demo, path="/classic", css=css, js="() => { document.body.classList.add('dark'); }",
                            theme=gr.themes.Default(primary_hue="indigo", secondary_hue="cyan", neutral_hue="slate"),
                            footer_links=[])
        # Routes added after startup land behind the SPA catch-all; move them to the front so /classic is reachable.
        new = app.router.routes[before:]
        del app.router.routes[before:]
        app.router.routes[0:0] = new
        CLASSIC["state"] = "ready"
    except Exception as e:
        CLASSIC.update(state="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ================================================================================================ UI (single-page app)
@app.get("/")
def index():
    p = os.path.join(UI_DIST, "index.html")
    if not os.path.exists(p):
        return JSONResponse({"error": "UI not built. Run: cd ui && npm install && npm run build"}, status_code=503)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


if os.path.isdir(os.path.join(UI_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(UI_DIST, "assets")), name="assets")


@app.get("/{path:path}")
def spa_fallback(path: str):
    if path.startswith(("api/", "classic", "gradio_api", "assets/")):
        raise HTTPException(404)
    f = os.path.realpath(os.path.join(UI_DIST, path))
    if os.path.isfile(f) and f.startswith(os.path.realpath(UI_DIST) + os.sep):
        return FileResponse(f)
    return index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("TB_PORT", 7860)))
    p.add_argument("--no-classic", action="store_true", help="skip loading the advanced Gradio tools")
    p.add_argument("--open", action="store_true", help="open the app in your browser once it is up")
    a = p.parse_args()
    load_settings()
    if not a.no_classic:
        threading.Thread(target=_mount_classic, daemon=True).start()
    else:
        CLASSIC["state"] = "off"
    print(f"--> TrainerBoi on http://127.0.0.1:{a.port}", flush=True)
    if a.open:
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

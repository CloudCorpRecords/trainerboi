"""
models_registry.py - finds every model you've trained and turns it into something shareable.

scan()                      -> list of model records (Guided runs, Live runs, Math Lab runs, exports)
model_card(rec)             -> markdown model card (what it is, how it was trained, results, how to use it)
build_package(rec, kind)    -> folder ready to share (model files + README card + chat script [+ Ollama Modelfile])
make_zip(pkg_dir)           -> a .zip next to the exports
publish_to_hub(pkg, repo, private, token)
"""

import os
import re
import json
import math
import time
import glob
import shutil
import subprocess
import sys

from model_ids import resolve_base

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
EXPORTS = os.path.join(ROOT, "exports")

TASK_LABEL = {"sft": "Fine-tuned chat model", "dpo": "Preference-tuned (DPO)", "orpo": "ORPO-tuned", "kto": "KTO-tuned",
              "grpo": "Reinforcement-learned (GRPO)", "pretrain": "Built from scratch", "math": "Math Lab model",
              "exported": "Exported model", "embedding": "Embedding model"}


def _j(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 ** 2


def _has_weights(path):
    return bool(path) and os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, f)) for f in ("adapter_model.safetensors", "model.safetensors", "pytorch_model.bin",
                                                         "adapter_model.bin")) or bool(glob.glob(os.path.join(path or "", "model-*.safetensors")))


def _base_of(path):
    cfg = _j(os.path.join(path, "adapter_config.json"))
    return resolve_base(cfg.get("base_model_name_or_path")) if cfg else None


def _ts(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def scan():
    recs, seen = [], set()

    def add(rec):
        p = os.path.abspath(rec["path"])
        if p in seen or not _has_weights(p):
            return
        seen.add(p)
        rec["path"] = p
        rec["adapter"] = os.path.exists(os.path.join(p, "adapter_config.json"))
        cfg = _j(os.path.join(p, "config.json"), {}) or {}
        archs = " ".join(cfg.get("architectures") or [])
        rec["chat_ok"] = rec["adapter"] or "CausalLM" in archs or "LMHead" in archs or not archs
        if not rec["chat_ok"]:
            rec["kind"] = "embedding"
        rec["base"] = rec.get("base") or _base_of(p)
        rec["size_mb"] = round(dir_mb(p), 1)
        rec["created"] = _ts(p)
        meta = _j(os.path.join(RUNS, rec["run"], "model_meta.json"), {}) or {}
        rec["title"] = meta.get("title") or rec["title"]
        rec["notes"] = meta.get("notes", "")
        rec["id"] = f"{rec['run']}::{rec['kind']}"
        recs.append(rec)

    for run_dir in sorted(glob.glob(os.path.join(RUNS, "*")), key=_ts, reverse=True):
        if not os.path.isdir(run_dir):
            continue
        run = os.path.basename(run_dir)
        if run == "exported":
            for sub in glob.glob(os.path.join(run_dir, "*")):
                add({"run": f"exported/{os.path.basename(sub)}", "kind": "exported", "title": os.path.basename(sub), "path": sub,
                     "metric": "", "status": "done", "goal": "Exported model"})
            continue
        # Math Lab
        ms = _j(os.path.join(run_dir, "math", "status.json"))
        if ms and (ms.get("final_model") or ms.get("current_model")):
            r = ms.get("rounds", [])
            metric = f"maths exam {100 * r[0]['overall']:.0f}% → {100 * r[-1]['overall']:.0f}%" if len(r) > 1 else ""
            add({"run": run, "kind": "math", "title": run, "path": ms.get("final_model") or ms.get("current_model"),
                 "base": ms.get("model"), "metric": metric, "status": ms.get("state", ""), "goal": "Maths practice"})
        # Guided (autopilot) run
        ap = _j(os.path.join(run_dir, "autopilot.json"))
        live = _j(os.path.join(run_dir, "live", "status.json"), {}) or {}
        final = (ap or {}).get("train_summary", {}).get("final_dir") or live.get("final_dir") or os.path.join(run_dir, "final")
        final = final if os.path.isabs(final) else os.path.join(ROOT, final)
        loss = (ap or {}).get("train_summary", {}).get("best_eval_loss") or live.get("best_eval_loss")
        task = (live.get("task") or ((ap or {}).get("plan") or {}).get("task") or "sft")
        metric = ""
        if loss:
            metric = f"perplexity {math.exp(min(loss, 20)):.1f}" if task == "pretrain" else f"test loss {loss:.3f}"
        if os.path.isdir(final):
            from autopilot import GOALS
            goal = GOALS.get((ap or {}).get("goal"), {}).get("title") if ap else None
            add({"run": run, "kind": task if task in TASK_LABEL else "sft", "title": run, "path": final,
                 "base": ((ap or {}).get("plan") or {}).get("model") or (live.get("config") or {}).get("model"),
                 "metric": metric, "status": (ap or {}).get("state") or live.get("state", ""), "goal": goal or TASK_LABEL.get(task, task),
                 "guided": bool(ap), "results": (ap or {}).get("results"), "dataset": (ap or {}).get("dataset"),
                 "plan": (ap or {}).get("plan"), "elapsed_s": live.get("elapsed_s")})
    recs.sort(key=lambda r: -r["created"])
    return recs


def by_id(mid):
    return next((r for r in scan() if r["id"] == mid), None)


def save_meta(rec, title=None, notes=None):
    run = rec["run"]
    if run.startswith("exported/"):
        return
    p = os.path.join(RUNS, run, "model_meta.json")
    meta = _j(p, {}) or {}
    if title is not None:
        meta["title"] = title.strip() or rec["run"]
    if notes is not None:
        meta["notes"] = notes
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


# ------------------------------------------------------------------------------------------------ model card
def _license(base):
    if not base or os.path.isdir(base):
        return None
    try:
        from huggingface_hub import model_info
        cd = model_info(base, timeout=6).card_data
        return getattr(cd, "license", None) if cd else None
    except Exception:
        return None


def model_card(rec, with_license_lookup=True):
    base = rec.get("base")
    lic = _license(base) if with_license_lookup else None
    kind = TASK_LABEL.get(rec["kind"], rec["kind"])
    lines = [f"# {rec['title']}", "", f"{kind}, made with [TrainerBoi](https://github.com/) on a single consumer GPU.", ""]
    lines += ["| | |", "|---|---|", f"| Type | {kind}{' (LoRA adapter)' if rec['adapter'] else ''} |"]
    if base:
        lines.append(f"| Base model | `{base}` |")
    if rec.get("goal"):
        lines.append(f"| Goal | {rec['goal']} |")
    if rec.get("metric"):
        lines.append(f"| Result | {rec['metric']} |")
    if rec.get("dataset"):
        lines.append(f"| Training data | `{rec['dataset']}` |")
    plan = rec.get("plan") or {}
    if plan.get("max_steps"):
        lines.append(f"| Training | {plan['max_steps']:,} steps, learning rate {plan.get('lr')} |")
    lines.append(f"| Size | {rec['size_mb']:.0f} MB |")
    if lic:
        lines.append(f"| Base model licence | {lic} |")
    lines.append("")
    if rec.get("notes"):
        lines += [rec["notes"], ""]
    res = (rec.get("results") or {}).get("pairs") or []
    if res:
        lines += ["## Before vs. after training", ""]
        for p in res[:3]:
            lines += [f"**Prompt:** {p['prompt'][:300]}", "", f"- *Before:* {p.get('before', '')[:400]}", f"- *After:* {p.get('after', '')[:400]}", ""]
    lines += ["## How to use", ""]
    if rec["adapter"]:
        lines += ["```python", "from transformers import AutoTokenizer, AutoModelForCausalLM", "from peft import PeftModel",
                  f"base = AutoModelForCausalLM.from_pretrained(\"{base}\", device_map=\"auto\")",
                  "model = PeftModel.from_pretrained(base, \".\")   # this folder", f"tok = AutoTokenizer.from_pretrained(\".\")", "```", ""]
    else:
        lines += ["```python", "from transformers import AutoTokenizer, AutoModelForCausalLM",
                  "tok = AutoTokenizer.from_pretrained(\".\")   # this folder",
                  "model = AutoModelForCausalLM.from_pretrained(\".\", device_map=\"auto\")", "```", ""]
    lines += ["Or run `python use_this_model.py` in this folder for a simple chat. Full models can also be converted to GGUF "
              "for Ollama / LM Studio (see the included `Modelfile`).", ""]
    if base:
        lines += [f"**Licence note:** this model builds on `{base}`; check and follow its licence"
                  + (f" (**{lic}**)" if lic else "") + " before sharing or using it commercially.", ""]
    return "\n".join(lines)


def _front_matter(rec):
    fm = ["---", "library_name: " + ("peft" if rec["adapter"] else "transformers"), "tags:", "- trainerboi"]
    if rec.get("base"):
        fm.insert(1, f"base_model: {rec['base']}")
    fm.append("---\n")
    return "\n".join(fm)


CHAT_SCRIPT = '''"""Chat with this model:  python use_this_model.py   (needs: pip install torch transformers peft)"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

here = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained(here)
cfg = os.path.join(here, "adapter_config.json")
dev = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists(cfg):
    from peft import PeftModel
    base = json.load(open(cfg))["base_model_name_or_path"]
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to(dev), here)
else:
    model = AutoModelForCausalLM.from_pretrained(here, torch_dtype=torch.bfloat16).to(dev)
model.eval()
history = []
print("Type a message (Ctrl+C to quit).")
while True:
    history.append({"role": "user", "content": input("you > ")})
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
    else:
        ids = tok(history[-1]["content"], return_tensors="pt").to(dev)
    stream = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(**ids, streamer=stream, max_new_tokens=400, do_sample=True, temperature=0.7)).start()
    reply = ""
    print("bot > ", end="", flush=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
        reply += chunk
    print()
    history.append({"role": "assistant", "content": reply})
'''

MODELFILE = '''FROM .

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.7
'''


def _safe(name):
    return re.sub(r"[^\w.-]+", "-", name).strip("-") or "model"


def build_package(rec, kind="auto", log=print):
    """kind: 'adapter' (small, needs the base model), 'full' (merged standalone) or 'auto'. Returns the package folder."""
    name = _safe(rec["title"])
    pkg = os.path.join(EXPORTS, name + ("-adapter" if rec["adapter"] and kind == "adapter" else ""))
    shutil.rmtree(pkg, ignore_errors=True)
    os.makedirs(EXPORTS, exist_ok=True)
    src = rec["path"]
    full = (rec["adapter"] and kind in ("full", "auto"))
    if rec["adapter"] and kind == "adapter":
        shutil.copytree(src, pkg)
        cfg_path = os.path.join(pkg, "adapter_config.json")      # point at the normal base model, not Unsloth's 4-bit copy
        cfg = _j(cfg_path, {}) or {}
        if cfg.get("base_model_name_or_path"):
            cfg["base_model_name_or_path"] = resolve_base(cfg["base_model_name_or_path"])
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        card_rec = dict(rec)
    elif full:
        log("Merging the LoRA adapter into the base model (about a minute)…")
        merged = os.path.join(RUNS, rec["run"], "share_full") if not rec["run"].startswith("exported/") else pkg
        r = subprocess.run([sys.executable, "merge_lora.py", "--adapter", src, "--out", merged], cwd=ROOT, capture_output=True,
                           text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore"))
        if r.returncode != 0:
            raise RuntimeError("Merging failed:\n" + (r.stderr or r.stdout)[-1200:])
        shutil.copytree(merged, pkg, dirs_exist_ok=True)
        card_rec = dict(rec, adapter=False)
    else:
        shutil.copytree(src, pkg)
        card_rec = dict(rec)
    for junk in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"):
        try:
            os.remove(os.path.join(pkg, junk))
        except OSError:
            pass
    card_rec["size_mb"] = round(dir_mb(pkg), 1)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as f:
        f.write(_front_matter(card_rec) + model_card(card_rec))
    with open(os.path.join(pkg, "use_this_model.py"), "w", encoding="utf-8") as f:
        f.write(CHAT_SCRIPT)
    if not card_rec["adapter"]:
        with open(os.path.join(pkg, "Modelfile"), "w", encoding="utf-8") as f:
            f.write(MODELFILE)
    return pkg


def make_zip(pkg):
    return shutil.make_archive(pkg, "zip", os.path.dirname(pkg), os.path.basename(pkg))


def publish_to_hub(pkg, repo_id, private=True, token=None):
    from huggingface_hub import HfApi
    api = HfApi(token=token or None)
    try:
        who = api.whoami()["name"]
    except Exception:
        raise RuntimeError("Not signed in to Hugging Face. Paste an access token with *write* permission "
                           "(huggingface.co/settings/tokens) or run `.venv\\Scripts\\hf.exe auth login` once.")
    if "/" not in repo_id:
        repo_id = f"{who}/{repo_id}"
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=pkg, repo_id=repo_id, repo_type="model", commit_message="Upload from TrainerBoi")
    return f"https://huggingface.co/{repo_id}"

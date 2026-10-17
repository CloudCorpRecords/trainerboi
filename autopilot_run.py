"""
autopilot_run.py - Runs a Quick Start job end to end, hands-free:

  1. Check      GPU, disk space
  2. Prepare    build a clean dataset from the user's data or a suggested public dataset (dataset_builder.py)
  3. Plan       pick model + all settings from the real dataset stats (autopilot.train_plan)
  4. Train      run the right trainer (SFT / DPO / pretrain) -> live telemetry in runs/<name>/live/
  5. Test       answer the same questions before and after training, side by side

Progress is written to runs/<name>/autopilot.json for the Quick Start tab.

  python autopilot_run.py --config runs/<name>/autopilot_config.json
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import traceback

import autopilot

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [("check", "Check your computer"), ("data", "Prepare the data"), ("plan", "Choose the best settings"),
         ("train", "Train the model"), ("test", "Test: before vs. after")]


class Job:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = os.path.join(ROOT, "runs", cfg["name"])
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, "autopilot.json")
        self.log = open(os.path.join(self.run_dir, "autopilot.log"), "a", encoding="utf-8")
        self.s = {"name": cfg["name"], "goal": cfg["goal"], "effort": cfg["effort"], "pid": os.getpid(),
                  "state": "running", "started_at": time.time(), "current": None, "error": None,
                  "steps": [{"id": i, "title": t, "state": "pending", "detail": ""} for i, t in STEPS],
                  "plan": None, "results": None, "dataset": None}
        self.save()

    def save(self):
        self.s["updated_at"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.s, f, indent=1)
        for _ in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05)

    def step(self, sid, state, detail=None):
        for st in self.s["steps"]:
            if st["id"] == sid:
                st["state"] = state
                if detail is not None:
                    st["detail"] = detail
                st["started" if state == "running" else "ended"] = time.time()
        if state == "running":
            self.s["current"] = sid
        self.say(f"[{sid}] {state}: {detail or ''}")
        self.save()

    def say(self, msg):
        print(msg, flush=True)
        self.log.write(msg + "\n")
        self.log.flush()

    def run(self, cmd, sid):
        """Run a child process, streaming its output into the log."""
        self.say("$ " + " ".join(cmd))
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                             text=True, encoding="utf-8", errors="replace",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.s["child_pid"] = p.pid
        self.save()
        tail = []
        for line in p.stdout:
            self.log.write(line)
            tail = (tail + [line.rstrip()])[-40:]
        self.log.flush()
        if p.wait() != 0:
            raise RuntimeError(f"The {sid} step failed. Last output:\n" + "\n".join(tail[-15:]))


def main(cfg):
    job = Job(cfg)
    try:
        _main(job, cfg)
        job.s["state"] = "done"
        job.s["current"] = None
        job.save()
    except Exception as e:
        cur = job.s.get("current")
        if cur:
            job.step(cur, "error", str(e).splitlines()[0][:300])
        job.s.update(state="error", error=f"{e}\n\n{traceback.format_exc()}")
        job.save()
        raise


def _arg(cmd, flag):
    return int(cmd[cmd.index(flag) + 1]) if flag in cmd else None


def _set(cmd, flag, value):
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(int(value))


def shrink_for_memory(cmd):
    """Halve the per-step batch (doubling accumulation so learning is unchanged); at batch 1 shorten sequences instead."""
    cmd = list(cmd)
    batch, accum = _arg(cmd, "--batch_size"), _arg(cmd, "--grad_accum")
    if batch and batch > 1:
        _set(cmd, "--batch_size", max(1, batch // 2))
        _set(cmd, "--grad_accum", (accum or 1) * 2)
        return cmd
    for flag in ("--max_seq_len", "--max_len", "--seq_len"):
        n = _arg(cmd, flag)
        if n and n > 384:
            _set(cmd, flag, max(384, int(n * 0.75) // 64 * 64))
            return cmd
    return cmd

def _main(job, cfg):
    goal, effort = cfg["goal"], cfg["effort"]
    g = autopilot.GOALS[goal]
    name = cfg["name"]

    # ------------------------------------------------------------------ 1. check
    job.step("check", "running", "Looking at your GPU and free disk space…")
    gpu = autopilot.gpu_info()
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if not gpu["vram_gb"]:
        raise RuntimeError("No NVIDIA GPU was found by PyTorch — training needs one.")
    if free_gb < 8:
        raise RuntimeError(f"Only {free_gb:.1f} GB of disk space free — please free at least 8 GB (models and data need room).")
    job.s["gpu"] = gpu
    import gpu_guard
    health = json.load(open(gpu_guard.HEALTH_FILE)) if os.path.exists(gpu_guard.HEALTH_FILE) else None
    note = f" · cooling guard on (pauses at {gpu_guard.max_temp_setting():.0f} °C)"
    if health and not health["stable"]:
        note += " · ⚠️ last health check found GPU errors when hot"
    job.step("check", "done", f"{gpu['name']} · {gpu['vram_gb']:.0f} GB GPU memory · {free_gb:.0f} GB free disk{note}")

    # ------------------------------------------------------------------ 2. data
    has_user = bool(cfg.get("sources") or cfg.get("urls") or cfg.get("github") or cfg.get("hf_user")
                    or cfg.get("wikipedia") or cfg.get("arxiv") or cfg.get("youtube"))
    if cfg.get("dataset_ref"):
        ds_name = cfg["dataset_ref"]
        job.step("data", "done", f"Using your existing dataset **{ds_name}** from the Library.")
    else:
        dp = autopilot.data_plan(goal, effort, has_user)
        ds_name = f"{name}-data"
        what = ("Reading your files and writing practice questions from them" if dp["synthetic"] else
                "Reading and cleaning your data" if has_user else f"Downloading {dp['hf'][0].split('|')[0]}")
        job.s["dataset"] = ds_name
        job.step("data", "running", what + "…")
        max_rows = min(dp["max_rows"], cfg.get("max_rows_cap") or cfg.get("max_rows") or dp["max_rows"])
        cmd = [sys.executable, "-u", "dataset_builder.py", "--name", ds_name, "--overwrite", "--format", dp["format"],
               "--max_rows", str(max_rows), "--tokenizer", "Qwen/Qwen2.5-0.5B-Instruct", "--eval_ratio", "0.03"]
        system_prompt = cfg.get("persona") or g["system_prompt"]
        if system_prompt and dp["format"] == "sft":
            cmd += ["--system_prompt", system_prompt]  # bake the persona into every training example
        if cfg.get("synth_chunks_cap"):
            dp["max_synth_chunks"] = min(dp["max_synth_chunks"], cfg["synth_chunks_cap"])
        for s in cfg.get("sources", []):
            cmd += ["--source", s]
        for u in cfg.get("urls", []):
            cmd += ["--url", u]
        if cfg.get("crawl_depth"):
            cmd += ["--crawl_depth", str(int(cfg["crawl_depth"]))]
        if cfg.get("max_pages"):
            cmd += ["--max_pages", str(int(cfg["max_pages"]))]
        if cfg.get("sitemap"):
            cmd.append("--sitemap")
        for gh in cfg.get("github", []):
            cmd += ["--github", gh]
        for h in cfg.get("hf_user", []) + dp["hf"]:
            cmd += ["--hf", h]
        for w in cfg.get("wikipedia", []):
            cmd += ["--wikipedia", w]
        if cfg.get("wiki_per_query"):
            cmd += ["--wiki_per_query", str(int(cfg["wiki_per_query"]))]
        for ar in cfg.get("arxiv", []):
            cmd += ["--arxiv", ar]
        for yt in cfg.get("youtube", []):
            cmd += ["--youtube", yt]
        if cfg.get("max_videos"):
            cmd += ["--max_videos", str(int(cfg["max_videos"]))]
        if cfg.get("ocr"):
            cmd.append("--ocr")
        if cfg.get("transcribe"):
            cmd.append("--transcribe")
        if dp["synthetic"]:
            cmd += ["--synthetic", "--gen_model", dp["gen_model"], "--max_synth_chunks", str(dp["max_synth_chunks"])]
            if "7B" in dp["gen_model"]:
                cmd.append("--gen_4bit")
        if goal == "scratch":
            cmd += ["--chunk_tokens", "512", "--min_chars", "40"]
        job.run(cmd, "data")
    stats = json.load(open(os.path.join(ROOT, "data", ds_name, "stats.json"), encoding="utf-8"))
    job.s["dataset"] = ds_name
    if not cfg.get("dataset_ref"):
        quality = ""
        try:
            import data_analyzer
            rep = data_analyzer.analyze(ds_name, sample=2000)
            quality = f" · data quality {rep['grade']} ({rep['health']:.0f}/100)"
            job.s["data_quality"] = {"grade": rep["grade"], "health": rep["health"],
                                     "top_issue": rep["recommendations"][0]["text"] if rep["recommendations"] else None}
        except Exception as e:
            job.say(f"(data check skipped: {e})")
        job.step("data", "done", f"{stats['train_examples']:,} training examples ready "
                                 f"({stats['total_tokens'] / 1e6:.1f}M tokens) · {stats['eval_examples']:,} kept aside for testing{quality}")
    if stats["format"] != g["format"]:
        raise RuntimeError(f"This goal needs '{g['format']}' data but the dataset turned out to be '{stats['format']}'. "
                           "Try a different goal or different data.")

    # ------------------------------------------------------------------ 3. plan
    job.step("plan", "running", "Picking the model and settings for your data…")
    plan = autopilot.train_plan(goal, effort, stats, gpu["vram_gb"], cfg.get("persona"), cfg.get("model_override"))
    if cfg.get("max_steps_cap"):  # used for quick smoke tests
        plan["max_steps"] = min(plan["max_steps"], cfg["max_steps_cap"])
        plan["sample_every"] = max(5, plan["max_steps"] // 2)
        plan["save_steps"] = max(10, plan["max_steps"])
    job.s["plan"] = plan
    if plan["task"] == "pretrain":
        detail = f"New {plan['size']} model · {plan['max_steps']:,} steps"
    else:
        import device_utils
        method = "QLoRA" if device_utils.supports_4bit() else "LoRA"
        detail = f"{plan['model'].split('/')[-1]} + {method} · {plan['max_steps']:,} steps · ~{autopilot._fmt_min(plan['estimate']['train_sec'])}"
    job.step("plan", "done", detail)

    # ------------------------------------------------------------------ 4. train
    out_dir = os.path.join("runs", name)
    train_path = stats["train_path"]
    job.step("train", "running", "Training… watch the loss go down below.")
    if plan["task"] == "pretrain":
        cmd = [sys.executable, "-u", "pretrain_llm.py", "--dataset", train_path, "--size", plan["size"],
               "--max_steps", str(plan["max_steps"]), "--batch_size", str(plan["batch_size"]),
               "--grad_accum", str(plan["grad_accum"]), "--seq_len", str(plan["seq_len"]), "--lr", str(plan["lr"]),
               "--max_samples", "0", "--save_steps", str(max(200, plan["max_steps"] // 5)),
               "--sample_every", str(plan["sample_every"]), "--sample_prompt", plan["sample_prompt"], "--output_dir", out_dir]
    elif plan["task"] == "dpo":
        cmd = [sys.executable, "-u", "train_llm_dpo.py", "--model", plan["model"], "--dataset", train_path,
               "--max_steps", str(plan["max_steps"]), "--batch_size", str(plan["batch_size"]),
               "--grad_accum", str(plan["grad_accum"]), "--lr", str(plan["lr"]), "--max_len", str(plan["max_seq_len"]),
               "--lora_r", str(plan["lora_r"]), "--save_steps", str(plan["save_steps"]), "--load_4bit",
               "--sample_every", str(plan["sample_every"]), "--sample_prompt", plan["sample_prompt"], "--output_dir", out_dir]
    else:
        cmd = [sys.executable, "-u", "train_llm_sft.py", "--model", plan["model"], "--dataset", train_path,
               "--mode", "qlora", "--max_steps", str(plan["max_steps"]), "--batch_size", str(plan["batch_size"]),
               "--grad_accum", str(plan["grad_accum"]), "--lr", str(plan["lr"]), "--max_seq_len", str(plan["max_seq_len"]),
               "--lora_r", str(plan["lora_r"]), "--lora_alpha", str(plan["lora_r"] * 2), "--save_steps", str(plan["save_steps"]),
               "--sample_every", str(plan["sample_every"]), "--sample_prompt", plan["sample_prompt"],
               "--output_dir", out_dir, "--eval_ratio", "0.02"]
        if plan.get("packing"):
            cmd.append("--packing")
    # Checkpoint often enough that a crash costs at most ~1/8 of the run, then retry automatically.
    save_every = str(max(25, plan["max_steps"] // 8))
    cmd[cmd.index("--save_steps") + 1] = save_every
    import gpu_guard
    import friendly_errors
    resume = False
    for attempt in range(1, 5):
        try:
            job.run(cmd + (["--resume"] if resume and "--resume" not in cmd else []), "train")
            break
        except RuntimeError as e:
            kind = (friendly_errors.explain(str(e)) or {}).get("key")
            if attempt == 4 or kind not in ("gpu_hw", "oom"):
                raise
            job.s.setdefault("recoveries", []).append({"attempt": attempt, "error": str(e).splitlines()[-1][:200]})
            if kind == "oom":
                # Out of memory: keep the same effective batch but use less memory per step, and start over (shapes changed).
                cmd = shrink_for_memory(cmd)
                resume = False
                job.step("train", "running", f"⚠️ The GPU ran out of memory. Retrying with smaller steps "
                                             f"(batch {_arg(cmd, '--batch_size')}) — attempt {attempt + 1}…")
                import trainer_utils
                trainer_utils.clear_gpu_memory()
            else:
                resume = True
                job.step("train", "running", f"⚠️ The GPU hit an error (usually from overheating). Cooling down, then "
                                             f"continuing from the last checkpoint — retry {attempt} of 3…")
                gpu_guard.wait_until_cool(68)
    live = json.load(open(os.path.join(ROOT, out_dir, "live", "status.json"), encoding="utf-8"))
    latest = live.get("latest", {})
    job.s["train_summary"] = {"final_dir": live.get("final_dir"), "best_eval_loss": live.get("best_eval_loss"),
                              "train_loss": live.get("train_loss"), "elapsed_s": live.get("elapsed_s"),
                              "eval_loss": latest.get("eval_loss")}
    job.step("train", "done", f"Finished in {autopilot._fmt_min(live.get('elapsed_s') or 0)}"
             + (f" · test loss {live['best_eval_loss']:.3f}" if live.get("best_eval_loss") else ""))

    # ------------------------------------------------------------------ 5. test
    # Runs in a fresh process: right after a long training the GPU is hot, and a hot GPU can poison the CUDA context of the process
    # that did the training. A clean process after a cool-down (and up to 3 tries) makes this step reliable.
    job.step("test", "running", "Asking the model questions before and after training…")
    import gpu_guard
    cfg_path = os.path.join(ROOT, "runs", name, "autopilot_config.json")
    last_err = ""
    for attempt in range(1, 4):
        gpu_guard.wait_until_cool(66)
        p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--config", cfg_path, "--test-only"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore"),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        job.say((p.stdout or "")[-1500:])
        if p.returncode == 0:
            break
        last_err = (p.stderr or p.stdout or "")[-1500:]
        job.say(f"(test attempt {attempt} failed)\n{last_err}")
        job.step("test", "running", f"⚠️ The GPU hit an error while testing - cooling down and retrying ({attempt}/3)…")
    else:
        raise RuntimeError("The model trained fine, but the before/after test could not run:\n" + last_err)
    with open(job.path, encoding="utf-8") as f:   # the test process wrote the results
        job.s = json.load(f)
    job.s["pid"] = os.getpid()
    job.save()


def test_only(cfg):
    """Entry point for the test step (also used to re-run just the test of a finished training)."""
    run_dir = os.path.join(ROOT, "runs", cfg["name"])
    path = os.path.join(run_dir, "autopilot.json")
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    job = Job.__new__(Job)
    job.cfg, job.run_dir, job.path = cfg, run_dir, path
    job.log = open(os.path.join(run_dir, "autopilot.log"), "a", encoding="utf-8")
    job.s = state
    job.s.update(state="running", error=None, pid=os.getpid())
    job.step("test", "running", "Letting the GPU cool down, then asking the model questions before and after training…")
    import gpu_guard
    gpu_guard.wait_until_cool(66)
    job.step("test", "running", "Asking the model questions before and after training…")
    g = autopilot.GOALS[cfg["goal"]]
    final_dir = (state.get("train_summary") or {}).get("final_dir") or os.path.join("runs", cfg["name"], "final")
    job.s["results"] = compare(job, state["plan"], final_dir, state["dataset"], g)
    job.step("test", "done", f"{len(job.s['results']['pairs'])} questions answered by both versions — see below.")
    job.s["state"] = "done"
    job.s["current"] = None
    job.save()


# ----------------------------------------------------------------------------- before / after
def _test_prompts(ds_name, g, plan, k=4):
    """Held-out questions from the dataset's eval split (with reference answers), else the goal's defaults."""
    prompts = []
    if plan["task"] == "sft":
        import dataset_store
        r = dataset_store.SplitReader(ds_name, "eval")
        for _, ex in r.rows(0, 40):
            msgs = ex.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            ref = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
            if user and len(user) < 600:
                prompts.append((user, ref))
            if len(prompts) >= k:
                break
        r.close()
    for p in g["test_prompts"]:
        if len(prompts) >= k + 1:
            break
        prompts.append((p, None))
    return prompts


def compare(job, plan, final_dir, ds_name, g):
    import torch
    import device_utils
    from transformers import AutoTokenizer, AutoModelForCausalLM
    prompts = _test_prompts(ds_name, g, plan)
    pairs = [{"prompt": p, "reference": ref} for p, ref in prompts]

    def gen(model, tok, chat=True):
        outs = []
        for p in pairs:
            if chat:
                msgs = ([{"role": "system", "content": plan["system_prompt"]}] if plan.get("system_prompt") else []) + \
                       [{"role": "user", "content": p["prompt"]}]
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            else:
                enc = tok(p["prompt"], return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=220, do_sample=False, repetition_penalty=1.1,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs.append(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        return outs

    if plan["task"] == "pretrain":
        tok = AutoTokenizer.from_pretrained(final_dir)
        model = device_utils.to_device(AutoModelForCausalLM.from_pretrained(
            final_dir, **device_utils.model_load_kwargs()).eval())
        for p, o in zip(pairs, gen(model, tok, chat=False)):
            p["before"] = "(a brand-new model knows nothing before training — it would output random tokens)"
            p["after"] = o
        return {"pairs": pairs, "model_dir": final_dir}

    quant, note = device_utils.quant_config_4bit("7B" in plan["model"], "the before/after test")
    if note:
        job.say(note)
    tok = AutoTokenizer.from_pretrained(final_dir)
    base = device_utils.to_device(AutoModelForCausalLM.from_pretrained(
        plan["model"], quantization_config=quant, **device_utils.model_load_kwargs()).eval())
    job.say("generating BEFORE answers")
    before = gen(base, tok)
    from peft import PeftModel
    tuned = PeftModel.from_pretrained(base, final_dir).eval()
    job.say("generating AFTER answers")
    after = gen(tuned, tok)
    for p, b, a in zip(pairs, before, after):
        p["before"], p["after"] = b, a
    return {"pairs": pairs, "model_dir": final_dir}


if __name__ == "__main__":
    # Never crash on printing non-ASCII (°, emoji) when stdout is a file with a legacy Windows code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-only", action="store_true", help="only (re)run the before/after test of a finished training")
    a = ap.parse_args()
    with open(a.config, encoding="utf-8") as f:
        cfg_ = json.load(f)
    if a.test_only:
        test_only(cfg_)
    else:
        main(cfg_)

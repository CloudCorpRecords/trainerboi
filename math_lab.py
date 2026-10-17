"""
math_lab.py - Verified self-improvement for maths ("Math Lab").

Each round:  1) test the model on a fixed exam (per-skill accuracy)  2) generate NEW practice problems aimed at its weakest skills
             3) let it attempt each problem several times  4) keep ONLY the attempts a checker proves correct (shortest first)
             5) fine-tune on those verified solutions, merge, repeat.

No dataset, no human labels, no teacher model: the answers are known because the problems are generated (math_gym.py), so the model
improves from its own verified successes and its practice is steered at what it is bad at.  Everything runs on one consumer GPU.

  python math_lab.py --model Qwen/Qwen2.5-1.5B-Instruct --rounds 3 --problems 600 --name math1
"""

import os
import re
import sys
import json
import time
import random
import argparse
import subprocess
import traceback
from collections import defaultdict

import math_gym as G

ROOT = os.path.dirname(os.path.abspath(__file__))


class Lab:
    def __init__(self, args):
        self.a = args
        self.dir = os.path.join(ROOT, "runs", args.name)
        self.mdir = os.path.join(self.dir, "math")
        os.makedirs(self.mdir, exist_ok=True)
        self.s = {"pid": os.getpid(), "state": "running", "phase": "Starting", "started_at": time.time(), "model": args.model,
                  "rounds_total": args.rounds, "rounds": [], "exam": [], "error": None, "config": vars(args), "kept": 0}
        prev = os.path.join(self.mdir, "status.json")
        if args.resume and os.path.exists(prev):
            try:
                with open(prev, encoding="utf-8") as f:
                    old = json.load(f)
                for k in ("rounds", "kept", "baseline", "current_model", "latest", "recoveries"):
                    if k in old:
                        self.s[k] = old[k]
                self.s["started_at"] = old.get("started_at", self.s["started_at"])
            except Exception:
                pass
        self.save()

    def save(self):
        self.s["updated_at"] = time.time()
        tmp = os.path.join(self.mdir, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.s, f)
        for _ in range(5):
            try:
                os.replace(tmp, os.path.join(self.mdir, "status.json"))
                break
            except PermissionError:
                time.sleep(0.05)

    def phase(self, text, **extra):
        print(f"--> {text}", flush=True)
        self.s["phase"] = text
        self.s.update(extra)
        self.save()


# ------------------------------------------------------------------------------------------------ model helpers
def load(model_path):
    import device_utils
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa",
                                                 **device_utils.model_load_kwargs()).eval()
    return tok, device_utils.to_device(model)


def free(model):
    import device_utils
    del model
    device_utils.empty_cache()


def generate(tok, model, prompts, k=1, temperature=0.0, max_new=320, bs=24, guard=None, progress=None):
    """-> list (per prompt) of k completion strings."""
    import torch
    out = []
    for i in range(0, len(prompts), bs):
        if guard:
            guard.maybe_cool()
        chunk = prompts[i:i + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k, remove_invalid_values=True)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            seq = model.generate(**enc, **kw)
        dec = tok.batch_decode(seq[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        per = k if temperature > 0 else 1
        out += [dec[j * per:(j + 1) * per] for j in range(len(chunk))]
        if progress:
            progress(min(i + bs, len(prompts)), len(prompts))
    return out


def run_exam(lab, tok, model, exam, label, guard):
    """Greedy accuracy per skill on the fixed exam."""
    def prog(d, t):
        lab.phase(f"{label}: exam {d}/{t}")
    outs = generate(tok, model, [G.prompt_text(p) for p in exam], k=1, temperature=0.0, max_new=lab.a.max_new, guard=guard, progress=prog)
    by = defaultdict(lambda: [0, 0])
    boxed = 0
    for p, o in zip(exam, outs):
        ok = G.is_correct(o[0], p["answer"])
        by[p["skill"]][0] += ok
        by[p["skill"]][1] += 1
        boxed += G.has_boxed(o[0])
    skills = {s: round(c / max(n, 1), 3) for s, (c, n) in by.items()}
    overall = round(sum(c for c, _ in by.values()) / max(len(exam), 1), 3)
    return {"label": label, "overall": overall, "skills": skills, "boxed_pct": round(boxed / max(len(exam), 1), 3),
            "samples": [{"q": p["question"], "gold": p["answer"], "out": o[0][-400:], "ok": G.is_correct(o[0], p["answer"])}
                        for p, o in list(zip(exam, outs))[:6]]}


# ------------------------------------------------------------------------------------------------ main loop
def main(a):
    lab = Lab(a)
    try:
        _main(lab, a)
        lab.s.update(state="done", phase="Finished")
    except KeyboardInterrupt:
        lab.s.update(state="stopped", phase="Stopped")
        raise
    except Exception as e:
        lab.s.update(state="error", phase="Crashed", error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        lab.save()
        raise
    lab.save()


def _main(lab, a):
    from gpu_guard import ThermalGuard
    guard = ThermalGuard(max_temp=min(76.0, float(os.environ.get("TB_MAX_GPU_TEMP", 76)) or 76.0), hysteresis=8.0, check_every_s=1.0)
    levels = tuple(range(a.min_level, a.max_level + 1))
    # exam: balanced across skills, fixed seed -> every round is graded on the same questions
    per_skill = max(2, a.exam // len(G.SKILLS))
    exam = []
    for i, s in enumerate(G.SKILLS):
        r = random.Random(1000 + i)
        seen = set()
        while len([p for p in exam if p["skill"] == s]) < per_skill:
            p = G.problem(s, r.choice(levels), r)
            if p["question"] not in seen:
                seen.add(p["question"])
                exam.append(p)
    random.Random(7).shuffle(exam)
    lab.s["exam_size"] = len(exam)

    resumed = bool(lab.s["rounds"])
    base = lab.s.get("current_model") or a.model if resumed else a.model
    lab.phase("Loading model" + (" (continuing after a GPU hiccup)" if resumed else ""))
    tok, model = load(base)
    if resumed:
        res = lab.s["rounds"][-1]
    else:
        res = run_exam(lab, tok, model, exam, "Before training", guard)
        lab.s["baseline"] = res
        lab.s["rounds"].append({"round": 0, **{k: res[k] for k in ("label", "overall", "skills", "boxed_pct")}, "kept": 0})
        lab.save()
    weights = {s: (1 - acc) + 0.15 for s, acc in res["skills"].items()}

    for rnd in range(len(lab.s["rounds"]), a.rounds + 1):
        lab.phase(f"Round {rnd}/{a.rounds}: writing practice problems")
        # steer practice at weak skills; raise difficulty when the model is doing well overall
        lv = tuple(range(a.min_level, min(a.max_level, a.min_level + 1 + (1 if res["overall"] > 0.6 else 0) + rnd // 2) + 1))
        probs = G.make_set(a.problems, levels=lv, weights=weights, seed=rnd * 101 + 5)
        exam_q = {p["question"] for p in exam}
        probs = [p for p in probs if p["question"] not in exam_q]  # never practise on the exam

        lab.phase(f"Round {rnd}/{a.rounds}: solving {len(probs)} problems × {a.samples} attempts")
        outs = generate(tok, model, [G.prompt_text(p) for p in probs], k=a.samples, temperature=a.temperature, max_new=a.max_new,
                        bs=max(4, 96 // a.samples), guard=guard,
                        progress=lambda d, t: lab.phase(f"Round {rnd}/{a.rounds}: practising {d}/{t} problems"))
        rows, solved, hard = [], 0, 0
        for p, cands in zip(probs, outs):
            good = [c for c in cands if G.is_correct(c, p["answer"]) and G.has_boxed(c) and len(c) > 20]
            if not good:
                hard += 1
                continue
            solved += 1
            good = sorted(set(good), key=len)[:a.keep_per_problem]
            for c in good:
                rows.append({"messages": [{"role": "user", "content": G.prompt_text(p)}, {"role": "assistant", "content": c.strip()}]})
        random.Random(rnd).shuffle(rows)
        lab.s["kept"] += len(rows)
        lab.phase(f"Round {rnd}/{a.rounds}: {len(rows)} verified solutions from {solved}/{len(probs)} problems ({hard} too hard)",
                  last_round={"round": rnd, "problems": len(probs), "solved": solved, "kept": len(rows), "hard": hard,
                              "levels": list(lv)})
        if len(rows) < 20:
            lab.phase(f"Round {rnd}: too few correct solutions to learn from - stopping early")
            break
        data_path = os.path.join(lab.mdir, f"round{rnd}.jsonl")
        with open(data_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        free(model)
        del tok
        out_dir = os.path.join("runs", a.name, f"round{rnd}")
        steps = max(30, min(400, int(len(rows) * a.epochs / 16)))
        lab.phase(f"Round {rnd}/{a.rounds}: learning from its own correct solutions ({steps} steps)")
        cmd = [sys.executable, "-u", "train_llm_sft.py", "--model", base, "--dataset", data_path, "--backend", "hf", "--mode", "lora",
               "--max_steps", str(steps), "--batch_size", "4", "--grad_accum", "4", "--lr", str(a.lr), "--max_seq_len", "768",
               "--lora_r", "32", "--lora_alpha", "64", "--save_steps", "100000", "--sample_every", "0", "--eval_ratio", "0.02",
               "--output_dir", out_dir]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
        import friendly_errors
        import gpu_guard
        gpu_guard.wait_until_cool(66)  # generating hundreds of answers heats the GPU; start training cool
        for attempt in range(1, 5):
            log = open(os.path.join(lab.mdir, f"round{rnd}.log"), "w", encoding="utf-8")
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
            log.close()
            if rc == 0:
                break
            tail = open(os.path.join(lab.mdir, f"round{rnd}.log"), encoding="utf-8", errors="replace").read()[-2500:]
            kind = (friendly_errors.explain(tail) or {}).get("key")
            if attempt == 4 or kind not in ("gpu_hw", "oom"):
                raise RuntimeError(f"Training in round {rnd} failed:\n{tail}")
            lab.phase(f"Round {rnd}/{a.rounds}: the GPU hit an error ({kind}); cooling down and retrying ({attempt}/3)")
            if kind == "oom":
                cmd[cmd.index("--batch_size") + 1] = "2"
                cmd[cmd.index("--grad_accum") + 1] = "8"
            gpu_guard.wait_until_cool(62)
        lab.phase(f"Round {rnd}/{a.rounds}: merging the new skills into the model")
        merged = os.path.join(ROOT, "runs", a.name, f"round{rnd}", "merged")
        rc = subprocess.run([sys.executable, "merge_lora.py", "--adapter", os.path.join(out_dir, "final"), "--out", merged], cwd=ROOT,
                            capture_output=True, text=True, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if rc.returncode != 0:
            raise RuntimeError("Merge failed:\n" + rc.stderr[-1200:])
        base = merged
        lab.s["current_model"] = merged
        tok, model = load(base)
        res = run_exam(lab, tok, model, exam, f"After round {rnd}", guard)
        lab.s["rounds"].append({"round": rnd, **{k: res[k] for k in ("label", "overall", "skills", "boxed_pct")}, "kept": len(rows),
                                "solved_pct": round(solved / max(len(probs), 1), 3)})
        lab.s["latest"] = res
        lab.save()
        weights = {s: (1 - acc) + 0.15 for s, acc in res["skills"].items()}
    lab.s["final_model"] = base
    lab.phase("Finished", final_model=base)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Math Lab: verified self-improvement")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--name", default=f"math-{time.strftime('%m%d-%H%M')}")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--problems", type=int, default=400, help="new practice problems per round")
    p.add_argument("--samples", type=int, default=4, help="attempts per problem")
    p.add_argument("--keep_per_problem", type=int, default=2)
    p.add_argument("--exam", type=int, default=170, help="exam questions (spread over all skills)")
    p.add_argument("--min_level", type=int, default=1)
    p.add_argument("--max_level", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new", type=int, default=320)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        main(args)
    else:
        # Supervisor: a hot laptop GPU can corrupt the CUDA context mid-run. Restart the worker after a cool-down; it resumes at the
        # last finished round (the merged model is saved each round).
        import gpu_guard
        from friendly_errors import explain
        argv = [x for x in sys.argv[1:] if x != "--resume"]
        for attempt in range(1, 5):
            rc = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), *argv, "--child"] + (["--resume"] if attempt > 1 else []))
            if rc == 0:
                break
            try:
                with open(os.path.join(ROOT, "runs", args.name, "math", "status.json"), encoding="utf-8") as f:
                    err = json.load(f).get("error") or ""
            except Exception:
                err = ""
            kind = (explain(err) or {}).get("key")
            if kind not in ("gpu_hw", "oom") or attempt == 4:
                sys.exit(rc)
            print(f"--> GPU error ({kind}); cooling down, then resuming (retry {attempt}/3)", flush=True)
            gpu_guard.wait_until_cool(62)
        sys.exit(0 if rc == 0 else rc)

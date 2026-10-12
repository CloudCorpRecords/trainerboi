"""
data_analyzer.py - the Data Doctor.

analyze(name)  -> health report (grade, per-problem counts with examples, duplicates, conflicts, train/eval leakage, diversity,
                  language mix, length stats, plain-English recommendations). Saved to data/<name>/analysis.json.
clean(...)     -> writes a NEW cleaned dataset (original untouched). CLI:
                  python data_analyzer.py clean --name src --out src-clean --drop ai_boilerplate,truncated --min_score 60 --dedup_prompts
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import argparse
import traceback
from collections import Counter

import data_quality as dq
import dataset_store as store

SAMPLE_BLOCK = 250


def _h(s):
    norm = re.sub(r"\W+", " ", (s or "").lower()).strip()
    return hashlib.blake2b(norm.encode(), digest_size=8).hexdigest()


def _read_sample(reader, sample):
    """Rows for analysis: everything if small, else random contiguous blocks (cheap on parquet/jsonl, and datasets are shuffled)."""
    n = len(reader)
    if n <= sample:
        return reader.rows(0, n)
    rng = random.Random(1)
    blocks = max(1, sample // SAMPLE_BLOCK)
    starts = sorted(rng.sample(range(0, max(1, n - SAMPLE_BLOCK)), min(blocks, max(1, n - SAMPLE_BLOCK))))
    out, seen = [], set()
    for s in starts:
        for i, ex in reader.rows(s, SAMPLE_BLOCK):
            if i not in seen:
                seen.add(i)
                out.append((i, ex))
    return out


def _snip(p, n=220):
    t = (p.get("response") or p.get("text") or p.get("prompt") or "")
    q = p.get("prompt") if p.get("kind") != "text" else ""
    return {"prompt": (q or "")[:n], "response": t[:n]}


def _grade(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"


def analyze(name, sample=4000, progress=None):
    stats = store.read_json(os.path.join(store.ds_dir(name), "stats.json"), {}) or {}
    r = store.SplitReader(name, "train")
    n_train = len(r)
    if not n_train:
        raise ValueError(f"'{name}' has no training examples.")
    rows = _read_sample(r, sample)
    r.close()

    flag_count, flag_ex, scores = Counter(), {}, []
    prompts, resp_by_prompt, resp_count, first_words = Counter(), {}, Counter(), Counter()
    scripts, resp_len, prompt_len, bigrams, total_bi = Counter(), [], [], set(), 0
    worst = []
    for k, (idx, ex) in enumerate(rows):
        score, flags = dq.check(ex)
        scores.append(score)
        p = dq.parts(ex)
        for f in flags:
            flag_count[f] += 1
            flag_ex.setdefault(f, [])
            if len(flag_ex[f]) < 3:
                flag_ex[f].append({"row": idx, **_snip(p)})
        if score < 70 and len(worst) < 12:
            worst.append({"row": idx, "score": score, "flags": flags, **_snip(p)})
        body = p.get("response") if p["kind"] != "text" else p.get("text", "")
        (resp_len if p["kind"] != "text" else resp_len).append(len(body or ""))
        if p["kind"] != "text":
            hp = _h(p.get("prompt", ""))
            prompts[hp] += 1
            resp_by_prompt.setdefault(hp, set()).add(_h(body))
            resp_count[_h(body)] += 1
            prompt_len.append(len(p.get("prompt", "")))
            fw = (p.get("prompt", "").split() or [""])[0].lower()
            first_words[fw] += 1
        else:
            prompts[_h(body[:400])] += 1
        sc = dq.script_of(body or "")
        if sc:
            scripts[sc] += 1
        w = (body or "").lower().split()
        if len(w) >= 2 and len(bigrams) < 400_000:
            bg = list(zip(w, w[1:]))
            bigrams.update(bg)
            total_bi += len(bg)
        if progress and k % 500 == 0:
            progress(k, len(rows))

    m = max(len(rows), 1)
    dup_prompts = sum(c - 1 for c in prompts.values() if c > 1)
    conflicts = sum(1 for hp, rs in resp_by_prompt.items() if len(rs) > 1 and prompts[hp] > 1)
    canned = sum(c for c in resp_count.values() if c >= 3)
    distinct2 = len(bigrams) / total_bi if total_bi else None
    top_fw = first_words.most_common(1)[0] if first_words else ("", 0)

    # train/eval leakage
    leak, leak_note = 0, ""
    ev = store.SplitReader(name, "eval")
    n_eval = len(ev)
    if n_eval:
        eval_hashes = {_h(dq.parts(ex).get("prompt") or dq.parts(ex).get("text", "")[:400]) for _, ex in ev.rows(0, min(n_eval, 5000))}
        train_hashes = set(prompts)
        if n_train <= 150_000 and len(rows) < n_train:
            tr = store.SplitReader(name, "train")
            train_hashes = set()
            for _, raw in tr.iter_raw():
                try:
                    pp = dq.parts(json.loads(raw))
                except Exception:
                    continue
                train_hashes.add(_h(pp.get("prompt") or pp.get("text", "")[:400]))
            tr.close()
        else:
            leak_note = " (checked on a sample)" if len(rows) < n_train else ""
        leak = len(eval_hashes & train_hashes)
        leak_pct = 100 * leak / max(len(eval_hashes), 1)
    else:
        leak_pct = 0
    ev.close()

    mean_score = sum(scores) / m
    penalty = 0.0
    penalty += min(25, 100 * dup_prompts / m * 0.6)
    penalty += min(10, 100 * conflicts / m * 2)
    penalty += min(10, leak_pct * 0.5)
    penalty += min(10, 100 * canned / m * 0.3)
    if distinct2 is not None and distinct2 < 0.25 and total_bi > 5000:
        penalty += 8
    health = max(0.0, mean_score - penalty)

    recs = []
    for f, c in flag_count.most_common():
        pct = 100 * c / m
        if pct < 0.5:
            continue
        sev = "high" if dq.FLAGS[f][2] and pct >= 3 else "medium" if pct >= 3 else "low"
        recs.append({"severity": sev, "text": f"{pct:.1f}% of examples: {dq.FLAGS[f][1].lower()}.", "action": {"drop": [f]}})
    if dup_prompts / m > 0.03:
        recs.append({"severity": "high" if dup_prompts / m > 0.1 else "medium",
                     "text": f"{100 * dup_prompts / m:.1f}% of examples repeat an earlier question. Duplicates make the model memorise instead of learn.",
                     "action": {"dedup_prompts": True}})
    if conflicts:
        recs.append({"severity": "medium", "text": f"{conflicts} questions appear with different answers; the model gets mixed signals.",
                     "action": {"dedup_prompts": True}})
    if leak:
        recs.append({"severity": "high", "text": f"{leak} test questions also appear in the training data{leak_note}. The test score would look better than reality.",
                     "action": {"dedup_prompts": True}})
    if canned / m > 0.05:
        recs.append({"severity": "medium", "text": f"{100 * canned / m:.0f}% of answers are the same canned reply used 3+ times.", "action": None})
    if distinct2 is not None and distinct2 < 0.25 and total_bi > 5000:
        recs.append({"severity": "medium", "text": "The answers use very similar wording throughout (low diversity). Add data from more varied sources.", "action": None})
    if top_fw[1] / m > 0.35 and top_fw[0]:
        recs.append({"severity": "low", "text": f"{100 * top_fw[1] / m:.0f}% of questions start with '{top_fw[0]}'. Variety in questions helps the model generalise.", "action": None})
    if len(scripts) > 1 and scripts.most_common(2)[1][1] / m > 0.1:
        mix = ", ".join(f"{k} {100 * v / m:.0f}%" for k, v in scripts.most_common(3))
        recs.append({"severity": "low", "text": f"Multiple writing systems found ({mix}). Fine if intended; otherwise filter by language.", "action": None})
    if n_train < 300:
        recs.append({"severity": "medium", "text": f"Only {n_train} training examples. Small data works for style but not knowledge; consider adding more.", "action": None})
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda x: order[x["severity"]])
    if not recs:
        recs.append({"severity": "low", "text": "No problems found. This dataset looks ready to train on. ✅", "action": None})

    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    report = {
        "name": name, "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sampled": len(rows), "train_examples": n_train,
        "eval_examples": n_eval, "format": stats.get("format"), "health": round(health, 1), "grade": _grade(health),
        "mean_example_score": round(mean_score, 1), "flags": {f: {"count": c, "pct": round(100 * c / m, 2), "label": dq.FLAGS[f][1],
                                                                  "default_drop": dq.FLAGS[f][2], "examples": flag_ex.get(f, [])}
                                                              for f, c in flag_count.most_common()},
        "score_hist": [{"score": f"{b}-{b + 19 if b < 80 else 100}", "examples": sum(1 for s in scores if b <= s < b + 20 or (b == 80 and s == 100))}
                       for b in range(0, 100, 20)],
        "duplicate_prompt_pct": round(100 * dup_prompts / m, 2), "conflicting_answers": conflicts,
        "leaked_eval_prompts": leak, "leak_pct": round(leak_pct, 1), "canned_answer_pct": round(100 * canned / m, 2),
        "distinct_2gram": round(distinct2, 3) if distinct2 is not None else None,
        "scripts": {k: round(100 * v / m, 1) for k, v in scripts.most_common(5)},
        "response_chars": {"p5": pct(resp_len, 0.05), "p50": pct(resp_len, 0.5), "p95": pct(resp_len, 0.95)},
        "prompt_chars": {"p50": pct(prompt_len, 0.5)} if prompt_len else {},
        "worst": worst, "recommendations": recs,
    }
    with open(os.path.join(store.ds_dir(name), "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return report


def load_report(name):
    return store.read_json(os.path.join(store.ds_dir(name), "analysis.json"), None)


# ----------------------------------------------------------------------------------------- clean (CLI job)
def clean(args):
    import dataset_builder as db
    src = store.read_json(os.path.join(store.ds_dir(args.name), "stats.json"))
    out_dir = store.ds_dir(args.out)
    build_dir = os.path.join(out_dir, "build")
    if os.path.exists(os.path.join(out_dir, "stats.json")):
        raise SystemExit(f"Dataset '{args.out}' already exists - pick another name.")
    os.makedirs(build_dir, exist_ok=True)
    st = db.Status(build_dir, args)
    try:
        drop = {x for x in (args.drop or "").split(",") if x}
        seen = set()
        kept_path = os.path.join(build_dir, "kept.jsonl")
        total = src["train_examples"] + src["eval_examples"]
        st.phase("Checking every example", total=total, unit="examples")
        kept = 0
        eval_first = []  # eval prompts go first so train copies of them are the ones removed (no leakage)
        with open(kept_path, "w", encoding="utf-8") as kf:
            for split in ("eval", "train"):
                r = store.SplitReader(args.name, split)
                for _, raw in r.iter_raw():
                    st.tick()
                    try:
                        ex = json.loads(raw)
                    except Exception:
                        st.drop("unreadable row")
                        continue
                    score, flags = dq.check(ex)
                    hit = [f for f in flags if f in drop]
                    if hit:
                        st.drop(dq.FLAGS[hit[0]][1])
                        continue
                    if args.min_score and score < args.min_score:
                        st.drop(f"quality score below {args.min_score}")
                        continue
                    if args.dedup_prompts:
                        p = dq.parts(ex)
                        key = _h(p.get("prompt") or p.get("text", "")[:400])
                        if key in seen:
                            st.drop("duplicate question")
                            continue
                        seen.add(key)
                    kf.write(raw.strip() + "\n")
                    kept += 1
                r.close()
        if kept < 10:
            raise ValueError(f"Only {kept} examples would remain - these filters are too strict.")
        ratio = src["eval_examples"] / max(total, 1)
        stats = db.finalize(kept_path, out_dir, build_dir, st, src["format"], name=args.out, tokenizer_id=src["tokenizer"],
                            eval_ratio=max(ratio, 0.01), seed=42, output_format=src.get("output_format", "jsonl"),
                            extra={"cleaned_from": args.name, "clean_rules": {"drop": sorted(drop), "min_score": args.min_score,
                                                                              "dedup_prompts": bool(args.dedup_prompts)}})
        os.remove(kept_path)
        st.s.update(state="done", phase="Finished", format=src["format"], stats=stats)
        st.write(True)
        print(f"==> Cleaned '{args.out}': kept {kept:,} of {total:,} examples", flush=True)
    except Exception as e:
        st.s.update(state="error", phase="Crashed", error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        st.write(True)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Data Doctor")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--name", required=True)
    a.add_argument("--sample", type=int, default=4000)
    c = sub.add_parser("clean")
    c.add_argument("--name", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--drop", default="")
    c.add_argument("--min_score", type=int, default=0)
    c.add_argument("--dedup_prompts", action="store_true")
    args = p.parse_args()
    if args.cmd == "analyze":
        rep = analyze(args.name, args.sample)
        print(json.dumps({k: rep[k] for k in ("grade", "health", "sampled")}, indent=1))
    else:
        clean(args)

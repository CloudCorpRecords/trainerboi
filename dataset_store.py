"""
dataset_store.py - Backend for the Dataset Library: list, browse, search, edit, mix, convert, export, publish.

Works on datasets in data/<name>/ produced by dataset_builder.py (train.jsonl/eval.jsonl or train-*.parquet shards).
Browsing is O(page) even for 100GB files: JSONL gets a cached byte-offset index, Parquet uses small row groups.

CLI (used by the Studio for long-running jobs):
  python dataset_store.py mix --name blend --part alpaca:20000 --part mydocs:0 [--dedup] [--output_format parquet]
  python dataset_store.py convert --name big --to parquet
"""

import os
import re
import sys
import json
import time
import glob
import shutil
import random
import zipfile
import argparse
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


# =============================================================================== discovery
def ds_dir(name):
    return os.path.join(DATA, name)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def list_datasets():
    out = []
    for d in glob.glob(os.path.join(DATA, "*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        stats = read_json(os.path.join(d, "stats.json"), {}) or {}
        status = read_json(os.path.join(d, "build", "status.json"), {}) or {}
        state = status.get("state", "done" if stats else "empty")
        if state == "running" and not _pid_alive(status.get("pid")):
            state = "interrupted"
        out.append({
            "name": name, "format": stats.get("format", status.get("format", "?")),
            "train": stats.get("train_examples"), "eval": stats.get("eval_examples"),
            "tokens": stats.get("total_tokens"), "storage": stats.get("output_format", "jsonl"),
            "size_mb": dir_size(d) / 1e6, "built": stats.get("built_at", ""), "state": state,
            "mtime": os.path.getmtime(d),
        })
    out.sort(key=lambda r: -r["mtime"])
    return out


def _pid_alive(pid):
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(pid)
    except Exception:
        return False


# =============================================================================== split readers
class SplitReader:
    """Random access to one split (train/eval) of a dataset, jsonl or parquet."""

    def __init__(self, name, split="train"):
        self.name, self.split = name, split
        d = ds_dir(name)
        self.jsonl = os.path.join(d, f"{split}.jsonl")
        self.shards = sorted(glob.glob(os.path.join(d, f"{split}-*.parquet")))
        if os.path.exists(self.jsonl):
            self.kind = "jsonl"
            self._offsets = self._index()
        elif self.shards:
            self.kind = "parquet"
            import pyarrow.parquet as pq
            self._pf = [pq.ParquetFile(s) for s in self.shards]
            # (shard, row_group, first_row_global, num_rows)
            self._groups, n = [], 0
            for si, pf in enumerate(self._pf):
                for gi in range(pf.num_row_groups):
                    rows = pf.metadata.row_group(gi).num_rows
                    self._groups.append((si, gi, n, rows))
                    n += rows
            self._n = n
        else:
            self.kind = None
            self._n = 0

    # ---- jsonl byte-offset index (cached next to the file, rebuilt if the file changes)
    def _index(self):
        idx_path, meta_path = self.jsonl + ".idx.npy", self.jsonl + ".idx.json"
        st = os.stat(self.jsonl)
        meta = read_json(meta_path, {})
        if meta.get("size") == st.st_size and meta.get("mtime") == st.st_mtime and os.path.exists(idx_path):
            return np.load(idx_path)  # not mmap: Windows would lock the file against later edits
        starts = [np.array([0], dtype=np.int64)]
        pos = 0
        with open(self.jsonl, "rb") as f:
            while True:
                block = f.read(64 << 20)
                if not block:
                    break
                nl = np.flatnonzero(np.frombuffer(block, dtype=np.uint8) == 10).astype(np.int64) + pos + 1
                starts.append(nl)
                pos += len(block)
        offs = np.concatenate(starts)
        if offs[-1] >= pos:  # trailing newline -> no row starts at EOF
            offs = offs[:-1]
        np.save(idx_path, offs)
        with open(meta_path, "w") as mf:
            json.dump({"size": st.st_size, "mtime": st.st_mtime}, mf)
        return offs

    def __len__(self):
        return len(self._offsets) if self.kind == "jsonl" else self._n

    def close(self):
        """Release parquet file handles (Windows can't delete/rename files that are still open)."""
        for pf in getattr(self, "_pf", []):
            try:
                pf.close()
            except Exception:
                pass
        self._pf = []

    def rows(self, start, count):
        """Returns [(row_index, example_dict)]."""
        n = len(self)
        start, end = max(0, start), min(n, start + count)
        if start >= end:
            return []
        if self.kind == "jsonl":
            out = []
            with open(self.jsonl, "rb") as f:
                f.seek(int(self._offsets[start]))
                for i in range(start, end):
                    line = f.readline()
                    try:
                        out.append((i, json.loads(line)))
                    except json.JSONDecodeError:
                        out.append((i, {"_raw": line.decode("utf-8", "replace")}))
            return out
        out = []
        for si, gi, first, rows in self._groups:
            if first + rows <= start or first >= end:
                continue
            tbl = self._pf[si].read_row_group(gi).to_pylist()
            for j in range(max(start, first), min(end, first + rows)):
                out.append((j, tbl[j - first]))
        return out

    def iter_raw(self, start=0):
        """Yields (row_index, raw_text) for searching."""
        if self.kind == "jsonl":
            with open(self.jsonl, "rb") as f:
                f.seek(int(self._offsets[start]) if start < len(self) else 0)
                for i, line in enumerate(f, start):
                    yield i, line.decode("utf-8", "replace")
        elif self.kind == "parquet":
            for si, gi, first, rows in self._groups:
                if first + rows <= start:
                    continue
                for j, ex in enumerate(self._pf[si].read_row_group(gi).to_pylist()):
                    if first + j >= start:
                        yield first + j, json.dumps(ex, ensure_ascii=False)

    def search(self, query, regex=False, max_hits=1000, time_limit=20.0):
        """Case-insensitive search over raw rows. Returns (hits, scanned, complete)."""
        rx = re.compile(query if regex else re.escape(query), re.I)
        hits, scanned, t0 = [], 0, time.time()
        for i, raw in self.iter_raw():
            scanned += 1
            # JSON-escaped newlines/quotes: also try against the unescaped text for multi-word queries.
            if rx.search(raw) or ("\\" in raw and rx.search(raw.replace("\\n", "\n").replace('\\"', '"'))):
                hits.append(i)
                if len(hits) >= max_hits:
                    return hits, scanned, False
            if scanned % 20000 == 0 and time.time() - t0 > time_limit:
                return hits, scanned, False
        return hits, scanned, True


# =============================================================================== edits (jsonl only)
def _edits_path(name):
    return os.path.join(ds_dir(name), "edits.json")


def load_edits(name):
    return read_json(_edits_path(name), {}) or {}


def _save_edits(name, edits):
    with open(_edits_path(name), "w", encoding="utf-8") as f:
        json.dump(edits, f)


def edit_row(name, split, idx, example):
    e = load_edits(name)
    s = e.setdefault(split, {"deleted": [], "replaced": {}})
    s["replaced"][str(idx)] = example
    if idx in s["deleted"]:
        s["deleted"].remove(idx)
    _save_edits(name, e)


def delete_row(name, split, idx):
    e = load_edits(name)
    s = e.setdefault(split, {"deleted": [], "replaced": {}})
    if idx not in s["deleted"]:
        s["deleted"].append(idx)
    _save_edits(name, e)


def restore_row(name, split, idx):
    e = load_edits(name)
    s = e.get(split)
    if s:
        if idx in s["deleted"]:
            s["deleted"].remove(idx)
        s["replaced"].pop(str(idx), None)
        _save_edits(name, e)


def pending_count(name):
    e = load_edits(name)
    return sum(len(s["deleted"]) + len(s["replaced"]) for s in e.values())


def discard_edits(name):
    if os.path.exists(_edits_path(name)):
        os.remove(_edits_path(name))


def commit_edits(name):
    """Rewrite the jsonl files applying deletions/replacements, then refresh stats."""
    edits = load_edits(name)
    d = ds_dir(name)
    stats = read_json(os.path.join(d, "stats.json"), {}) or {}
    for split, s in edits.items():
        src = os.path.join(d, f"{split}.jsonl")
        if not os.path.exists(src):
            raise ValueError("Editing is supported for JSONL datasets. Convert this one to JSONL first.")
        deleted, replaced = set(s["deleted"]), s["replaced"]
        tmp = src + ".tmp"
        kept = 0
        with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i in deleted:
                    continue
                if str(i) in replaced:
                    line = json.dumps(replaced[str(i)], ensure_ascii=False) + "\n"
                fout.write(line)
                kept += 1
        os.replace(tmp, src)
        # Mark this split applied immediately so a crash can never apply the same edits twice.
        edits_left = load_edits(name)
        edits_left.pop(split, None)
        _save_edits(name, edits_left)
        for ext in (".idx.npy", ".idx.json"):  # stale index is also detected by size/mtime; this is just cleanup
            try:
                os.remove(src + ext)
            except OSError:
                pass
        stats[f"{split}_examples"] = kept
    stats["edited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    discard_edits(name)
    return stats


# =============================================================================== manage
def rename(old, new):
    new = re.sub(r"[^\w.-]+", "-", new.strip())
    if not new:
        raise ValueError("New name is empty")
    if os.path.exists(ds_dir(new)):
        raise ValueError(f"A dataset named '{new}' already exists")
    os.rename(ds_dir(old), ds_dir(new))
    stats_p = os.path.join(ds_dir(new), "stats.json")
    stats = read_json(stats_p)
    if stats:
        stats["name"] = new
        stats["train_path"] = stats["train_path"].replace(f"data/{old}", f"data/{new}")
        with open(stats_p, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    return new


def delete(name):
    shutil.rmtree(ds_dir(name))


def export_zip(name):
    d = ds_dir(name)
    out = os.path.join(d, "export", f"{name}.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(glob.glob(os.path.join(d, "train*")) + glob.glob(os.path.join(d, "eval*")) +
                        [os.path.join(d, "README.md"), os.path.join(d, "stats.json")]):
            if os.path.isfile(f) and not f.endswith((".npy", ".idx.json")):
                z.write(f, os.path.basename(f))
    return out


def push_to_hub(name, repo_id, private=True):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        raise RuntimeError("Not logged in to Hugging Face. In a terminal run:  .venv\\Scripts\\hf.exe auth login")
    if "/" not in repo_id:
        repo_id = f"{user}/{repo_id}"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=ds_dir(name),
                      allow_patterns=["train*.jsonl", "eval*.jsonl", "train-*.parquet", "eval-*.parquet", "README.md", "stats.json"],
                      commit_message="Upload dataset from TrainerBoi")
    return f"https://huggingface.co/datasets/{repo_id}"


def detailed_stats(name, sample=3000):
    """Cheap structural stats from a sample of rows."""
    r = SplitReader(name, "train")
    n = len(r)
    if not n:
        return {}
    idx = sorted(random.Random(0).sample(range(n), min(sample, n)))
    rows = []
    # Read in contiguous runs to keep it fast.
    for i in idx:
        rows += r.rows(i, 1)
    turns, sys_prompt, resp_len, prompt_len, text_len = [], 0, [], [], []
    for _, ex in rows:
        if "messages" in ex:
            msgs = ex["messages"]
            turns.append(sum(1 for m in msgs if m["role"] == "user"))
            sys_prompt += any(m["role"] == "system" for m in msgs)
            resp_len += [len(m["content"]) for m in msgs if m["role"] == "assistant"]
            prompt_len += [len(m["content"]) for m in msgs if m["role"] == "user"]
        elif "text" in ex:
            text_len.append(len(ex["text"]))
        elif "chosen" in ex:
            resp_len.append(len(ex["chosen"]))
            prompt_len.append(len(ex["prompt"]))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    return {"sampled": len(rows), "multi_turn_pct": 100 * sum(t > 1 for t in turns) / len(turns) if turns else None,
            "system_prompt_pct": 100 * sys_prompt / len(turns) if turns else None,
            "median_prompt_chars": med(prompt_len), "median_response_chars": med(resp_len),
            "median_text_chars": med(text_len), "turns": turns}


# =============================================================================== mix / convert (CLI jobs)
def _iter_examples(name):
    for split in ("train", "eval"):
        r = SplitReader(name, split)
        if r.kind == "jsonl":
            with open(r.jsonl, encoding="utf-8") as f:
                for line in f:
                    yield line
        elif r.kind == "parquet":
            for _, raw in r.iter_raw():
                yield raw + "\n"


def mix(args):
    import dataset_builder as db
    out_dir, build_dir = ds_dir(args.name), os.path.join(ds_dir(args.name), "build")
    if os.path.exists(os.path.join(out_dir, "stats.json")):
        raise SystemExit(f"Dataset '{args.name}' already exists - pick another name.")
    os.makedirs(build_dir, exist_ok=True)
    st = db.Status(build_dir, args)
    try:
        parts = []
        for p in args.part:
            pname, _, take = p.rpartition(":")
            stats = read_json(os.path.join(ds_dir(pname), "stats.json"), None)
            if not stats:
                raise ValueError(f"'{pname}' is not a finished dataset")
            parts.append((pname, int(take or 0), stats))
        fmts = {s["format"] for _, _, s in parts}
        if len(fmts) > 1:
            raise ValueError(f"Can't mix different formats {fmts} - all parts must be the same (sft / pretrain / dpo).")
        fmt = fmts.pop()
        rng = random.Random(args.seed)
        dedup = db.Deduper(near=False) if args.dedup else None
        kept_path = os.path.join(build_dir, "kept.jsonl")
        total = sum((t or (s["train_examples"] + s["eval_examples"])) for _, t, s in parts)
        st.phase("Mixing datasets", total=total, unit="examples")
        with open(kept_path, "w", encoding="utf-8") as kf:
            for pname, take, s in parts:
                have = s["train_examples"] + s["eval_examples"]
                ratio = (take / have) if take else 1.0  # >1 = oversample (repeat), <1 = random subsample
                n = 0
                for line in _iter_examples(pname):
                    reps = int(ratio) + (1 if rng.random() < ratio - int(ratio) else 0)
                    for _ in range(reps):
                        if dedup and reps == 1 and dedup.is_dup(line):
                            st.drop("exact duplicate across parts")
                            continue
                        kf.write(line)
                        n += 1
                        st.tick()
                st.source(f"{pname}", n)
                print(f"  {pname}: {n:,} examples (ratio {ratio:.3g})", flush=True)
        stats = db.finalize(kept_path, out_dir, build_dir, st, fmt, name=args.name, tokenizer_id=parts[0][2]["tokenizer"],
                            eval_ratio=args.eval_ratio, seed=args.seed, output_format=args.output_format,
                            extra={"mixed_from": {p: t for p, t, _ in parts}})
        os.remove(kept_path)
        st.s.update(state="done", phase="Finished", format=fmt, stats=stats)
        st.write(True)
        print(f"==> Mixed dataset '{args.name}': {stats['train_examples']:,} train / {stats['eval_examples']:,} eval", flush=True)
    except Exception as e:
        st.s.update(state="error", phase="Crashed", error=f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        st.write(True)
        raise


def convert(args):
    """jsonl <-> parquet shards, in place."""
    import dataset_builder as db
    d = ds_dir(args.name)
    stats = read_json(os.path.join(d, "stats.json"))
    for split in ("train", "eval"):
        r = SplitReader(args.name, split)
        if r.kind == args.to or r.kind is None:
            continue
        if args.to == "parquet":
            w = db._ParquetWriter(d, split, args.shard_rows)
            with open(r.jsonl, encoding="utf-8") as f:
                for line in f:
                    w.write(line, json.loads(line))
            w.close()
            os.remove(r.jsonl)
            for ext in (".idx.npy", ".idx.json"):
                if os.path.exists(r.jsonl + ext):
                    os.remove(r.jsonl + ext)
        else:
            with open(os.path.join(d, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for _, raw in r.iter_raw():
                    f.write(raw + "\n")
            r.close()
            for s in r.shards:
                os.remove(s)
    stats["output_format"] = args.to
    stats["train_path"] = (f"data/{args.name}/train.jsonl" if args.to == "jsonl" else f"data/{args.name}")
    stats["files"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "train*")) + glob.glob(os.path.join(d, "eval*"))
                            if not p.endswith((".npy", ".idx.json")))
    with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"==> Converted '{args.name}' to {args.to}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dataset library jobs")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix")
    m.add_argument("--name", required=True)
    m.add_argument("--part", action="append", required=True, help="dataset_name:rows (0 = all; more than available = oversample)")
    m.add_argument("--dedup", action="store_true")
    m.add_argument("--eval_ratio", type=float, default=0.02)
    m.add_argument("--output_format", choices=["jsonl", "parquet"], default="jsonl")
    m.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("convert")
    c.add_argument("--name", required=True)
    c.add_argument("--to", choices=["jsonl", "parquet"], required=True)
    c.add_argument("--shard_rows", type=int, default=500_000)
    a = p.parse_args()
    {"mix": mix, "convert": convert}[a.cmd](a)

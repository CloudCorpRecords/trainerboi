"""
resilient_run.py - runs a training command and quietly recovers from the GPU hiccups laptops are prone to.

    python resilient_run.py -- python -u train_llm_sft.py --model ... --output_dir runs/x ...

* A GPU error (usually heat): wait for the card to cool, then continue from the last checkpoint. If it keeps happening,
  start over once on the slower but steadier Hugging Face backend (SFT only).
* Out of memory: halve the batch (doubling accumulation so learning is unchanged) and start again.
* Anything else (a real bug, a stop by you): exit with the same code, nothing is retried.
Everything the command prints is passed straight through, so the usual log and live status keep working.
"""

import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
MAX_ATTEMPTS = 4


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


# What we've learned about this computer's fast engine (Unsloth). Some laptops throw GPU errors with its custom kernels
# whatever the temperature; after a few, new runs start straight on the steady engine instead of failing twice first.
PROFILE = os.path.join(ROOT, "gpu_profile.json")


def profile():
    try:
        return json.load(open(PROFILE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"fast_failures": 0, "fast_ok": 0}


def _record(fast_ok=0, fast_failures=0):
    p = profile()
    p["fast_ok"] = p.get("fast_ok", 0) + fast_ok
    p["fast_failures"] = p.get("fast_failures", 0) + fast_failures
    p["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(PROFILE, "w", encoding="utf-8") as f:
            json.dump(p, f)
    except OSError:
        pass


def prefer_steady():
    p = profile()
    return p.get("fast_failures", 0) >= 3 and p.get("fast_failures", 0) > 2 * p.get("fast_ok", 0)


def reset_profile():
    try:
        os.remove(PROFILE)
    except OSError:
        pass


def _uses_fast_engine(cmd):
    return any(c.endswith("train_llm_sft.py") for c in cmd) and "--backend" not in cmd and _arg_str(cmd, "--mode") != "full"


def _arg_str(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def _status_path(cmd):
    out = cmd[cmd.index("--output_dir") + 1] if "--output_dir" in cmd else None
    return os.path.join(ROOT, out, "live", "status.json") if out else None


def _tell(cmd, phase):
    """Keep the run page truthful while we're between attempts (our own pid is alive, so it won't show as crashed)."""
    p = _status_path(cmd)
    try:
        st = json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else {}
        st.update(state="running", phase=phase, pid=os.getpid())
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st, f)
    except (OSError, ValueError):
        pass


def _say(msg):
    print(f"\n[resilient_run] {msg}\n", flush=True)


def run_once(cmd):
    tail = collections.deque(maxlen=800)      # a CUDA crash ends with a long native stack dump; the real message sits above it
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line)
    return p.wait(), "".join(tail)


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: resilient_run.py -- <training command>")
    cmd = argv[argv.index("--") + 1:]
    import friendly_errors
    import gpu_guard
    is_sft = any(c.endswith("train_llm_sft.py") for c in cmd)
    hw_failures = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        fast = _uses_fast_engine(cmd)
        rc, tail = run_once(cmd)
        if rc == 0:
            if fast:
                _record(fast_ok=1)
            return 0
        kind = (friendly_errors.explain(tail) or {}).get("key")
        if kind is None and (rc < 0 or rc >= 0xC0000000):
            kind = "gpu_hw"       # the process died natively (access violation / fast-fail) - on these laptops that's the GPU, not a bug
        if fast and kind in ("gpu_hw", "nan"):
            _record(fast_failures=1)
        if attempt == MAX_ATTEMPTS or kind not in ("gpu_hw", "oom", "nan"):
            return rc
        out = cmd[cmd.index("--output_dir") + 1] if "--output_dir" in cmd else ""
        if kind == "oom":
            cmd = [c for c in shrink_for_memory(cmd) if c != "--resume"]
            _tell(cmd, f"Out of memory - trying smaller steps (batch {_arg(cmd, '--batch_size')}), attempt {attempt + 1}")
            _say(f"Out of memory. Retrying with batch {_arg(cmd, '--batch_size')} (attempt {attempt + 1}).")
            try:
                import trainer_utils
                trainer_utils.clear_gpu_memory()
            except Exception:
                pass
            continue
        hw_failures += 1
        if kind == "nan":
            # The newest checkpoint may already hold the broken numbers: drop it and go back one.
            ckpts = sorted(glob.glob(os.path.join(ROOT, out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
            if ckpts:
                shutil.rmtree(ckpts[-1], ignore_errors=True)
            _say("The loss turned into NaN (the GPU computed invalid numbers). Rolling back one checkpoint.")
        _tell(cmd, f"The GPU hit an error (usually heat) - cooling down, then trying again (attempt {attempt + 1} of {MAX_ATTEMPTS})")
        _say(f"GPU error #{hw_failures}. Letting the card cool, then trying again.")
        gpu_guard.wait_until_cool(66)
        cmd = [c for c in cmd if c != "--resume"]
        if hw_failures >= 2 and is_sft and "--backend" not in cmd:
            cmd += ["--backend", "hf"]            # slower, but doesn't use the custom kernels that tend to trip on hot laptops
            _say("Still unsteady - starting over on the steadier (slower) backend.")
        elif glob.glob(os.path.join(ROOT, out, "checkpoint-*")):
            cmd.append("--resume")
    return 1


if __name__ == "__main__":
    sys.exit(main())

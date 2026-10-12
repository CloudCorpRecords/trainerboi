"""
gpu_guard.py - Keep training correct on GPUs that misbehave when hot (common on laptops).

Some GPUs return silently wrong results or crash ("illegal memory access", "CUBLAS_STATUS_INTERNAL_ERROR")
once they run very hot. Two tools:

  ThermalGuard   pauses training when the GPU reaches TB_MAX_GPU_TEMP (default 82 °C) and resumes once it has
                 cooled by 5 °C. Used automatically by live_monitor.py in every training script. Set
                 TB_MAX_GPU_TEMP=0 to disable.
  health_check   a stress test that repeats identical GPU work and verifies the results never change.
                 python gpu_guard.py --check 180          (writes gpu_health.json)
"""

import os
import sys
import json
import time
import argparse
import subprocess

HEALTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_health.json")


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def max_temp_setting():
    try:
        return float(os.environ.get("TB_MAX_GPU_TEMP", "82"))
    except ValueError:
        return 82.0


class ThermalGuard:
    def __init__(self, max_temp=None, hysteresis=5.0, check_every_s=3.0, on_pause=None, on_resume=None):
        self.max_temp = max_temp_setting() if max_temp is None else max_temp
        self.hyst = hysteresis
        self.check_every = check_every_s
        self.on_pause, self.on_resume = on_pause, on_resume
        self._last = 0.0
        self.paused_s = 0.0
        self.pauses = 0
        self.last_temp = None

    @property
    def enabled(self):
        return bool(self.max_temp)

    def maybe_cool(self):
        """Call often (e.g. every training step). Cheap: checks the temperature at most every few seconds."""
        if not self.enabled or time.time() - self._last < self.check_every:
            return
        self._last = time.time()
        t = gpu_temp()
        self.last_temp = t
        if t is None or t < self.max_temp:
            return
        self.pauses += 1
        start = time.time()
        if self.on_pause:
            self.on_pause(t)
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        while True:
            time.sleep(1.0)
            t = gpu_temp()
            if t is None or t <= self.max_temp - self.hyst or time.time() - start > 600:
                break
        self.paused_s += time.time() - start
        self.last_temp = t
        if self.on_resume:
            self.on_resume(t, time.time() - start)


def wait_until_cool(target=70.0, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        t = gpu_temp()
        if t is None or t <= target:
            return t
        time.sleep(2)
    return gpu_temp()


def health_check(seconds=180, progress=None):
    """Repeat identical GPU work; any change in results means the GPU is computing incorrectly."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    q, k, v = (torch.randn(8, 14, 1024, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    mem = torch.randn(128 * 1024 * 1024 // 2, device="cuda", dtype=torch.bfloat16)

    def work():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return a @ b, o, mem * 1.5 + 0.25

    ref = [t.clone() for t in work()]
    torch.cuda.synchronize()
    names = ["matrix math", "attention", "memory"]
    bad = dict.fromkeys(names, 0)
    t0, it, temps, first_error_temp = time.time(), 0, [], None
    while time.time() - t0 < seconds:
        for name, r, o in zip(names, ref, work()):
            if not torch.equal(r, o):
                bad[name] += 1
                if first_error_temp is None:
                    first_error_temp = temps[-1] if temps else gpu_temp()
        it += 1
        if it % 100 == 0:
            temps.append(gpu_temp())
            if progress:
                progress(time.time() - t0, seconds, temps[-1], sum(bad.values()))
    errors = sum(bad.values())
    result = {"stable": errors == 0, "errors": errors, "by_type": bad, "iterations": it, "seconds": round(time.time() - t0),
              "max_temp": max((t for t in temps if t), default=None), "first_error_temp": first_error_temp,
              "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(HEALTH_FILE, "w") as f:
        json.dump(result, f, indent=1)
    return result


def explain(result):
    """Plain-English verdict + advice."""
    if not result:
        return "No health check yet."
    if result["stable"]:
        return (f"✅ **Your GPU computed correctly** — {result['iterations']:,} identical runs over {result['seconds']}s, "
                f"peaking at {result['max_temp']:.0f} °C.")
    t = result.get("first_error_temp")
    return (f"⚠️ **Your GPU made {result['errors']} calculation error(s)** while under full load"
            + (f" (first one at about {t:.0f} °C)" if t else "") + ". This is a hardware/cooling problem, not your data or "
            "settings — it causes crashes and can silently ruin training.\n\n"
            "**What we do automatically:** training pauses whenever the GPU reaches "
            f"{max_temp_setting():.0f} °C and resumes after it cools, and Quick Start retries from the last checkpoint if a crash still happens.\n\n"
            "**What fixes it for good (pick any):**\n"
            "- Raise the back of the laptop or use a cooling pad; keep the vents clear.\n"
            "- In your laptop's control app (e.g. Razer Synapse / Armoury Crate / Lenovo Vantage), switch from a "
            "*Boost / Turbo / overclocked* GPU mode to *Balanced*, and remove any GPU overclock (e.g. MSI Afterburner).\n"
            "- Clean dust from the fans; update or roll back the NVIDIA driver if problems began after an update.\n"
            "- Then run this check again.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, default=180, help="seconds to stress-test")
    a = ap.parse_args()
    r = health_check(a.check, progress=lambda e, s, t, n: print(f"{e:4.0f}/{s:.0f}s  {t} °C  errors {n}", flush=True))
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["stable"] else 1)

"""
gpu_guard.py - Keep training correct on GPUs that misbehave when hot (common on laptops).

Some GPUs return silently wrong results or crash ("illegal memory access", "CUBLAS_STATUS_INTERNAL_ERROR")
once they run very hot. Two tools:

  ThermalGuard   a thermostat that rests briefly between steps to hold the GPU just under TB_MAX_GPU_TEMP (default
                 82 °C), plus a full pause at the limit until it has cooled by 5 °C. Used automatically by
                 live_monitor.py in every training script. Set
                 TB_MAX_GPU_TEMP=0 to disable. (On Macs the temperature can't be read without root, so the
                 guard stays idle; macOS throttles a hot chip itself.)
  health_check   a stress test that repeats identical GPU work and verifies the results never change.
                 Runs on CUDA, Apple MPS, or CPU.
                 python gpu_guard.py --check 180          (writes gpu_health.json)
"""

import os
import sys
import json
import time
import argparse
import subprocess

import device_utils

HEALTH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_health.json")


def gpu_temp():
    """GPU temperature in °C, or None when it can't be read (Macs, CPU-only machines)."""
    try:                                   # straight from the driver: microseconds, so the thermostat can check every second
        import hw_info
        import pynvml
        dev = hw_info._nvml()
        if dev is not None:
            return float(pynvml.nvmlDeviceGetTemperature(dev, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        pass
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
    """Two layers. The thermostat: as the GPU nears the limit it rests briefly after each step (a share of the time that grows
    with the heat), holding the temperature just under the limit instead of bouncing off it - laptops that live at 85+ °C
    keep training smoothly. The backstop: at the limit itself, a full pause until the GPU has cooled by `hysteresis`."""

    HOLD_BELOW = 3.0      # the thermostat aims this many °C under the limit
    GAIN = 0.03           # rest share added per second, per °C over the aim (removed the same way when under it)
    RISE_GAIN = 0.08      # extra rest per °C/s of warming once near the aim: laptops heat ~1 °C/s, so react early
    MAX_REST = 0.6        # at most 60% rest - beyond that a full pause (the backstop) is the better tool

    def __init__(self, max_temp=None, hysteresis=5.0, check_every_s=1.0, on_pause=None, on_resume=None, thermostat=True):
        self.max_temp = max_temp_setting() if max_temp is None else max_temp
        self.hyst = hysteresis
        self.check_every = check_every_s
        self.on_pause, self.on_resume = on_pause, on_resume
        self.thermostat = thermostat
        self._last = 0.0
        self._step_end = None
        self.paused_s = 0.0
        self.pauses = 0
        self.rest = 0.0          # current share of time spent resting (0..MAX_REST)
        self.rested_s = 0.0
        self.last_temp = None

    @property
    def enabled(self):
        return bool(self.max_temp)

    def maybe_cool(self):
        """Call once per training step. Cheap: reads the temperature at most every few seconds."""
        now = time.time()
        step_s = now - self._step_end if self._step_end else None
        self._step_end = now
        if not self.enabled:
            return
        if now - self._last >= self.check_every:
            dt = min(10.0, now - self._last) if self._last else self.check_every
            self._last = now
            prev, t = self.last_temp, gpu_temp()
            self.last_temp = t
            if t is not None and t >= self.max_temp:
                self._pause(t)
                self.rest = max(self.rest, 0.4)       # come back gently rather than at full heat
                self._step_end = time.time()
                return
            if t is not None and self.thermostat:
                aim = self.max_temp - self.HOLD_BELOW
                rise = (t - prev) / dt if prev is not None and dt > 0 else 0.0
                change = self.GAIN * (t - aim) * dt
                if t > aim - 4 and rise > 0:
                    change += self.RISE_GAIN * rise * dt
                self.rest = min(self.MAX_REST, max(0.0, self.rest + change))
        if self.rest > 0.01 and step_s:
            device_utils.synchronize()                # let queued GPU work finish so the card really idles
            pause = min(5.0, step_s * self.rest / (1 - self.rest))
            time.sleep(pause)
            self.rested_s += pause
            self._step_end = time.time()              # resting doesn't count as step time

    def _pause(self, t):
        self.pauses += 1
        start = time.time()
        if self.on_pause:
            self.on_pause(t)
        device_utils.synchronize()
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
    """Repeat identical accelerator work; any change in results means it is computing incorrectly."""
    import contextlib
    import torch
    import torch.nn.functional as F
    dev = device_utils.accelerator()
    if dev == "cpu":
        result = {"stable": True, "skipped": "no GPU", "device": device_utils.device_name(),
                  "errors": 0, "iterations": 0, "seconds": 0, "max_temp": None, "first_error_temp": None,
                  "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(HEALTH_FILE, "w") as f:
            json.dump(result, f, indent=1)
        return result
    # bf16 matmul is solid on CUDA; MPS is most reliable in fp16.
    dtype = torch.bfloat16 if dev == "cuda" else torch.float16
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device=dev, dtype=dtype)
    b = torch.randn(4096, 4096, device=dev, dtype=dtype)
    q, k, v = (torch.randn(8, 14, 1024, 64, device=dev, dtype=dtype) for _ in range(3))
    mem = torch.randn(128 * 1024 * 1024 // 2, device=dev, dtype=dtype)

    if dev == "cuda":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        attn = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    else:
        attn = contextlib.nullcontext()  # MPS uses its own SDPA path

    def work():
        with attn:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return a @ b, o, mem * 1.5 + 0.25

    ref = [t.clone() for t in work()]
    device_utils.synchronize()
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
    result = {"stable": errors == 0, "device": device_utils.device_name(), "errors": errors, "by_type": bad,
              "iterations": it, "seconds": round(time.time() - t0),
              "max_temp": max((t for t in temps if t), default=None), "first_error_temp": first_error_temp,
              "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(HEALTH_FILE, "w") as f:
        json.dump(result, f, indent=1)
    return result


def explain(result):
    """Plain-English verdict + advice."""
    if not result:
        return "No health check yet."
    if result.get("skipped"):
        return "No GPU found, so there was nothing to stress-test — training will use the CPU (slow but fine for tiny models)."
    if result["stable"]:
        temp = f", peaking at {result['max_temp']:.0f} °C" if result.get("max_temp") else ""
        return (f"✅ **Your {result.get('device') or 'GPU'} computed correctly** — {result['iterations']:,} identical runs "
                f"over {result['seconds']}s{temp}.")
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

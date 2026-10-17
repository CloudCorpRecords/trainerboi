"""
hw_info.py - lightweight, torch-free hardware + job facts (safe to import in any process, starts instantly).

snapshot()      live GPU / CPU / RAM / disk numbers (NVML, <1 ms)
hw_identity()   cached static facts: GPU name, VRAM, power limit, throttle temperature, CPU name and cores
active_jobs()   everything running right now (training, guided runs, Math Lab, dataset builds), verified alive
"""

import os
import re
import json
import glob
import time
import platform
import subprocess
import sys

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")

_NV = {"ok": None, "dev": None}
_ID = {}
_MPS = {"probed": False, "ok": False}


def _mps_available():
    """True when PyTorch can use Apple Metal. Probed lazily so this module stays torch-free to import."""
    if not _MPS["probed"]:
        _MPS["probed"] = True
        try:
            import torch
            mps = getattr(torch.backends, "mps", None)
            _MPS["ok"] = bool(mps is not None and mps.is_available())
        except Exception:
            _MPS["ok"] = False
    return _MPS["ok"]


def _mac_chip():
    for exe in ("/usr/sbin/sysctl", "/sbin/sysctl", "sysctl"):
        try:
            out = subprocess.run([exe, "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        except Exception:
            continue
    return None


def _nvml():
    if _NV["ok"] is None:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NV["dev"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            _NV["ok"] = True
        except Exception:
            _NV["ok"] = False
    return _NV["dev"] if _NV["ok"] else None


def hw_identity():
    if _ID:
        return _ID
    info = {"gpu": None, "vram_gb": None, "power_limit_w": None, "slowdown_c": None, "cpu": None,
            "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(logical=True)}
    dev = _nvml()
    if dev is not None:
        import pynvml
        try:
            n = pynvml.nvmlDeviceGetName(dev)
            info["gpu"] = n.decode() if isinstance(n, bytes) else n
            info["vram_gb"] = pynvml.nvmlDeviceGetMemoryInfo(dev).total / 1024 ** 3
        except Exception:
            pass
        try:
            info["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(dev) / 1000.0
        except Exception:
            pass
        try:
            info["slowdown_c"] = float(pynvml.nvmlDeviceGetTemperatureThreshold(dev, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
        except Exception:
            pass
    if sys.platform == "darwin" and info["gpu"] is None:
        # Apple Silicon: the CPU and GPU share one pool of unified memory.
        chip = _mac_chip()
        if chip:
            info["gpu"] = f"{chip} (MPS)"
            info["vram_gb"] = psutil.virtual_memory().total / 1024 ** 3
            info["unified_memory"] = True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            info["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        info["cpu"] = (_mac_chip() if sys.platform == "darwin" else None) or platform.processor() or "CPU"
    _ID.update(info)
    return _ID


def short_gpu_name(name):
    return re.sub(r"\s*(Laptop GPU|Laptop)\b", "", (name or "GPU").replace("NVIDIA ", "").replace("GeForce ", "")).strip()


def snapshot():
    """Fresh numbers right now. Fields are None when the GPU can't be read (never a fake 0)."""
    hw = hw_identity()
    g = {"name": short_gpu_name(hw.get("gpu")), "full_name": hw.get("gpu"), "ok": False, "vram_total_gb": hw.get("vram_gb"),
         "vram_used_gb": None, "util": None, "temp": None, "power": None, "power_limit": hw.get("power_limit_w"),
         "clock_mhz": None, "slowdown_c": hw.get("slowdown_c")}
    dev = _nvml()
    if dev is not None:
        import pynvml
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(dev)
            g.update(vram_used_gb=mem.used / 1024 ** 3, vram_total_gb=mem.total / 1024 ** 3, ok=True,
                     util=float(pynvml.nvmlDeviceGetUtilizationRates(dev).gpu),
                     temp=float(pynvml.nvmlDeviceGetTemperature(dev, pynvml.NVML_TEMPERATURE_GPU)))
            try:
                g["power"] = pynvml.nvmlDeviceGetPowerUsage(dev) / 1000.0
                g["clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(dev, pynvml.NVML_CLOCK_SM))
            except Exception:
                pass
        except Exception:
            pass
    elif hw.get("unified_memory") and _mps_available():
        # Apple Silicon: report the shared pool; per-process MPS usage when torch is loaded.
        g["ok"] = True
        try:
            import torch
            g["vram_used_gb"] = torch.mps.current_allocated_memory() / 1024 ** 3
        except Exception:
            pass
    ram = psutil.virtual_memory()
    drive = os.path.splitdrive(ROOT)[0]
    disk = psutil.disk_usage(drive + "\\" if drive else "/")
    return {"t": time.time(), "gpu": g,
            "cpu": {"name": hw.get("cpu"), "cores": hw.get("cores"), "threads": hw.get("threads"), "load": psutil.cpu_percent(interval=None)},
            "ram": {"used_gb": ram.used / 1024 ** 3, "total_gb": ram.total / 1024 ** 3},
            "disk_free_gb": disk.free / 1024 ** 3}


def _alive(pid):
    try:
        return bool(pid) and psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def active_jobs():
    jobs, live_runs = [], set()
    for p in glob.glob(os.path.join(RUNS_DIR, "*", "live", "status.json")):
        s = _read(p)
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if not s or s.get("state") != "running" or not _alive(s.get("pid")):
            continue
        live_runs.add(name)
        mx, step = s.get("max_steps") or 0, s.get("step") or 0
        jobs.append({"kind": "cooling" if s.get("cooling") else "train", "name": name, "phase": s.get("phase") or "Training",
                     "step": step, "max_steps": mx, "loss": (s.get("latest") or {}).get("loss"), "eta_s": s.get("eta_s"),
                     "progress": (step / mx) if mx else None})
    for p in glob.glob(os.path.join(RUNS_DIR, "*", "autopilot.json")):
        s = _read(p)
        name = os.path.basename(os.path.dirname(p))
        if s and s.get("state") == "running" and _alive(s.get("pid")) and name not in live_runs:
            cur = next((x for x in s.get("steps", []) if x.get("id") == s.get("current")), {})
            jobs.append({"kind": "guided", "name": name, "phase": cur.get("title", "Working"), "progress": None})
    for p in glob.glob(os.path.join(RUNS_DIR, "*", "math", "status.json")):
        s = _read(p)
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if s and s.get("state") == "running" and _alive(s.get("pid")):
            jobs.append({"kind": "math", "name": name, "phase": s.get("phase", "Math Lab"), "progress": None})
    for p in glob.glob(os.path.join(ROOT, "data", "*", "build", "status.json")):
        s = _read(p)
        name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if s and s.get("state") == "running" and _alive(s.get("pid")):
            pr = s.get("progress") or {}
            jobs.append({"kind": "data", "name": name, "phase": s.get("phase", "Building dataset"),
                         "progress": (pr["done"] / pr["total"]) if pr.get("total") else None})
    return jobs

"""
stats_manager.py - Comprehensive real-time hardware telemetry and experiment tracking for TrainerBoi.
"""

import os
import re
import glob
import json
import time
import platform
from datetime import datetime
from collections import deque
import psutil
import torch
import pandas as pd
import trainer_utils
from live_monitor import nvidia_smi

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")

from hw_info import hw_identity, short_gpu_name, active_jobs  # noqa: E402  (shared, torch-free)

# 5-minute rolling telemetry history at 1-second resolution
TELEMETRY_HISTORY = deque(maxlen=300)

def record_telemetry_snapshot():
    """Captures a snapshot of GPU and Host system metrics."""
    smi = nvidia_smi() or {}
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    disk = psutil.disk_usage(os.path.splitdrive(ROOT)[0] + "\\" if os.path.splitdrive(ROOT)[0] else "/")
    
    vram_used = smi.get("vram_used_gb", 0.0) or 0.0
    vram_total = smi.get("vram_total_gb", 16.0) or 16.0
    vram_pct = round((vram_used / vram_total) * 100, 1) if vram_total else 0.0
    
    temp = smi.get("temp_c", 0.0) or 0.0
    power = smi.get("power_w", 0.0) or 0.0
    gpu_util = smi.get("gpu_util", 0.0) or 0.0
    clock = smi.get("sm_clock_mhz", 0.0) or 0.0
    
    snapshot = {
        "timestamp": time.time(),
        "time_str": datetime.now().strftime("%H:%M:%S"),
        "gpu_util": gpu_util,
        "vram_used_gb": round(vram_used, 2),
        "vram_total_gb": round(vram_total, 2),
        "vram_pct": vram_pct,
        "temp_c": temp,
        "power_w": round(power, 1),
        "power_pct": round((power / (hw_identity().get("power_limit_w") or 150.0)) * 100, 1),
        "gpu_ok": bool(smi),
        "clock_mhz": clock,
        "cpu_pct": cpu,
        "ram_used_gb": round(ram.used / (1024 ** 3), 2),
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_pct": ram.percent,
        "disk_free_gb": round(disk.free / (1024 ** 3), 1)
    }
    TELEMETRY_HISTORY.append(snapshot)
    return snapshot

def get_latest_telemetry():
    """Always fresh: NVML reads take <1 ms, so never serve a stale reading."""
    if not TELEMETRY_HISTORY or time.time() - TELEMETRY_HISTORY[-1]["timestamp"] > 0.8:
        return record_telemetry_snapshot()
    return TELEMETRY_HISTORY[-1]

def get_telemetry_dataframe():
    """Returns a pandas DataFrame of the last 300 seconds for plotting."""
    if not TELEMETRY_HISTORY:
        record_telemetry_snapshot()
    return pd.DataFrame(list(TELEMETRY_HISTORY))

def scan_all_experiment_runs():
    """
    Scans the runs/ directory and compiles comprehensive metrics for every training run.
    """
    runs = []
    if not os.path.exists(RUNS_DIR):
        return pd.DataFrame()
        
    for item in os.listdir(RUNS_DIR):
        run_path = os.path.join(RUNS_DIR, item)
        if not os.path.isdir(run_path):
            continue
            
        autopilot_file = os.path.join(run_path, "autopilot.json")
        status_file = os.path.join(run_path, "live", "status.json")
        metrics_file = os.path.join(run_path, "live", "metrics.jsonl")
        
        state = "unknown"
        task = "Custom / Script"
        model = "-"
        dataset = "-"
        steps = 0
        final_loss = None
        duration_s = 0
        created_at = datetime.fromtimestamp(os.path.getctime(run_path)).strftime("%Y-%m-%d %H:%M")
        
        # Check autopilot info
        if os.path.exists(autopilot_file):
            try:
                with open(autopilot_file, "r", encoding="utf-8") as f:
                    ap_data = json.load(f)
                    state = ap_data.get("state", "unknown")
                    task = f"Autopilot ({ap_data.get('goal', 'N/A')})"
                    plan = ap_data.get("plan", {})
                    model = plan.get("model", "-")
                    dataset = ap_data.get("dataset", "-")
                    started = ap_data.get("started_at", 0)
                    updated = ap_data.get("updated_at", started)
                    duration_s = max(0, updated - started)
            except Exception:
                pass
                
        # Check live monitor status info
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    st_data = json.load(f)
                    if state == "unknown" or state == "done":
                        state = st_data.get("state", state)
                    cfg = st_data.get("config", {})
                    task = st_data.get("task", task)
                    if model == "-":
                        model = cfg.get("model", "-")
                    if dataset == "-":
                        dataset = cfg.get("dataset", "-")
                    steps = st_data.get("step", 0)
                    latest = st_data.get("latest", {})
                    final_loss = latest.get("loss")
                    if not duration_s and st_data.get("started_at"):
                        end_t = st_data.get("finished_at", time.time())
                        duration_s = max(0, end_t - st_data["started_at"])
            except Exception:
                pass
                
        # Check metrics.jsonl if loss still none
        if final_loss is None and os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        last_metric = json.loads(lines[-1])
                        final_loss = last_metric.get("loss")
                        if not steps:
                            steps = last_metric.get("step", 0)
            except Exception:
                pass
                
        runs.append({
            "Run Name": item,
            "Task": task,
            "Base Model": os.path.basename(model) if "/" in model else model,
            "Dataset": os.path.basename(dataset) if "/" in dataset or "\\" in dataset else dataset,
            "Steps": steps,
            "Final Loss": round(float(final_loss), 4) if final_loss is not None else "-",
            "Duration": _format_duration(duration_s),
            "Status": state.upper(),
            "Created": created_at,
            "duration_s": duration_s
        })
        
    df = pd.DataFrame(runs)
    if not df.empty:
        df = df.sort_values(by="Created", ascending=False).reset_index(drop=True)
    return df

def get_cumulative_summary_stats():
    """Calculates lifetime training metrics across all experiments."""
    df = scan_all_experiment_runs()
    total_runs = len(df)
    completed_runs = len(df[df["Status"] == "DONE"]) if not df.empty else 0
    total_compute_sec = df["duration_s"].sum() if not df.empty else 0
    
    # Check saved model weights/adapters
    models_found = len(glob.glob(os.path.join(RUNS_DIR, "**/adapter_config.json"), recursive=True))
    if os.path.exists(os.path.join(ROOT, "output_qlora", "final_adapter")):
        models_found += 1
    if os.path.exists(os.path.join(ROOT, "model_xgboost_classifier.json")):
        models_found += 1
    if os.path.exists(os.path.join(ROOT, "model_vision_resnet18.pt")):
        models_found += 1
        
    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "total_compute_time": _format_duration(total_compute_sec),
        "saved_checkpoints": models_found
    }

def _format_duration(sec):
    sec = int(sec or 0)
    if sec < 60:
        return f"{sec}s"
    elif sec < 3600:
        return f"{sec // 60}m {sec % 60}s"
    else:
        return f"{sec // 3600}h {(sec % 3600) // 60}m"

# Render HTML for persistent Top Bar
def render_persistent_top_bar(current_run_name=None):
    snap = get_latest_telemetry()
    hw = hw_identity()
    ok = snap.get("gpu_ok")
    temp = snap["temp_c"]
    slow = hw.get("slowdown_c") or 87.0
    temp_color = "#10b981" if temp < 70 else ("#f59e0b" if temp < 80 else "#ef4444")
    jobs = active_jobs()
    if current_run_name and not jobs:
        jobs = [{"kind": "train", "name": current_run_name, "text": "Training"}]

    if jobs:
        j = jobs[0]
        icon, color = {"train": ("🔵", "#60a5fa"), "cooling": ("🟠", "#fbbf24"), "guided": ("🔵", "#60a5fa"),
                       "math": ("🧮", "#a78bfa"), "data": ("📚", "#34d399")}.get(j["kind"], ("🔵", "#60a5fa"))
        more = f" +{len(jobs) - 1} more" if len(jobs) > 1 else ""
        active_tag = f"{icon} {j['name']} · {j['text']}{more}"
        badge_bg, badge_border, badge_color = f"{color}26", f"{color}66", color
    else:
        active_tag, badge_bg, badge_border, badge_color = "🟢 Idle - nothing running", "rgba(16, 185, 129, 0.15)", "rgba(16, 185, 129, 0.4)", "#34d399"
    if ok and temp >= slow - 3:
        active_tag += f" · 🔥 {temp:.0f} °C - near the throttle limit"

    gpu_name = short_gpu_name(hw.get("gpu") or snap.get("gpu_name"))
    total = snap["vram_total_gb"] or hw.get("vram_gb") or 0
    na = '<span style="opacity:.6">n/a</span>'
    if ok:
        vram = f'<b>{snap["vram_used_gb"]}</b> / {total:.0f} GB'
        util = f'<b>{snap["gpu_util"]:.0f}</b>%'
        temp_s = f'<span style="color:{temp_color}; font-weight:700;">{temp:.0f} °C</span>'
        power = f'<b>{snap["power_w"]}</b> W'
        clock = f'<b>{snap["clock_mhz"]:.0f}</b> MHz'
    else:
        vram = util = temp_s = power = clock = na
    return f"""
    <div class="top-nav-bar">
        <div class="nav-left">
            <div class="brand-logo">🚀 <span>TrainerBoi</span> Studio</div>
            <div class="device-pill"><span class="dot"></span><b>{gpu_name}</b> ({total:.0f} GB)</div>
            <div class="status-pill" style="background:{badge_bg}; border:1px solid {badge_border}; color:{badge_color}">{active_tag}</div>
        </div>
        <div class="nav-metrics">
            <div class="metric-pill" title="Memory used on the whole GPU, by every program"><span class="icon">💾</span><span class="label">VRAM:</span>
                <span class="val">{vram}</span><div class="mini-bar"><div style="width:{snap["vram_pct"] if ok else 0}%;"></div></div></div>
            <div class="metric-pill"><span class="icon">📈</span><span class="label">GPU:</span><span class="val">{util}</span></div>
            <div class="metric-pill"><span class="icon">🌡️</span><span class="label">Temp:</span><span class="val">{temp_s}</span></div>
            <div class="metric-pill"><span class="icon">⚡</span><span class="label">Power:</span><span class="val">{power}</span></div>
            <div class="metric-pill"><span class="icon">🧠</span><span class="label">RAM:</span><span class="val"><b>{snap['ram_used_gb']}</b> / {snap['ram_total_gb']:.0f} GB</span></div>
        </div>
    </div>
    """


def render_hardware_cards():
    snap = get_latest_telemetry()
    temp = snap["temp_c"]
    temp_badge = "Optimal" if temp < 70 else ("Warm" if temp < 80 else "Critical")
    temp_color = "#10b981" if temp < 70 else ("#f59e0b" if temp < 80 else "#ef4444")
    
    html = f"""
    <div class="telemetry-grid">
        <!-- CARD 1: VRAM -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">GPU Dedicated VRAM</span>
                <span class="badge blue">{snap['vram_pct']}% Used</span>
            </div>
            <div class="card-num">{snap['vram_used_gb']} <span class="unit">/ {snap['vram_total_gb']} GB</span></div>
            <div class="progress-bar">
                <div class="fill blue" style="width: {snap['vram_pct']}%;"></div>
            </div>
            <div class="card-sub">{round(snap['vram_total_gb'] - snap['vram_used_gb'], 2)} GB free · {short_gpu_name(hw_identity().get('gpu'))}</div>
        </div>

        <!-- CARD 2: GPU UTILIZATION -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">GPU SM Compute Engine</span>
                <span class="badge purple">{snap['clock_mhz']} MHz</span>
            </div>
            <div class="card-num">{snap['gpu_util']} <span class="unit">%</span></div>
            <div class="progress-bar">
                <div class="fill purple" style="width: {snap['gpu_util']}%;"></div>
            </div>
            <div class="card-sub">Live GPU utilisation (all programs)</div>
        </div>

        <!-- CARD 3: TEMPERATURE -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">GPU Core Temperature</span>
                <span class="badge" style="background:{temp_color}22; color:{temp_color}; border:1px solid {temp_color}44;">{temp_badge}</span>
            </div>
            <div class="card-num" style="color:{temp_color};">{temp} <span class="unit">°C</span></div>
            <div class="progress-bar">
                <div class="fill" style="width: {min(100, (temp/90)*100)}%; background:{temp_color};"></div>
            </div>
            <div class="card-sub">Throttle limit: {hw_identity().get('slowdown_c') or 'unknown'} °C · training pauses at 82 °C</div>
        </div>

        <!-- CARD 4: POWER DRAW -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">GPU Power Consumption</span>
                <span class="badge amber">{snap['power_pct']}% TDP</span>
            </div>
            <div class="card-num">{snap['power_w']} <span class="unit">/ {(hw_identity().get('power_limit_w') or 0):.0f} W</span></div>
            <div class="progress-bar">
                <div class="fill amber" style="width: {snap['power_pct']}%;"></div>
            </div>
            <div class="card-sub">Power limit set by the laptop/driver: {(hw_identity().get('power_limit_w') or 0):.0f} W</div>
        </div>

        <!-- CARD 5: SYSTEM RAM -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">Host System Memory</span>
                <span class="badge green">{snap['ram_pct']}%</span>
            </div>
            <div class="card-num">{snap['ram_used_gb']} <span class="unit">/ {snap['ram_total_gb']} GB</span></div>
            <div class="progress-bar">
                <div class="fill green" style="width: {snap['ram_pct']}%;"></div>
            </div>
            <div class="card-sub">{round(snap['ram_total_gb'] - snap['ram_used_gb'], 2)} GB free system RAM</div>
        </div>

        <!-- CARD 6: HOST CPU -->
        <div class="stat-card">
            <div class="card-head">
                <span class="title">{hw_identity().get("cpu")}</span>
                <span class="badge cyan">{hw_identity().get("cores")} cores / {hw_identity().get("threads")} threads</span>
            </div>
            <div class="card-num">{snap['cpu_pct']} <span class="unit">% Load</span></div>
            <div class="progress-bar">
                <div class="fill cyan" style="width: {snap['cpu_pct']}%;"></div>
            </div>
            <div class="card-sub">Disk free: {snap['disk_free_gb']} GB</div>
        </div>
    </div>
    """
    return html

def render_lifetime_banner():
    stats = get_cumulative_summary_stats()
    html = f"""
    <div class="lifetime-banner">
        <div class="banner-item">
            <div class="val">{stats['total_runs']}</div>
            <div class="lbl">Total Experiments</div>
        </div>
        <div class="banner-item">
            <div class="val" style="color:#10b981;">{stats['completed_runs']}</div>
            <div class="lbl">Completed Models</div>
        </div>
        <div class="banner-item">
            <div class="val" style="color:#6366f1;">{stats['total_compute_time']}</div>
            <div class="lbl">GPU Training Time</div>
        </div>
        <div class="banner-item">
            <div class="val" style="color:#f59e0b;">{stats['saved_checkpoints']}</div>
            <div class="lbl">Saved Adapters & Weights</div>
        </div>
    </div>
    """
    return html

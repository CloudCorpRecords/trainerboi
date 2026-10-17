"""
trainer_utils.py - High-Efficiency GPU Optimization, Memory Management & Direct NVML Telemetry.

Works on NVIDIA/CUDA, Apple Silicon/MPS and CPU-only machines (see device_utils).
"""

import os
import gc
import torch
import subprocess

import device_utils

_NVML_INITIALIZED = False
_NVML_HANDLE = None

def _init_nvml():
    global _NVML_INITIALIZED, _NVML_HANDLE
    if not _NVML_INITIALIZED:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            _NVML_INITIALIZED = True
        except Exception:
            _NVML_INITIALIZED = False

def enable_maximum_gpu_performance():
    """
    Applies production-grade hardware acceleration & memory allocation flags:
    - expandable_segments:True (Prevents VRAM fragmentation and eliminates spurious OOMs)
    - TF32 on Ampere tensor cores (3x-5x FP32 matmul speedup)
    - cuDNN benchmarking for fast kernel auto-tuning
    - CUDA Lazy Module Loading (saves 300MB+ VRAM at startup)
    - MPS fallback on Apple Silicon (ops without Metal kernels run on CPU instead of failing)
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    acc = device_utils.accelerator()
    if acc == "mps":
        # Silently-slow beats crashing: fall back to CPU for the few ops with no Metal kernel.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Override hook if individual SDPA kernels need toggling
        for name in filter(None, os.environ.get("TB_SDPA_DISABLE", "").split(",")):
            getattr(torch.backends.cuda, f"enable_{'mem_efficient' if name == 'efficient' else name}_sdp")(False)

def clear_gpu_memory():
    """Flushes GPU VRAM cache, resets peak stats, and forces garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    elif device_utils.has_mps():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def get_gpu_telemetry():
    """
    Returns real-time GPU statistics in <1ms using direct C-API NVML calls,
    with subprocess fallback if NVML is unavailable. On Apple Silicon returns
    MPS memory stats (temperature/power are not readable without root).
    """
    acc = device_utils.accelerator()

    if acc == "cpu":
        return {
            "available": False,
            "accelerator": "cpu",
            "device_name": device_utils.device_name(),
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_allocated_gb": 0.0,
            "vram_reserved_gb": 0.0,
            "vram_free_gb": 0.0,
            "temperature_c": None,
            "utilization_pct": None,
            "power_watts": None
        }

    if acc == "mps":
        try:
            import psutil
            total_mem = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            total_mem = 0.0
        allocated = device_utils.allocated_gb() or 0.0
        reserved = device_utils.reserved_gb() or 0.0
        return {
            "available": True,
            "accelerator": "mps",
            "device_name": device_utils.device_name(),
            "vram_total_gb": round(total_mem, 2),
            "vram_allocated_gb": round(allocated, 2),
            "vram_reserved_gb": round(reserved, 2),
            "vram_free_gb": round(max(total_mem - reserved, 0.0), 2),
            "temperature_c": None,
            "utilization_pct": None,
            "power_watts": None,
            "clock_mhz": None,
            "bf16_supported": False,
            "unified_memory": True,
        }

    device_idx = 0
    props = torch.cuda.get_device_properties(device_idx)
    total_mem = props.total_memory / (1024 ** 3)
    allocated_mem = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
    reserved_mem = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
    free_mem = total_mem - reserved_mem

    temp_c = None
    util_pct = None
    power_w = None
    clock_mhz = None

    _init_nvml()
    if _NVML_INITIALIZED:
        import pynvml
        try:
            temp_c = float(pynvml.nvmlDeviceGetTemperature(_NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
            util_pct = float(pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu)
            power_w = round(float(pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0), 1)
            clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(_NVML_HANDLE, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass

    if temp_c is None:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                temp_c = float(parts[0])
                util_pct = float(parts[1])
                power_w = float(parts[2])
        except Exception:
            pass

    return {
        "available": True,
        "accelerator": "cuda",
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_total_gb": round(total_mem, 2),
        "vram_allocated_gb": round(allocated_mem, 2),
        "vram_reserved_gb": round(reserved_mem, 2),
        "vram_free_gb": round(free_mem, 2),
        "temperature_c": temp_c,
        "utilization_pct": util_pct,
        "power_watts": power_w,
        "clock_mhz": clock_mhz,
        "bf16_supported": torch.cuda.is_bf16_supported()
    }

def get_optimal_dataloader_kwargs():
    """Returns high-throughput DataLoader arguments for this machine."""
    if device_utils.accelerator() == "cuda":
        return {
            "pin_memory": True,
            "num_workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 2
        }
    # Pinned memory only helps CUDA copies; worker processes are fragile on macOS (spawn).
    return {
        "pin_memory": False,
        "num_workers": 0,
    }

# Auto-apply maximum optimizations upon import
enable_maximum_gpu_performance()


def benchmark_gpu(size=4096, iters=20):
    """Quick matmul throughput test on this machine's accelerator; returns a human-readable report."""
    import time
    import torch
    clear_gpu_memory()
    acc = device_utils.accelerator()
    if acc == "cpu":
        # A full-size CPU benchmark would take minutes; use a small one.
        size, iters, dtype = 1024, 5, torch.float32
    else:
        dtype = torch.float16
    device = acc if acc != "cpu" else "cpu"
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    torch.matmul(a, b)
    device_utils.synchronize()
    t0 = time.time()
    for _ in range(iters):
        torch.matmul(a, b)
    device_utils.synchronize()
    elapsed = time.time() - t0
    tflops = (iters * 2 * size ** 3 / elapsed) / 1e12
    return (f"Benchmark successful!\n- Device: {device_utils.device_name()}\n- Matrix: {size}x{size} ({dtype})\n- Iterations: {iters}\n"
            f"- Elapsed: {elapsed * 1000:.2f} ms\n- Throughput: ~{tflops:.2f} TFLOPS")

"""
device_utils.py - one place that knows which accelerator this computer has.

Windows/Linux machines with an NVIDIA GPU train on CUDA. Apple Silicon Macs have
no NVIDIA GPU and no CUDA: PyTorch on macOS uses MPS (Metal). Older machines
have only a CPU.

Every training / inference script asks this module instead of calling
``torch.cuda`` directly, so the same code runs on CUDA, MPS and CPU::

    import device_utils
    print(device_utils.accelerator())          # "cuda" | "mps" | "cpu"
    model = AutoModelForCausalLM.from_pretrained(mid, **device_utils.model_load_kwargs())
    model = device_utils.to_device(model)

Set TB_DEVICE=cuda|mps|cpu to force a device (for testing).

This module never imports torch at import time, so it is safe (and instant) to
import anywhere, including torch-free processes like the desktop backend check.
"""

import contextlib
import importlib.util
import os
import platform
import subprocess
import sys
import time


def has_package(name):
    """True if ``import name`` would succeed (without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def has_cuda():
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_mps():
    torch = _torch()
    if torch is None:
        return False
    try:
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def accelerator():
    """Best available device: "cuda" | "mps" | "cpu"."""
    forced = (os.environ.get("TB_DEVICE") or "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        if forced == "cuda" and not has_cuda():
            print("WARNING: TB_DEVICE=cuda but CUDA is not available; falling back.")
        elif forced == "mps" and not has_mps():
            print("WARNING: TB_DEVICE=mps but MPS is not available; falling back.")
        else:
            return forced
    if has_cuda():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"


def device_label():
    return {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (Metal/MPS)", "cpu": "CPU"}[accelerator()]


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


def device_name():
    """Human-readable compute device, e.g. 'NVIDIA GeForce RTX 3080 Ti' or 'Apple M1 Pro (MPS)'."""
    acc = accelerator()
    torch = _torch()
    if acc == "cuda" and torch is not None:
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return "NVIDIA GPU"
    if acc == "mps":
        return f"{_mac_chip() or 'Apple Silicon'} (MPS)"
    cpu = _mac_chip() if sys.platform == "darwin" else None
    return cpu or platform.processor() or "CPU"


def usable_memory_gb():
    """Memory (GB) training can realistically use.

    On CUDA this is the GPU's VRAM. On Apple Silicon the CPU and GPU share one
    pool of unified memory, so only part of it is usable for training.
    """
    acc = accelerator()
    torch = _torch()
    if acc == "cuda" and torch is not None:
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            pass
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        ram = 16.0
    return ram * (0.6 if acc == "mps" else 0.5)


def is_bf16_supported():
    """True only on CUDA GPUs with bf16 support.

    MPS bf16 training is unreliable (many ops lack bf16 Metal kernels), so MPS
    uses fp16 and CPU uses fp32 instead.
    """
    if accelerator() != "cuda":
        return False
    torch = _torch()
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def default_dtype():
    """Torch dtype for loading model weights on this machine."""
    torch = _torch()
    acc = accelerator()
    if acc == "cuda":
        return torch.bfloat16 if is_bf16_supported() else torch.float16
    if acc == "mps":
        return torch.float16
    return torch.float32


def low_precision_dtype():
    """Weights dtype when the user explicitly asks for low precision (e.g. --pure_bf16)."""
    torch = _torch()
    if accelerator() == "mps":
        return torch.float16
    return torch.bfloat16


def supports_4bit():
    """bitsandbytes 4-bit (QLoRA) only works on CUDA GPUs (compute capability 7.0+)."""
    if accelerator() != "cuda" or not has_package("bitsandbytes"):
        return False
    torch = _torch()
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 7
    except Exception:
        return False


def quant_config_4bit(load_4bit, what="this run"):
    """Returns (config_or_None, note_or_None).

    When 4-bit was requested but is unavailable (Mac / CPU), returns None plus a
    plain-English note explaining the automatic fallback to full-precision LoRA.
    """
    if not load_4bit:
        return None, None
    if supports_4bit():
        from transformers import BitsAndBytesConfig
        torch = _torch()
        return (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16), None)
    return (None, f"--> 4-bit loading is only available on NVIDIA GPUs; {what} will use full-precision "
                  f"weights on {device_label()} instead (same steps, a little more memory).")


def model_load_kwargs(dtype=None):
    """kwargs for ``AutoModelForCausalLM.from_pretrained`` on this machine.

    On CUDA the model loads straight onto the GPU. Everywhere else it loads on
    CPU and the caller moves it with :func:`to_device`.
    """
    torch = _torch()
    kw = {"dtype": default_dtype() if dtype is None else dtype}
    if accelerator() == "cuda":
        kw["device_map"] = {"": 0}
    return kw


def to_device(model):
    """Move a loaded model onto this machine's accelerator (no-op on CPU/CUDA-mapped)."""
    if accelerator() == "mps":
        return model.to("mps")
    return model


def training_optim(want_8bit=True):
    """Optimizer name for HF TrainingArguments.

    8-bit optimizers (paged_adamw_8bit) need bitsandbytes + CUDA; fused AdamW
    needs CUDA. Everything else uses plain adamw_torch.
    """
    if accelerator() != "cuda" or not has_package("bitsandbytes"):
        return "adamw_torch"
    return "paged_adamw_8bit" if want_8bit else "adamw_torch_fused"


def mixed_precision():
    """{"bf16": ..., "fp16": ...} flags for HF TrainingArguments on this machine."""
    acc = accelerator()
    if acc == "cuda":
        bf16 = is_bf16_supported()
        return {"bf16": bf16, "fp16": not bf16}
    if acc == "mps":
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def extra_training_args():
    """Extra HF TrainingArguments on this machine (MPS flag when supported)."""
    if accelerator() != "mps":
        return {}
    try:
        from transformers import TrainingArguments
        if "use_mps_device" in getattr(TrainingArguments, "__dataclass_fields__", {}):
            return {"use_mps_device": True}
    except Exception:
        pass
    return {}


def allow_tf32():
    return accelerator() == "cuda"


def allow_compile():
    """torch.compile only reliably speeds up CUDA training (inductor has no MPS backend)."""
    return accelerator() == "cuda"


def allow_liger():
    return has_package("liger_kernel")


def empty_cache():
    """Free unused accelerator memory. Never raises."""
    import gc
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        if has_mps():
            torch.mps.empty_cache()
    except Exception:
        pass


def _mps_mem(which):
    torch = _torch()
    try:
        if which == "current":
            return torch.mps.current_allocated_memory() / 1024 ** 3
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    except Exception:
        return None


def allocated_gb():
    """Accelerator memory currently held by torch (GB), or None on CPU."""
    torch = _torch()
    acc = accelerator()
    try:
        if acc == "cuda":
            return torch.cuda.memory_allocated() / 1024 ** 3
        if acc == "mps":
            return _mps_mem("current")
    except Exception:
        pass
    return None


def reserved_gb():
    torch = _torch()
    acc = accelerator()
    try:
        if acc == "cuda":
            return torch.cuda.memory_reserved() / 1024 ** 3
        if acc == "mps":
            return _mps_mem("driver")
    except Exception:
        pass
    return None


def peak_gb():
    """Peak accelerator memory this process used (GB).

    MPS does not track a peak counter, so the current allocation is returned as
    the best available approximation. None on CPU.
    """
    torch = _torch()
    try:
        if accelerator() == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    if accelerator() == "mps":
        return allocated_gb()
    return None


def peak_gb_text():
    p = peak_gb()
    return f"{p:.2f} GB" if p is not None else "n/a (CPU)"


def synchronize():
    torch = _torch()
    if torch is None:
        return
    try:
        if accelerator() == "cuda":
            torch.cuda.synchronize()
        elif accelerator() == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def autocast():
    """Autocast for inference sampling: bf16 on CUDA, otherwise no-op.

    (MPS/CPU models already run in their load dtype; wrapping them adds nothing.)
    """
    torch = _torch()
    if torch is not None and accelerator() == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def xgb_device():
    return "cuda" if accelerator() == "cuda" else "cpu"


def speed_factor():
    """Rough training throughput vs. the 16 GB NVIDIA laptop this was tuned on."""
    return {"cuda": 1.0, "mps": 0.35, "cpu": 0.06}[accelerator()]


# ---------------------------------------------------------------- processes (shared, torch-free)
def kill_pid(pid):
    """Stop a process by pid (and its children on Windows). Never raises."""
    if not pid:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            except Exception:
                pass
            time.sleep(0.5)
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return
            except Exception:
                return
            try:
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass


def kill_tree(proc):
    """Stop a subprocess (and its children on Windows). Never raises."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    except Exception:
        pass


def reveal_in_file_manager(path):
    """Show a file in Explorer / Finder / the Linux file manager. Never raises."""
    try:
        if not os.path.exists(path):
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path) or "."])
    except Exception:
        pass

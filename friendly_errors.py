"""
friendly_errors.py - turns scary tracebacks into plain-English "what happened / what to do".
explain(text) -> {"key", "title", "advice"} or None. Used by the Live and Guided training screens and the auto-retry logic.
"""

import re
import sys

# (key, regex, title, advice) - first match wins, so order from specific to generic.
RULES = [
    ("oom", r"out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|"
            r"MPS backend out of memory|Metal.*out of memory",
     "The GPU ran out of memory",
     "The model plus your settings needed more GPU memory than you have free. Close other programs that use the GPU "
     "(games, browsers with video, other training runs), then try a smaller model, a shorter max length, or batch size 1. "
     "Guided training does this automatically."),
    ("gpu_hw", r"CUBLAS_STATUS_INTERNAL_ERROR|illegal memory access|misaligned address|unspecified launch failure|AcceleratorError|device-side assert",
     "The GPU hit a calculation error",
     "This is almost always caused by the GPU running too hot (laptop GPUs above ~85 °C compute wrong answers). "
     "Let it cool, raise the fan speed or use a cooling pad, set the laptop power mode to Balanced, and resume from the last "
     "checkpoint. Run the GPU health check in Guided training to confirm."),
    ("gated", r"gated repo|401 Client Error|403 Client Error|Access to model .* is restricted|You must be authenticated|Cannot access gated",
     "This model needs a Hugging Face login",
     "The model is gated (you have to accept its licence). Create a free Hugging Face account, accept the licence on the model "
     "page, create an access token and set it as the HF_TOKEN environment variable - or pick an open model such as "
     "Qwen2.5 (no login needed)."),
    ("offline", r"ConnectionError|Max retries exceeded|Temporary failure in name resolution|getaddrinfo failed|Failed to resolve|"
                r"couldn't connect|Can't load .* offline|LocalEntryNotFoundError|ConnectTimeout|ReadTimeout",
     "Couldn't reach the internet",
     "Downloading the model or dataset failed. Check your connection (or a VPN/proxy/firewall) and try again. Models already "
     "downloaded once work offline."),
    ("disk", r"No space left on device|not enough space|OSError: \[Errno 28\]|Only [\d.]+ GB of disk",
     "You're out of disk space",
     "Models and datasets are big. Free at least 10 GB (delete old runs in the runs folder or old datasets in the Library) and try again."),
    ("nan", r"loss is nan|nan loss|'loss': 'nan'|grad_norm.*nan",
     "The loss turned into NaN",
     "Training numerically broke. Common causes: a learning rate that is too high (try 5x lower), a GPU that is too hot, or "
     "empty/garbled examples in the data. Check the data in the Library and lower the learning rate."),
    ("format", r"Can't infer dataset format|needs 'chosen' and 'rejected'|KTO needs|GRPO needs|Unsupported file type|No .* data \(",
     "Your data isn't in a format this method understands",
     "Use the Dataset Builder (or Guided training) to convert it - it accepts almost any file. For preference methods you need "
     "'chosen' and 'rejected' answers; for fine-tuning you need question/answer or chat data."),
    ("filtered", r"Every record was filtered out|No usable (text|data|examples)",
     "Nothing usable was left after tidying",
     "Every piece of your data was filtered out as too short, repeated, garbled or off-format. Try more (or more varied) text, a different kind of example, "
     "or - for plain documents - let the helper AI turn them into questions and answers."),
    ("tokenizer", r"chat_template|apply_chat_template|Tokenizer .* has no|pad_token",
     "The model's chat format couldn't be applied",
     "This base model has no chat template. Pick an *-Instruct* version of the model, or use a different model."),
    ("missing_pkg", r"ModuleNotFoundError|No module named|ImportError: cannot import|DLL load failed",
     "A required library is missing or broken",
     "In the desktop app use TrainerBoi -> Repair installation. When running from source, re-run the installer for requirements-app.txt."),
    ("no_gpu", r"No NVIDIA GPU|CUDA is not available|Torch not compiled with CUDA|no CUDA GPUs are available",
     "No usable NVIDIA GPU was found",
     "Training needs an NVIDIA GPU with up-to-date drivers. Update the driver from nvidia.com, restart, and make sure the "
     "laptop is using the NVIDIA GPU (not integrated graphics) for this app."),
    ("locked", r"PermissionError|WinError 32|being used by another process|Access is denied",
     "A file is in use by another program",
     "Close other programs that may have the file open (Explorer previews, antivirus scans, another training run) and retry."),
    ("killed", r"MemoryError|Killed|paging file is too small|WinError 1455",
     "Your computer ran out of RAM",
     "Close other programs, or use a smaller model / dataset. Increasing the Windows paging file also helps."),
]

# Advice that differs on macOS (Macs share one pool of unified memory; there is no NVIDIA driver).
_MAC_ADVICE = {
    "oom": ("The model plus your settings needed more memory than your Mac has free (Macs share one pool of memory "
            "between the CPU and GPU). Quit other apps (browsers with video, other training runs), then try a smaller "
            "model, a shorter max length, or batch size 1. Guided training does this automatically."),
    "no_gpu": ("Training needs a GPU. On Apple Silicon Macs this means PyTorch with Metal (MPS) support — re-run setup.sh "
               "to install the correct PyTorch build. CPU-only training works for tiny models but is very slow."),
    "killed": ("Close other programs, or use a smaller model / dataset. Macs share one pool of memory between the CPU "
               "and GPU, so quitting memory-hungry apps (browsers, video editors) helps the most."),
    "locked": ("Close other programs that may have the file open (Finder previews, cloud-sync apps, another training run) "
               "and retry."),
}


def explain(text):
    if not text:
        return None
    for key, rx, title, advice in RULES:
        if re.search(rx, text, re.I):
            if sys.platform == "darwin" and key in _MAC_ADVICE:
                advice = _MAC_ADVICE[key]
            return {"key": key, "title": title, "advice": advice}
    return None


def html(text):
    """Small HTML panel (or '' if the error is not recognised)."""
    e = explain(text)
    if not e:
        return ""
    return (f"<div class='tb-insights' style='border-color:#f59e0b'><b>💡 {e['title']}</b><br>{e['advice']}</div>")

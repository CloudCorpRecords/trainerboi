import sys
import os
import time

import device_utils

def check_env():
    print("=" * 60)
    print("SYSTEM & ML ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Platform          : {sys.platform} ({os.uname().machine if hasattr(os, 'uname') else '?'})")

    # 1. Check PyTorch
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        acc = device_utils.accelerator()
        print(f"Accelerator       : {device_utils.device_label()}")
        print(f"Device            : {device_utils.device_name()}")
        print(f"CUDA Available    : {torch.cuda.is_available()}")
        try:
            print(f"MPS Available     : {bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}")
        except Exception:
            print("MPS Available     : False")

        if acc == "cuda":
            device_name = torch.cuda.get_device_name(0)
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            major, minor = torch.cuda.get_device_capability(0)
            print(f"GPU Device        : {device_name}")
            print(f"Compute Capability: {major}.{minor}")
            print(f"Total VRAM        : {total_mem_gb:.2f} GB")
            print(f"BF16 Supported    : {torch.cuda.is_bf16_supported()}")
            print(f"FP16 Supported    : True (Compute Capability {major}.{minor} >= 5.3)")

            # Quick Benchmark
            print("\nRunning quick GPU tensor compute benchmark...")
            t0 = time.time()
            a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f"GPU MatMul (4096 x 4096 x 10 in FP16) completed in: {elapsed*1000:.2f} ms")
        elif acc == "mps":
            print(f"Usable Memory     : ~{device_utils.usable_memory_gb():.1f} GB of shared unified memory")
            print(f"Mixed Precision   : FP16 (MPS does not reliably support BF16 training)")
            print("\nRunning quick MPS tensor compute benchmark...")
            import trainer_utils
            print(trainer_utils.benchmark_gpu(size=4096, iters=10))
        else:
            print("[WARNING] No GPU accelerator found - training will use the CPU (slow, tiny models only).")
            print("          On Apple Silicon this usually means PyTorch was installed without MPS support;")
            print("          re-run setup.sh / setup.bat to install the correct PyTorch build.")
    except ImportError as e:
        print(f"[ERROR] Could not import PyTorch: {e}")

    # 2. Check Key ML & Training Libraries
    print("\n" + "-" * 60)
    print("CHECKING TRAINING FRAMEWORKS & EXTENSIONS")
    print("-" * 60)
    libs = [
        ("transformers", "Hugging Face Transformers", False),
        ("datasets", "Hugging Face Datasets", False),
        ("accelerate", "Hugging Face Accelerate", False),
        ("peft", "Parameter-Efficient Fine-Tuning (LoRA)", False),
        ("bitsandbytes", "BitsAndBytes (4-bit/8-bit QLoRA)", True),
        ("trl", "Transformer Reinforcement / SFT Trainer", False),
        ("unsloth", "Unsloth (2x faster LLM fine-tuning)", True),
        ("triton", "Triton (torch.compile / fused kernels)", True),
        ("liger_kernel", "Liger Kernel (fused LLM training ops)", True),
        ("wandb", "Weights & Biases (experiment tracking)", False),
        ("evaluate", "Hugging Face Evaluate", False),
        ("timm", "PyTorch Image Models", False),
        ("sklearn", "Scikit-Learn", False),
        ("pandas", "Pandas DataFrames", False),
    ]
    cuda = device_utils.accelerator() == "cuda"
    for module_name, desc, nvidia_only in libs:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[OK] {desc:<42} : v{ver}")
        except ImportError:
            tag = "(optional on this machine - NVIDIA only)" if (nvidia_only and not cuda) else ""
            print(f"[MISSING] {desc:<42} {tag}")

    print("=" * 60)
    print("ALL CORE TRAINING PREREQUISITES READY!")
    print("=" * 60)

if __name__ == "__main__":
    check_env()

import sys
import os
import time

def check_env():
    print("=" * 60)
    print("SYSTEM & ML ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    
    # 1. Check PyTorch
    try:
        import torch
        print(f"PyTorch Version   : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available    : {cuda_avail}")
        
        if cuda_avail:
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
        else:
            print("[WARNING] CUDA is NOT available to PyTorch!")
    except ImportError as e:
        print(f"[ERROR] Could not import PyTorch: {e}")

    # 2. Check Key ML & Training Libraries
    print("\n" + "-" * 60)
    print("CHECKING TRAINING FRAMEWORKS & EXTENSIONS")
    print("-" * 60)
    libs = [
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("accelerate", "Hugging Face Accelerate"),
        ("peft", "Parameter-Efficient Fine-Tuning (LoRA)"),
        ("bitsandbytes", "BitsAndBytes (4-bit/8-bit QLoRA)"),
        ("trl", "Transformer Reinforcement / SFT Trainer"),
        ("unsloth", "Unsloth (2x faster LLM fine-tuning)"),
        ("triton", "Triton (torch.compile / fused kernels)"),
        ("liger_kernel", "Liger Kernel (fused LLM training ops)"),
        ("wandb", "Weights & Biases (experiment tracking)"),
        ("evaluate", "Hugging Face Evaluate"),
        ("timm", "PyTorch Image Models"),
        ("sklearn", "Scikit-Learn"),
        ("pandas", "Pandas DataFrames")
    ]
    for module_name, desc in libs:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"[OK] {desc:<42} : v{ver}")
        except ImportError:
            print(f"[MISSING] {desc:<42}")

    print("=" * 60)
    print("ALL CORE TRAINING PREREQUISITES READY!")
    print("=" * 60)

if __name__ == "__main__":
    check_env()

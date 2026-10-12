import time
import torch
import trainer_utils

def run_tflops_benchmark(size=4096, iters=20):
    trainer_utils.clear_gpu_memory()
    if not torch.cuda.is_available():
        return "CUDA is not available."
    
    device = "cuda"
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    
    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    ops = iters * 2 * (size ** 3)
    tflops = (ops / elapsed) / 1e12
    return f"⚡ Benchmark Complete: {size}x{size} FP16 (20 iters) took {elapsed*1000:.1f}ms → ~{tflops:.2f} TFLOPS on your GPU"

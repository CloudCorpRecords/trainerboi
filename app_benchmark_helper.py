import time
import torch
import device_utils
import trainer_utils

def run_tflops_benchmark(size=4096, iters=20):
    trainer_utils.clear_gpu_memory()
    acc = device_utils.accelerator()
    if acc == "cpu":
        size, iters, dtype = 1024, 5, torch.float32
    else:
        dtype = torch.float16

    device = acc
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    torch.matmul(a, b)
    device_utils.synchronize()

    t0 = time.time()
    for _ in range(iters):
        c = torch.matmul(a, b)
    device_utils.synchronize()
    elapsed = time.time() - t0

    ops = iters * 2 * (size ** 3)
    tflops = (ops / elapsed) / 1e12
    return (f"⚡ Benchmark Complete: {size}x{size} {dtype} ({iters} iters) took {elapsed*1000:.1f}ms → "
            f"~{tflops:.2f} TFLOPS on {device_utils.device_name()}")

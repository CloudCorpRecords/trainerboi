import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import friendly_errors  # noqa: E402
import resilient_run  # noqa: E402


class ShrinkForMemory(unittest.TestCase):
    def test_halves_batch_and_keeps_effective_batch(self):
        cmd = ["python", "t.py", "--batch_size", "16", "--grad_accum", "1", "--max_seq_len", "1024"]
        out = resilient_run.shrink_for_memory(cmd)
        self.assertEqual(resilient_run._arg(out, "--batch_size"), 8)
        self.assertEqual(resilient_run._arg(out, "--grad_accum"), 2)

    def test_at_batch_one_shortens_sequences(self):
        cmd = ["python", "t.py", "--batch_size", "1", "--grad_accum", "8", "--max_seq_len", "1024"]
        out = resilient_run.shrink_for_memory(cmd)
        self.assertEqual(resilient_run._arg(out, "--batch_size"), 1)
        self.assertLess(resilient_run._arg(out, "--max_seq_len"), 1024)

    def test_does_not_modify_the_original(self):
        cmd = ["--batch_size", "4", "--grad_accum", "4"]
        resilient_run.shrink_for_memory(cmd)
        self.assertEqual(cmd, ["--batch_size", "4", "--grad_accum", "4"])


class ErrorKinds(unittest.TestCase):
    def kind(self, text):
        return (friendly_errors.explain(text) or {}).get("key")

    def test_gpu_hiccups_are_recognised(self):
        for text in ("torch.AcceleratorError: CUDA error: an illegal instruction was encountered",
                     "CUDA error: an illegal memory access was encountered",
                     "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"):
            self.assertEqual(self.kind(text), "gpu_hw", text)

    def test_out_of_memory_wins_over_generic_cuda_error(self):
        self.assertEqual(self.kind("CUDA error: out of memory"), "oom")
        self.assertEqual(self.kind("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"), "oom")

    def test_ordinary_bugs_are_not_retried(self):
        self.assertNotIn(self.kind("KeyError: 'prompt'"), ("gpu_hw", "oom"))


if __name__ == "__main__":
    unittest.main()

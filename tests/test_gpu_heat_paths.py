import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_guard  # noqa: E402


class HealthCheckHeatAbort(unittest.TestCase):
    def test_aborts_instead_of_cooking_a_hot_gpu(self):
        if gpu_guard.device_utils.accelerator() != "cuda":
            self.skipTest("needs a CUDA GPU")
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(gpu_guard, "HEALTH_FILE", os.path.join(td, "gpu_health.json")), \
                mock.patch.object(gpu_guard, "gpu_temp", return_value=99.0):
            result = gpu_guard.health_check(seconds=30)
        self.assertTrue(result.get("aborted_too_hot"), result)
        self.assertFalse(result["stable"])
        self.assertLess(result["seconds"], 30)


class SharedThermalCallback(unittest.TestCase):
    def test_step_end_calls_the_guard(self):
        try:
            import trainer_utils
        except ImportError:
            self.skipTest("needs torch (imported by trainer_utils)")
        guard = mock.Mock()
        cb = trainer_utils.make_thermal_callback(guard=guard)
        cb.on_step_end(None, None, None)
        guard.maybe_cool.assert_called()

    def test_every_hf_style_trainer_wires_the_callback(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for fname in ("train_llm_qlora.py", "train_embeddings.py"):
            with open(os.path.join(here, fname), encoding="utf-8") as f:
                self.assertIn("make_thermal_callback", f.read(), fname)


class VisionAndTabularGuards(unittest.TestCase):
    def test_vision_loop_calls_the_guard(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "train_vision.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("ThermalGuard", src)
        self.assertIn("maybe_cool()", src)

    def test_tabular_callback_calls_the_guard(self):
        try:
            import train_tabular
        except ImportError:
            self.skipTest("needs xgboost (imported by train_tabular)")
        guard = mock.Mock()
        cb = train_tabular.TabularThermalCallback(guard=guard)
        cb.after_iteration(None, 0, None)
        guard.maybe_cool.assert_called()


if __name__ == "__main__":
    unittest.main()

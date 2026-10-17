"""Tests for the cross-platform layer (device_utils + friendly_errors).

Stdlib unittest only, so CI runs them with plain `python -m unittest` on every OS.
Nothing here needs torch, a GPU, or the network; torch-dependent behavior is
covered by the smoke runs in CI's bigger jobs... (there are none yet - see README).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import device_utils
import friendly_errors


class DeviceSelectionTest(unittest.TestCase):
    def test_accelerator_is_known_value(self):
        self.assertIn(device_utils.accelerator(), ("cuda", "mps", "cpu"))

    def test_tb_device_override(self):
        old = os.environ.get("TB_DEVICE")
        os.environ["TB_DEVICE"] = "cpu"
        try:
            self.assertEqual(device_utils.accelerator(), "cpu")
        finally:
            if old is None:
                os.environ.pop("TB_DEVICE", None)
            else:
                os.environ["TB_DEVICE"] = old

    def test_device_name_non_empty(self):
        self.assertTrue(isinstance(device_utils.device_name(), str) and device_utils.device_name())

    def test_usable_memory_positive(self):
        self.assertGreater(device_utils.usable_memory_gb(), 0)


class TrainingFlagsTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("TB_DEVICE")
        os.environ["TB_DEVICE"] = "cpu"  # deterministic: no GPU

    def tearDown(self):
        if self._old is None:
            os.environ.pop("TB_DEVICE", None)
        else:
            os.environ["TB_DEVICE"] = self._old

    def test_mixed_precision_shape(self):
        mp = device_utils.mixed_precision()
        self.assertEqual(set(mp), {"bf16", "fp16"})
        self.assertFalse(mp["bf16"] and mp["fp16"], "bf16 and fp16 must never both be on")
        self.assertEqual(mp, {"bf16": False, "fp16": False})

    def test_cpu_optim_is_plain_adamw(self):
        self.assertEqual(device_utils.training_optim(), "adamw_torch")
        self.assertEqual(device_utils.training_optim(want_8bit=False), "adamw_torch")

    def test_no_4bit_without_cuda(self):
        self.assertFalse(device_utils.supports_4bit())

    def test_quant_config_decline_explains_itself(self):
        cfg, note = device_utils.quant_config_4bit(True, "the test")
        self.assertIsNone(cfg)
        self.assertIn("NVIDIA", note)

    def test_quant_config_not_requested_is_silent(self):
        self.assertEqual(device_utils.quant_config_4bit(False), (None, None))

    def test_no_tf32_or_compile_without_cuda(self):
        self.assertFalse(device_utils.allow_tf32())
        self.assertFalse(device_utils.allow_compile())

    def test_xgb_device_cpu(self):
        self.assertEqual(device_utils.xgb_device(), "cpu")

    def test_speed_factor_positive(self):
        self.assertGreater(device_utils.speed_factor(), 0)

    def test_extra_training_args_is_dict(self):
        self.assertIsInstance(device_utils.extra_training_args(), dict)


class ProcessHelpersTest(unittest.TestCase):
    def test_kill_pid_never_raises(self):
        device_utils.kill_pid(None)
        device_utils.kill_pid(0)
        device_utils.kill_pid(99999999)  # almost certainly not a real pid

    def test_kill_tree_none_ok(self):
        device_utils.kill_tree(None)

    def test_reveal_missing_path_ok(self):
        device_utils.reveal_in_file_manager("/definitely/not/here.txt")


class FriendlyErrorsTest(unittest.TestCase):
    def test_mps_oom_recognised(self):
        e = friendly_errors.explain("RuntimeError: MPS backend out of memory")
        self.assertIsNotNone(e)
        self.assertEqual(e["key"], "oom")

    def test_cuda_oom_still_recognised(self):
        e = friendly_errors.explain("torch.OutOfMemoryError: CUDA out of memory")
        self.assertIsNotNone(e)
        self.assertEqual(e["key"], "oom")

    def test_unknown_error_is_none(self):
        self.assertIsNone(friendly_errors.explain("some totally novel failure mode xyzzy"))


if __name__ == "__main__":
    unittest.main()

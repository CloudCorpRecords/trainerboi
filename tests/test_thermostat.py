import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_guard  # noqa: E402


class Clock:
    """Fake time: steps advance it, sleeps advance it, nothing really waits."""
    def __init__(self):
        self.t = 1000.0
        self.slept = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += s


class Thermostat(unittest.TestCase):
    def run_steps(self, temps, max_temp=82.0, step_s=0.5):
        clock, guard = Clock(), gpu_guard.ThermalGuard(max_temp=max_temp, check_every_s=0.0)
        rests = []
        with mock.patch.object(gpu_guard.time, "time", clock.time), mock.patch.object(gpu_guard.time, "sleep", clock.sleep), \
                mock.patch.object(gpu_guard.device_utils, "synchronize", lambda: None):
            for t in temps:
                with mock.patch.object(gpu_guard, "gpu_temp", return_value=t):
                    clock.t += step_s
                    guard.maybe_cool()
                    rests.append(guard.rest)
        return guard, rests, clock

    def test_cool_gpu_never_rests(self):
        guard, rests, clock = self.run_steps([60, 65, 70, 72] * 5)
        self.assertEqual(max(rests), 0.0)
        self.assertEqual(clock.slept, 0.0)

    def test_rests_more_as_it_heats_and_stops_when_cool(self):
        guard, rests, _ = self.run_steps([78, 80, 81, 81, 81])
        self.assertGreater(rests[-1], rests[0])
        self.assertGreater(rests[-1], 0.0)
        self.assertEqual(guard.pauses, 0)                       # stayed under the limit: no hard pause
        guard2, rests2, _ = self.run_steps([81] * 5 + [70] * 10)
        self.assertEqual(rests2[-1], 0.0)

    def test_rest_is_capped(self):
        _, rests, _ = self.run_steps([81.9] * 200)
        self.assertLessEqual(max(rests), gpu_guard.ThermalGuard.MAX_REST)

    def test_disabled_when_limit_is_zero(self):
        guard, rests, clock = self.run_steps([95] * 5, max_temp=0)
        self.assertEqual(clock.slept, 0.0)
        self.assertEqual(guard.pauses, 0)


if __name__ == "__main__":
    unittest.main()

import subprocess
import unittest
from unittest.mock import patch

from scrl.telemetry import SpeedSampler


class SpeedSamplerTests(unittest.TestCase):
    def test_separates_capacity_from_busy_and_filters_devices(self):
        with patch("scrl.telemetry.subprocess.check_output", return_value=
                   "2, GPU-a, 94, 48, 3000, 12000\n3, GPU-b, N/A, N/A, 0, 12000\n") as query:
            sampler = SpeedSampler([2, 3])
            sampler.sample(5)
            row = sampler.sample(5)
        self.assertIn("2,3",query.call_args.args[0])
        a, b = row["gpus"]
        self.assertEqual((a["memory_occupied_pct"], a["memory_busy_pct"]), (25, 48))
        self.assertIsNone(b["utilization"])
        self.assertIn("network_bytes_per_sec",row)
        self.assertNotIn("power_watts",a)

    def test_probe_failure_is_not_a_training_failure(self):
        with patch("scrl.telemetry.subprocess.check_output", side_effect=subprocess.TimeoutExpired("probe",3)):
            row = SpeedSampler([2]).sample(5)
        self.assertEqual(row["gpu_sample_error"],"TimeoutExpired")

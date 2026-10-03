import unittest

from benchmarks.analyze_runner import evaluate_gate, summarize_intervals, union_duration


class RunnerAnalysisTest(unittest.TestCase):

    def test_overlapping_gpu_intervals_are_not_double_counted(self):
        self.assertEqual(union_duration([(0, 5), (3, 8), (9, 15)], 2, 10), 7)

    def test_gate_uses_cpu_and_gpu_bounds(self):
        ranges = [(0, 100000, "nanovllm::decode_step_0"),
                  (0, 20000, "nanovllm::decode_metadata")]
        activities = [(10000, 90000, "kernel"), (50000, 80000, "memcpy")]
        report = summarize_intervals(ranges, activities, discard_steps=0)
        self.assertEqual(report["gpu_idle_median_us"], 20)
        self.assertEqual(report["overlap_upper_bound_ratio"], 0.2)
        self.assertEqual(report["activity_counts"], {"kernel": 1, "memcpy": 1})

    def test_missing_metadata_is_not_a_failed_gate(self):
        with self.assertRaises(ValueError):
            summarize_intervals([(0, 10, "nanovllm::decode_step_0")], [], 0)

    def test_gate_requires_repeated_stable_evidence(self):
        reports = [{"steps": 100, "overlap_upper_bound_median_us": value,
                    "overlap_upper_bound_ratio": value / 1000} for value in (80, 90, 85)]
        self.assertTrue(evaluate_gate(reports)["passed"])
        reports[0]["overlap_upper_bound_ratio"] = 0.01
        reports[1]["overlap_upper_bound_ratio"] = 0.01
        self.assertFalse(evaluate_gate(reports)["passed"])
        with self.assertRaises(ValueError):
            evaluate_gate(reports[:2])


if __name__ == "__main__":
    unittest.main()

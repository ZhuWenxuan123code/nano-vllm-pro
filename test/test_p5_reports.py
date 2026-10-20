import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_p5 import summarize


class P5ReportsTest(unittest.TestCase):

    def report(self, policy, goodput):
        return dict(workload="steady", rate=10, profile_only=False, trace_sha256="same", tokens_sha256="same",
                    measurement_sha256="same", gpu_uuid="same", code=dict(source_sha256="same"),
                    config=dict(scheduler_policy=policy), cache_memory=dict(blocks=512),
                    metrics=dict(timed_out=False, goodput_requests=goodput / 128,
                                 goodput_tokens=goodput, e2e_throughput=1000, slo_miss_fraction=0.5,
                                 ttft_ms=dict(p50=50, p99=100), itl_ms=dict(p50=20, p99=30),
                                 queue_wait_ms=dict(p50=20, p99=40), submission_lag_ms=dict(p50=1, p99=2),
                                 scheduler_stats=dict(preemptions=1, recompute_tokens=20, prefix_hit_tokens=0,
                                                      admission_pauses=3, protected_prefill_steps=4),
                                 phases=dict(prefill=1, mixed=2, decode=7), prefill_tokens=200,
                                 kv_used_blocks_peak=256, graph_steps=7, mixed_eager_steps=2,
                                 p2_tokens=80, decode_tokens=100))

    def test_medians_and_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "main").mkdir()
            for i, value in enumerate((100, 110, 200)):
                (root / "main" / f"legacy-{i}.json").write_text(json.dumps(self.report("legacy", value)))
            result = summarize(root)
            self.assertEqual(result[0]["goodput_tokens"]["median"], 110)
            self.assertEqual(result[0]["runs"], 3)
            self.assertEqual(result[0]["diagnostics"]["recompute_fraction"]["median"], 0.1)
            self.assertEqual(result[0]["diagnostics"]["graph_step_fraction"]["median"], 0.7)
            self.assertIn("Scheduling and execution diagnostics", (root / "comparison.md").read_text())
            coverage = json.loads((root / "coverage.json").read_text())
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["completed_runs"], 3)
            self.assertEqual(coverage["expected_runs"], 180)
            self.assertIn("PARTIAL matrix", (root / "comparison.md").read_text())
            other = self.report("mixed", 120)
            other["trace_sha256"] = "different"
            (root / "main" / "mixed.json").write_text(json.dumps(other))
            with self.assertRaisesRegex(ValueError, "different"):
                summarize(root)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "combined").mkdir()
            for workload in ("burst", "mixed"):
                for policy in ("legacy", "mixed", "adaptive"):
                    for repeat in range(5):
                        report = self.report(policy, 100)
                        report["workload"] = workload
                        path = root / "combined" / f"{workload}-{policy}-{repeat}.json"
                        path.write_text(json.dumps(report))
            summarize(root, "combined")
            coverage = json.loads((root / "combined-coverage.json").read_text())
            self.assertTrue(coverage["complete"])
            self.assertEqual(coverage["completed_runs"], 30)
            self.assertIn("Complete matrix.", (root / "combined-comparison.md").read_text())
            summarize(root, "combined", expected_runs=6)
            coverage = json.loads((root / "combined-coverage.json").read_text())
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["expected_runs"], 36)


if __name__ == "__main__":
    unittest.main()

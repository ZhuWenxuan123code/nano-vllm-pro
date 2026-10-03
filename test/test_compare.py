import unittest

from benchmarks.compare import compare_cases, improvement_percent


def report(throughput, ttft):
    return {
        "metrics": {
            "e2e_throughput": throughput,
            "prefill_throughput": throughput,
            "decode_throughput": throughput,
            "ttft_ms": {"p50": ttft, "p99": ttft},
            "tpot_ms": {"p50": ttft, "p99": ttft},
        }
    }


class CompareTest(unittest.TestCase):

    def test_improvement_accounts_for_metric_direction(self):
        self.assertEqual(improvement_percent(100, 125, True), 25)
        self.assertEqual(improvement_percent(100, 80, False), 25)

    def test_compare_uses_medians_and_formats_markdown(self):
        table = compare_cases(
            {"decode": [report(100, 10), report(120, 12), report(140, 14)]},
            {"decode": [report(150, 9), report(180, 8), report(210, 7)]},
        )

        self.assertIn("| E2E tok/s | 120.00 | 180.00 | +50.00% |", table)
        self.assertIn("| TTFT P50 ms | 12.00 | 8.00 | +50.00% |", table)

    def test_compare_warns_on_config_mismatch(self):
        baseline = report(100, 10)
        candidate = report(100, 10)
        baseline["config"] = {"max_num_seqs": 64}
        candidate["config"] = {"max_num_seqs": 128}

        table = compare_cases({"decode": [baseline]}, {"decode": [candidate]})

        self.assertIn("Warning: configuration differs for max_num_seqs", table)

    def test_backend_difference_is_expected_in_ab_comparison(self):
        baseline = report(100, 10)
        candidate = report(105, 10)
        baseline["config"] = {"max_num_seqs": 64, "rms_norm_backend": "compiled"}
        candidate["config"] = {"max_num_seqs": 64, "rms_norm_backend": "triton"}

        table = compare_cases({"balanced": [baseline]}, {"balanced": [candidate]})

        self.assertIn("RMSNorm backend: compiled -> triton", table)
        self.assertNotIn("Warning: configuration differs", table)

    def test_decode_fusion_difference_is_expected(self):
        baseline = report(100, 10)
        candidate = report(105, 9)
        baseline["config"] = {"rms_norm_backend": "compiled", "fuse_decode_qk_rope_cache": False}
        candidate["config"] = {"rms_norm_backend": "compiled", "fuse_decode_qk_rope_cache": True}
        table = compare_cases({"decode": [baseline]}, {"decode": [candidate]})
        self.assertIn("Decode QKV fusion: False -> True", table)
        self.assertNotIn("Warning: configuration differs", table)


if __name__ == "__main__":
    unittest.main()

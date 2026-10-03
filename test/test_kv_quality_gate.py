import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.check_kv_quality import check


class KVQualityGateTest(unittest.TestCase):

    def fixture(self, root, delta=0.005):
        for length in (2048, 8192):
            for mode in ("flash", "triton", "int8"):
                report = {"window": length, "scored_tokens": 32768, "shared_prefix": False,
                          "prefill_budget": 16384, "data": {"token_hash": "fixed"},
                          "cache_memory": {"format": "int8-k32-vhead-v1" if mode == "int8" else "torch.bfloat16"},
                          "continuation_perplexity": 20 * (1 + delta) if mode == "int8" else 20}
                (root / f"{mode}-{length}.json").write_text(json.dumps(report))

    def test_both_lengths_required_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertTrue(check(root)["passed"])
            self.fixture(root, delta=.016)
            self.assertFalse(check(root)["passed"])

    def test_smoke_or_diagnostic_cannot_pass_full_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "int8-8192.json"
            report = json.loads(path.read_text())
            report["scored_tokens"] = 128
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                check(root)

    def test_old_format_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root / "int8-2048.json"
            report = json.loads(path.read_text())
            report["cache_memory"]["format"] = "int8-perhead-v0"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "format"):
                check(root)

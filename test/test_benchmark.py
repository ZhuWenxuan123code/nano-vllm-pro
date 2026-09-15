import unittest
from types import SimpleNamespace

from bench import parse_args, percentile, run_benchmark


class BenchmarkHelpersTest(unittest.TestCase):

    def test_percentile_interpolates_values(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile(values, 0), 1.0)
        self.assertEqual(percentile(values, 50), 2.5)
        self.assertEqual(percentile(values, 100), 4.0)

    def test_percentile_returns_none_for_empty_input(self):
        self.assertIsNone(percentile([], 50))

    def test_fixed_lengths_override_random_ranges(self):
        args = parse_args([
            "--input-len", "512",
            "--output-len", "128",
            "--min-input-len", "10",
            "--max-input-len", "20",
            "--min-output-len", "10",
            "--max-output-len", "20",
        ])
        self.assertEqual((args.min_input_len, args.max_input_len), (512, 512))
        self.assertEqual((args.min_output_len, args.max_output_len), (128, 128))

    def test_single_token_output_has_no_decode_metrics(self):
        class FakeCuda:

            @staticmethod
            def synchronize():
                pass

            @staticmethod
            def reset_peak_memory_stats():
                pass

            @staticmethod
            def max_memory_allocated():
                return 1024**2

        class FakeLLM:

            def __init__(self):
                self.done = False
                self.next_seq_id = 0

            def add_request(self, prompt, params):
                seq_id = self.next_seq_id
                self.next_seq_id += 1
                return seq_id

            def is_finished(self):
                return self.done

            def step_with_info(self):
                self.done = True
                info = SimpleNamespace(
                    is_prefill=True,
                    num_scheduled_tokens=4,
                    generated_seq_ids=[0, 1],
                    finished_seq_ids=[0, 1],
                )
                return [], info

        torch = SimpleNamespace(cuda=FakeCuda())
        params = [SimpleNamespace(max_tokens=1), SimpleNamespace(max_tokens=1)]
        metrics = run_benchmark(FakeLLM(), [[1, 2], [3, 4]], params, torch)

        self.assertEqual(metrics["total_input_tokens"], 4)
        self.assertEqual(metrics["total_output_tokens"], 2)
        self.assertIsNone(metrics["decode_throughput"])
        self.assertIsNone(metrics["tpot_ms"]["p50"])
        self.assertEqual(metrics["peak_memory_mb"], 1.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

from benchmarks.online import make_trace, materialize_trace, run_online, summarize_records, trace_hash
from benchmarks.bench_online import parse_args


class OnlineTest(unittest.TestCase):

    def test_trace_reproducibility_and_poisson_shapes(self):
        trace = make_trace("mixed", 32, 10, seed=7, vocab_size=128)
        again = make_trace("mixed", 32, 10, seed=7, vocab_size=128)
        self.assertEqual(trace_hash(trace), trace_hash(again))
        rows, digest = materialize_trace(trace)
        self.assertEqual(digest, materialize_trace(again)[1])
        self.assertEqual({r["input_len"] for r in rows}, {256, 8192})
        self.assertNotEqual(trace_hash(trace), trace_hash(make_trace("mixed", 32, 11, seed=7)))

    def test_late_submission_uses_planned_arrival(self):
        now = [10.0]
        class FakeLLM:
            active = []
            count = 0
            def add_request(self, prompt, params):
                identity = self.count
                self.count += 1
                self.active.append(identity)
                return identity
            def is_finished(self):
                return not self.active
            def step_with_info(self):
                identities, self.active = self.active, []
                first = {identity: now[0] for identity in identities}
                now[0] += 0.2
                return [(i, [42]) for i in identities], SimpleNamespace(
                    first_scheduled=first, generated_seq_ids=identities, finished_seq_ids=identities,
                    phase="prefill", prefill_tokens=len(identities), decode_tokens=0,
                    execution_path="prefill_eager", p2_tokens=0, prefill_budget=0,
                    budget_reason="legacy", kv_used_blocks=0, scheduler_stats={})
        rows = [dict(request_id=i, arrival_s=t, prompt=[1], output_len=1) for i, t in enumerate((0, 0.1))]
        result = run_online(FakeLLM(), rows, lambda n: n, clock=lambda: now[0], sleeper=lambda t: None)
        self.assertAlmostEqual(result["requests"][1]["ttft_ms"], 300)
        self.assertAlmostEqual(result["requests"][1]["submitted_s"], 0.2)
        self.assertEqual(result["metrics"]["slo_requests"], 2)
        self.assertIsNone(result["metrics"]["itl_ms"]["p99"])

    def test_every_gap_not_average_tpot_determines_goodput(self):
        rows = [dict(request_id=0, arrival_s=0, submitted_s=0, first_scheduled_s=0,
                     token_times_s=[0.1, 0.11, 0.2], finished_s=0.2)]
        metrics = summarize_records(rows, 0.2, 2000, 50)
        self.assertEqual(metrics["slo_requests"], 0)
        self.assertEqual(metrics["total_output_tokens"], 3)
        self.assertAlmostEqual(rows[0]["itl_max_ms"], 90)

    def test_bad_trace_and_cli_rejected(self):
        trace = make_trace("steady", 2, 10)
        trace["requests"][1]["arrival_s"] = -1
        with self.assertRaises(ValueError):
            materialize_trace(trace)
        args = parse_args(["--output-json", "unused.json"])
        self.assertEqual(args.scheduler_policy, "legacy")
        self.assertEqual(args.kv_cache_block_limit, 512)


if __name__ == "__main__":
    unittest.main()

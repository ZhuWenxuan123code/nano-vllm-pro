"""Scheduling and feedback tests deliberately run without a GPU/model."""

import unittest
import pickle
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduled_batch import PrefillBudgetController, ScheduledBatch, tp_shm_capacity
from nanovllm.sampling_params import SamplingParams


def config(policy="mixed", blocks=12, seqs=3, tokens=128):
    return SimpleNamespace(scheduler_policy=policy, num_kvcache_blocks=blocks,
        kvcache_block_size=256, max_num_seqs=seqs, max_num_batched_tokens=tokens,
        max_model_len=1024, eos=2, prefill_token_budget=64, prefill_budget_min=64,
        prefill_budget_max=1024, itl_slo_ms=50, prefill_max_wait_ms=1000)


class SchedulerTest(unittest.TestCase):

    def add(self, scheduler, length, outputs=4):
        seq = Sequence([20 + i % 100 for i in range(length)], SamplingParams(ignore_eos=True, max_tokens=outputs))
        scheduler.add(seq)
        return seq

    def drain(self, scheduler):
        events = {}
        for _ in range(500):
            if scheduler.is_finished():
                break
            batch = scheduler.schedule_batch()
            self.assertLessEqual(sum(batch.query_lens), scheduler.max_num_batched_tokens)
            self.assertLessEqual(len(batch.seqs), scheduler.max_num_seqs)
            self.assertEqual(len({s.seq_id for s in batch.seqs}), len(batch.seqs))
            generated = scheduler.postprocess_batch(batch, [42] * len(batch.seqs))
            for identity in generated:
                events[identity] = events.get(identity, 0) + 1
        else:
            self.fail("Scheduler failed to make finite progress")
        self.assertEqual(len(scheduler.block_manager.free_block_ids), len(scheduler.block_manager.blocks))
        self.assertFalse(scheduler.block_manager.used_block_ids)
        return events

    def test_chunk_does_not_emit_and_decode_is_first_in_mixed(self):
        scheduler = Scheduler(config())
        first = self.add(scheduler, 20)
        batch = scheduler.schedule_batch()
        scheduler.postprocess_batch(batch, [42])
        second = self.add(scheduler, 257)
        batch = scheduler.schedule_batch()
        self.assertEqual(batch.phase, "mixed")
        self.assertEqual(batch.seqs, [first, second])
        self.assertEqual(batch.prefill, (False, True))
        self.assertEqual(batch.query_lens, (1, 64))
        self.assertEqual(scheduler.postprocess_batch(batch, [43, 44]), [first.seq_id])
        events = self.drain(scheduler)
        self.assertEqual(first.num_completion_tokens, 4)
        self.assertEqual(events[second.seq_id], 4)

    def test_multiple_chunks_and_prefix_release_reuse(self):
        scheduler = Scheduler(config(tokens=1024))
        for cycle in range(2):
            seqs = [self.add(scheduler, n) for n in (255, 256, 257)]
            events = self.drain(scheduler)
            self.assertEqual([events[s.seq_id] for s in seqs], [4] * 3)
        self.assertGreater(scheduler.stats["prefix_hit_tokens"], 0)

    def test_pressure_preemption_and_recompute(self):
        scheduler = Scheduler(config(blocks=4, seqs=3))
        seqs = [self.add(scheduler, 255, 8) for _ in range(5)]
        events = self.drain(scheduler)
        self.assertEqual([events[s.seq_id] for s in seqs], [8] * 5)

    def test_preempt_preserves_completion_history(self):
        scheduler = Scheduler(config())
        seq = self.add(scheduler, 255, 4)
        while not seq.num_completion_tokens:
            batch = scheduler.schedule_batch()
            scheduler.postprocess_batch(batch, [42])
        scheduler.running.remove(seq)
        scheduler.preempt(seq)
        events = self.drain(scheduler)
        self.assertEqual(events[seq.seq_id], 3)
        self.assertEqual(seq.num_completion_tokens, 4)
        self.assertGreater(scheduler.stats["recompute_tokens"], 0)

    def test_unschedulable_request_is_rejected(self):
        scheduler = Scheduler(config(blocks=1))
        with self.assertRaisesRegex(ValueError, "fit"):
            self.add(scheduler, 257)
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            self.add(scheduler, 1024)

    def test_eos_releases_blocks(self):
        scheduler = Scheduler(config())
        seq = Sequence([20], SamplingParams(max_tokens=4))
        scheduler.add(seq)
        batch = scheduler.schedule_batch()
        self.assertEqual(scheduler.postprocess_batch(batch, [2]), [seq.seq_id])
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 12)

    def test_adaptive_overdue_resident_gets_service(self):
        scheduler = Scheduler(config("adaptive"))
        self.add(scheduler, 20)
        scheduler.postprocess_batch(scheduler.schedule_batch(), [42])
        waiting = self.add(scheduler, 257)
        with patch("nanovllm.engine.scheduler.perf_counter", return_value=waiting.arrived_at + 2):
            batch = scheduler.schedule_batch()
        self.assertIn(waiting, batch.seqs)
        self.assertGreaterEqual(batch.prefill_tokens, 64)
        self.assertEqual(scheduler.stats["protected_prefill_steps"], 1)

    def test_wrong_sample_row_count_rejected(self):
        scheduler = Scheduler(config())
        self.add(scheduler, 20)
        with self.assertRaises(ValueError):
            scheduler.postprocess_batch(scheduler.schedule_batch(), [])


class ControllerTest(unittest.TestCase):

    def test_aimd_bounds_and_hysteresis(self):
        cfg = config("adaptive")
        cfg.prefill_token_budget = 256
        control = PrefillBudgetController(cfg)
        for _ in range(8):
            control.observe(20)
            control.choose(True, 0.4)
        self.assertEqual(control.budget, 320)
        control.observe(51)
        self.assertEqual(control.budget, 128)
        for _ in range(10):
            control.observe(100)
        self.assertEqual(control.budget, 64)

    def test_high_pressure_and_no_queue_do_not_increase(self):
        control = PrefillBudgetController(config("adaptive"))
        for _ in range(16):
            control.observe(10)
            control.choose(True, 0.95)
        self.assertEqual(control.budget, 64)
        self.assertEqual(control.choose(False, 0.1)[0], 0)

    def test_tp_capacity_accounts_for_full_cached_history(self):
        seqs = [Sequence([100000] * 7937) for _ in range(63)]
        batch = ScheduledBatch(seqs, (True,) * 63, (1,) * 63)
        size = len(pickle.dumps(["run_batch", batch])) + 4
        self.assertGreater(size, 2**20)
        self.assertLess(size, tp_shm_capacity(64, 8448, 256, "mixed"))
        self.assertEqual(tp_shm_capacity(64, 8448, 256), 2**20)


if __name__ == "__main__":
    unittest.main()

"""Teacher-forced model and cache lifecycle checks, with bounded storage."""

import atexit
import os
import unittest
from unittest.mock import patch

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context


class P4SmallCacheRunner(ModelRunner):

    def allocate_kv_cache(self):
        self.allocate_cache_storage(12)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NANOVLLM_TEST_MODEL"),
                     "CUDA and NANOVLLM_TEST_MODEL required")
class P4IntegrationTest(unittest.TestCase):

    def setUp(self):
        limit = torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit = 64
        self.addCleanup(setattr, torch._dynamo.config, "recompile_limit", limit)

    def assert_model_close(self, left, right):
        # Across 28 BF16 layers, one-ULP Attention differences accumulate.
        # Bound both the worst case and RMS error instead of near-zero relative error.
        delta = left.float() - right.float()
        self.assertLess(delta.square().mean().sqrt().item(), 0.08)
        torch.testing.assert_close(left, right, atol=0.4, rtol=0.03)

    def assert_cache_close(self, left, right):
        # Qwen K-norm has outlier channels (layer 0 max weight 96.5).
        # Report/bound error relative to each KV vector's amplitude, rather
        # than applying a constant absolute threshold to all heads/layers.
        amplitude = left.float().abs().amax(-1, keepdim=True).clamp_min(1.)
        normalized = (left.float() - right.float()) / amplitude
        self.assertLess(normalized.abs().max().item(), 0.06)
        self.assertLess(normalized.square().mean().sqrt().item(), 0.005)

    def trajectory(self, backend, quantized=False, eager=True, fused=False, buffered=False):
        torch._dynamo.reset()
        config = Config(model=os.environ["NANOVLLM_TEST_MODEL"], attention_backend=backend,
                        kv_cache_dtype="int8" if quantized else "auto", max_num_seqs=3,
                        max_num_batched_tokens=128, max_model_len=512, enforce_eager=eager,
                        fuse_decode_qk_rope_cache=fused, execution_mode="buffered" if buffered else "original")
        runner = P4SmallCacheRunner(config, 0, [])
        scheduler = Scheduler(config)
        records = []
        try:
            # Repeat a 257-token prompt after release to exercise prefix hits and
            # reuse stale physical pages/scales. Different tails cross page edges.
            for cycle in range(2):
                preempted = False
                for length in (255, 256, 257):
                    scheduler.add(Sequence([20 + i % 100 for i in range(length)],
                        SamplingParams(ignore_eos=True, max_tokens=4)))
                while not scheduler.is_finished():
                    seqs, prefill = scheduler.schedule()
                    with torch.inference_mode():
                        inputs, positions = runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                        logits = runner.run_model(inputs, positions, prefill)
                        saved = []
                        for seq in seqs:
                            for offset in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens):
                                block, slot = seq.block_table[offset // 256], offset % 256
                                data = runner.kv_cache[:, :, block, slot]
                                if quantized:
                                    ks = runner.k_scales[:, block, slot]
                                    vs = runner.v_scales[:, block, slot]
                                    key = (data[0].float().reshape(*ks.shape, 32) * ks[..., None]).flatten(-2)
                                    value = data[1].float() * vs[..., None]
                                    data = torch.stack((key, value)).to(config.hf_config.dtype)
                                saved.append(data.cpu())
                        records.append((prefill, logits.cpu(), torch.stack(saved),
                                        [(s.num_tokens, s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs]))
                    reset_context()
                    scheduler.postprocess(seqs, [40 + s.num_completion_tokens for s in seqs], prefill)
                    if not preempted and scheduler.running:
                        # Force release/recompute while preserving teacher-forced
                        # token history, then compare the resulting trajectory.
                        victim = scheduler.running.pop()
                        scheduler.preempt(victim)
                        preempted = True
                self.assertEqual(len(scheduler.block_manager.free_block_ids), 12)
        finally:
            reset_context()
            runner.exit()
        return records

    def test_high_precision_logits_and_cache(self):
        baseline = self.trajectory("flash")
        for eager in (True, False):
            candidate = self.trajectory("triton", eager=eager)
            self.assertEqual(len(baseline), len(candidate))
            for left, right in zip(baseline, candidate):
                self.assert_model_close(left[1], right[1])
                self.assertEqual(left[3], right[3])
                self.assert_cache_close(left[2], right[2])

    def test_int8_graph_p2_buffered_logits_and_cache(self):
        baseline = self.trajectory("triton", quantized=True)
        for fused in (False, True):
            candidate = self.trajectory("triton", quantized=True, eager=False, fused=fused, buffered=True)
            self.assertEqual(len(baseline), len(candidate))
            for left, right in zip(baseline, candidate):
                self.assert_model_close(left[1], right[1])
                self.assertEqual(left[3], right[3])
                self.assert_cache_close(left[2], right[2])

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_TP") == "2" and torch.cuda.device_count() >= 2,
                         "NANOVLLM_TEST_TP=2 and two free GPUs required")
    def test_tp2_int8_generation(self):
        with patch("nanovllm.engine.llm_engine.ModelRunner", P4SmallCacheRunner):
            llm = LLM(os.environ["NANOVLLM_TEST_MODEL"], tensor_parallel_size=2,
                attention_backend="triton", kv_cache_dtype="int8", execution_mode="buffered",
                fuse_decode_qk_rope_cache=True, max_num_seqs=3, max_model_len=512,
                max_num_batched_tokens=128)
            try:
                ranks = llm.model_runner.call("cache_metadata")
                self.assertEqual([r["rank"] for r in ranks], [0, 1])
                for metadata in ranks:
                    self.assertEqual(metadata["attention_backend"], "triton")
                    self.assertEqual(metadata["format"], "int8-k32-vhead-v1")
                    self.assertEqual(metadata["blocks"], 12)
                    self.assertEqual(metadata["kv_heads"], 4)
                    self.assertEqual(metadata["k_quant_group_size"], 32)
                outputs = llm.generate([[20] * 255, [30] * 257, [40] * 10],
                    SamplingParams(ignore_eos=True, max_tokens=4), use_tqdm=False)
                self.assertEqual([len(o["token_ids"]) for o in outputs], [4, 4, 4])
                self.assertEqual(len(llm.scheduler.block_manager.free_block_ids), 12)
                self.assertEqual(llm.model_runner.kv_cache.dtype, torch.int8)
            finally:
                atexit.unregister(llm.exit)
                llm.exit()


class P4ConfigTest(unittest.TestCase):

    def test_incompatible_flags_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            Config(model="missing", attention_backend="flash", kv_cache_dtype="int8")
        for kwargs in ({"attention_backend": "invalid"}, {"kv_cache_dtype": "fp8"}):
            with self.assertRaises(ValueError):
                Config(model="missing", **kwargs)


if __name__ == "__main__":
    unittest.main()

"""Real-model P3 checks with a small fixed cache, not a performance benchmark."""

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


class SmallCacheRunner(ModelRunner):

    def allocate_kv_cache(self):
        config, hf = self.config, self.config.hf_config
        config.num_kvcache_blocks = 16
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        self.kv_cache = torch.zeros(
            2, hf.num_hidden_layers, config.num_kvcache_blocks, self.block_size,
            hf.num_key_value_heads // self.world_size, head_dim,
            dtype=hf.dtype, device=f"cuda:{self.rank}",
        )
        layer = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache, module.v_cache = self.kv_cache[0, layer], self.kv_cache[1, layer]
                layer += 1


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NANOVLLM_TEST_MODEL"),
                     "CUDA and NANOVLLM_TEST_MODEL required")
class RunnerIntegrationTest(unittest.TestCase):

    def trajectory(self, mode, eager, fused):
        # Give both modes the same compilation history. Otherwise earlier shape
        # sweeps can exhaust Dynamo's limit and compare compiled vs eager math.
        torch._dynamo.reset()
        config = Config(model=os.environ["NANOVLLM_TEST_MODEL"], execution_mode=mode,
                        max_num_seqs=3, max_num_batched_tokens=128, max_model_len=512,
                        enforce_eager=eager, fuse_decode_qk_rope_cache=fused)
        runner = SmallCacheRunner(config, 0, [])
        records = []
        try:
            scheduler = Scheduler(config)
            for length, outputs in ((255, 4), (256, 3), (257, 6)):
                seq = Sequence([20 + i % 100 for i in range(length)],
                               SamplingParams(ignore_eos=True, max_tokens=outputs))
                scheduler.add(seq)
            while not scheduler.is_finished():
                seqs, prefill = scheduler.schedule()
                with torch.inference_mode():
                    inputs, positions = (runner.prepare_prefill(seqs) if prefill
                                         else runner.prepare_decode(seqs))
                    logits = runner.run_model(inputs, positions, prefill).cpu()
                    cache = []
                    for seq in seqs:
                        end = seq.num_cached_tokens + seq.num_scheduled_tokens
                        for offset in range(seq.num_cached_tokens, end):
                            block, slot = seq.block_table[offset // 256], offset % 256
                            cache.append(runner.kv_cache[:, :, block, slot].cpu())
                    records.append((prefill, logits, torch.stack(cache)))
                reset_context()
                # Teacher-forced continuation isolates runner correctness from RNG.
                tokens = [30 + seq.num_completion_tokens for seq in seqs]
                scheduler.postprocess(seqs, tokens, prefill)
            self.assertEqual(len(scheduler.block_manager.free_block_ids), 16)
        finally:
            reset_context()
            runner.exit()
        return records

    def test_logits_cache_and_chunked_prefill_match(self):
        for eager in (True, False):
            for fused in (False, True):
                with self.subTest(eager=eager, fused=fused):
                    original = self.trajectory("original", eager, fused)
                    buffered = self.trajectory("buffered", eager, fused)
                    self.assertEqual(len(original), len(buffered))
                    for step, (left, right) in enumerate(zip(original, buffered)):
                        with self.subTest(step=step):
                            self.assertEqual(left[0], right[0])
                            torch.testing.assert_close(left[2], right[2], atol=0.02, rtol=0.02)
                            torch.testing.assert_close(left[1], right[1], atol=0.1, rtol=0.02)

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_TP") == "2" and torch.cuda.device_count() >= 2,
                         "NANOVLLM_TEST_TP=2 and two GPUs required")
    def test_tp2_buffered_generation(self):
        with patch("nanovllm.engine.llm_engine.ModelRunner", SmallCacheRunner):
            llm = LLM(os.environ["NANOVLLM_TEST_MODEL"], execution_mode="buffered",
                      tensor_parallel_size=2, max_num_seqs=3, max_model_len=512,
                      max_num_batched_tokens=128)
            try:
                outputs = llm.generate([[20] * 255, [30] * 257, [40] * 10],
                                       SamplingParams(ignore_eos=True, max_tokens=6), use_tqdm=False)
                self.assertEqual([len(output["token_ids"]) for output in outputs], [6, 6, 6])
                self.assertEqual(len(llm.scheduler.block_manager.free_block_ids), 16)
            finally:
                # The public engine registers exit at construction; remove that callback.
                import atexit
                atexit.unregister(llm.exit)
                llm.exit()


if __name__ == "__main__":
    unittest.main()

"""Optional real-model comparison: set NANOVLLM_TEST_MODEL to the local model path."""

import os
import unittest

import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.qwen3 import Qwen3Attention
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NANOVLLM_TEST_MODEL"),
                     "CUDA and NANOVLLM_TEST_MODEL required")
class DecodeQKVIntegrationTest(unittest.TestCase):

    def test_real_model_logits_and_cache_match(self):
        config = Config(
            model=os.environ["NANOVLLM_TEST_MODEL"],
            max_num_seqs=4,
            max_num_batched_tokens=128,
            max_model_len=128,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
        )
        runner = ModelRunner(config, rank=0, event=[])
        try:
            runner.model.eval()
            input_ids = torch.tensor([1], dtype=torch.int64, device="cuda")
            positions = torch.tensor([0], dtype=torch.int64, device="cuda")
            slots = torch.tensor([0], dtype=torch.int32, device="cuda")
            context_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
            block_tables = torch.tensor([[0]], dtype=torch.int32, device="cuda")
            set_context(False, slot_mapping=slots, context_lens=context_lens,
                        block_tables=block_tables)
            with torch.inference_mode():
                original_logits = runner.model.compute_logits(runner.model(input_ids, positions))
                original_cache = runner.kv_cache[:, :, 0, 0].clone()
                for module in runner.model.modules():
                    if isinstance(module, Qwen3Attention):
                        module.fuse_decode_qk_rope_cache = True
                fused_logits = runner.model.compute_logits(runner.model(input_ids, positions))
                fused_cache = runner.kv_cache[:, :, 0, 0]
            torch.testing.assert_close(fused_cache, original_cache, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(fused_logits, original_logits, atol=0.1, rtol=2e-2)
            self.assertEqual(fused_logits.argmax(dim=-1).item(),
                             original_logits.argmax(dim=-1).item())
        finally:
            reset_context()
            runner.exit()


if __name__ == "__main__":
    unittest.main()

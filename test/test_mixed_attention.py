"""Dispatcher isolation: identical projected Q/K/V, pages and chunk boundaries."""

import unittest

import torch

from nanovllm.layers.attention import Attention
from nanovllm.layers.kv_quantization import store_int8_kvcache
from nanovllm.layers.paged_attention import AttentionWorkspace
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MixedAttentionTest(unittest.TestCase):

    def test_mixed_equals_separate_without_projection_rounding(self):
        for dtype in (torch.float16, torch.bfloat16):
            for backend, quantized in (("flash", False), ("triton", False), ("triton", True)):
                with self.subTest(dtype=dtype, backend=backend, quantized=quantized):
                    torch.manual_seed(41)
                    layer = Attention(4, 64, 64 ** -0.5, 2).cuda()
                    layer.backend = backend
                    shape = (4, 256, 2, 64)
                    key, value = [torch.randn(shape, dtype=dtype, device="cuda") for _ in range(2)]
                    if quantized:
                        layer.k_cache = torch.empty(shape, dtype=torch.int8, device="cuda")
                        layer.v_cache = torch.empty_like(layer.k_cache)
                        layer.k_scale = torch.empty(4, 256, 2, 2, device="cuda", dtype=torch.float32)
                        layer.v_scale = torch.empty(4, 256, 2, device="cuda", dtype=torch.float32)
                        store_int8_kvcache(key.flatten(0, 1), value.flatten(0, 1), layer.k_cache,
                            layer.v_cache, layer.k_scale, layer.v_scale, torch.arange(1024, device="cuda", dtype=torch.int32))
                    else:
                        layer.k_cache, layer.v_cache = key, value
                    layer.workspace = AttentionWorkspace(3, 4, 64)
                    # Two Decode rows, followed by queries of lengths 3 and 5.
                    q = torch.randn(10, 4, 64, dtype=dtype, device="cuda")
                    k, v = [torch.randn(10, 2, 64, dtype=dtype, device="cuda") for _ in range(2)]
                    tables = torch.tensor([[0], [1], [2], [3]], device="cuda", dtype=torch.int32)
                    lengths = torch.tensor([20, 30], device="cuda", dtype=torch.int32)
                    slots = torch.tensor([19, 256 + 29, *range(512 + 10, 512 + 13),
                                          *range(768 + 15, 768 + 20)], device="cuda", dtype=torch.int32)
                    cuq = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
                    cuk = torch.tensor([0, 13, 33], device="cuda", dtype=torch.int32)
                    set_context(False, slot_mapping=slots[:2], context_lens=lengths, block_tables=tables[:2])
                    with torch.inference_mode():
                        decode = layer(q[:2], k[:2], v[:2]).reshape(2, 4, 64)
                        set_context(True, cuq, cuk, 5, 20, slots[2:], None, tables[2:])
                        prefill = layer(q[2:], k[2:], v[2:])
                        before = layer.k_cache.clone()
                        set_context(True, slot_mapping=slots, context_lens=lengths, block_tables=tables,
                                    phase="mixed", num_decode=2, prefill_cu_q=cuq, prefill_cu_k=cuk,
                                    prefill_max_q=5, prefill_max_k=20, prefill_uses_cache=True)
                        mixed = layer(q, k, v)
                    torch.testing.assert_close(mixed, torch.cat((decode, prefill)), atol=0, rtol=0)
                    torch.testing.assert_close(layer.k_cache, before, atol=0, rtol=0)
                    reset_context()


if __name__ == "__main__":
    unittest.main()

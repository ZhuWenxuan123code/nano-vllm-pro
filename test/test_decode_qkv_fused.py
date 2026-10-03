import unittest

import torch

from nanovllm.layers.decode_qkv_fused import decode_qkv_fused
from nanovllm.layers.rotary_embedding import apply_rotary_emb, RotaryEmbedding


def reference(qkv, q_weight, k_weight, rope, positions, slots, k_cache, v_cache):
    batch, kv_heads, head_dim = qkv.shape[0], k_cache.shape[-2], k_cache.shape[-1]
    q_heads = qkv.shape[1] // head_dim - 2 * kv_heads
    q, k, v = qkv.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1)
    q, k, v = (t.view(batch, -1, head_dim) for t in (q, k, v))

    def rms(x, weight):
        values = x.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        return (values * torch.rsqrt(variance + 1e-6)).to(x.dtype).mul_(weight)

    cos, sin = rope[positions].chunk(2, dim=-1)
    q = apply_rotary_emb(rms(q, q_weight), cos, sin)
    k = apply_rotary_emb(rms(k, k_weight), cos, sin)
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            k_cache.view(-1, kv_heads, head_dim)[slot].copy_(k[row])
            v_cache.view(-1, kv_heads, head_dim)[slot].copy_(v[row])
        else:
            q[row].zero_()
    return q


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeQKVFusedTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(9)

    def _case(self, dtype, batch, q_heads, kv_heads, head_dim, invalid=False,
              strided_columns=False):
        width = (q_heads + 2 * kv_heads) * head_dim
        if strided_columns:
            qkv = torch.randn(batch, width, 2, device="cuda", dtype=dtype)[:, :, 0]
            self.assertEqual(qkv.stride(1), 2)
        else:
            qkv = torch.randn(batch, width + 16, device="cuda", dtype=dtype)[:, :width]
        if batch > 1:
            self.assertFalse(qkv.is_contiguous())
        q_before = qkv.clone()
        q_weight = torch.rand(head_dim, device="cuda", dtype=dtype) + 0.5
        k_weight = torch.rand(head_dim, device="cuda", dtype=dtype) + 0.5
        rope = RotaryEmbedding(head_dim, head_dim, 1024, 1000000).cuda().cos_sin_cache
        positions = torch.arange(batch, device="cuda", dtype=torch.int64) + 5
        slots = torch.arange(batch, device="cuda", dtype=torch.int32) * 2
        if invalid:
            slots[-1] = -1
        k_cache = torch.full((2, 128, kv_heads, head_dim), -7, device="cuda", dtype=dtype)
        v_cache = torch.full_like(k_cache, -7)
        expected_k, expected_v = k_cache.clone(), v_cache.clone()
        expected_q = reference(qkv, q_weight, k_weight, rope, positions, slots, expected_k, expected_v)
        actual_q = decode_qkv_fused(qkv, q_weight, k_weight, rope, positions, slots, k_cache, v_cache)
        tol = 1e-2 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(actual_q, expected_q, atol=tol, rtol=tol)
        torch.testing.assert_close(k_cache, expected_k, atol=tol, rtol=tol)
        torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)
        torch.testing.assert_close(qkv, q_before, atol=0, rtol=0)
        if invalid:
            self.assertEqual(torch.count_nonzero(actual_q[-1]).item(), 0)

    def test_dtypes_head_dims_and_gqa(self):
        for dtype in (torch.float16, torch.bfloat16):
            for q_heads, kv_heads, head_dim in ((4, 4, 64), (8, 4, 128), (16, 4, 256)):
                with self.subTest(dtype=dtype, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim):
                    self._case(dtype, 3, q_heads, kv_heads, head_dim, invalid=True)

    def test_decode_batch_64(self):
        self._case(torch.bfloat16, 64, 16, 8, 128)

    def test_strided_qkv_columns(self):
        self._case(torch.float16, 4, 8, 4, 128, strided_columns=True)

    def test_cuda_graph_replay_with_padding(self):
        batch, q_heads, kv_heads, head_dim = 8, 8, 4, 128
        width = (q_heads + 2 * kv_heads) * head_dim
        qkv = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
        q_weight = torch.ones(head_dim, device="cuda", dtype=qkv.dtype)
        k_weight = torch.ones_like(q_weight)
        rope = RotaryEmbedding(head_dim, head_dim, 1024, 1000000).cuda().cos_sin_cache
        positions = torch.arange(batch, device="cuda", dtype=torch.int64)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        slots[-2:] = -1
        k_cache = torch.full((1, 128, kv_heads, head_dim), -7, device="cuda", dtype=qkv.dtype)
        v_cache = torch.full_like(k_cache, -7)
        decode_qkv_fused(qkv, q_weight, k_weight, rope, positions, slots, k_cache, v_cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = decode_qkv_fused(qkv, q_weight, k_weight, rope, positions, slots, k_cache, v_cache)
        for _ in range(2):
            qkv.normal_()
            k_cache.fill_(-7)
            v_cache.fill_(-7)
            expected_k, expected_v = k_cache.clone(), v_cache.clone()
            expected_q = reference(qkv, q_weight, k_weight, rope, positions, slots, expected_k, expected_v)
            graph.replay()
            torch.testing.assert_close(output, expected_q, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(k_cache, expected_k, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

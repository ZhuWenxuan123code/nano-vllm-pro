import unittest

import torch
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func

from nanovllm.layers.kv_quantization import store_int8_kvcache
from nanovllm.layers.paged_attention import paged_decode_attention, paged_prefill_attention


def quantize_reference(x, group_size=None):
    values = x.float()
    if group_size:
        values = values.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    maximum = values.abs().amax(-1)
    # A tensor divisor avoids PyTorch's scalar reciprocal-multiply shortcut.
    scale = torch.where(maximum == 0, 1., maximum / torch.full_like(maximum, 127.))
    data = (values / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return data.reshape(x.shape), scale


def dequantize_key(data, scale):
    return (data.float().reshape(*scale.shape, 32) * scale[..., None]).flatten(-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KVQuantizationTest(unittest.TestCase):

    def test_store_scales_rounding_strides_and_invalid_slots(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128, 256):
                k = torch.randn(5, 2, dim * 2, device="cuda", dtype=dtype)[..., ::2]
                v = torch.randn_like(k)
                k[0].zero_()
                k[1, :, 0] = 1000
                # Exact ties with scale=1 exercise nearest-even, including negatives.
                k[2].zero_()
                k[2, :, :7] = torch.tensor([127, .5, 1.5, 2.5, -.5, -1.5, -2.5], device="cuda", dtype=dtype)
                before = (k.clone(), v.clone())
                kc = torch.full((2, 256, 2, dim), -19, device="cuda", dtype=torch.int8)
                vc = kc.clone()
                ks = torch.full(kc.shape[:-1] + (dim // 32,), -9., device="cuda", dtype=torch.float32)
                vs = torch.full(kc.shape[:-1], -9., device="cuda", dtype=torch.float32)
                slots = torch.tensor([255, 256, 257, 0, -1], device="cuda", dtype=torch.int32)
                expected = [kc.clone(), vc.clone(), ks.clone(), vs.clone()]
                ki, kscale = quantize_reference(k, 32)
                vi, vscale = quantize_reference(v)
                for row, slot in enumerate(slots.tolist()):
                    if slot >= 0:
                        expected[0].flatten(0, 1)[slot].copy_(ki[row])
                        expected[1].flatten(0, 1)[slot].copy_(vi[row])
                        expected[2].flatten(0, 1)[slot].copy_(kscale[row])
                        expected[3].flatten(0, 1)[slot].copy_(vscale[row])
                store_int8_kvcache(k, v, kc, vc, ks, vs, slots)
                for actual, ref in zip((kc, vc, ks, vs), expected):
                    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
                for actual, ref in zip((k, v), before):
                    torch.testing.assert_close(actual, ref, atol=0, rtol=0)
                rebuilt = dequantize_key(ki, kscale)
                bound = kscale.repeat_interleave(32, dim=-1) / 2 + 1e-4
                self.assertTrue(((rebuilt - k.float()).abs() <= bound).all())
                self.assertTrue((kscale[1, :, 1:] < kscale[1, :, :1]).all())

    def test_int8_attention_implementation_error(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128, 256):
                for group in (1, 2, 4, 8, 16):
                    with self.subTest(dtype=dtype, dim=dim, group=group):
                        q = torch.randn(6, group * 2, dim, device="cuda", dtype=dtype)
                        k = torch.randn(12, 256, 2, dim, device="cuda", dtype=dtype)
                        v = torch.randn_like(k)
                        ki, ks = quantize_reference(k, 32)
                        vi, vs = quantize_reference(v)
                        kd = dequantize_key(ki, ks).to(dtype)
                        vd = (vi.float() * vs[..., None]).to(dtype)
                        tables = torch.randperm(12, device="cuda").reshape(6, 2).to(torch.int32)
                        lens = torch.tensor([0, 1, 255, 256, 257, 512], device="cuda", dtype=torch.int32)
                        actual = paged_decode_attention(q, ki, vi, tables, lens, k_scale=ks, v_scale=vs, num_splits=4)
                        ref = flash_attn_with_kvcache(q[:, None], kd, vd, cache_seqlens=lens, block_table=tables)[:, 0]
                        torch.testing.assert_close(actual, ref, atol=0.003 if dtype == torch.float16 else 0.02, rtol=0.02)

    def test_cached_causal_prefill(self):
        for dtype in (torch.float16, torch.bfloat16):
            q = torch.randn(20, 4, 128 * 2, device="cuda", dtype=dtype)[..., ::2]
            k = torch.randn(4, 256, 2, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            ki, ks = quantize_reference(k, 32)
            vi, vs = quantize_reference(v)
            kd, vd = dequantize_key(ki, ks).to(dtype), (vi.float() * vs[..., None]).to(dtype)
            tables = torch.tensor([[3, 0], [1, 2]], device="cuda", dtype=torch.int32)
            cuq = torch.tensor([0, 3, 20], device="cuda", dtype=torch.int32)
            cuk = torch.tensor([0, 257, 520], device="cuda", dtype=torch.int32)
            for quantized in (False, True):
                actual = paged_prefill_attention(q, ki if quantized else kd, vi if quantized else vd,
                    tables, cuq, cuk, 17, k_scale=ks if quantized else None, v_scale=vs if quantized else None)
                ref = flash_attn_varlen_func(q, kd, vd, cuq, cuk, 17, 263, causal=True, block_table=tables)
                torch.testing.assert_close(actual, ref, atol=0.003 if dtype == torch.float16 else 0.02, rtol=0.02)

    def test_quantized_graph_replay(self):
        q = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn(3, 2, 128, device="cuda", dtype=q.dtype), torch.randn(3, 2, 128, device="cuda", dtype=q.dtype)
        kc = torch.zeros(3, 256, 2, 128, device="cuda", dtype=torch.int8)
        vc = kc.clone()
        ks = torch.ones(kc.shape[:-1] + (4,), device="cuda", dtype=torch.float32)
        vs = torch.ones(kc.shape[:-1], device="cuda", dtype=torch.float32)
        slots = torch.tensor([0, 256, -1], device="cuda", dtype=torch.int32)
        tables = torch.arange(3, device="cuda", dtype=torch.int32)[:, None]
        lens = torch.tensor([1, 1, 0], device="cuda", dtype=torch.int32)
        from nanovllm.layers.paged_attention import AttentionWorkspace
        workspace = AttentionWorkspace(3, 4, 128)
        def fn():
            store_int8_kvcache(k, v, kc, vc, ks, vs, slots)
            return paged_decode_attention(q, kc, vc, tables, lens, k_scale=ks, v_scale=vs, workspace=workspace)
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        for _ in range(2):
            k.normal_()
            v.normal_()
            graph.replay()
            torch.testing.assert_close(output, fn(), atol=0, rtol=0)

    def test_p2_fused_quantization_matches_rounded_key(self):
        from nanovllm.layers.decode_qkv_fused import decode_qkv_fused
        from nanovllm.layers.rotary_embedding import RotaryEmbedding
        for dtype in (torch.float16, torch.bfloat16):
            qkv = torch.randn(3, (4 + 2 * 2) * 128 + 8, device="cuda", dtype=dtype)[:, :-8]
            weights = torch.rand(128, device="cuda", dtype=dtype) + .5
            rope = RotaryEmbedding(128, 128, 512, 1000000).cuda().cos_sin_cache
            positions = torch.tensor([255, 256, 257], device="cuda", dtype=torch.int64)
            slots = torch.tensor([255, 256, -1], device="cuda", dtype=torch.int32)
            k = torch.zeros(2, 256, 2, 128, device="cuda", dtype=dtype)
            v = torch.zeros_like(k)
            qref = decode_qkv_fused(qkv, weights, weights, rope, positions, slots, k, v)
            ki = torch.zeros_like(k, dtype=torch.int8)
            vi = ki.clone()
            ks = torch.ones(k.shape[:-1] + (4,), device="cuda", dtype=torch.float32)
            vs = torch.ones(k.shape[:-1], device="cuda", dtype=torch.float32)
            actual = decode_qkv_fused(qkv, weights, weights, rope, positions, slots, ki, vi, k_scale=ks, v_scale=vs)
            kr, ksr = quantize_reference(k, 32)
            vr, vsr = quantize_reference(v)
            for left, right in ((actual, qref), (ki, kr), (vi, vr), (ks, ksr), (vs, vsr)):
                torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_old_scale_layout_is_rejected(self):
        k = torch.zeros(1, 2, 128, device="cuda", dtype=torch.bfloat16)
        cache = torch.zeros(1, 256, 2, 128, device="cuda", dtype=torch.int8)
        old = torch.ones(cache.shape[:-1], device="cuda", dtype=torch.float32)
        slots = torch.zeros(1, device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "D/32"):
            store_int8_kvcache(k, k, cache, cache.clone(), old, old, slots)


if __name__ == "__main__":
    unittest.main()

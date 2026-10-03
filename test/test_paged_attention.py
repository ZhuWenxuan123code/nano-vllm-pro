import unittest

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.paged_attention import AttentionWorkspace, paged_decode_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PagedAttentionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(41)

    def _case(self, dtype, dim, group, splits, lengths=(0, 1, 255, 256, 257)):
        batch, kv_heads = len(lengths), 2
        q = torch.randn(batch, kv_heads * group, dim * 2, device="cuda", dtype=dtype)[..., ::2]
        k = torch.randn(batch * 2, 256, kv_heads, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        # Physical pages deliberately differ from logical order.
        table = torch.randperm(batch * 2, device="cuda").reshape(batch, 2).to(torch.int32)
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        original = (q.clone(), k.clone(), v.clone())
        output = paged_decode_attention(q, k, v, table, lens, num_splits=splits)
        reference = flash_attn_with_kvcache(q[:, None], k, v, cache_seqlens=lens,
                                          block_table=table)[:, 0]
        tolerance = 3e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
        for row, length in enumerate(lengths):
            if not length:
                self.assertEqual(output[row].count_nonzero().item(), 0)
                continue
            pages = table[row].long()
            keys = k[pages].flatten(0, 1)[:length].repeat_interleave(group, dim=1).float()
            values = v[pages].flatten(0, 1)[:length].repeat_interleave(group, dim=1).float()
            scores = torch.einsum("hd,thd->ht", q[row].float(), keys) * dim ** -0.5
            expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values).to(dtype)
            torch.testing.assert_close(output[row], expected, atol=tolerance, rtol=tolerance)
        for actual, before in zip((q, k, v), original):
            torch.testing.assert_close(actual, before, atol=0, rtol=0)
        return q, k, v, table, lens

    def test_dimensions_gqa_and_boundaries(self):
        for dtype in (torch.float16, torch.bfloat16):
            for dim in (64, 128, 256):
                for group in (1, 2, 4, 8, 16):
                    with self.subTest(dtype=dtype, dim=dim, group=group):
                        self._case(dtype, dim, group, 1)

    def test_split_kv(self):
        for splits in (2, 4, 8, 16):
            self._case(torch.bfloat16, 128, 2, splits)

    def test_graph_replay_ragged_padding(self):
        q, k, v, table, lens = self._case(torch.bfloat16, 128, 2, 4)
        workspace = AttentionWorkspace(5, 4, 128)
        fn = lambda: paged_decode_attention(q, k, v, table, lens, workspace=workspace, num_splits=4)
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        for lengths in ((257, 0, 1, 256, 0), (0, 255, 0, 1, 257)):
            lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            q.normal_()
            graph.replay()
            reference = fn()
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()

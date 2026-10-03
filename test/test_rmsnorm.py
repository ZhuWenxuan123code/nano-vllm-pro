import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


def reference(x, weight, eps, residual=None):
    combined = x.float() if residual is None else x.float() + residual.float()
    new_residual = combined.to(x.dtype) if residual is not None else None
    var = combined.square().mean(dim=-1, keepdim=True)
    out = (combined * torch.rsqrt(var + eps)).to(x.dtype).mul_(weight)
    return out, new_residual


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton RMSNorm")
class RMSNormTritonTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        # Other model tests populate this shared compiler cache. Keep the
        # compiled reference from silently falling back to eager in a suite.
        torch._dynamo.reset()
        name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
        limit = getattr(torch._dynamo.config, name)
        setattr(torch._dynamo.config, name, 64)
        self.addCleanup(setattr, torch._dynamo.config, name, limit)

    def _check(self, x, residual):
        norm = RMSNorm(x.shape[-1], eps=1e-6).to(device=x.device, dtype=x.dtype)
        norm.set_backend("triton")
        with torch.no_grad():
            norm.weight.uniform_(0.75, 1.25)
            before_x = x.clone()
            before_residual = residual.clone() if residual is not None else None
            expected, expected_residual = reference(x, norm.weight, norm.eps, residual)
            actual = norm(x, residual)
            if residual is None:
                actual_out = actual
            else:
                actual_out, actual_residual = actual
            tolerance = 1e-2 if x.dtype == torch.float16 else 2e-2
            torch.testing.assert_close(actual_out, expected, atol=tolerance, rtol=tolerance)
            if residual is not None:
                torch.testing.assert_close(actual_residual, expected_residual, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(residual, before_residual, atol=0, rtol=0)
            torch.testing.assert_close(x, before_x, atol=0, rtol=0)
            self.assertEqual(actual_out.dtype, x.dtype)
            self.assertEqual(actual_out.shape, x.shape)

    def test_plain_and_add_varied_shapes_and_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows, hidden_size in ((1, 128), (64, 1024), (8192, 1024), (4, 1536)):
                with self.subTest(dtype=dtype, rows=rows, hidden_size=hidden_size):
                    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
                    residual = torch.randn_like(x)
                    self._check(x, None)
                    self._check(x, residual)

    def test_qk_noncontiguous_views(self):
        for dtype in (torch.float16, torch.bfloat16):
            for x in (
                torch.randn(16, 3 * 128, device="cuda", dtype=dtype)[:, :128].view(16, 1, 128),
                torch.randn(16, 8 * 128, device="cuda", dtype=dtype)[:, :4 * 128].view(16, 4, 128),
            ):
                with self.subTest(dtype=dtype, shape=x.shape):
                    self.assertFalse(x.is_contiguous())
                    self._check(x, None)
                    self._check(x, torch.randn_like(x))

    def test_other_supported_strides_and_single_row(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(8, 256, device="cuda", dtype=dtype)[:, ::2]
            residual = torch.randn(8, 256, device="cuda", dtype=dtype)[:, 1::2]
            self._check(x, residual)
            self._check(x[0], None)

    def test_fp32_weight_and_mixed_residual(self):
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(32, 128, device="cuda", dtype=torch.float32)
        norm = RMSNorm(128).to(device="cuda")
        norm.set_backend("triton")
        expected, expected_residual = reference(x, norm.weight, norm.eps, residual)
        actual, actual_residual = norm(x, residual)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_residual, expected_residual, atol=2e-2, rtol=2e-2)

    def test_matches_existing_compiled_backend(self):
        for dtype in (torch.float16, torch.bfloat16):
            projection = torch.randn(16, 8 * 128, device="cuda", dtype=dtype)
            q = projection[:, :4 * 128].view(16, 4, 128)
            plain = RMSNorm(128).to(device="cuda", dtype=dtype)
            compiled_q = plain(q)
            plain.set_backend("triton")
            triton_q = plain(q)
            tolerance = 1e-2 if dtype == torch.float16 else 2e-2
            torch.testing.assert_close(triton_q, compiled_q, atol=tolerance, rtol=tolerance)

            x = torch.randn(64, 1024, device="cuda", dtype=dtype)
            residual = torch.randn_like(x)
            fused = RMSNorm(1024).to(device="cuda", dtype=dtype)
            compiled_out, compiled_residual = fused(x, residual)
            fused.set_backend("triton")
            triton_out, triton_residual = fused(x, residual)
            torch.testing.assert_close(triton_out, compiled_out, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(triton_residual, compiled_residual, atol=tolerance, rtol=tolerance)

    def test_cuda_graph_capture_and_replay(self):
        x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        norm = RMSNorm(1024).to(device="cuda", dtype=x.dtype)
        norm.set_backend("triton")
        with torch.no_grad():
            norm(x, residual)  # Compile the Triton kernel before capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out, updated = norm(x, residual)
            for _ in range(2):
                x.uniform_(-1, 1)
                residual.uniform_(-1, 1)
                graph.replay()
                expected, expected_residual = reference(x, norm.weight, norm.eps, residual)
                torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(updated, expected_residual, atol=2e-2, rtol=2e-2)

    def test_backend_validation(self):
        norm = RMSNorm(128)
        self.assertEqual(norm.backend, "compiled")
        with self.assertRaises(ValueError):
            norm.set_backend("other")
        norm.set_backend("triton")
        with self.assertRaises(ValueError):
            norm(torch.empty(2, 128, device="cpu", dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()

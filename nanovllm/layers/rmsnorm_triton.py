"""Inference-only Triton implementations of RMSNorm and Add-RMSNorm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    num_heads: tl.constexpr, hidden_size: tl.constexpr, eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    x_offset = (row // num_heads) * x_stride0 + (row % num_heads) * x_stride1
    x = tl.load(x_ptr + x_offset + cols * x_stride2, cols < hidden_size, other=0).to(tl.float32)
    var = tl.sum(x * x, 0) / hidden_size
    norm = (x * tl.rsqrt(var + eps)).to(out_ptr.dtype.element_ty).to(tl.float32)
    weight = tl.load(weight_ptr + cols, cols < hidden_size, other=0).to(tl.float32)
    tl.store(out_ptr + row * hidden_size + cols, norm * weight, cols < hidden_size)


@triton.jit
def _add_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr,
    residual_stride0: tl.constexpr, residual_stride1: tl.constexpr,
    residual_stride2: tl.constexpr, num_heads: tl.constexpr,
    hidden_size: tl.constexpr, eps: tl.constexpr, block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    x_offset = (row // num_heads) * x_stride0 + (row % num_heads) * x_stride1
    residual_offset = (row // num_heads) * residual_stride0 + (row % num_heads) * residual_stride1
    x = tl.load(x_ptr + x_offset + cols * x_stride2, cols < hidden_size, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offset + cols * residual_stride2,
                       cols < hidden_size, other=0).to(tl.float32)
    combined = x + residual
    tl.store(new_residual_ptr + row * hidden_size + cols, combined, cols < hidden_size)
    var = tl.sum(combined * combined, 0) / hidden_size
    norm = (combined * tl.rsqrt(var + eps)).to(out_ptr.dtype.element_ty).to(tl.float32)
    weight = tl.load(weight_ptr + cols, cols < hidden_size, other=0).to(tl.float32)
    tl.store(out_ptr + row * hidden_size + cols, norm * weight, cols < hidden_size)


def _layout(x: torch.Tensor) -> tuple[int, int, int, int, int]:
    if x.ndim == 1:
        return 1, 1, 0, 0, x.stride(0)
    if x.ndim == 2:
        return x.shape[0], 1, x.stride(0), 0, x.stride(1)
    if x.ndim == 3:
        return x.shape[0] * x.shape[1], x.shape[1], *x.stride()
    raise ValueError("Triton RMSNorm supports tensors with 1, 2 or 3 dimensions")


def _check(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None = None) -> None:
    if not x.is_cuda or not weight.is_cuda or (residual is not None and not residual.is_cuda):
        raise ValueError("Triton RMSNorm requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype not in (x.dtype, torch.float32):
        raise TypeError("Triton RMSNorm requires FP16/BF16 input and matching or FP32 weight")
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or weight.stride(0) != 1:
        raise ValueError("weight must be a contiguous vector of hidden_size elements")
    if residual is not None and (residual.shape != x.shape or residual.dtype not in
                                 (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("residual must have the same shape and a floating-point dtype")


def _launch_config(hidden_size: int, add: bool) -> tuple[int, int]:
    block_size = triton.next_power_of_2(hidden_size)
    # Measured on RTX 3090: Q/K's H=128 benefits from one warp for plain RMSNorm;
    # the fused path and H=1024 are more balanced with two warps.
    num_warps = (2 if add else 1) if block_size <= 256 else (2 if block_size <= 2048 else 8)
    return block_size, num_warps


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _check(x, weight)
    rows, num_heads, stride0, stride1, stride2 = _layout(x)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows:
        block_size, num_warps = _launch_config(x.shape[-1], add=False)
        _rmsnorm_kernel[(rows,)](
            x, weight, out, stride0, stride1, stride2, num_heads,
            x.shape[-1], eps, block_size, num_warps=num_warps,
        )
    return out


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check(x, weight, residual)
    rows, num_heads, stride0, stride1, stride2 = _layout(x)
    _, _, res_stride0, res_stride1, res_stride2 = _layout(residual)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    new_residual = torch.empty_like(out)
    if rows:
        block_size, num_warps = _launch_config(x.shape[-1], add=True)
        _add_rmsnorm_kernel[(rows,)](
            x, residual, weight, out, new_residual,
            stride0, stride1, stride2, res_stride0, res_stride1, res_stride2,
            num_heads, x.shape[-1], eps, block_size, num_warps=num_warps,
        )
    return out, new_residual

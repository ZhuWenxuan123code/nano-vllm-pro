"""Decode-only Q/K RMSNorm + RoPE + paged KV cache write prototype."""

import torch
import triton
import triton.language as tl
from nanovllm.layers.kv_quantization import quantize_key, quantize_vector, validate_scales


@triton.jit
def _norm_rope_halves(
    x_ptr, weight_ptr, cos_sin_ptr, x_offset, x_col_stride, rope_offset,
    rope_col_stride, eps: tl.constexpr, half: tl.constexpr,
    round_before_weight: tl.constexpr = True,
):
    cols = tl.arange(0, half)
    x1 = tl.load(x_ptr + x_offset + cols * x_col_stride).to(tl.float32)
    x2 = tl.load(x_ptr + x_offset + (cols + half) * x_col_stride).to(tl.float32)
    variance = tl.sum(x1 * x1 + x2 * x2, 0) / (2 * half)
    inv_rms = tl.rsqrt(variance + eps)
    w1 = tl.load(weight_ptr + cols).to(tl.float32)
    w2 = tl.load(weight_ptr + cols + half).to(tl.float32)
    n1, n2 = x1 * inv_rms, x2 * inv_rms
    if round_before_weight:
        n1 = n1.to(x_ptr.dtype.element_ty).to(tl.float32)
        n2 = n2.to(x_ptr.dtype.element_ty).to(tl.float32)
    n1 = (n1 * w1).to(x_ptr.dtype.element_ty).to(tl.float32)
    n2 = (n2 * w2).to(x_ptr.dtype.element_ty).to(tl.float32)
    cos = tl.load(cos_sin_ptr + rope_offset + cols * rope_col_stride).to(tl.float32)
    sin = tl.load(cos_sin_ptr + rope_offset + (cols + half) * rope_col_stride).to(tl.float32)
    return n1 * cos - n2 * sin, n2 * cos + n1 * sin


@triton.jit
def _decode_qkv_kernel(
    qkv_ptr, q_weight_ptr, k_weight_ptr, cos_sin_ptr, positions_ptr,
    slots_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr, ks_ptr, vs_ptr,
    qkv_row_stride: tl.constexpr, qkv_col_stride: tl.constexpr,
    rope_row_stride: tl.constexpr, rope_col_stride: tl.constexpr,
    k_slot_stride: tl.constexpr, v_slot_stride: tl.constexpr,
    q_heads: tl.constexpr, kv_heads: tl.constexpr, head_dim: tl.constexpr,
    q_eps: tl.constexpr, k_eps: tl.constexpr,
    INT8: tl.constexpr,
    ROUND_BEFORE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots_ptr + row)
    half: tl.constexpr = head_dim // 2
    cols = tl.arange(0, half)

    if head < q_heads:
        q_offset = row * q_heads * head_dim + head * head_dim
        if slot >= 0:
            position = tl.load(positions_ptr + row)
            rope_offset = position * rope_row_stride
            x_offset = row * qkv_row_stride + head * head_dim * qkv_col_stride
            q1, q2 = _norm_rope_halves(
                qkv_ptr, q_weight_ptr, cos_sin_ptr, x_offset, qkv_col_stride,
                rope_offset, rope_col_stride, q_eps, half, ROUND_BEFORE_WEIGHT,
            )
            tl.store(q_out_ptr + q_offset + cols, q1)
            tl.store(q_out_ptr + q_offset + cols + half, q2)
        else:
            tl.store(q_out_ptr + q_offset + cols, tl.full((half,), 0, tl.float32))
            tl.store(q_out_ptr + q_offset + cols + half, tl.full((half,), 0, tl.float32))

    if head < kv_heads and slot >= 0:
        position = tl.load(positions_ptr + row)
        rope_offset = position * rope_row_stride
        k_offset = row * qkv_row_stride + (q_heads + head) * head_dim * qkv_col_stride
        k1, k2 = _norm_rope_halves(
            qkv_ptr, k_weight_ptr, cos_sin_ptr, k_offset, qkv_col_stride,
            rope_offset, rope_col_stride, k_eps, half, ROUND_BEFORE_WEIGHT,
        )
        k_cache_offset = slot * k_slot_stride + head * head_dim
        v_cache_offset = slot * v_slot_stride + head * head_dim
        v_offset = row * qkv_row_stride + (q_heads + kv_heads + head) * head_dim * qkv_col_stride
        v1 = tl.load(qkv_ptr + v_offset + cols * qkv_col_stride)
        v2 = tl.load(qkv_ptr + v_offset + (cols + half) * qkv_col_stride)
        if INT8:
            # RoPE must round to the input dtype before computing the scale.
            key = tl.cat(k1.to(qkv_ptr.dtype.element_ty), k2.to(qkv_ptr.dtype.element_ty))
            value = tl.cat(v1, v2)
            ki, ks = quantize_key(key, head_dim)
            vi, vs = quantize_vector(value)
            full = tl.arange(0, head_dim)
            tl.store(k_cache_ptr + k_cache_offset + full, ki)
            tl.store(v_cache_ptr + v_cache_offset + full, vi)
            groups = tl.arange(0, head_dim // 32)
            tl.store(ks_ptr + (slot * kv_heads + head) * (head_dim // 32) + groups, ks)
            tl.store(vs_ptr + slot * kv_heads + head, vs)
        else:
            tl.store(k_cache_ptr + k_cache_offset + cols, k1)
            tl.store(k_cache_ptr + k_cache_offset + cols + half, k2)
            tl.store(v_cache_ptr + v_cache_offset + cols, v1)
            tl.store(v_cache_ptr + v_cache_offset + cols + half, v2)


def decode_qkv_fused(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor, q_eps: float = 1e-6,
    k_eps: float = 1e-6, num_warps: int = 4, k_scale=None, v_scale=None,
    round_before_weight: bool = True,
) -> torch.Tensor:
    """Write K/V into the paged cache and return contiguous, rotated Q."""
    if qkv.ndim != 2 or qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("qkv must be a 2D FP16/BF16 tensor")
    if not all(t.is_cuda for t in (qkv, q_weight, k_weight, cos_sin_cache,
                                   positions, slot_mapping, k_cache, v_cache)):
        raise ValueError("all inputs must be CUDA tensors")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V cache must have matching [blocks, block_size, kv_heads, head_dim] shapes")
    kv_heads, head_dim = k_cache.shape[-2:]
    if head_dim not in (64, 128, 256) or kv_heads < 1:
        raise ValueError("head_dim must be 64, 128 or 256 and kv_heads must be positive")
    if qkv.shape[1] % head_dim or (qkv.shape[1] // head_dim - 2 * kv_heads) < 1:
        raise ValueError("qkv width is incompatible with the KV cache")
    q_heads = qkv.shape[1] // head_dim - 2 * kv_heads
    if q_heads % kv_heads or q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError("Q heads must be a multiple of KV heads; norm weights must match head_dim")
    if q_weight.dtype != qkv.dtype or k_weight.dtype != qkv.dtype:
        raise ValueError("norm weights must match qkv dtype")
    if positions.shape != (qkv.shape[0],) or slot_mapping.shape != positions.shape:
        raise ValueError("positions and slot_mapping must have one entry per token")
    if positions.dtype not in (torch.int32, torch.int64) or slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions and slot_mapping must contain integer indices")
    if positions.stride(0) != 1 or slot_mapping.stride(0) != 1:
        raise ValueError("positions and slot_mapping must be contiguous")
    if cos_sin_cache.ndim != 3 or cos_sin_cache.shape[1:] != (1, head_dim):
        raise ValueError("RoPE cache must have shape [max_position, 1, head_dim]")
    quantized = k_cache.dtype == torch.int8
    if cos_sin_cache.dtype != torch.float32 or v_cache.dtype != k_cache.dtype or (not quantized and k_cache.dtype != qkv.dtype):
        raise ValueError("RoPE must be FP32; K/V must match qkv dtype or both be INT8")
    if quantized:
        validate_scales(k_cache, k_scale, v_scale)
    if any(t.device != qkv.device for t in (q_weight, k_weight, cos_sin_cache, positions, slot_mapping, k_cache, v_cache)):
        raise ValueError("All tensors must use the same CUDA device")
    if q_weight.stride(0) != 1 or k_weight.stride(0) != 1 or cos_sin_cache.stride(-1) != 1:
        raise ValueError("norm weights and RoPE cache must be contiguous in the head dimension")
    if any(cache.stride(-4) != cache.shape[-3] * cache.stride(-3) or
           cache.stride(-3) != kv_heads * head_dim or cache.stride(-2) != head_dim or
           cache.stride(-1) != 1 for cache in (k_cache, v_cache)):
        raise ValueError("K/V cache must have contiguous physical token slots")

    q_out = torch.empty((qkv.shape[0], q_heads, head_dim), device=qkv.device, dtype=qkv.dtype)
    if qkv.shape[0]:
        _decode_qkv_kernel[(qkv.shape[0], max(q_heads, kv_heads))](
            qkv, q_weight, k_weight, cos_sin_cache, positions, slot_mapping,
            q_out, k_cache, v_cache, k_scale if quantized else k_cache,
            v_scale if quantized else v_cache, qkv.stride(0), qkv.stride(1),
            cos_sin_cache.stride(0), cos_sin_cache.stride(-1),
            k_cache.stride(-3), v_cache.stride(-3), q_heads, kv_heads, head_dim,
            q_eps, k_eps, quantized, round_before_weight, num_warps=num_warps,
        )
    return q_out

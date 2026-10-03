"""INT8 KV storage: K scales per 32 channels, V scales per token and head."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


K_QUANT_GROUP_SIZE = 32
INT8_CACHE_FORMAT = "int8-k32-vhead-v1"


@triton.jit
def quantize_vector(x):
    values = x.to(tl.float32)
    maximum = tl.max(tl.abs(values), 0)
    scale = tl.where(maximum == 0., 1., tl.div_rn(maximum, 127.))
    quantized = libdevice.nearbyint(tl.div_rn(values, scale))
    return tl.minimum(127., tl.maximum(-127., quantized)).to(tl.int8), scale


@triton.jit
def quantize_key(x, D: tl.constexpr):
    values = x.to(tl.float32).reshape((D // 32, 32))
    maximum = tl.max(tl.abs(values), 1)
    scales = tl.where(maximum == 0., 1., tl.div_rn(maximum, 127.))
    quantized = libdevice.nearbyint(tl.div_rn(values, scales[:, None]))
    return tl.minimum(127., tl.maximum(-127., quantized)).to(tl.int8).reshape((D,)), scales


@triton.jit
def _store_int8(K, V, KC, VC, KS, VS, Slots,
                k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr,
                v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr,
                HEADS: tl.constexpr, D: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + token)
    if slot >= 0:
        cols = tl.arange(0, D)
        k = tl.load(K + token * k0 + head * k1 + cols * k2)
        v = tl.load(V + token * v0 + head * v1 + cols * v2)
        ki, ks = quantize_key(k, D)
        vi, vs = quantize_vector(v)
        tl.store(KC + (slot * HEADS + head) * D + cols, ki)
        tl.store(VC + (slot * HEADS + head) * D + cols, vi)
        groups = tl.arange(0, D // 32)
        tl.store(KS + (slot * HEADS + head) * (D // 32) + groups, ks)
        tl.store(VS + slot * HEADS + head, vs)


def validate_scales(k_cache, k_scale, v_scale):
    shapes = (k_cache.shape[:-1] + (k_cache.shape[-1] // K_QUANT_GROUP_SIZE,), k_cache.shape[:-1])
    for scale, shape in zip((k_scale, v_scale), shapes):
        if (scale is None or scale.shape != shape or
                scale.dtype != torch.float32 or scale.device != k_cache.device or
                not scale.is_contiguous()):
            raise ValueError("INT8 requires FP32 K scales [blocks,page,heads,D/32] and V scales [blocks,page,heads]")


def store_int8_kvcache(k, v, k_cache, v_cache, k_scale, v_scale, slots):
    if k.ndim != 3 or v.shape != k.shape or k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
        raise ValueError("K/V must be matching [tokens, KV heads, head_dim] FP16/BF16 tensors")
    if (k_cache.ndim != 4 or v_cache.shape != k_cache.shape or
            k_cache.shape[-2:] != k.shape[-2:] or k.shape[-1] not in (64, 128, 256) or
            k_cache.dtype != torch.int8 or v_cache.dtype != torch.int8 or
            not k_cache.is_contiguous() or not v_cache.is_contiguous()):
        raise ValueError("Invalid INT8 cache shape, dtype or layout")
    if not all(t.is_cuda and t.device == k.device for t in (k, v, k_cache, v_cache, slots)):
        raise ValueError("All tensors must use the same CUDA device")
    if slots.shape != (k.shape[0],) or slots.dtype not in (torch.int32, torch.int64) or not slots.is_contiguous():
        raise ValueError("slot_mapping must be a contiguous integer vector")
    validate_scales(k_cache, k_scale, v_scale)
    if k.shape[0]:
        _store_int8[(k.shape[0], k.shape[1])](k, v, k_cache, v_cache, k_scale, v_scale,
                                             slots, *k.stride(), *v.stride(),
                                             k.shape[1], k.shape[2], num_warps=4,
                                             enable_fp_fusion=False)

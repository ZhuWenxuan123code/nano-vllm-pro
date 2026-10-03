"""Inference-only paged GQA attention, with optional INT8 cache dequantization."""

import torch
import triton
import triton.language as tl
from nanovllm.layers.kv_quantization import validate_scales


def choose_splits(batch, kv_heads, max_context):
    occupancy = triton.next_power_of_2(triton.cdiv(128, max(1, batch * kv_heads)))
    capacity = triton.next_power_of_2(max(1, triton.cdiv(max_context, 512)))
    return min(16, occupancy, capacity)


def choose_decode_config(batch, kv_heads, max_context):
    # Offline 3090 sweep (9 shapes, 30 configurations each): tile=32/warps=4
    # is within 1.9% geometric mean of each shape's best. Never autotune here.
    splits = 16 if batch * kv_heads <= 8 and max_context >= 2048 else choose_splits(batch, kv_heads, max_context)
    return 32, 4, splits


class AttentionWorkspace:

    def __init__(self, max_seqs, q_heads, head_dim, device="cuda"):
        self.partial = torch.empty((max_seqs, q_heads, 16, head_dim + 2),
                                   dtype=torch.float32, device=device)

    @property
    def nbytes(self):
        return self.partial.numel() * self.partial.element_size()


@triton.jit
def _decode_attention(
    Q, K, V, KS, VS, Tables, Lengths, Out, Partial,
    q0: tl.constexpr, q1: tl.constexpr, q2: tl.constexpr,
    table0: tl.constexpr, table1: tl.constexpr,
    p0: tl.constexpr, p1: tl.constexpr, p2: tl.constexpr,
    KV_HEADS: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr,
    INT8: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
):
    seq, kv_head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows, cols, keys = tl.arange(0, M), tl.arange(0, D), tl.arange(0, N)
    heads = kv_head * GROUP + rows
    q = tl.load(Q + seq * q0 + heads[:, None] * q1 + cols[None, :] * q2,
                rows[:, None] < GROUP, 0)
    length = tl.load(Lengths + seq)
    chunk = tl.cdiv(length, SPLITS * N) * N
    start, stop = split * chunk, tl.minimum((split + 1) * chunk, length)
    acc = tl.full((M, D), 0, tl.float32)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.full((M,), 0, tl.float32)
    for offset in range(start, stop, N):
        pos = offset + keys
        valid = pos < stop
        page = tl.load(Tables + seq * table0 + (pos // PAGE) * table1, valid, 0)
        slots = page * PAGE + pos % PAGE
        k = tl.load(K + (slots[None, :] * KV_HEADS + kv_head) * D + cols[:, None],
                    valid[None, :], 0)
        v = tl.load(V + (slots[:, None] * KV_HEADS + kv_head) * D + cols[None, :],
                    valid[:, None], 0)
        if INT8:
            ks = tl.load(KS + (slots[None, :] * KV_HEADS + kv_head) * (D // 32) + cols[:, None] // 32,
                         valid[None, :], 0)
            vs = tl.load(VS + slots * KV_HEADS + kv_head, valid, 0)
            k = (k.to(tl.float32) * ks).to(q.dtype)
            v = (v.to(tl.float32) * vs[:, None]).to(q.dtype)
        scores = tl.dot(q, k) * SCALE
        mask = (rows[:, None] < GROUP) & valid[None, :]
        scores = tl.where(mask, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(tl.where(new_max == -float("inf"), 0., maximum - new_max))
        probs = tl.exp(tl.where(mask, scores - new_max[:, None], -float("inf")))
        acc = acc * correction[:, None] + tl.dot(probs.to(q.dtype), v)
        denominator = denominator * correction + tl.sum(probs, 1)
        maximum = new_max
    if SPLITS == 1:
        output = tl.where(denominator[:, None] > 0, acc / denominator[:, None], 0.)
        tl.store(Out + (seq * KV_HEADS * GROUP + heads[:, None]) * D + cols[None, :],
                 output, rows[:, None] < GROUP)
    else:
        base = Partial + seq * p0 + heads * p1 + split * p2
        tl.store(base[:, None] + cols[None, :], acc, rows[:, None] < GROUP)
        tl.store(base + D, maximum, rows < GROUP)
        tl.store(base + D + 1, denominator, rows < GROUP)


@triton.jit
def _merge_attention(Partial, Out, p0: tl.constexpr, p1: tl.constexpr, p2: tl.constexpr,
                     HEADS: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr):
    seq, head = tl.program_id(0), tl.program_id(1)
    splits, cols = tl.arange(0, SPLITS), tl.arange(0, D)
    base = Partial + seq * p0 + head * p1 + splits * p2
    m, l = tl.load(base + D), tl.load(base + D + 1)
    largest = tl.max(m, 0)
    weights = tl.where(l > 0, tl.exp(m - largest), 0.)
    values = tl.load(base[:, None] + cols[None, :])
    numerator = tl.sum(values * weights[:, None], 0)
    denominator = tl.sum(l * weights, 0)
    result = tl.where(denominator > 0, numerator / denominator, 0.)
    tl.store(Out + (seq * HEADS + head) * D + cols, result)


def paged_decode_attention(q, k_cache, v_cache, block_tables, context_lens,
                           softmax_scale=None, k_scale=None, v_scale=None,
                           workspace=None, num_splits=None, block_n=None, num_warps=None):
    if q.ndim != 3 or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q must be [batch, heads, dimension], FP16/BF16")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V must use matching [blocks, page, KV heads, dimension] caches")
    batch, heads, dim = q.shape
    blocks, page, kv_heads, cache_dim = k_cache.shape
    if kv_heads < 1 or page < 1:
        raise ValueError("Page size and KV head count must be positive")
    group = heads // kv_heads
    if cache_dim != dim or dim not in (64, 128, 256) or heads % kv_heads or group not in (1, 2, 4, 8, 16):
        raise ValueError("Unsupported head dimension or GQA ratio")
    tensors = (q, k_cache, v_cache, block_tables, context_lens)
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        raise ValueError("All tensors must be on the same CUDA device")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous() or k_cache.dtype != v_cache.dtype:
        raise ValueError("K/V cache must be contiguous and use the same dtype")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch or context_lens.shape != (batch,):
        raise ValueError("Invalid block table/context length shape")
    if block_tables.dtype != torch.int32 or context_lens.dtype != torch.int32 or context_lens.stride(0) != 1:
        raise ValueError("Page indices and context lengths must be int32; lengths contiguous")
    quantized = k_cache.dtype == torch.int8
    if quantized:
        validate_scales(k_cache, k_scale, v_scale)
    elif k_cache.dtype != q.dtype:
        raise ValueError("High-precision cache must match Q dtype")
    tile, warps, splits = choose_decode_config(batch, kv_heads, block_tables.shape[1] * page)
    num_splits = splits if num_splits is None else num_splits
    block_n = tile if block_n is None else block_n
    num_warps = warps if num_warps is None else num_warps
    if num_splits not in (1, 2, 4, 8, 16) or block_n not in (32, 64, 128) or num_warps not in (4, 8):
        raise ValueError("Unsupported split/tile/warp configuration")
    if workspace is None:
        workspace = AttentionWorkspace(batch, heads, dim, q.device)
    partial = workspace.partial
    if (partial.shape[0] < batch or partial.shape[1:] != (heads, 16, dim + 2) or
            partial.device != q.device or partial.dtype != torch.float32):
        raise ValueError("Workspace does not match Q")
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if batch:
        _decode_attention[(batch, kv_heads, num_splits)](
            q, k_cache, v_cache, k_scale if quantized else k_cache,
            v_scale if quantized else v_cache, block_tables, context_lens, out, partial,
            *q.stride(), *block_tables.stride(), *partial.stride()[:3],
            kv_heads, group, dim, page, softmax_scale or dim ** -0.5, num_splits,
            quantized, max(16, triton.next_power_of_2(group)), block_n,
            num_warps=num_warps, num_stages=2,
        )
        if num_splits > 1:
            _merge_attention[(batch, heads)](
                partial, out, *partial.stride()[:3], heads, dim, num_splits, num_warps=4,
            )
    return out


@triton.jit
def _prefill_attention(Q, K, V, KS, VS, Tables, CUQ, CUK, Out,
                       q0: tl.constexpr, q1: tl.constexpr, q2: tl.constexpr,
                       t0: tl.constexpr, t1: tl.constexpr,
                       KV_HEADS: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                       PAGE: tl.constexpr, SCALE: tl.constexpr, INT8: tl.constexpr,
                       M: tl.constexpr, N: tl.constexpr):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows, cols, keys = tl.arange(0, M), tl.arange(0, D), tl.arange(0, N)
    begin, end = tl.load(CUQ + seq), tl.load(CUQ + seq + 1)
    length = tl.load(CUK + seq + 1) - tl.load(CUK + seq)
    query = (tile * M + rows) // GROUP
    q_head = head * GROUP + (tile * M + rows) % GROUP
    valid_q = query < end - begin
    position = length - (end - begin) + query
    q = tl.load(Q + (begin + query[:, None]) * q0 + q_head[:, None] * q1 + cols[None, :] * q2,
                valid_q[:, None], 0)
    acc = tl.full((M, D), 0, tl.float32)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.full((M,), 0, tl.float32)
    stop = tl.minimum(length, length - (end - begin) + (tile * M + M - 1) // GROUP + 1)
    for offset in range(0, stop, N):
        pos = offset + keys
        valid = pos < length
        block = tl.load(Tables + seq * t0 + pos // PAGE * t1, valid, 0)
        slots = block * PAGE + pos % PAGE
        k = tl.load(K + (slots[None, :] * KV_HEADS + head) * D + cols[:, None], valid[None, :], 0)
        v = tl.load(V + (slots[:, None] * KV_HEADS + head) * D + cols[None, :], valid[:, None], 0)
        if INT8:
            ks = tl.load(KS + (slots[None, :] * KV_HEADS + head) * (D // 32) + cols[:, None] // 32,
                         valid[None, :], 0)
            vs = tl.load(VS + slots * KV_HEADS + head, valid, 0)
            k = (k.to(tl.float32) * ks).to(q.dtype)
            v = (v.to(tl.float32) * vs[:, None]).to(q.dtype)
        mask = valid_q[:, None] & valid[None, :] & (pos[None, :] <= position[:, None])
        scores = tl.where(mask, tl.dot(q, k) * SCALE, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp(tl.where(new_max == -float("inf"), 0., maximum - new_max))
        probs = tl.exp(tl.where(mask, scores - new_max[:, None], -float("inf")))
        acc = acc * correction[:, None] + tl.dot(probs.to(q.dtype), v)
        denominator = denominator * correction + tl.sum(probs, 1)
        maximum = new_max
    output = tl.where(denominator[:, None] > 0, acc / denominator[:, None], 0.)
    tl.store(Out + ((begin + query[:, None]) * KV_HEADS * GROUP + q_head[:, None]) * D + cols[None, :],
             output, valid_q[:, None])


def paged_prefill_attention(q, k_cache, v_cache, block_tables, cu_q, cu_k, max_query,
                            softmax_scale=None, k_scale=None, v_scale=None):
    """Ragged causal queries attend directly to paged cache, including cached prefixes."""
    # Reuse Decode's structural validation without reading any device metadata.
    batch = cu_q.numel() - 1
    if cu_q.dtype != torch.int32 or cu_k.dtype != torch.int32 or cu_k.shape != cu_q.shape:
        raise ValueError("Prefill cumulative lengths must be matching int32 vectors")
    if not all(t.is_cuda and t.device == q.device and t.is_contiguous() for t in (cu_q, cu_k)):
        raise ValueError("Cumulative lengths must be contiguous on Q's CUDA device")
    if q.ndim != 3 or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Prefill Q must be FP16/BF16 [tokens, heads, dimension]")
    kv_heads, dim = k_cache.shape[-2:]
    heads = q.shape[1]
    if kv_heads < 1 or heads % kv_heads or heads // kv_heads not in (1, 2, 4, 8, 16) or dim not in (64, 128, 256) or q.shape[2] != dim:
        raise ValueError("Unsupported Prefill head dimension/GQA")
    if (v_cache.shape != k_cache.shape or v_cache.dtype != k_cache.dtype or
            not k_cache.is_contiguous() or not v_cache.is_contiguous() or
            block_tables.ndim != 2 or block_tables.shape[0] != batch or block_tables.dtype != torch.int32 or
            not all(t.is_cuda and t.device == q.device for t in (k_cache, v_cache, block_tables))):
        raise ValueError("Invalid paged Prefill caches or block tables")
    quantized = k_cache.dtype == torch.int8
    if quantized:
        validate_scales(k_cache, k_scale, v_scale)
    elif k_cache.dtype != q.dtype:
        raise ValueError("High-precision cache must match Q dtype")
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    if batch and max_query:
        _prefill_attention[(batch, kv_heads, triton.cdiv(max_query * (heads // kv_heads), 32))](
            q, k_cache, v_cache, k_scale if quantized else k_cache, v_scale if quantized else v_cache,
            block_tables, cu_q, cu_k, output, *q.stride(), *block_tables.stride(),
            kv_heads, heads // kv_heads, dim, k_cache.shape[1], softmax_scale or dim ** -0.5,
            quantized, 32, 64, num_warps=4, num_stages=2)
    return output

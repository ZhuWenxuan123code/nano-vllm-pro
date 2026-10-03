"""CUDA Graph microbenchmark for the Decode Q/K Norm + RoPE + cache-write chain."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.testing

from nanovllm.layers.attention import store_kvcache
from nanovllm.layers.decode_qkv_fused import decode_qkv_fused
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=25)
    parser.add_argument("--capture-nsys", choices=("original", "fused"),
                        help="Capture 20 CUDA Graph replays of one backend for Nsight Systems.")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.kv_heads < 1 or args.q_heads % args.kv_heads:
        parser.error("batch size must be positive and Q heads a multiple of KV heads")
    if args.rounds < 2 or args.rep_ms < 1 or any(w not in (1, 2, 4, 8) for w in args.warps):
        parser.error("rounds must be at least 2, rep-ms positive, and warps one of 1, 2, 4, 8")

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    batch, q_heads, kv_heads, head_dim = args.batch_size, args.q_heads, args.kv_heads, args.head_dim
    qkv = torch.randn(batch, (q_heads + 2 * kv_heads) * head_dim, device="cuda", dtype=dtype)
    q_norm = RMSNorm(head_dim).to(device="cuda", dtype=dtype)
    k_norm = RMSNorm(head_dim).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        q_norm.weight.uniform_(0.75, 1.25)
        k_norm.weight.uniform_(0.75, 1.25)
    rope = RotaryEmbedding(head_dim, head_dim, 4096, 1000000).cuda()
    positions = torch.arange(batch, device="cuda", dtype=torch.int64) + 128
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    k_original = torch.zeros((1, 256, kv_heads, head_dim), device="cuda", dtype=dtype)
    v_original = torch.zeros_like(k_original)
    k_fused = torch.zeros_like(k_original)
    v_fused = torch.zeros_like(k_original)

    def original():
        q, k, v = qkv.split((q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim), dim=-1)
        q, k, v = (x.view(batch, -1, head_dim) for x in (q, k, v))
        q, k = q_norm(q), k_norm(k)
        q, k = rope(positions, q, k)
        store_kvcache(k, v, k_original, v_original, slots)
        return q

    def fused(warps):
        return decode_qkv_fused(qkv, q_norm.weight, k_norm.weight, rope.cos_sin_cache,
                                positions, slots, k_fused, v_fused, num_warps=warps)

    with torch.inference_mode():
        reference = original()
        if args.capture_nsys:
            mode = args.capture_nsys
            fn = original if mode == "original" else (lambda: fused(args.warps[0]))
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"Captured 20 {mode} CUDA Graph replays", flush=True)
            return
        results = []
        for warps in args.warps:
            actual = fused(warps)
            tolerance = 1e-2 if dtype == torch.float16 else 2e-2
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(k_fused, k_original, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(v_fused, v_original, atol=0, rtol=0)
            original_ms = []
            fused_ms = []
            for round_index in range(args.rounds):
                # Alternate order to reduce temperature/order bias.
                modes = ("original", "fused") if round_index % 2 == 0 else ("fused", "original")
                for mode in modes:
                    fn = original if mode == "original" else (lambda: fused(warps))
                    latency = triton.testing.do_bench_cudagraph(
                        fn, rep=args.rep_ms, return_mode="median",
                    ) * 1000
                    (original_ms if mode == "original" else fused_ms).append(latency)
            original_median = statistics.median(original_ms)
            fused_median = statistics.median(fused_ms)
            result = {
                "warps": warps,
                "original_us": original_ms,
                "fused_us": fused_ms,
                "original_median_us": original_median,
                "fused_median_us": fused_median,
                "improvement_percent": (original_median / fused_median - 1) * 100,
                "original_std_us": statistics.stdev(original_ms),
                "fused_std_us": statistics.stdev(fused_ms),
            }
            results.append(result)
            print(f"warps={warps}: original={original_median:.3f} us "
                  f"fused={fused_median:.3f} us gain={result['improvement_percent']:+.2f}% "
                  f"std=({result['original_std_us']:.3f},{result['fused_std_us']:.3f}) us", flush=True)

    if args.output_json:
        path = Path(args.output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "batch_size": batch,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "dtype": args.dtype,
            "rounds": args.rounds,
            "rep_ms": args.rep_ms,
            "results": results,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Reproducible CUDA Graph Attention measurements; tuning is outside timed regions."""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.testing
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.paged_attention import AttentionWorkspace, choose_decode_config, paged_decode_attention
from nanovllm.layers.kv_quantization import INT8_CACHE_FORMAT, store_int8_kvcache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192, 16384])
    parser.add_argument("--head-dims", type=int, nargs="+", default=[128])
    parser.add_argument("--groups", type=int, nargs="+", default=[2])
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--include-int8", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--capture-nsys", choices=("flash", "triton", "int8"))
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.rounds < 2 or args.rep_ms < 1 or args.kv_heads < 1 or min(args.batch_sizes + args.lengths) < 1:
        parser.error("Positive shapes/rep-ms and at least two rounds required")
    torch.manual_seed(41)
    results = []
    for batch, length, dim, group in itertools.product(args.batch_sizes, args.lengths, args.head_dims, args.groups):
        heads, page = args.kv_heads * group, 256
        pages = triton.cdiv(length, page)
        dtype = getattr(torch, args.dtype)
        q = torch.randn(batch, heads, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch * pages, page, args.kv_heads, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        tables = torch.arange(batch * pages, device="cuda", dtype=torch.int32).reshape(batch, pages)
        lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        workspace = AttentionWorkspace(batch, heads, dim)
        functions = {"flash": lambda: flash_attn_with_kvcache(q[:, None], k, v, cache_seqlens=lens, block_table=tables)[:, 0]}
        if args.include_int8 or args.capture_nsys == "int8":
            ki, vi = torch.empty_like(k, dtype=torch.int8), torch.empty_like(v, dtype=torch.int8)
            ks = torch.empty(k.shape[:-1] + (dim // 32,), device="cuda", dtype=torch.float32)
            vs = torch.empty(k.shape[:-1], device="cuda", dtype=torch.float32)
            slots = torch.arange(k.shape[0] * page, device="cuda", dtype=torch.int32)
            store_int8_kvcache(k.flatten(0, 1), v.flatten(0, 1), ki, vi, ks, vs, slots)
            kd = (ki.float().reshape(*ks.shape, 32) * ks[..., None]).flatten(-2).to(dtype)
            vd = (vi.float() * vs[..., None]).to(dtype)
            int8_reference = flash_attn_with_kvcache(q[:, None], kd, vd, cache_seqlens=lens, block_table=tables)[:, 0]
            del kd, vd, slots
        configurations = list(itertools.product((32, 64, 128), (4, 8), (1, 2, 4, 8, 16))) if args.tune else [choose_decode_config(batch, args.kv_heads, pages * page)]
        reference = functions["flash"]()
        for tile, warps, splits in configurations:
            functions["triton"] = lambda: paged_decode_attention(q, k, v, tables, lens, workspace=workspace, block_n=tile, num_warps=warps, num_splits=splits)
            if args.include_int8 or args.capture_nsys == "int8":
                functions["int8"] = lambda: paged_decode_attention(q, ki, vi, tables, lens, k_scale=ks, v_scale=vs, workspace=workspace, block_n=tile, num_warps=warps, num_splits=splits)
            row = {"batch": batch, "length": length, "head_dim": dim, "group": group,
                   "kv_heads": args.kv_heads, "tile": tile, "warps": warps, "splits": splits,
                   "workspace_bytes": workspace.nbytes, "triton_kernels": 1 + (splits > 1)}
            try:
                actual = functions["triton"]()
                torch.testing.assert_close(actual, reference, atol=0.003 if dtype == torch.float16 else 0.02, rtol=0.02)
                row["max_abs_error"] = (actual - reference).abs().max().item()
                if "int8" in functions:
                    quantized_actual = functions["int8"]()
                    torch.testing.assert_close(quantized_actual, int8_reference,
                                               atol=0.003 if dtype == torch.float16 else 0.02, rtol=0.02)
                    row["int8_implementation_max_abs_error"] = (quantized_actual - int8_reference).abs().max().item()
                if args.capture_nsys:
                    fn = functions[args.capture_nsys]
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
                    return
                times = {name: [] for name in functions}
                for repeat in range(args.rounds):
                    names = list(functions)
                    names = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in names:
                        times[name].append(triton.testing.do_bench_cudagraph(functions[name], rep=args.rep_ms, return_mode="median") * 1000)
                row["latency_us"] = times
                row["median_us"] = {name: statistics.median(values) for name, values in times.items()}
                row["std_us"] = {name: statistics.stdev(values) for name, values in times.items()}
            except (RuntimeError, AssertionError) as error:
                row["error"] = str(error)
            results.append(row)
            print(f"B={batch} L={length} D={dim} G={group} tile={tile} w={warps} splits={splits}: {row.get('median_us', row.get('error'))}", flush=True)
        # Store completed shapes even if a later shape fails or the run is interrupted.
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": vars(args), "gpu": torch.cuda.get_device_name(),
                                    "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                                    "torch": torch.__version__, "triton": triton.__version__, "int8_cache_format": INT8_CACHE_FORMAT,
                                    "results": results}, indent=2) + "\n")
        del q, k, v, workspace, reference, actual, functions
        if args.include_int8:
            del ki, vi, ks, vs, int8_reference, quantized_actual


if __name__ == "__main__":
    main()

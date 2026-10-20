"""Controlled BF16 Decode tile ablation; profiling is separate from timing."""

import argparse
import hashlib
import json
import statistics
from functools import partial
from pathlib import Path

import flash_attn
import torch
import triton
import triton.testing
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.paged_attention import AttentionWorkspace, paged_decode_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', nargs='+', type=int, default=[128, 129, 256, 257, 512, 640])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--rep-ms', type=int, default=100)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--profile', choices=['nsys', 'ncu'])
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()
    if args.rounds < 5 or args.rep_ms < 1 or min(args.lengths) < 1:
        parser.error('Use at least five rounds and positive lengths/rep-ms')
    root = Path(__file__).resolve().parents[1]
    sources = ['benchmarks/bench_attention_tiles.py', 'nanovllm/layers/paged_attention.py']
    report = {'config': vars(args), 'gpu': torch.cuda.get_device_name(),
              'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'triton': triton.__version__, 'flash_attn': flash_attn.__version__,
              'sources': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
              'batch': 64, 'q_heads': 16, 'kv_heads': 8, 'head_dim': 128,
              'dtype': 'bfloat16', 'page_size': 256, 'splits': 1, 'warps': 4,
              'profile_only': args.profile is not None, 'results': []}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.lengths:
        torch.manual_seed(args.seed + length)
        batch, kv_heads, group, dim, page = 64, 8, 2, 128, 256
        pages = triton.cdiv(length, page)
        q = torch.randn(batch, kv_heads * group, dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch * pages, page, kv_heads, dim, device='cuda', dtype=q.dtype)
        v = torch.randn_like(k)
        tables = torch.arange(batch * pages, device='cuda', dtype=torch.int32).reshape(batch, pages)
        lens = torch.full((batch,), length, device='cuda', dtype=torch.int32)
        workspace = AttentionWorkspace(batch, kv_heads * group, dim)
        inputs = (q, k, v, tables, lens)
        snapshots = [tensor.clone() for tensor in inputs]
        # Independent FP32 reference; no full repeated-head K/V copy is needed.
        k_ref = k.reshape(batch, pages * page, kv_heads, dim)[:, :length].float()
        v_ref = v.reshape(batch, pages * page, kv_heads, dim)[:, :length].float()
        scores = torch.einsum('bhgd,blhd->bhgl', q.float().reshape(batch, kv_heads, group, dim), k_ref)
        reference = torch.einsum('bhgl,blhd->bhgd', (scores * dim ** -0.5).softmax(-1), v_ref).reshape_as(q)
        functions = {'flash': lambda: flash_attn_with_kvcache(
            q[:, None], k, v, cache_seqlens=lens, block_table=tables,
            causal=True, num_splits=1)[:, 0]}
        for m, n in ((16, 32), (64, 32), (16, 128), (64, 128)):
            functions[f'triton_m{m}_n{n}'] = partial(
                paged_decode_attention, q, k, v, tables, lens, workspace=workspace,
                block_m=m, block_n=n, num_warps=4, num_splits=1)
        row = {'length': length, 'variants': {}, 'order': []}
        graphs, graph_outputs = {}, {}
        for name, fn in functions.items():
            entry = row['variants'][name] = {}
            try:
                actual = fn()
                torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
                entry['max_abs_error_fp32'] = (actual.float() - reference).abs().max().item()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = fn()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured.float(), reference, atol=0.02, rtol=0.02)
                graphs[name], graph_outputs[name] = graph, captured
                entry['correctness_passed'] = True
            except (RuntimeError, AssertionError) as error:
                entry['error'] = str(error)
        for tensor, snapshot in zip(inputs, snapshots):
            assert torch.equal(tensor, snapshot), 'Input or KV cache was modified'
        row['inputs_unchanged'] = True
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for name in graphs:
                with torch.cuda.nvtx.range(name):
                    if args.profile == 'nsys':
                        for _ in range(20):
                            graphs[name].replay()
                    else:
                        functions[name]()
                    torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        else:
            # All variants compiled/validated before any formal measurement.
            names = list(graphs)
            for name in names:
                row['variants'][name]['latency_us'] = []
            for repeat in range(args.rounds):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                row['order'].append(order)
                for name in order:
                    elapsed = triton.testing.do_bench_cudagraph(
                        functions[name], rep=args.rep_ms, return_mode='median') * 1000
                    row['variants'][name]['latency_us'].append(elapsed)
            for entry in row['variants'].values():
                if 'latency_us' in entry:
                    entry['median_us'] = statistics.median(entry['latency_us'])
                    entry['std_us'] = statistics.stdev(entry['latency_us'])
        report['results'].append(row)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        del graphs, graph_outputs, functions, snapshots, inputs, reference, k_ref, v_ref, scores, q, k, v
    if any('error' in entry for row in report['results'] for entry in row['variants'].values()):
        raise SystemExit('Some variants failed; see saved errors')


if __name__ == '__main__':
    main()

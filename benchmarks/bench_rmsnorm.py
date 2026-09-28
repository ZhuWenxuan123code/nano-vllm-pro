"""Compare PyTorch eager, torch.compile and Triton RMSNorm on one GPU."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.testing

from nanovllm.layers.layernorm import RMSNorm


def eager_rmsnorm(x, weight, eps, residual=None):
    combined = x.float() if residual is None else x.float() + residual.float()
    updated = combined.to(x.dtype) if residual is not None else None
    var = combined.square().mean(dim=-1, keepdim=True)
    out = (combined * torch.rsqrt(var + eps)).to(x.dtype).mul_(weight)
    return out if residual is None else (out, updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 8192])
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"])
    parser.add_argument("--layout", choices=["contiguous", "qk"], default="contiguous")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if any(value <= 0 for value in args.rows + args.hidden_sizes):
        parser.error("rows and hidden sizes must be positive")
    if args.layout == "qk" and (args.hidden_sizes != [128] or any(rows % 4 for rows in args.rows)):
        parser.error("qk layout requires --hidden-sizes 128 and rows divisible by 4")

    results = []
    accuracy = []
    torch.manual_seed(0)
    with torch.inference_mode():
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for rows in args.rows:
                for hidden_size in args.hidden_sizes:
                    if args.layout == "qk":
                        projection = torch.randn(rows // 4, 8 * hidden_size, device="cuda", dtype=dtype)
                        x = projection[:, :4 * hidden_size].view(rows // 4, 4, hidden_size)
                    else:
                        x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
                    residual = torch.randn_like(x)
                    norm = RMSNorm(hidden_size).to(device="cuda", dtype=dtype)
                    for mode in ("plain", "add"):
                        values = {}
                        other = None if mode == "plain" else residual
                        compiled_fn = (
                            (lambda: norm.rms_forward(x)) if other is None
                            else (lambda: norm.add_rms_forward(x, other))
                        )
                        for backend in ("eager", "compiled", "triton"):
                            if backend == "eager":
                                fn = lambda: eager_rmsnorm(x, norm.weight, norm.eps, other)
                            elif backend == "compiled":
                                fn = compiled_fn
                            else:
                                norm.set_backend("triton")
                                fn = lambda: norm(x, other)
                            value = fn()  # Compile outside of timed measurements.
                            values[backend] = value if isinstance(value, tuple) else (value,)
                            ms = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms)
                            row = {
                                "dtype": dtype_name,
                                "mode": mode,
                                "rows": rows,
                                "hidden_size": hidden_size,
                                "backend": backend,
                                "layout": args.layout,
                                "latency_us": round(ms * 1000, 3),
                            }
                            results.append(row)
                            print(f"{dtype_name:8s} {mode:5s} M={rows:5d} H={hidden_size:4d} "
                                  f"{backend:8s} {row['latency_us']:9.3f} us", flush=True)
                        for reference in ("eager", "compiled"):
                            error = {
                                "dtype": dtype_name,
                                "mode": mode,
                                "rows": rows,
                                "hidden_size": hidden_size,
                                "layout": args.layout,
                                "reference": reference,
                                "max_abs_output": (values["triton"][0] - values[reference][0]).abs().max().item(),
                            }
                            if other is not None:
                                error["max_abs_residual"] = (
                                    values["triton"][1] - values[reference][1]
                                ).abs().max().item()
                            accuracy.append(error)
    if args.output_json:
        path = Path(args.output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "results": results,
            "accuracy": accuracy,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

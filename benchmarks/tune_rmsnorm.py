"""Measure RMSNorm kernel warp choices, excluding Python output allocation."""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.testing

from nanovllm.layers.rmsnorm_triton import _add_rmsnorm_kernel, _rmsnorm_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--rep-ms", type=int, default=200)
    args = parser.parse_args()
    torch.manual_seed(0)
    results = []
    for hidden_size in (128, 1024):
        for rows in (64, 8192):
            x = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            weight = torch.ones(hidden_size, device="cuda", dtype=x.dtype)
            out = torch.empty_like(x)
            new_residual = torch.empty_like(x)
            for mode in ("plain", "add"):
                for num_warps in (1, 2, 4, 8):
                    if mode == "plain":
                        fn = lambda: _rmsnorm_kernel[(rows,)](
                            x, weight, out, hidden_size, 0, 1, 1, hidden_size,
                            1e-6, triton.next_power_of_2(hidden_size), num_warps=num_warps,
                        )
                    else:
                        fn = lambda: _add_rmsnorm_kernel[(rows,)](
                            x, residual, weight, out, new_residual,
                            hidden_size, 0, 1, hidden_size, 0, 1, 1,
                            hidden_size, 1e-6, triton.next_power_of_2(hidden_size),
                            num_warps=num_warps,
                        )
                    fn()
                    ms = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms)
                    result = {
                        "mode": mode,
                        "rows": rows,
                        "hidden_size": hidden_size,
                        "num_warps": num_warps,
                        "latency_us": round(ms * 1000, 3),
                    }
                    results.append(result)
                    print(result, flush=True)
    path = Path(args.output_json).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

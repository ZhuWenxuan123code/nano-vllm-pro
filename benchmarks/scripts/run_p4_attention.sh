#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python_bin="${PYTHON_BIN:-python}"
root="${RESULT_ROOT:-benchmarks/results/p4-k32-gpu1}"
"$python_bin" benchmarks/bench_paged_attention.py --tune --batch-sizes 1 8 64 \
  --lengths 128 2048 8192 --output-json "$root/tuning.json"
"$python_bin" benchmarks/bench_paged_attention.py --include-int8 --output-json "$root/layers.json"
"$python_bin" benchmarks/bench_paged_attention.py --include-int8 --batch-sizes 1 8 \
  --lengths 128 2048 --head-dims 64 256 --groups 1 4 8 16 --kv-heads 2 \
  --output-json "$root/variants.json"

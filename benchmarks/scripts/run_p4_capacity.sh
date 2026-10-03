#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python_bin="${PYTHON_BIN:-python}"
root="${RESULT_ROOT:-benchmarks/results/p4-k32-gpu1/capacity}"
for backend in flash triton int8; do
  "$python_bin" benchmarks/bench_kv_capacity.py --backend "$backend" \
    --model "${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" --output-json "$root/$backend.json"
done

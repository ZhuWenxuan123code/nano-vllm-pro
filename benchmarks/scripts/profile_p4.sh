#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
root="${PROFILE_ROOT:-benchmarks/profiles/p4-k32-gpu1}"
export FUSE_DECODE_QK_ROPE_CACHE=0 EXECUTION_MODE=original
for backend in flash triton int8; do
  export ATTENTION_BACKEND=triton KV_CACHE_DTYPE=auto PROFILE_DIR="$root/$backend"
  [[ "$backend" != "flash" ]] || export ATTENTION_BACKEND=flash
  [[ "$backend" != "int8" ]] || export KV_CACHE_DTYPE=int8
  bash benchmarks/scripts/profile_nsys.sh decode
done

# Isolate one Attention layer from the rest of the model. These profiled
# timings are diagnostic only, not substitutes for run_p4_attention.sh.
mkdir -p "$root/layers"
for backend in flash triton int8; do
  report="$root/layers/b64-l8192-$backend"
  nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="$report" \
    "$PYTHON_BIN" benchmarks/bench_paged_attention.py --batch-sizes 64 --lengths 8192 \
      --capture-nsys "$backend" --output-json "$report.json"
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum \
    --format csv "$report.nsys-rep" > "$report.nsys.csv"
done

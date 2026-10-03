#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
result_root="${RESULT_ROOT:-benchmarks/results/p3-runner-gpu1}"
runs="${RUNS:-5}"
export RMS_NORM_BACKEND=compiled MEASUREMENT_MODE=runtime
export WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Rotate order every repetition; never overwrite the previous repetition.
for fused in 0 1; do
  workloads=(decode_heavy balanced prefill_heavy)
  if [[ "$fused" == "1" ]]; then workloads=(decode_heavy balanced); fi
  for workload in "${workloads[@]}"; do
    for ((run = 1; run <= runs; run++)); do
      modes=(original buffered)
      if ((run % 2 == 0)); then modes=(buffered original); fi
      for mode in "${modes[@]}"; do
        echo "P3: p2=$fused workload=$workload mode=$mode run=$run"
        EXECUTION_MODE="$mode" FUSE_DECODE_QK_ROPE_CACHE="$fused" \
          RUNS=1 RUN_START="$run" RESULT_DIR="$result_root/p2-$fused/$mode" \
          bash "benchmarks/scripts/run_$workload.sh"
      done
    done
  done
  "$python_bin" benchmarks/compare.py "$result_root/p2-$fused/original" \
    "$result_root/p2-$fused/buffered" --output "$result_root/p2-$fused/comparison.md"
  for mode in original buffered; do
    "$python_bin" benchmarks/summarize.py "$result_root/p2-$fused/$mode"
  done
done

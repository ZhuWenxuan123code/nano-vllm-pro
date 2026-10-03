#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
result_root="${RESULT_ROOT:-benchmarks/results/p2-decode-qkv}"

for backend in original fused; do
  fused="0"
  if [[ "$backend" == "fused" ]]; then
    fused="1"
  fi
  for workload in decode_heavy balanced prefill_heavy; do
    echo "Running $backend / $workload"
    RESULT_DIR="$result_root/$backend" RMS_NORM_BACKEND=compiled \
      FUSE_DECODE_QK_ROPE_CACHE="$fused" \
      bash "benchmarks/scripts/run_${workload}.sh"
  done
done

"$python_bin" benchmarks/compare.py \
  "$result_root/original" "$result_root/fused" \
  --output "$result_root/comparison.md"

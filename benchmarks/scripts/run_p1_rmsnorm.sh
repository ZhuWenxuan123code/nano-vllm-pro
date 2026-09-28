#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
result_root="${RESULT_ROOT:-benchmarks/results/p1-rmsnorm}"

for backend in compiled triton; do
  for workload in balanced decode_heavy prefill_heavy; do
    echo "Running $backend / $workload"
    RESULT_DIR="$result_root/$backend" RMS_NORM_BACKEND="$backend" \
      bash "benchmarks/scripts/run_${workload}.sh"
  done
done

"$python_bin" benchmarks/compare.py \
  "$result_root/compiled" "$result_root/triton" \
  --output "$result_root/comparison.md"

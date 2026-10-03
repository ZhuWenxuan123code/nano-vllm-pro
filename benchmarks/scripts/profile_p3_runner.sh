#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
profile_root="${PROFILE_DIR:-benchmarks/profiles/p3-runner-gpu1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export EXECUTION_MODE=buffered PROFILE_STAGES=1 FUSE_DECODE_QK_ROPE_CACHE=0
export ENFORCE_EAGER=0 BATCH_SIZE=64 INPUT_LEN=128 PROFILE_STEPS=105 TP_SIZE=1
traces=()
for run in 1 2 3; do
  PROFILE_DIR="$profile_root/gate-$run" bash benchmarks/scripts/profile_nsys.sh decode
  traces+=("$profile_root/gate-$run/nsys/decode-cudagraph-buffered.sqlite")
done
"$python_bin" benchmarks/analyze_runner.py "${traces[@]}" \
  --output-json "$profile_root/gate.json"

#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
"${PYTHON_BIN:-python}" benchmarks/run_p5.py \
  --stage "${P5_STAGE:-all}" --root "${RESULT_ROOT:-benchmarks/results/p5-scheduling-controlled-gpu1}" \
  --model "${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}" --runs "${RUNS:-5}" \
  --requests "${P5_REQUESTS:-256}" --tune-requests "${P5_TUNE_REQUESTS:-128}" \
  --tune-runs "${P5_TUNE_RUNS:-3}"

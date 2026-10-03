#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python_bin="${PYTHON_BIN:-python}"
model="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
data="${DATA_DIR:-benchmarks/results/p4-data}"
root="${QUALITY_DIR:-benchmarks/results/p4-k32-gpu1/quality}"
mkdir -p "$root"
if [[ ! -f "$data/manifest.json" ]]; then
  "$python_bin" benchmarks/eval_kv_quality.py --prepare --model "$model" --data-dir "$data"
fi
for length in 2048 8192; do
  for backend in flash triton int8; do
    "$python_bin" benchmarks/eval_kv_quality.py --model "$model" --data-dir "$data" \
      --backend "$backend" --window "$length" --output-json "$root/$backend-$length.json"
  done
done
"$python_bin" benchmarks/check_kv_quality.py "$root"
# Real-text regressions are separate from the full 32768-target quality gate.
for backend in flash triton int8; do
  "$python_bin" benchmarks/eval_kv_quality.py --model "$model" --data-dir "$data" \
    --backend "$backend" --limit-windows 1 --tail-tokens 128 --prefill-budget 128 \
    --shared-prefix --output-json "$root/$backend-prefix-chunk.json"
done

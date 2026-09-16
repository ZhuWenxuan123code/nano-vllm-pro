#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
device_ids="${CUDA_VISIBLE_DEVICES:-1}"
result_dir="${RESULT_DIR:-benchmarks/results/baseline}"
runs="${RUNS:-5}"

for ((run = 1; run <= runs; run++)); do
  CUDA_VISIBLE_DEVICES="$device_ids" "$python_bin" bench.py \
    --model "$model_path" \
    --num-prompts 256 \
    --max-num-seqs 64 \
    --input-len 1024 \
    --output-len 128 \
    --max-model-len 4096 \
    --seed 0 \
    --warmup-runs 1 \
    --output-json "$result_dir/balanced-${run}.json"
done

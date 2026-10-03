#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
device_ids="${CUDA_VISIBLE_DEVICES:-1}"
result_dir="${RESULT_DIR:-benchmarks/results/baseline}"
runs="${RUNS:-5}"
run_start="${RUN_START:-1}"

for ((run = run_start; run < run_start + runs; run++)); do
  CUDA_VISIBLE_DEVICES="$device_ids" "$python_bin" bench.py \
    --model "$model_path" \
    --execution-mode "${EXECUTION_MODE:-original}" \
    --measurement-mode "${MEASUREMENT_MODE:-sync}" \
    --warmup-decode-steps "${WARMUP_DECODE_STEPS:-0}" \
    --num-prompts 256 \
    --max-num-seqs 512 \
    --min-input-len 100 \
    --max-input-len 1024 \
    --min-output-len 100 \
    --max-output-len 1024 \
    --max-model-len 4096 \
    --seed 0 \
    --warmup-runs 1 \
    --output-json "$result_dir/random-${run}.json"
done

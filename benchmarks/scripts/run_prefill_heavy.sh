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
rms_norm_backend="${RMS_NORM_BACKEND:-compiled}"
fuse_decode="${FUSE_DECODE_QK_ROPE_CACHE:-0}"
extra_args=()
if [[ "$fuse_decode" == "1" ]]; then
  extra_args+=(--fuse-decode-qk-rope-cache)
fi

for ((run = run_start; run < run_start + runs; run++)); do
  CUDA_VISIBLE_DEVICES="$device_ids" "$python_bin" bench.py \
    --model "$model_path" \
    --execution-mode "${EXECUTION_MODE:-original}" \
    --measurement-mode "${MEASUREMENT_MODE:-sync}" \
    --warmup-decode-steps "${WARMUP_DECODE_STEPS:-0}" \
    --num-prompts 128 \
    --max-num-seqs 32 \
    --input-len 2048 \
    --output-len 32 \
    --max-model-len 4096 \
    --rms-norm-backend "$rms_norm_backend" \
    "${extra_args[@]}" \
    --seed 0 \
    --warmup-runs 1 \
    --output-json "$result_dir/prefill-${run}.json"
done

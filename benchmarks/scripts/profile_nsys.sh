#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

phase="${1:-decode}"
if [[ "$phase" != "prefill" && "$phase" != "decode" ]]; then
  echo "usage: $0 [prefill|decode]" >&2
  exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found in PATH" >&2
  exit 127
fi

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
device_ids="${CUDA_VISIBLE_DEVICES:-1}"
output_dir="${PROFILE_DIR:-benchmarks/profiles/baseline}/nsys"
eager="${ENFORCE_EAGER:-0}"

if [[ "$phase" == "prefill" ]]; then
  batch_size="${BATCH_SIZE:-8}"
  input_len="${INPUT_LEN:-1024}"
  profile_steps="${PROFILE_STEPS:-3}"
else
  batch_size="${BATCH_SIZE:-64}"
  input_len="${INPUT_LEN:-128}"
  profile_steps="${PROFILE_STEPS:-20}"
fi

mode="eager"
extra_args=()
if [[ "$eager" == "1" ]]; then
  extra_args+=(--enforce-eager)
elif [[ "$phase" == "decode" ]]; then
  mode="cudagraph"
fi
mkdir -p "$output_dir"
report_base="$output_dir/$phase-$mode"

CUDA_VISIBLE_DEVICES="$device_ids" nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$report_base" \
  "$python_bin" benchmarks/profile_workload.py \
    --backend nsys \
    --phase "$phase" \
    --model "$model_path" \
    --batch-size "$batch_size" \
    --input-len "$input_len" \
    --profile-steps "$profile_steps" \
    --warmup-runs "${WARMUP_RUNS:-1}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-16384}" \
    --tensor-parallel-size "${TP_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    --seed "${SEED:-0}" \
    --output-dir "$output_dir" \
    "${extra_args[@]}"

nsys stats \
  --force-export=true \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  "$report_base.nsys-rep" > "$report_base.nsys.csv"

echo "Nsight report: $report_base.nsys-rep"
echo "Nsight summary: $report_base.nsys.csv"

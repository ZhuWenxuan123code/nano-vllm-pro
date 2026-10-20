#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
root="${PROFILE_DIR:-benchmarks/profiles/p5-scheduling-gpu1}"
mkdir -p "$root"
: "${P5_TRACE:?Set P5_TRACE to a frozen online trace JSON}"
for policy in legacy mixed adaptive; do
  nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
    --output="$root/$policy" "${PYTHON_BIN:-python}" benchmarks/bench_online.py \
    --model "${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}" --trace "$P5_TRACE" \
    --scheduler-policy "$policy" --prefill-token-budget "${PREFILL_TOKEN_BUDGET:-256}" \
    --kv-cache-block-limit "${KV_BLOCKS:-512}" --capture-nsys --output-json "$root/$policy.json"
  nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
    "$root/$policy.nsys-rep" > "$root/$policy.nsys.csv"
done

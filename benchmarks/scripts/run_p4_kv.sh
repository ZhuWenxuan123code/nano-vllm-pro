#!/usr/bin/env bash
# Same-load timing only. Run run_p4_quality.sh first; no profiler in these runs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export MEASUREMENT_MODE=runtime WARMUP_DECODE_STEPS=32 RMS_NORM_BACKEND=compiled
root="${RESULT_ROOT:-benchmarks/results/p4-k32-gpu1/e2e}"
runs="${P4_RUNS:-5}"
model="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
"$PYTHON_BIN" benchmarks/check_kv_quality.py "${QUALITY_DIR:-benchmarks/results/p4-k32-gpu1/quality}"
for suite in main combined; do
  if [[ "$suite" == "main" ]]; then
    export FUSE_DECODE_QK_ROPE_CACHE=0 EXECUTION_MODE=original
    workloads=(balanced decode prefill long8k long16k)
    backends=(flash triton int8)
  else
    export FUSE_DECODE_QK_ROPE_CACHE=1 EXECUTION_MODE=buffered
    workloads=(decode long8k)
    backends=(flash int8)
  fi
  for workload in "${workloads[@]}"; do
    for ((repeat=1; repeat<=runs; repeat++)); do
      for ((offset=0; offset<${#backends[@]}; offset++)); do
        backend="${backends[$(((repeat-1+offset)%${#backends[@]}))]}"
        export ATTENTION_BACKEND=triton KV_CACHE_DTYPE=auto
        [[ "$backend" != "flash" ]] || export ATTENTION_BACKEND=flash
        [[ "$backend" != "int8" ]] || export KV_CACHE_DTYPE=int8
        export RESULT_DIR="$root/$suite/$backend" RUNS=1 RUN_START="$repeat"
        mkdir -p "$RESULT_DIR"
        case "$workload" in
          balanced) bash benchmarks/scripts/run_balanced.sh ;;
          decode) bash benchmarks/scripts/run_decode_heavy.sh ;;
          prefill) bash benchmarks/scripts/run_prefill_heavy.sh ;;
          long8k|long16k)
            batch=8; length=8192
            [[ "$workload" != "long16k" ]] || { batch=4; length=16384; }
            extra=()
            [[ "$FUSE_DECODE_QK_ROPE_CACHE" != "1" ]] || extra+=(--fuse-decode-qk-rope-cache)
            "$PYTHON_BIN" bench.py --model "$model" --attention-backend "$ATTENTION_BACKEND" \
              --kv-cache-dtype "$KV_CACHE_DTYPE" --execution-mode "$EXECUTION_MODE" \
              --measurement-mode runtime --warmup-decode-steps 32 --warmup-runs 1 --seed 0 \
              --num-prompts "$batch" --max-num-seqs "$batch" --input-len "$length" --output-len 256 \
              --max-model-len "$((length+256))" "${extra[@]}" \
              --output-json "$RESULT_DIR/$workload-$repeat.json" ;;
        esac
      done
    done
  done
  "$PYTHON_BIN" benchmarks/compare.py "$root/$suite/flash" "$root/$suite/int8" --output "$root/$suite/flash-vs-int8.md"
  if [[ "$suite" == "main" ]]; then
    "$PYTHON_BIN" benchmarks/compare.py "$root/$suite/flash" "$root/$suite/triton" --output "$root/$suite/flash-vs-triton.md"
  fi
done

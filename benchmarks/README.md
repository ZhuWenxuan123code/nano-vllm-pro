# Benchmark Workloads

`scripts/` 保存可复现的 benchmark 工作负载，`results/` 保存本地生成的 JSON
结果。结果目录由 Git 忽略；需要长期保留的汇总和结论应写入 `docs/`。

所有脚本默认使用物理 GPU 1、`~/huggingface/Qwen3-0.6B`、运行 5 次，并将
结果写入 `benchmarks/results/baseline/`。可通过环境变量覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/Qwen3-0.6B \
RUNS=3 \
bash benchmarks/scripts/run_random.sh
bash benchmarks/scripts/run_balanced.sh
bash benchmarks/scripts/run_decode_heavy.sh
bash benchmarks/scripts/run_prefill_heavy.sh
```

优化后使用独立结果目录运行同一工作负载：

```bash
RESULT_DIR=benchmarks/results/p1-rmsnorm \
bash benchmarks/scripts/run_decode_heavy.sh
```

P1 RMSNorm 对照可设置 `RMS_NORM_BACKEND=compiled`（默认）或 `triton`；
`bench.py --rms-norm-backend` 也接受这两个值，结果 JSON 会记录实际后端。
单算子延迟使用 `benchmarks/bench_rmsnorm.py`，warp 扫描使用
`benchmarks/tune_rmsnorm.py`；两者生成的 JSON 保存在忽略提交的 `results/` 中。
三组固定负载完整 A/B（各 5 次）可运行
`bash benchmarks/scripts/run_p1_rmsnorm.sh`，并通过 `PYTHON_BIN`、
`CUDA_VISIBLE_DEVICES`、`RUNS`、`RESULT_ROOT` 覆盖默认环境。

P2 Decode Q/K Norm + RoPE + KV 写入融合可通过
`bench.py --fuse-decode-qk-rope-cache` 启用；默认仍用原路径，Prefill 不变。
单层 CUDA Graph 对照：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_decode_qkv.py \
  --output-json benchmarks/results/p2-decode-qkv-gpu1/prototype-b64.json
```

完整原/新路径 A/B（固定 `RMS_NORM_BACKEND=compiled`）使用
`CUDA_VISIBLE_DEVICES=1 RESULT_ROOT=benchmarks/results/p2-decode-qkv-gpu1 bash benchmarks/scripts/run_p2_decode_qkv.sh`。现有 profiling 脚本也接受
`FUSE_DECODE_QK_ROPE_CACHE=1`；请给两组设置不同的 `PROFILE_DIR`，
避免混淆报告。

汇总某个目录中的多次结果：

```bash
python benchmarks/summarize.py benchmarks/results/baseline
```

比较优化前后结果并生成 Markdown 表格；吞吐指标越高越好，延迟指标越低越好：

```bash
python benchmarks/compare.py \
  benchmarks/results/baseline \
  benchmarks/results/p1-rmsnorm \
  --output benchmarks/results/p1-rmsnorm/comparison.md
```

脚本应从任意工作目录正常执行。请保持优化前后的模型、硬件、依赖、随机种子和
命令参数一致。

## Prefill / Decode Profiling

`profile_workload.py` 会先完成模型初始化和 workload warmup，再单独采集指定阶段。
Torch Profiler 默认生成 Chrome Trace、JSON 汇总和文本表格：

```bash
CUDA_VISIBLE_DEVICES=2 # 默认使用 device 2
INPUT_LEN=1024
bash benchmarks/scripts/profile_torch.sh prefill
bash benchmarks/scripts/profile_torch.sh decode
ENFORCE_EAGER=1 bash benchmarks/scripts/profile_torch.sh decode
```

Nsight Systems 通过 `cudaProfilerStart/Stop` 只捕获目标 step，不包含模型加载和
CUDA Graph capture：

```bash
INPUT_LEN=1024
bash benchmarks/scripts/profile_nsys.sh prefill
bash benchmarks/scripts/profile_nsys.sh decode
ENFORCE_EAGER=1 bash benchmarks/scripts/profile_nsys.sh decode
```

默认结果分别位于 `benchmarks/profiles/baseline/torch/` 和
`benchmarks/profiles/baseline/nsys/`。`PROFILE_DIR` 指定两者共同的父目录；该目录由
Git 忽略，需要提交的热点结论应整理到 `docs/性能.md`。常用环境变量包括：

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=~/huggingface/Qwen3-0.6B \
PROFILE_DIR=benchmarks/profiles/p1-rmsnorm \
ENFORCE_EAGER=1 \
BATCH_SIZE=64 INPUT_LEN=128 PROFILE_STEPS=20 \
bash benchmarks/scripts/profile_torch.sh decode
```

`ENFORCE_EAGER=0` 表示 Decode 使用 CUDA Graph，设为 `1` 表示 Eager。Prefill 始终
走 Eager 路径。Torch 的 `*.summary.json` 提供 CUDA 时间占比、kernel activity 和
launch API 次数；Nsight 的 `*.nsys.csv` 提供更权威的 kernel/API 汇总。Profiler
会引入额外开销，只用于热点归因；端到端性能仍以 `bench.py` 为准。

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

## P3 输入缓冲对照

`EXECUTION_MODE=buffered` 启用持久化 Decode 输入；默认 `original`。
`MEASUREMENT_MODE=runtime` 仅在 benchmark 整轮边界显式同步。

```bash
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/run_p3_runner.sh
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/profile_p3_runner.sh
```

前者交替执行两种模式，保存 P2 关闭的三组负载以及 P2 开启的两组叠加对照。
后者生成三次 CUDA Graph node 时间线和 `gate.json`，用于决定是否进入异步阶段。
两者都需要空闲 GPU；性能测试不带 profiler。具体状态和计时定义见
[P3 文档](../docs/优化/P3-输入缓冲与异步执行.md)。

## P5 在线混合调度

`bash benchmarks/scripts/run_p5.sh` 在物理 GPU1 上分阶段执行在线标定、
固定预算扫描、180 次主对照和 30 次组合对照。默认结果目录为
`benchmarks/results/p5-scheduling-controlled-gpu1/`；使用 `P5_STAGE` 分阶段执行，
使用 `PYTHON_BIN` 指定解释器。完整矩阵包含长上下文，运行时间明显长于离线单次 bench。

`P5_STAGE=summarize` 可生成阶段性汇总，表头和 `coverage.json` 明确标注
尚未完成的矩阵，不能把不足五轮的档位当作最终对照。完成后再刷新汇总；
使用过自定义 `RUNS` 时，汇总也需透传相同值。

当前 GPU1 正式矩阵已完成全部 282 次，主/组合共 42 个档位各重复五次，
两份 coverage 均为 complete，并完成全结果审计。固定混合策略仅在部分负载获益，
自适应策略有明显回退；实验完成不代表所有策略满足 SLO，默认仍使用 legacy。

`benchmarks/bench_online.py --scheduler-policy legacy|mixed|adaptive` 支持
固定到达轨迹，报告真实逐 token ITL 和满足 TTFT/全部 ITL 目标的有效吞吐。
这些在线指标与旧离线 TPOT 不是同一口径。混合批次走 Eager，纯 Decode 保留 Graph/P2。
详见 [P5 实现与验收](../docs/优化/P5-混合调度与自适应预算.md)。

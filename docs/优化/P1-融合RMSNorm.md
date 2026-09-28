# P1：Triton 融合 Add + RMSNorm

## 实现与切换

算子实现在 `nanovllm/layers/rmsnorm_triton.py`，由 `nanovllm/layers/layernorm.py` 分发。普通 RMSNorm 接受 Q/K 的非连续视图；Add + RMSNorm 一次读取 `x` 与 `residual`，返回归一化输出和新的 residual。归约及残差求和使用 FP32，输入/输出支持 FP16、BF16。保持现有 `torch.compile` 路径为默认实现，显式传入 `--rms-norm-backend triton` 才启用新算子；所有 TP rank 使用同一后端，benchmark JSON 记录后端名称。未实现 backward。

## 正确性与调优

运行 `python -m unittest discover -s test -p 'test_*.py'`（本次 19 项通过）。测试覆盖两种 dtype、不同 token 数、非连续 Q/K 及 residual、输入不被修改、CUDA Graph 捕获/重放；使用 PyTorch 计算为参考，以 FP16 `atol=rtol=1e-2`、BF16 `atol=rtol=2e-2` 验证。微基准覆盖的形状中，Triton 对 PyTorch eager/`torch.compile` 的最大输出绝对误差均为 FP16 0.001953125、BF16 0.015625；返回 residual 的误差为 0。另用实际 Qwen3-0.6B 完成单卡 CUDA Graph 和 TP=2 Eager 冒烟测试。

单算子基准：

```bash
CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_rmsnorm.py \
  --output-json benchmarks/results/p1-rmsnorm/micro-tuned.json
CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_rmsnorm.py \
  --layout qk --rows 64 8192 --hidden-sizes 128 --dtypes bfloat16 \
  --output-json benchmarks/results/p1-rmsnorm/micro-qk.json
CUDA_VISIBLE_DEVICES=2 python benchmarks/tune_rmsnorm.py \
  --output-json benchmarks/results/p1-rmsnorm/warp-sweep.json
```

调优前的 `micro-initial.json`、warp 扫描 `warp-sweep.json` 和调优后的 `micro-tuned.json` 均保留在本地结果目录。RTX 3090 上，H=128 普通路径选 1 warp、融合路径选 2 warps；H=1024 两条路径均选 2 warps。`BLOCK_SIZE` 为不小于 hidden size 的最小 2 的幂，列维度连续加载，便于向量化。warp 扫描中，BF16、普通路径、M=8192/H=128 的 kernel 时间从 4 warps 的 11.883 μs 降至 1 warp 的 9.290 μs；这**不是**端到端加速比。

单次 microbenchmark 中，BF16 非连续 Q/K 视图的普通 RMSNorm 在 M=64 时为 compiled 4.089 μs、Triton 4.383 μs；M=8192 时分别为 8.779 μs、9.247 μs。BF16 融合路径 M=8192/H=1024 分别为 84.035 μs、82.745 μs。微秒级差异尚需重复测量，不能单凭这些数值认定稳定收益。

## 端到端对照

在同一工作树、同一张 RTX 3090（物理 GPU 2）上分别运行 compiled 与 Triton 后端；环境为 PyTorch 2.13.0+cu130、Triton 3.7.1。三个固定负载各重复 5 次，比较中位数。Prefill 与 Decode 均使用现有脚本默认配置（Decode 使用 CUDA Graph）。不要将本轮结果直接与旧提交的 baseline 混比。

```bash
CUDA_VISIBLE_DEVICES=2 bash benchmarks/scripts/run_p1_rmsnorm.sh
python benchmarks/summarize.py benchmarks/results/p1-rmsnorm/compiled
python benchmarks/summarize.py benchmarks/results/p1-rmsnorm/triton
```

对比表由脚本生成在 `benchmarks/results/p1-rmsnorm/comparison.md`，原始 JSON 不纳入 Git。结论应以该表中 E2E 吞吐、Prefill/Decode 吞吐、TTFT/TPOT 中位数为准；若 Triton 不优于现有编译融合，需如实说明 GEMM/Attention 占比和小算子调度开销限制，不单独宣称 microbenchmark 收益。

本次 RTX 3090、Qwen3-0.6B、单卡 CUDA Graph 的结果如下；每格为五次运行的中位数，箭头表示 compiled → Triton，吞吐越高越好、延迟越低越好：

| 负载 | E2E tok/s | Prefill tok/s | Decode tok/s | TTFT P50 ms | TPOT P50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 2560.41 → 2559.08 | 56003.77 → 55914.08 | 4299.38 → 4300.15 | 2515.24 → 2525.21 | 49.52 → 49.41 |
| decode-heavy | 7485.35 → 7471.18 | 48366.38 → 47405.53 | 7778.47 → 7770.31 | 473.59 → 486.48 | 20.98 → 21.01 |
| prefill-heavy | 558.82 → 557.92 | 50824.69 → 50709.37 | 2087.44 → 2090.19 | 2764.37 → 2771.82 | 105.29 → 105.64 |

E2E 中位数依次变化 -0.05%、-0.19%、-0.16%，没有显示稳定的整机收益；decode-heavy 的 Prefill 吞吐下降 1.99%，TTFT P50 增加 12.89 ms。TTFT P99 中位数分别为 9022.62 → 9036.45、677.54 → 691.27、6385.69 → 6398.60 ms。现有基线已经通过 `torch.compile` 融合 RMSNorm；两条路径结构上都是单 kernel，单纯替换不应显著降低 launch 次数，但本轮并未重新用 Nsight 统计。结合此前 GEMM/Attention 占较大 CUDA 时间的 profiling，端到端收益受限是合理解释，而非本轮单独证明的因果结论。P1 的成果应描述为可切换的算子实现、数值/Graph/TP 验证和完整的消融分析，**不应写成性能提升**。

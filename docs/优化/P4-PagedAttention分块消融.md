# P4：Paged Decode Attention 分块消融

## 问题与结论

2026-10-12～13 在 GPU1（RTX 3090）验证：将 Triton 的 query/KV tile 从当前 16×32 改成原 FlashAttention 路径的 64×128 后，单层延迟明显接近 FlashAttention。分块配置是所测 Decode 形状下性能差异的重要因素；不能将收益解释为 Triton 语言天然比 CUDA 快，或 FlashAttention 不具备 GQA/在线 softmax。

本实验用于解释此前约 12.5% 的端到端收益，不重新测量端到端吞吐，也不把单层延迟下降率直接替代简历里的吞吐提升。

## 控制变量与测量方法

- GPU1 UUID：`GPU-e9e4afed-ebf4-e4ec-28e4-598dada14eac`；PyTorch 2.13.0+cu130、CUDA 13.0、Triton 3.7.1、FlashAttention 2.8.3.post1。
- BF16，B=64，Q/KV heads=16/8，head_dim=128，page_size=256；两边固定 split=1、4 warps。FlashAttention 显式传 `num_splits=1`，Nsight 确认实际 kernel 的 M/N=64/128。
- 相同 Q/K/V、分页表与长度；每个长度按 seed=41+length 生成数据，同一长度内五组共享输入。均匀长度、连续 Q、顺序物理页，不等价于所有真实调度状态。
- 对照组为 FlashAttention；Triton 使用 M/N=16/32、64/32、16/128、64/128 四组，以分别改变 M、N 并观察交互。
- 所有组先编译预热并通过 FP32 PyTorch 参考、CUDA Graph 捕获/重放及输入不变检查，再开始正式计时。每个配置五轮，各组轮换执行顺序；每轮 `do_bench_cudagraph(rep=100, return_mode="median")`，报告五轮中位数，原始 JSON 保留每轮值及样本标准差。
- 主实验长度顺序为 128/129/256/257/512/640，第二个独立进程以逆序复验。所有 profiling 独立执行，不混入正式延迟。

## 主实验：单层延迟

单位 μs，越低越好。列名为 query tile M × KV tile N，所有组均为 4 warps、split=1。

| 上下文长度 | Flash 64×128 | Triton 16×32 | Triton 64×32 | Triton 16×128 | Triton 64×128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 53.32 | 42.13 | 48.69 | 46.74 | 53.48 |
| 129 | 83.02 | 45.32 | 57.91 | 59.66 | 82.87 |
| 256 | 97.56 | 80.26 | 93.81 | 84.04 | 94.93 |
| 257 | 129.20 | 83.53 | 104.35 | 96.44 | 127.03 |
| 512 | 193.45 | 156.90 | 190.98 | 164.10 | 185.35 |
| 640 | 237.30 | 195.73 | 231.47 | 203.77 | 228.26 |

当前 16×32 路径相对 Flash 延迟下降 17.52%～45.41%；对齐 64×128 后为 -0.31%～4.18%（负数表示变慢）。两次单变量改动在这六个形状均慢于当前 16×32，但 M/N 存在交互，不能把各自改善相加。

129/257 是有意义的边界点：KV tile=128 对应的覆盖长度分别向上取整为 256/384，而 tile=32 为 160/288。较大 tile 的尾部无效位置更多；masked load 不意味着这些位置都会产生真实 HBM 读取，也不能由此直接计算硬件指令或流量节省。

## 逆序复验

逆序复验同样支持差距收窄，完整结果如下，单位仍为 μs：

| 上下文长度 | Flash 64×128 | Triton 16×32 | Triton 64×32 | Triton 16×128 | Triton 64×128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 59.10 | 42.50 | 50.86 | 47.05 | 57.27 |
| 129 | 93.00 | 46.16 | 60.72 | 64.61 | 88.17 |
| 256 | 104.04 | 80.49 | 97.32 | 85.57 | 100.50 |
| 257 | 137.35 | 83.86 | 108.49 | 100.80 | 132.30 |
| 512 | 184.58 | 156.61 | 184.19 | 163.73 | 181.08 |
| 640 | 223.67 | 196.59 | 229.88 | 202.73 | 219.35 |

逆序复验中当前配置相对 Flash 延迟下降 12.11%～50.36%，对齐后为 1.89%～5.19%。两次合计 60 个形状/实现组、300 个轮次测量值；全部数值和 Graph 检查通过，FP32 参考最大绝对误差约 0.003697。不同实验批次绝对延迟有变化，结论以同批次对照为准，不将两批数值混算为新的端到端提升。

## Nsight：资源配置证据

L=129，每组 20 次 Graph replay。五组均记录 20 个 Attention kernel、每次 512 个 thread blocks、每 block 128 个线程；没有额外 split 合并 kernel。

| 路径 | 每线程寄存器 | 每 block 共享内存 | 报告的 local memory/thread |
| --- | ---: | ---: | ---: |
| Flash 64×128 | 244 | 80 KiB | 0 |
| Triton 16×32 | 64 | 21 KiB | 0 |
| Triton 64×32 | 154 | 36 KiB | 0 |
| Triton 16×128 | 96 | 72 KiB | 0 |
| Triton 64×128 | 170 | 96 KiB | 0 |

同一 Triton 实现增大 tile 后，寄存器和共享内存需求均增加。这为“较小分块降低资源压力”提供直接证据，也为更多 block 同时驻留提供可能；实际占用率仍需计数器验证。对齐 tile 后资源需求并未完全相同，说明相同 M/N/warps 不等于相同布局、流水线和生成指令。

Nsight Compute 尝试返回 `ERR_NVGPUCTRPERM`，未取得性能计数器权限；未修改系统权限。不能据本次实验宣称测得实际 occupancy、Tensor Core 利用率、指令数量或 HBM 流量。Nsight Systems 的 kernel/资源记录有效，其带 profiler 时间不用于上表。

## 代码、验证与复现

- 核心包装层 `nanovllm/layers/paged_attention.py` 新增可选 `block_m`，供离线实验覆盖 query tile；默认选择规则和模型调用不变。
- `benchmarks/bench_attention_tiles.py` 提供五组对照、FP32 数值验证、Graph 重放、输入不变检查和独立 profiling 模式，失败形状写入 JSON 并返回非零状态。
- `test/test_paged_attention.py` 增加 query tile 覆盖：M=16/32/64、split=1/4，包含长度零、跨页、非连续 Q、非顺序页；本轮四项测试全部通过，原有 dtype/head/GQA 与 Graph 回归通过。

从仓库根目录运行（需激活 nano-vllm 环境，GPU1 空闲）；请给新实验使用不同目录：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_attention_tiles.py \
  --output-json benchmarks/results/p4-tiles-new/tiles.json
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_attention_tiles.py \
  --lengths 640 512 257 256 129 128 \
  --output-json benchmarks/results/p4-tiles-new/tiles-reverse.json
CUDA_VISIBLE_DEVICES=1 python -m unittest discover -s test -p 'test_paged_attention.py' -v

mkdir -p benchmarks/profiles/p4-tiles-new
CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=benchmarks/profiles/p4-tiles-new/tiles-l129 \
  python benchmarks/bench_attention_tiles.py --lengths 129 --profile nsys \
  --output-json benchmarks/profiles/p4-tiles-new/nsys-l129.json
```

原始计时数据为 `benchmarks/results/p4-tiles-gpu1-20261012/tiles.json` 与 `tiles-reverse.json`；性能分析在 `benchmarks/profiles/p4-tiles-gpu1-20261012/`。JSON 保存软件版本、GPU UUID、脚本和 kernel 文件哈希。新结果目录不入 Git。

本机未锁时钟，其他卡有任务，存在共享主机环境和温度漂移；不把对齐后几个百分点的残余差异解释为普适实现优势。形状数量有限，当前结论限于所测单层 BF16 Decode。端到端约 12.5% 来自 [P4 完整模型实验](P4-INT8-KV与PagedAttention.md)，本轮没有以修改后的 64×128 配置重新跑整模型，不能给出它对应的端到端吞吐。

# P4：INT8 KV Cache 与融合反量化的 Paged Attention

## 当前结论：高精度路径提速，INT8 获得容量收益

P4.1～P4.3、单卡/TP=2 正确性、固定真实文本质量关卡、95 次端到端对照、容量扩展负载和 Nsight 分析均已完成。默认仍为 FlashAttention + 模型原始精度缓存，没有加入高精度保留缓存或自动提交 Git。

原 token × head 方案未通过 1% 门槛；经用户确认，K 改为每 32 维一组，V 保留每 token × KV head 一个 scale。当前 `int8-k32-vhead-v1` 的 2K/8K 固定子集续写困惑度分别退化 **0.3244%/0.6971%**，通过关卡。高精度 Triton 的 decode-heavy E2E 中位数提升 **12.52%**；INT8 每 token KV 存储（含 scale）减少 **46.09%**，同预算槽位增加 **85.65%**，但同负载吞吐回退。两类收益分别报告，不把容量收益写成速度收益。

主环境：物理 GPU1（第二张 RTX 3090，24 GiB），UUID `GPU-e9e4afed-ebf4-e4ec-28e4-598dada14eac`；Qwen3-0.6B，28 层、16 Q heads、8 KV heads、head_dim=128、BF16；PyTorch `2.13.0+cu130`。主对照固定 P2 关闭、P3 `original`、RMSNorm `compiled`。`CUDA_VISIBLE_DEVICES=1` 后，进程内使用 `cuda:0`；已用 UUID 核实对应关系。

## 实现与核心代码

- `nanovllm/layers/paged_attention.py`：按序列/KV head 复用同组 Q heads，在线 softmax、FP32 累加；Decode 支持单段及 split-KV 归约。INT8 在 kernel 内反量化并转换为 Q dtype，不生成完整高精度缓存。缓存前缀/chunked Prefill 使用分页因果 Attention，query 的绝对位置为 `KV长度 - query长度 + query索引`。
- `nanovllm/layers/kv_quantization.py`：对称 INT8；K scale 为 `[blocks,page,kv_heads,D/32]`，V scale 为 `[blocks,page,kv_heads]`，均为 FP32。`maxabs/127`、最近偶数舍入、限制 `[-127,127]`，全零组 scale=1。精确 FP32 除法避免倒数乘法改变舍入边界；旧 scale 布局显式拒绝。
- `nanovllm/layers/decode_qkv_fused.py`：P2 直接量化写入 K/V 和 scale；K 在 Norm/RoPE 后先舍入到模型 dtype，再量化。无效 slot 不写数据或 scale。
- `nanovllm/layers/attention.py`：无前缀 Prefill 保留连续高精度 FlashAttention，同时写 INT8 缓存；有前缀时读取分页 INT8。
- `nanovllm/engine/model_runner.py`：按 data + scale 计算缓存预算；预留一个所有层顺序复用的 FP32 workspace，数据/scale 共用物理 block 编号和生命周期。Graph 固定 workspace 地址与 split 数，不读取 GPU 长度到 CPU。

新增独立开关：`--attention-backend flash|triton`，默认 `flash`；`--kv-cache-dtype auto|int8`，默认 `auto`。INT8 + FlashAttention、未知后端及不支持的 Triton head/GQA/dtype/TP 配置在初始化时报错。`bench.py`、固定负载和 profiling 脚本记录并透传开关；profile 名称包含新后端和格式，避免覆盖原路径。

## 正确性验证与误差边界

测试覆盖 FP16/BF16、head_dim 64/128/256、GQA 1/2/4/8/16、非连续输入、长度 0/1/255/256/257、物理页乱序、未写入 slot、舍入边界、零值与离群值、Graph 捕获/重放。高精度对照 FlashAttention 和小形状 FP32 PyTorch；INT8 对照**同一量化数据显式反量化后的参考**，不把量化损失算成 kernel 实现误差。

模型测试使用 teacher forcing，覆盖跨页、释放/复用、前缀命中、强制抢占重计算以及 P2/buffered/Graph。多层 BF16 舍入误差可能累积，检查 logits RMS 和绝对误差，同时按每个 KV 向量的幅度检查缓存误差；这不替代真实文本质量关卡。P2 的融合量化还单独对照其高精度输出，验证整数及 scale 一致。

一次 P2+INT8 模型对照出现 logits RMS=0.1618（超过 0.08）。定位到当前 Inductor 的 `emulate_precision_casts=False`：它省略 RMSNorm 权重乘法前的中间 BF16 舍入，而旧 P2 kernel 显式保留该舍入；模型 K Norm 的离群权重会放大差异。INT8 融合路径现按所选 RMSNorm 后端/Inductor 配置匹配该行为，修复后重新通过模型对照；旧 P2 高精度路径不变。不同 PyTorch 版本或编译回退需要重新验证，不能只按 Python 表面代码推断实际舍入。

GPU2 空闲后，使用物理 GPU1+GPU2 完成 TP=2 验证：K32 + P2 + buffered + Graph 冒烟通过，两个 rank 均为 Triton/K32，分别持有 4 个 KV heads，固定 12 blocks，生成结束全部归还。另用正常分配器、0.1 显存预算分别跑 Triton BF16/INT8 短请求，分配 94/174 blocks，每 rank 为 57344/30912 B/token，均完成 3×4 token 生成，覆盖初始化的跨 rank 最小容量归约；这只是分配/生成冒烟，不是 TP 性能对照。正常分配冒烟的极小 Prefill budget=128 触发了 RMSNorm 编译次数限制回退，日志保留；因此也不将其视为量化质量验收。

CPU 测试发现 67 项（40 项执行通过、27 项需要 GPU/模型而跳过）；设置本地模型和两张卡后，完整 67 项全部通过，无跳过，包含 P1/P2/P3 回归和两个 TP=2 用例。新增分页/量化算子测试共 9 项通过。旧 RMSNorm/P3 用例受到共享编译缓存或编译次数限制影响，现每用例重置缓存并临时提高测试编译上限，最终全套无编译回退；推理默认配置未改动。集成数值边界：logits RMS <0.08，逐元素 `atol=0.4/rtol=0.03`；缓存按 KV 向量最大幅度（至少 1）归一化，最大误差 <0.06、RMS <0.005。算子测试使用更紧的 FP16/BF16 dtype 容差；量化数据和 scale 的写入测试要求与参考一致。语法编译、shell 语法和 `git diff --check` 通过。

## 固定真实文本质量结果

数据来源：[WikiText-2 raw test](https://huggingface.co/datasets/Salesforce/wikitext)，固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`。按原行顺序以两个换行拼接，用本地模型 tokenizer 编码，不加特殊 token；取前 65536 token。token 文件 SHA256 为 `9fdc95dd09b48c59da36c1969501c2404c55ff6eee145b23a56340968e54ca75`，manifest 同时保存文本、原始数据和 tokenizer 文件哈希。数据在忽略提交的 `benchmarks/results/p4-data/`。

每种长度使用非重叠窗口，前半段为 prompt，后半段逐 token teacher forcing；每条路径、每种长度评分 32768 个实际预测目标。下表仅表示**该固定子集的续写困惑度**，不是完整 WikiText-2 标准困惑度；参考模型均来自本项目，没有 Transformers 模型对照。

### 当前 K32 格式

同一数据、窗口和评分定义，所有路径重新运行。结果位于 `benchmarks/results/p4-k32-gpu1/quality/`，旧报告不会用于当前格式的关卡检查。

| 总窗口长度 | Flash BF16 | Triton BF16 | Triton INT8 K32 | INT8 相对 Flash | ≤1% 门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 | 18.43624 | 18.43442 | 18.49604 | +0.3244% | 通过 |
| 8192 | 16.87978 | 16.88256 | 16.99746 | +0.6971% | 通过 |

### 原 token × head 格式（保留负结果）

| 总窗口长度 | Flash BF16 | Triton BF16 | 原 Triton INT8 | INT8 相对 Flash | ≤1% 门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048 | 18.43624 | 18.43442 | 18.74118 | +1.6540% | 未通过 |
| 8192 | 16.87978 | 16.88256 | 17.15530 | +1.6323% | 未通过 |

### 原 token × head 的 K/V 量化消融

使用同一 2K 子集和高精度缓存，写入前仅对指定 K/V 做原 token × head 格式的量化/反量化，以隔离量化损失与 INT8 内核。**这是诊断工具，不是新增运行时后端或高精度保留方案。** 复现该旧消融时需要 `--diagnostic-k-group-size 128`；诊断工具默认已改为 K32。

| 路径 | 续写困惑度 |
| --- | ---: |
| Triton BF16，无量化 | 18.43442 |
| 仅 K 量化/反量化 | 18.73876 |
| 仅 V 量化/反量化 | 18.43327 |
| K/V 都量化/反量化 | 18.74648 |
| 实际 INT8 缓存 | 18.74118 |

退化主要来自 K 量化：只量化 K 已重现几乎全部退化，只量化 V 没有同量级损失；高精度 roundtrip 与真实 INT8 结果接近。模型第 0 层 K Norm 权重最大值为 96.5，说明同一 head 的通道幅度可能非常不均衡。结合严格算子对照，当前证据指向 token × head 的 K scale 精度不足，而非单纯 INT8 Attention 实现错误；不能据此声称穷尽了所有形状的错误。

用户已确认 K32 调整，1% 门槛保持不变。缓存格式在实例初始化时固定，不在请求途中切换。

补充真实文本回归：2048 窗口取第一个文本窗口，prompt=1024、续写评分 128 token，Prefill budget=128，先填充并释放同一前缀再请求，确认分块与前缀缓存读取完成且 blocks 全部归还。当前 Flash/Triton/K32 INT8 的续写困惑度分别为 5.21213/5.17933/5.22014；原 token × head INT8 为 5.20761。此小样本不是正式质量关卡，不替代上述 32768-target 结果。

## P4.1 初始测量与调参记录

原始数据：`benchmarks/results/p4-gpu1/p41-initial.json`（20 个 B/长度形状）及 `p41-tuning.json`（9 个形状 × 30 个配置，完整保留，无失败形状）。下面是调参前的示例，3 轮 CUDA Graph 测量，各轮使用多次 replay；不是最终 5 轮端到端结果。

初始脚本每轮使用 `do_bench_cudagraph` 默认 mean，再取三轮中位数；当前复现脚本已显式指定每轮 median。初始数据保留原定义，不将它与后续统计定义混合计算提升。

| B | 上下文 | Flash μs | Triton μs | split 数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 12.13 | 5.97 | 1 |
| 1 | 512 | 12.72 | 17.83 | 1 |
| 8 | 2048 | 99.75 | 82.12 | 2 |
| 64 | 8192 | 2660.84 | 2568.37 | 1 |

有收益也有回退，不用单个形状概括整体。离线搜索 `tile=32/64/128`、`warps=4/8`、`splits=1/2/4/8/16`；跨 9 个形状，tile=32/warps=4 在各形状选择最合适 split 后，相对逐形状最优的几何平均差距约 1.9%。因此首版固化该 tile/warps 和元数据驱动 split 规则，不在计时/Graph 捕获中 autotune。这仍是有限形状的选择，不承诺跨硬件最优。

### 当前 K32 单层矩阵

`benchmarks/results/p4-k32-gpu1/` 下 `tuning.json` 保留 270 组配置，`layers.json` 保留 20 个主形状，`variants.json` 保留 32 个 head/GQA 变体。全部数值对照通过，无失败形状。每个配置五轮测量、轮换后端顺序，每轮 CUDA Graph 多次 replay 取 median，再取五轮中位数；不在计时中调参。INT8 实现误差对照同一量化缓存显式反量化后的 FlashAttention。

| B | 上下文 | Flash BF16 μs | Triton BF16 μs | Triton K32 INT8 μs | split 数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 10.45 | 6.07 | 18.62 | 1 |
| 1 | 8192 | 60.21 | 43.21 | 115.42 | 16 |
| 8 | 8192 | 331.78 | 310.60 | 868.03 | 2 |
| 64 | 128 | 59.97 | 42.48 | 121.22 | 1 |
| 64 | 8192 | 3003.98 | 2438.98 | 6565.89 | 1 |

当前 INT8 在主矩阵 20 个形状中均慢于 Flash BF16，存储减少没有转化为单层提速。新增 scale 加载、INT8 转换和反量化都有计算/访存成本；本实现的 K32 scale 加载布局及 tile/split 规则还有优化空间，目前不能把回退单一归因于带宽或某一项开销。高精度 Triton 的部分单层收益也不能直接写成整机收益。

新 270 组搜索中，tile=32/warps=4 **允许每形状单独选择最优 split 时**，相对逐形状最优的几何平均差距约 1.67%；这不是当前默认规则的整体误差。默认规则在 B=1/L=128 选择 split=1（6.02 μs），而该轮 split=4 达到 3.81 μs，说明短序列规则并非最优。当前端到端实验沿用已验证的固定规则，不根据正在运行的结果临时改参。

Triton Decode 主 kernel 为 1 次，split>1 时另有 1 次归约；本轮 Nsight 的 B=64/split=1 实测每层 Attention 每步一个 kernel，其他 split 的两次 launch 属于实现结构说明，不混作这轮 profiler 数据。

## 五轮同负载端到端结果

主实验结果位于 `benchmarks/results/p4-k32-gpu1/e2e/main/`，5 负载 × 3 后端 × 5 轮共 75 份 JSON。每轮轮换后端顺序，同一物理 GPU1、同一推理源代码 SHA256 `1b41884dfe7a65e4a51988e8eda0aa38f6e1c17019201095710e78a251c461f8`。采用 runtime 模式、seed=0、一次代表性预热及 32 个 Decode step，无 profiler；P2 关闭、P3 original、RMSNorm compiled，Decode 使用 CUDA Graph。以下均为五轮中位数，吞吐变化为 `新/原 - 1`，延迟下降率为 `1 - 新/原`。比较工具已修正延迟计算，不再把倒数速率提升当作延迟下降。

负载配置：balanced 为 256 请求、最大 batch=64、1024→128；decode-heavy 为 256/64/128→512；prefill-heavy 为 128/32/2048→32；long8k 为 8/8/8192→256；long16k 为 4/4/16384→256。这里 batch 指每步最大序列数，不是请求总数，也不保证各缓存格式有相同驻留请求数。

| 负载 | Flash BF16 E2E tok/s | Triton BF16 | 相对 Flash | Triton K32 INT8 | 相对 Flash |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 2536.09 | 2715.74 | +7.08% | 1781.77 | -29.74% |
| decode-heavy | 7442.31 | 8374.09 | +12.52% | 4957.65 | -33.39% |
| prefill-heavy | 557.31 | 582.22 | +4.47% | 470.73 | -15.54% |
| long8k | 395.05 | 412.86 | +4.51% | 229.86 | -41.82% |
| long16k | 171.25 | 173.47 | +1.30% | 103.76 | -39.41% |

| 负载 | 后端 | Prefill tok/s | Decode tok/s | TTFT P50 / P99 ms | TPOT P50 / P99 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced | Flash BF16 | 55712.83 | 4245.77 | 2475.02 / 9098.73 | 50.23 / 66.04 |
| balanced | Triton BF16 | 55801.70 | 4774.41 | 2472.48 / 8671.65 | 45.44 / 61.02 |
| balanced | K32 INT8 | 55665.82 | 2377.11 | 2494.66 / 4709.38 | 84.02 / 114.71 |
| decode-heavy | Flash BF16 | 59605.02 | 7673.83 | 343.25 / 549.79 | 21.18 / 33.39 |
| decode-heavy | Triton BF16 | 59634.83 | 8669.97 | 342.48 / 549.53 | 18.86 / 29.56 |
| decode-heavy | K32 INT8 | 59033.96 | 5057.06 | 346.51 / 555.11 | 31.52 / 50.65 |
| prefill-heavy | Flash BF16 | 50783.38 | 2067.56 | 2718.07 / 6397.95 | 107.14 / 138.10 |
| prefill-heavy | Triton BF16 | 50809.91 | 2476.64 | 2718.05 / 6239.76 | 101.86 / 133.03 |
| prefill-heavy | K32 INT8 | 50707.83 | 1123.86 | 2736.23 / 5169.80 | 149.75 / 185.14 |
| long8k | Flash BF16 | 33087.70 | 636.81 | 1236.99 / 1980.70 | 15.48 / 18.39 |
| long8k | Triton BF16 | 33144.91 | 683.72 | 1233.70 / 1977.28 | 14.62 / 17.53 |
| long8k | K32 INT8 | 32956.11 | 294.72 | 1241.17 / 1988.61 | 30.08 / 33.01 |
| long16k | Flash BF16 | 22248.49 | 336.30 | 1837.71 / 2923.49 | 16.24 / 20.48 |
| long16k | Triton BF16 | 22260.92 | 344.87 | 1836.13 / 2921.88 | 15.95 / 20.19 |
| long16k | K32 INT8 | 22173.04 | 147.56 | 1843.35 / 2933.42 | 31.47 / 35.74 |

主实验各组 E2E 五轮样本标准差/均值为 0.06%～1.19%。完整八指标对比保留在 `flash-vs-triton.md` 和 `flash-vs-int8.md`。TTFT 为离线请求入队至观察到首 token 的时间；TPOT 是每请求首末 token 间的平均间隔，**TPOT P99 不是逐 token ITL P99**。Prefill/Decode 吞吐按完成 step 的墙钟时间统计，不是 profiler 的纯 kernel 时间。

高精度 Triton 的 decode-heavy E2E +12.52% 是本配置的整模型收益，不是所有场景的保证。INT8 在五组同负载中均回退，不推荐作为速度优化默认开启。INT8 的 balanced/prefill-heavy TTFT P99 更低，但更多 slots 改变了驻留与重计算：两条 BF16 路径分别处理 293408/275976 个 Prefill token，INT8 均为原始 262144 token。因此这些延迟改善不能全部归因于 kernel；同预算容量与单层延迟必须分别解释。16K 这里只测了合成负载性能，未验证 16K 真实文本困惑度。

### P2 + buffered 组合

另固定 P2 开启、P3 buffered，decode-heavy/long8k 各 5 轮，对照同样组合下的 Flash BF16，而不是把各阶段独立收益相加。结果位于 `e2e/combined/`，共 20 份 JSON，E2E 样本标准差/均值为 0.12%～1.22%。

| 负载 | 后端 | E2E tok/s | Prefill tok/s | Decode tok/s | TTFT P50 / P99 ms | TPOT P50 / P99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| decode-heavy | Flash BF16 | 7772.32 | 59611.39 | 8026.56 | 343.21 / 549.72 | 20.30 / 31.92 |
| decode-heavy | K32 INT8 | 5165.80 | 59207.80 | 5273.68 | 345.52 / 553.47 | 30.49 / 48.57 |
| long8k | Flash BF16 | 402.07 | 33253.74 | 653.58 | 1229.98 / 1970.81 | 15.15 / 18.05 |
| long8k | K32 INT8 | 233.01 | 33139.16 | 299.50 | 1234.86 / 1977.62 | 29.62 / 32.54 |

INT8 相对组合高精度基线的 E2E 分别为 -33.54%/-42.05%，P2/buffered 没有消除 Attention 反量化路径的回退。正式 32768-target 质量关卡固定 P2 关闭/P3 original；组合只补了短请求 logits/cache 正确性与这里的性能，不宣称已经通过相同规模的组合困惑度关卡。

## Nsight：实际 kernel 数与 CUDA 时间

结果位于 `benchmarks/profiles/p4-k32-gpu1/`，Nsight Systems 2025.3.2，`--cuda-graph-trace=node`，以 `cudaProfilerStart/Stop` 限定 capture。20 个完整模型 Decode step，B=64、初始 prompt=128，P2 关闭/P3 original；SQLite 的 `CUPTI_ACTIVITY_KIND_KERNEL` 按 `end-start` 汇总，图内外均计数。下面是**带 profiler 的累计 kernel CUDA 时间**，不是端到端墙钟时间或无 profiler 延迟。

| 路径 | 总 kernels | 图内 / 图外 | 总 kernel CUDA ms | Attention kernels / CUDA ms | Attention 时间占比 | KV 写入 CUDA ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash BF16 | 8700 | 8440 / 260 | 110.143 | 560 / 46.360 | 42.09% | 0.823 |
| Triton BF16 | 8700 | 8440 / 260 | 90.211 | 560 / 26.844 | 29.76% | 0.807 |
| K32 INT8 | 8700 | 8440 / 260 | 132.014 | 560 / 68.221 | 51.68% | 1.252 |

每步 435 kernels（图内 422 + 图外 13）；Attention 为 28 层 × 20 步=560，不把一个 Graph 当成一个 kernel。P4 的 B=64/split=1 只是替换 Attention 和缓存写入 kernel，**本轮没有减少 kernel 数**。高精度收益主要对应 Attention 执行时间缩短；INT8 的主要时间回退出现在 Attention，而不是独立量化写入。

独立单层 B=64/L=8192 也分别捕获 20 次 replay，三后端均为 20 kernels；累计 CUDA 时间分别为 Flash 53.673 ms、Triton BF16 48.641 ms、INT8 114.619 ms。它们不替代五轮无 profiler 单层延迟，GPU 时钟/采样方式不同，不跨两套测量定义计算加速比。

模型 Attention 的 profiler 元数据中，Triton BF16/INT8 每线程寄存器数为 72/128，共享内存为 21504/29824 B；独立长上下文单层为 64/128 个寄存器。可确认 INT8 资源需求增加，但没有采集实际 occupancy、HBM 流量或指令级瓶颈，不能断言具体瓶颈或将 KV 存储缩减等同于实际 HBM 访问缩减。CPU context switch tracing 不支持的警告只禁用该项 CPU 跟踪，本轮 CUDA kernel/node 数据正常生成。

## 存储与当前分配观察

当前 K32：INT8 data 为 56 KiB/token，K scale 为 3.5 KiB，V scale 为 0.875 KiB，总计 **60.375 KiB/token（61824 B）**，相对 BF16 的 112 KiB 降低 **46.09375%**，理论容量比为 1.8551×。K/V scale 分别分配；缓存预算先预留共享 workspace 与 768 B 对齐余量。TP 初始化使用各 rank 可分配 block 数的最小值，保证同一物理 block 编号在所有 rank 有效。

### K32 同预算容量与实际准入

`benchmarks/results/p4-k32-gpu1/capacity/`：同一 GPU1、gpu_memory_utilization=0.9、max_num_seqs=64、max_model_len=8196、Prefill budget=16384，P2 关闭/P3 original。请求各自使用不同 token 构造 8192-token prompt，输出固定 4 token，避免共享前缀虚增容量。每请求最多占用 33 pages；验证所有请求同时处于 running、零抢占、全部完成并归还缓存。

| 后端 | KV KiB/token（含 scale） | blocks | token slots | 实际 8K 同时准入 | 抢占 | 归还 blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash BF16 | 112 | 711 | 182016 | 21 | 0 | 711 |
| Triton BF16 | 112 | 711 | 182016 | 21 | 0 | 711 |
| K32 INT8 | 60.375 | 1320 | 337920 | 40 | 0 | 1320 |

INT8 分配槽位相对 Flash **增加 85.65%（1.8565×）**；该特定 8K→4 负载的同时准入由 21 增至 40（+90.48%，受页数取整影响），不是吞吐提升或所有长度下的并发保证。INT8 data 为 19377684480 B、K scale 为 1211105280 B、V scale 为 302776320 B，合计 20891566080 B；两条 Triton 路径共享 workspace 均为 8519680 B（8.125 MiB），Flash 不分配该 workspace。workspace 不计入上表 KV bytes/token，已经在容量预算之前预留。

分配器仍填满显存预算，总已分配显存并没有降低 46%；降低的是每 token 的 KV 数据与 scale 存储。这里初始化的模型长度/峰值预留不同于 4096-token 固定负载，所以 711 blocks 与端到端 balanced 的 703 blocks 不矛盾。

### 原格式历史观察

以下为**原 token × head 格式的历史容量观察**，不能当作 K32 当前容量：

Qwen3 每 token 的 BF16 KV 为 `2×28×8×128×2 = 114688 B`（112 KiB）；INT8 data 为 56 KiB，K/V FP32 scale 为 1.75 KiB，总计 **57.75 KiB/token**，降低 48.4375%。不包含共享 workspace、模型、Graph 和运行时临时张量，不能写成“总显存降低 48%”。

在质量评测的同一 0.9 显存预算下，Flash 分配 703 blocks / 179968 token slots，INT8 分配 1364 blocks / 349184 slots，约 **1.94× 分配容量**。INT8 data 为 20023607296 B、scale 为 625737728 B；2K/8K 评测 workspace 分别为 4259840/1064960 B。不同 max_num_seqs 会改变 workspace，所以容量实验必须固定配置。

上述原格式仅测了分配，没有完成容量扩展负载，也未通过质量门槛；当前 K32 的可调度性以新表格为准，不混用两种格式的结果。

后续固定负载即使请求数相同，更多 KV slots 也可能改变调度、抢占与同时驻留的请求数，尤其 prefill-heavy；需结合 block 数解释，不把同预算的整体收益全部归因于 Attention kernel。单层对照才用于隔离算子本身。

## 复现入口

在已激活的 nano-vllm 环境、仓库根目录运行；未激活时设置 `PYTHON_BIN=/home/zwx/miniconda3/envs/nano-vllm/bin/python`。数据准备额外需要可选依赖 `python -m pip install -e '.[eval]'`。

```bash
# 单卡算子与真实模型正确性；TP 测试默认跳过
CUDA_VISIBLE_DEVICES=1 NANOVLLM_TEST_MODEL="$HOME/huggingface/Qwen3-0.6B" \
  python -m unittest discover -s test -p 'test_*.py' -v
# 数据 + 六组完整质量评测 + 门槛检查；当前 K32 通过
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/run_p4_quality.sh
# 定位 K/V；不替代正式质量评测
CUDA_VISIBLE_DEVICES=1 python benchmarks/eval_kv_quality.py --backend triton \
  --diagnose k --window 2048 --output-json benchmarks/results/p4-k32-gpu1/quality/diagnose-k-2048.json
```

以下工具按关卡顺序运行；默认新结果目录为 `benchmarks/results/p4-k32-gpu1/`，profile 为 `benchmarks/profiles/p4-k32-gpu1/`：

```bash
# 单层三后端矩阵、调参、head/GQA 变体
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/run_p4_attention.sh
# 同负载：主对照 5 负载×3 后端×5 轮；组合 2 负载×2 后端×5 轮
# 脚本先检查当前 K32 的质量门槛
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/run_p4_kv.sh
# 相同预算的分配容量与扩展 8K 请求准入验证
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/run_p4_capacity.sh
# 无 profiler 计时之外的 Nsight Graph node 分析
CUDA_VISIBLE_DEVICES=1 bash benchmarks/scripts/profile_p4.sh
# 两张卡确认空闲后执行全套，包含 P3/P4 的 TP=2
CUDA_VISIBLE_DEVICES=1,2 NANOVLLM_TEST_TP=2 \
  NANOVLLM_TEST_MODEL="$HOME/huggingface/Qwen3-0.6B" \
  python -m unittest discover -s test -p 'test_*.py' -v
```

旧质量报告仍在 `benchmarks/results/p4-gpu1/quality/`，新 K32 报告单独保存，没有覆盖失败记录。未提交数据或性能产物。`bench.py` 同时记录推理源代码 SHA256，避免仅用 Git HEAD + dirty 标记混淆未提交版本。

## 局限与下一步

当前结论只覆盖 RTX 3090、Qwen3-0.6B、上述固定版本/负载；不推广为其他模型或在线服务的 SLO 收益。INT8 的主要交付是正确性、1% 内的限定质量退化、容量收益及完整负结果，默认后端不变。后续可在不改变 K32 格式的前提下，研究按 group 一次加载 scale 后广播、向量化转换及 INT8 专用 tile/warps/split 规则；需要重新做单层、质量和端到端对照，不因容量成功而省略验收。

## 参考

在线 softmax 的数学结构参考 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)；本实现使用适配 SM86 的普通 load/dot，不依赖 Hopper/Blackwell TMA 或新硬件特性。

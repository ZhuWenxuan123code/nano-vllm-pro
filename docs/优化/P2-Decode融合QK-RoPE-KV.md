# P2：融合 Decode Q/K Norm、RoPE 与 KV Cache 写入

## 设计与开关

Triton kernel 从 QKV 投影输出直接读取 Q/K/V：按原路径的舍入顺序完成 Q/K 的 FP32 RMSNorm 与 RoPE，将连续 Q 返回给 FlashAttention，并依据 `slot_mapping` 把旋转后的 K、原始 V 写入 Paged KV Cache。`slot=-1` 时不写缓存、Q 输出零值；支持 GQA、head_dim 64/128/256 与 FP16/BF16。该实现仅用于 Decode 推理前向。不具备 Q/K Norm 或不满足融合形状、dtype 条件时回退原路径。Prefill、QKV GEMM、FlashAttention 和原路径均保留，`--fuse-decode-qk-rope-cache` 为独立的显式开关，默认关闭。

## 单层关卡与正确性

以物理 GPU 1（第二张 RTX 3090）、Qwen3-0.6B 的 B=64、Q/KV 头数 16/8、head_dim=128、BF16 为原型。使用同一预投影 QKV 与缓存，预热后在 CUDA Graph 中重复测量 5 轮：原路径中位数 10.760 μs，融合路径 1 warp 中位数 2.854 μs，链路延迟下降 73.5%，优于预设的 ≥5% 单层关卡。warp 扫描和每轮原始数值见 `benchmarks/results/p2-decode-qkv-gpu1/prototype-b64.json`（本地生成，不提交）。

Nsight Systems 的 `--cuda-graph-trace=node` 确认，单层 20 次 Graph replay 中原路径为 100 个 kernel node（每次 5 个），融合路径为 20 个（每次 1 个）；两者 kernel CUDA 时间合计分别为 223.905 μs 与 59.937 μs。完整模型 Decode 的 20 个 step 中，kernel node 从 8700 降到 6460（-25.7%，每步少 112 个），累计 kernel CUDA 时间从 109.549 ms 降到 105.961 ms（-3.3%）。Profiler 时间不是端到端延迟，不可直接当作吞吐提升。

GPU 1 上重新运行的 27 项测试，对照现有路径的 Q、全部 K/V Cache 与真实模型 logits；覆盖两种 dtype、不同 head_dim/GQA、非连续 QKV、无效 slot、CUDA Graph 重放、输入和未命中 cache slot 不被改写。GPU 1 的单卡 Eager 冒烟也通过；此前 TP=2 Eager 冒烟未在本次重新运行，不计入单卡性能结果。不依赖 Hugging Face 输出。

按当前 Q=16×128、K=8×128、BF16 每元素 2 字节计算，融合去除的 Q/K 中间张量理论读写量约为 `2 × (Q元素数 + 2×K元素数) × 2字节 = 16 KiB/token/layer`，B=64、28 层约 28 MiB/step。这是**逻辑张量流量估算**，并非真实 HBM 传输量；缓存命中会使两者不同。

## 复现与端到端结果

先激活项目 Python 环境；端到端脚本默认从 `~/huggingface/Qwen3-0.6B` 加载模型。以下 `CUDA_VISIBLE_DEVICES=1` 指物理编号 1，即第二张显卡。

```bash
# 单层延迟对比
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_decode_qkv.py \
  --output-json benchmarks/results/p2-decode-qkv-gpu1/prototype-b64.json
mkdir -p benchmarks/profiles/p2-decode-qkv-gpu1
# 单层 Nsight 分析
for backend in original fused; do
  CUDA_VISIBLE_DEVICES=1 nsys profile --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node --sample=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true \
    --output="benchmarks/profiles/p2-decode-qkv-gpu1/prototype-$backend" \
    python benchmarks/bench_decode_qkv.py --capture-nsys "$backend" --warps 1
done

# 端到端 对比吞吐量
CUDA_VISIBLE_DEVICES=1 RESULT_ROOT=benchmarks/results/p2-decode-qkv-gpu1 \
  bash benchmarks/scripts/run_p2_decode_qkv.sh

# nsys
CUDA_VISIBLE_DEVICES=1 PROFILE_DIR=benchmarks/profiles/p2-decode-qkv-gpu1/original \
  bash benchmarks/scripts/profile_nsys.sh decode

# nsys
CUDA_VISIBLE_DEVICES=1 PROFILE_DIR=benchmarks/profiles/p2-decode-qkv-gpu1/fused \
  FUSE_DECODE_QK_ROPE_CACHE=1 bash benchmarks/scripts/profile_nsys.sh decode

CUDA_VISIBLE_DEVICES=1 PROFILE_STEPS=3 \
  PROFILE_DIR=benchmarks/profiles/p2-decode-qkv-gpu1/fused-torch \
  FUSE_DECODE_QK_ROPE_CACHE=1 bash benchmarks/scripts/profile_torch.sh decode
CUDA_VISIBLE_DEVICES=1 NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B \
  python -m unittest discover -s test -p 'test_*.py' -v
```

完整基准固定 RMSNorm 为 `compiled`，原/新路径在相同 GPU、模型、seed、CUDA Graph 设置下，对 decode-heavy、balanced、prefill-heavy 各重复 5 次。以下均为五次运行的中位数；吞吐越高越好，延迟越低越好。

| 负载 / 指标 | 原路径 | 融合路径 | 改善 |
| --- | ---: | ---: | ---: |
| Decode-heavy E2E tok/s | 7361.22 | 7521.30 | +2.17% |
| Decode-heavy Decode tok/s | 7647.52 | 7819.92 | +2.25% |
| Decode-heavy TTFT P50 / P99 ms | 477.84 / 684.63 | 484.44 / 690.57 | -1.36% / -0.86% |
| Decode-heavy TPOT P50 / P99 ms | 21.25 / 33.51 | 20.77 / 32.77 | +2.29% / +2.25% |
| Balanced E2E tok/s | 2513.26 | 2537.56 | +0.97% |
| Balanced Decode tok/s | 4227.14 | 4303.15 | +1.80% |
| Balanced TTFT P50 / P99 ms | 2557.75 / 9192.84 | 2559.90 / 9135.02 | -0.08% / +0.63% |
| Balanced TPOT P50 / P99 ms | 50.30 / 66.23 | 49.83 / 65.49 | +0.94% / +1.13% |
| Prefill-heavy E2E tok/s | 550.13 | 551.79 | +0.30% |
| Prefill-heavy Prefill tok/s | 50019.93 | 49885.26 | -0.27% |

Prefill-heavy 也包含短 Decode，故 E2E 可略有变化；Prefill 子指标基本持平。GPU 1 的 Decode-heavy 五次 E2E 吞吐区间有交叠（原路径 7250.05–7509.20，融合路径 7472.45–7528.54 tok/s），且原路径五次全部先于融合路径运行，可能存在时间顺序或 GPU 状态偏差；TTFT P50/P99 也略有回退。融合链路的单层提速并不等同于完整模型提速，后者仍受 GEMM、FlashAttention、采样和调度影响。上述结果仅代表 Qwen3-0.6B、第二张 RTX 3090、这些固定负载与 CUDA Graph；尚未证明其他模型或 GPU 上有相同收益。GPU 1 的完整 JSON 和对比表保留在忽略提交的 `benchmarks/results/p2-decode-qkv-gpu1/`；GPU 2 的旧结果仍保留在 `benchmarks/results/p2-decode-qkv/`，未用于上表。

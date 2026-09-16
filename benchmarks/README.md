# Benchmark Workloads

`scripts/` 保存可复现的 benchmark 工作负载，`results/` 保存本地生成的 JSON
结果。结果目录由 Git 忽略；需要长期保留的汇总和结论应写入 `docs/`。

所有脚本默认使用物理 GPU 1、`~/huggingface/Qwen3-0.6B`、运行 5 次，并将
结果写入 `benchmarks/results/baseline/`。可通过环境变量覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/path/to/Qwen3-0.6B \
RUNS=3 \
bash benchmarks/scripts/run_balanced.sh
```

优化后使用独立结果目录运行同一工作负载：

```bash
RESULT_DIR=benchmarks/results/p1-rmsnorm \
bash benchmarks/scripts/run_decode_heavy.sh
```

汇总某个目录中的多次结果：

```bash
python benchmarks/summarize.py benchmarks/results/baseline
```

脚本应从任意工作目录正常执行。请保持优化前后的模型、硬件、依赖、随机种子和
命令参数一致。

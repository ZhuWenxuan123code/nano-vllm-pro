import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from random import randint, seed

import torch
from nanovllm import LLM, SamplingParams


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark nano-vLLM throughput.")
    parser.add_argument(
        "--model",
        default="~/huggingface/Qwen3-0.6B/",
        help="Local model directory.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=256
    )  # 本次 benchmark 一共生成多少条请求/提示词。
    parser.add_argument(  # 引擎调度时，单个 batch 最多同时处理多少条序列。
        "--max-num-seqs",
        type=int,
        default=512,
        help="Maximum number of sequences scheduled in one batch.",
    )
    # 输入长度
    parser.add_argument(
        "--input-len",
        type=int,
        help="Fixed prompt length. Overrides --min/--max-input-len.",
    )
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    # 输出长度
    parser.add_argument(
        "--output-len",
        type=int,
        help="Fixed output length. Overrides --min/--max-output-len.",
    )
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument(
        "--max-model-len", type=int, default=4096
    )  # 单请求最大上下文长度（prompt + output）的目标上限
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=16384
    )  # 单个 batch 最多处理多少token
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graph and run decode in eager mode.",  # 是否启用 cuda graph
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--output-json", help="Write the benchmark report to this path."
    )
    args = parser.parse_args(argv)

    if args.input_len is not None:
        args.min_input_len = args.max_input_len = args.input_len
    if args.output_len is not None:
        args.min_output_len = args.max_output_len = args.output_len

    positive_values = {
        "num-prompts": args.num_prompts,
        "max-num-seqs": args.max_num_seqs,
        "min-input-len": args.min_input_len,
        "max-input-len": args.max_input_len,
        "min-output-len": args.min_output_len,
        "max-output-len": args.max_output_len,
        "max-model-len": args.max_model_len,
        "max-num-batched-tokens": args.max_num_batched_tokens,
        "tensor-parallel-size": args.tensor_parallel_size,
    }
    for name, value in positive_values.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.min_input_len > args.max_input_len:
        parser.error("--min-input-len cannot exceed --max-input-len")
    if args.min_output_len > args.max_output_len:
        parser.error("--min-output-len cannot exceed --max-output-len")
    if args.max_input_len + args.max_output_len > args.max_model_len:
        parser.error("maximum input and output lengths cannot exceed --max-model-len")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.temperature <= 1e-10:
        parser.error("--temperature must be greater than 1e-10")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def print_report(report):
    metrics = report["metrics"]

    def show(value, suffix=""):
        return "n/a" if value is None else f"{value:.2f}{suffix}"

    print(f"Total input tokens: {metrics['total_input_tokens']}")
    print(f"Total output tokens: {metrics['total_output_tokens']}")
    print(f"Elapsed: {metrics['elapsed_seconds']:.2f}s")
    print(f"E2E throughput: {metrics['e2e_throughput']:.2f} output tok/s")
    print(f"Prefill throughput: {show(metrics['prefill_throughput'], ' tok/s')}")
    print(f"Decode throughput: {show(metrics['decode_throughput'], ' tok/s')}")
    print(
        "TTFT: "
        f"P50={show(metrics['ttft_ms']['p50'], 'ms')}, "
        f"P95={show(metrics['ttft_ms']['p95'], 'ms')}, "
        f"P99={show(metrics['ttft_ms']['p99'], 'ms')}"
    )
    print(
        "TPOT: "
        f"P50={show(metrics['tpot_ms']['p50'], 'ms')}, "
        f"P95={show(metrics['tpot_ms']['p95'], 'ms')}, "
        f"P99={show(metrics['tpot_ms']['p99'], 'ms')}"
    )
    print(f"Peak allocated memory (rank 0): {metrics['peak_memory_mb']:.2f} MiB")


def run_benchmark(llm: LLM, prompts, sampling_params, torch):
    seq_ids = [
        llm.add_request(prompt, params)
        for prompt, params in zip(prompts, sampling_params)
    ]
    output_lengths = {
        seq_id: params.max_tokens for seq_id, params in zip(seq_ids, sampling_params)
    }

    first_token_times = {}
    finish_times = {}
    prefill_tokens = decode_tokens = 0
    prefill_steps = decode_steps = 0
    prefill_time = decode_time = 0.0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    benchmark_start = perf_counter()
    while not llm.is_finished():
        step_start = perf_counter()
        _, info = llm.step_with_info()
        torch.cuda.synchronize()
        step_end = perf_counter()
        step_elapsed = step_end - step_start

        if info.is_prefill:
            prefill_tokens += info.num_scheduled_tokens
            prefill_steps += 1
            prefill_time += step_elapsed
        else:
            decode_tokens += info.num_scheduled_tokens
            decode_steps += 1
            decode_time += step_elapsed

        for seq_id in info.generated_seq_ids:
            first_token_times.setdefault(seq_id, step_end) # 第一个 token 生成完成的时间
        for seq_id in info.finished_seq_ids:
            finish_times[seq_id] = step_end
    benchmark_end = perf_counter()

    ttft = [first_token_times[seq_id] - benchmark_start for seq_id in seq_ids]
    tpot = [
        (finish_times[seq_id] - first_token_times[seq_id])
        / (output_lengths[seq_id] - 1)
        for seq_id in seq_ids
        if output_lengths[seq_id] > 1
    ]
    elapsed = benchmark_end - benchmark_start
    total_output_tokens = sum(output_lengths.values())
    return {
        "total_input_tokens": sum(map(len, prompts)),
        "total_output_tokens": total_output_tokens,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "elapsed_seconds": elapsed,
        "prefill_seconds": prefill_time,
        "decode_seconds": decode_time,
        "e2e_throughput": total_output_tokens / elapsed,
        "prefill_throughput": prefill_tokens / prefill_time if prefill_time else None,
        "decode_throughput": decode_tokens / decode_time if decode_time else None,
        "ttft_ms": {
            "p50": percentile(ttft, 50) * 1000,
            "p95": percentile(ttft, 95) * 1000,
            "p99": percentile(ttft, 99) * 1000,
        },
        "tpot_ms": {
            "p50": percentile(tpot, 50) * 1000 if tpot else None,
            "p95": percentile(tpot, 95) * 1000 if tpot else None,
            "p99": percentile(tpot, 99) * 1000 if tpot else None,
        },
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    path = os.path.expanduser(args.model)
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    vocab_size = llm.tokenizer.vocab_size

    prompt_token_ids = [
        [
            randint(0, vocab_size - 1)
            for _ in range(randint(args.min_input_len, args.max_input_len))
        ]
        for _ in range(args.num_prompts)
    ]
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_prompts)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    for index in range(args.warmup_runs):
        llm.generate(
            [f"Benchmark warmup {index}: "],
            SamplingParams(
                temperature=args.temperature,
                ignore_eos=True,
                max_tokens=1,
            ),
            use_tqdm=False,
        )

    metrics = run_benchmark(llm, prompt_token_ids, sampling_params, torch)
    report = {
        "config": {
            "model": path,
            "num_prompts": args.num_prompts,
            "min_input_len": args.min_input_len,
            "max_input_len": args.max_input_len,
            "min_output_len": args.min_output_len,
            "max_output_len": args.max_output_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "temperature": args.temperature,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "gpu": torch.cuda.get_device_name(),
        },
        "metrics": metrics,
    }
    print_report(report)

    # 将结果保存到json
    if args.output_json:
        output_path = Path(os.path.expanduser(args.output_json))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()

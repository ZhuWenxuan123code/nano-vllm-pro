import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from nanovllm import LLM, SamplingParams


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile nano-vLLM Prefill or Decode after warmup."
    )
    parser.add_argument("--backend", choices=("torch", "nsys"), default="torch")
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/profiles/baseline"),
    )
    args = parser.parse_args(argv)

    positive = {
        "batch-size": args.batch_size,
        "input-len": args.input_len,
        "profile-steps": args.profile_steps,
        "max-model-len": args.max_model_len,
        "max-num-batched-tokens": args.max_num_batched_tokens,
        "tensor-parallel-size": args.tensor_parallel_size,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be greater than zero")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    required_length = args.input_len + (
        args.profile_steps + 1 if args.phase == "decode" else 1
    )
    if required_length > args.max_model_len:
        parser.error("input and profiled output lengths exceed --max-model-len")
    if args.phase == "prefill":
        tokens_per_step = args.batch_size * args.input_len
        if tokens_per_step > args.max_num_batched_tokens:
            parser.error(
                "Prefill profiling requires batch-size * input-len <= "
                "--max-num-batched-tokens so each capture step has a full batch"
            )
    return args


def make_prompts(rng, count, length, vocab_size):
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(length)]
        for _ in range(count)
    ]


def add_requests(llm, prompts, max_tokens):
    params = SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=True)
    for prompt in prompts:
        llm.add_request(prompt, params)


def run_warmup(llm, args, vocab_size):
    rng = Random(args.seed + 1)
    for _ in range(args.warmup_runs):
        prompts = make_prompts(rng, args.batch_size, args.input_len, vocab_size)
        llm.generate(
            prompts,
            SamplingParams(temperature=1.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
    torch.cuda.synchronize()


def prepare_capture(llm, args, vocab_size):
    rng = Random(args.seed)
    if args.phase == "prefill":
        prompts = make_prompts(
            rng,
            args.batch_size * args.profile_steps,
            args.input_len,
            vocab_size,
        )
        add_requests(llm, prompts, max_tokens=1)
        return

    prompts = make_prompts(rng, args.batch_size, args.input_len, vocab_size)
    add_requests(llm, prompts, max_tokens=args.profile_steps + 1)
    while llm.scheduler.waiting:
        _, info = llm.step_with_info()
        if not info.is_prefill:
            raise RuntimeError("Decode preparation unexpectedly entered a decode step")
    torch.cuda.synchronize()


def run_capture_steps(llm, args):
    scheduled_tokens = 0
    scheduled_sequences = 0
    for step in range(args.profile_steps):
        marker = f"nanovllm::{args.phase}_step_{step}"
        with record_function(marker), torch.cuda.nvtx.range(marker):
            _, info = llm.step_with_info()
        actual_phase = "prefill" if info.is_prefill else "decode"
        if actual_phase != args.phase:
            raise RuntimeError(
                f"expected a {args.phase} step, but scheduler ran {actual_phase}"
            )
        if info.num_scheduled_seqs != args.batch_size:
            raise RuntimeError(
                f"expected batch size {args.batch_size}, got {info.num_scheduled_seqs}"
            )
        scheduled_tokens += info.num_scheduled_tokens
        scheduled_sequences += info.num_scheduled_seqs
    torch.cuda.synchronize()
    return {
        "steps": args.profile_steps,
        "scheduled_tokens": scheduled_tokens,
        "scheduled_sequences": scheduled_sequences,
    }


def event_device_time_us(event):
    return float(getattr(event, "device_time_total", 0.0) or 0.0)


def is_cuda_event(event):
    return str(getattr(event, "device_type", "")).lower().endswith("cuda")


def summarize_torch_profiler(profiler, top_n=20):
    cuda_events = [
        event
        for event in profiler.events()
        if is_cuda_event(event)
        and getattr(event, "linked_correlation_id", 0) > 0
    ]
    kernel_events = []
    kernel_stats = defaultdict(lambda: {"count": 0, "cuda_time_us": 0.0})
    for event in cuda_events:
        name = event.name
        lower_name = name.lower()
        if "memcpy" in lower_name or "memset" in lower_name or name == "[memory]":
            continue
        kernel_events.append(event)
        kernel_stats[name]["count"] += 1
        kernel_stats[name]["cuda_time_us"] += event_device_time_us(event)

    total_kernel_time = sum(item["cuda_time_us"] for item in kernel_stats.values())
    top_kernels = []
    for name, item in sorted(
        kernel_stats.items(), key=lambda pair: pair[1]["cuda_time_us"], reverse=True
    )[:top_n]:
        top_kernels.append(
            {
                "name": name,
                "count": item["count"],
                "cuda_time_us": item["cuda_time_us"],
                "cuda_time_percent": (
                    item["cuda_time_us"] / total_kernel_time * 100
                    if total_kernel_time
                    else 0.0
                ),
            }
        )

    operator_stats = []
    launch_api_calls = 0
    for event in profiler.key_averages():
        self_device_time = float(
            getattr(event, "self_device_time_total", 0.0) or 0.0
        )
        if (
            self_device_time
            and not is_cuda_event(event)
            and not event.key.startswith("nanovllm::")
        ):
            operator_stats.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cuda_time_us": self_device_time,
                }
            )
        if any(
            name in event.key
            for name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")
        ):
            launch_api_calls += event.count

    total_operator_time = sum(item["self_cuda_time_us"] for item in operator_stats)
    operator_stats.sort(key=lambda item: item["self_cuda_time_us"], reverse=True)
    for item in operator_stats:
        item["cuda_time_percent"] = (
            item["self_cuda_time_us"] / total_operator_time * 100
            if total_operator_time
            else 0.0
        )

    return {
        "total_kernel_cuda_time_us": total_kernel_time,
        "cuda_activity_count": len(cuda_events),
        "kernel_activity_count": len(kernel_events),
        "cuda_launch_api_calls": launch_api_calls,
        "top_operators": operator_stats[:top_n],
        "top_kernels": top_kernels,
    }


def format_summary_table(summary):
    lines = [
        f"CUDA kernel activities: {summary['kernel_activity_count']}",
        f"CUDA launch API calls: {summary['cuda_launch_api_calls']}",
        f"Summed kernel CUDA time: {summary['total_kernel_cuda_time_us']:.3f} us",
        "",
        "Top CUDA kernels",
        f"{'CUDA us':>12} {'Percent':>9} {'Calls':>8}  Name",
    ]
    for item in summary["top_kernels"]:
        lines.append(
            f"{item['cuda_time_us']:12.3f} "
            f"{item['cuda_time_percent']:8.2f}% "
            f"{item['count']:8d}  {item['name']}"
        )
    lines.extend(
        [
            "",
            "Top PyTorch operators",
            f"{'Self CUDA us':>12} {'Percent':>9} {'Calls':>8}  Name",
        ]
    )
    for item in summary["top_operators"]:
        lines.append(
            f"{item['self_cuda_time_us']:12.3f} "
            f"{item['cuda_time_percent']:8.2f}% "
            f"{item['count']:8d}  {item['name']}"
        )
    return "\n".join(lines) + "\n"


def report_config(args):
    return {
        "backend": args.backend,
        "phase": args.phase,
        "model": os.path.expanduser(args.model),
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "profile_steps": args.profile_steps,
        "warmup_runs": args.warmup_runs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(),
    }


def output_prefix(args):
    mode = "eager" if args.phase == "prefill" or args.enforce_eager else "cudagraph"
    return args.output_dir / f"{args.phase}-{mode}"


def run_torch_profile(llm, args):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    started = perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        capture = run_capture_steps(llm, args)
    elapsed = perf_counter() - started

    prefix = output_prefix(args)
    profiler.export_chrome_trace(str(prefix.with_suffix(".trace.json")))
    profiler_summary = summarize_torch_profiler(profiler)
    report = {
        "config": report_config(args),
        "capture": {**capture, "elapsed_seconds": elapsed},
        "profiler": profiler_summary,
    }
    prefix.with_suffix(".table.txt").write_text(
        format_summary_table(profiler_summary), encoding="utf-8"
    )
    prefix.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prefix, report


def run_nsys_capture(llm, args):
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    started = perf_counter()
    try:
        capture = run_capture_steps(llm, args)
    finally:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    elapsed = perf_counter() - started

    prefix = output_prefix(args)
    report = {
        "config": report_config(args),
        "capture": {**capture, "elapsed_seconds": elapsed},
        "note": "Kernel statistics are generated from the .nsys-rep by profile_nsys.sh.",
    }
    prefix.with_suffix(".run.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prefix, report


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    vocab_size = llm.tokenizer.vocab_size
    run_warmup(llm, args, vocab_size)
    prepare_capture(llm, args, vocab_size)

    if args.backend == "torch":
        prefix, report = run_torch_profile(llm, args)
    else:
        prefix, report = run_nsys_capture(llm, args)

    capture = report["capture"]
    print(f"Profile: {args.phase} ({args.backend})")
    print(f"Steps: {capture['steps']}")
    print(f"Scheduled tokens: {capture['scheduled_tokens']}")
    print(f"Capture elapsed: {capture['elapsed_seconds'] * 1000:.2f} ms")
    print(f"Output prefix: {prefix}")


if __name__ == "__main__":
    main()

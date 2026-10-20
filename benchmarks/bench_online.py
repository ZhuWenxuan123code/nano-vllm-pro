"""Replay identical open-loop arrivals; no HTTP server or per-step device sync."""

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.online import make_trace, materialize_trace, run_online, summarize_records, trace_hash


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--write-trace", type=Path)
    parser.add_argument("--workload", choices=("steady", "burst", "mixed", "pressure"), default="steady")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--arrival-rate", type=float, default=10)
    parser.add_argument("--saturated", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-policy", choices=("legacy", "mixed", "adaptive"), default="legacy")
    parser.add_argument("--prefill-token-budget", type=int, default=256)
    parser.add_argument("--prefill-budget-min", type=int, default=64)
    parser.add_argument("--prefill-budget-max", type=int, default=1024)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000)
    parser.add_argument("--itl-slo-ms", type=float, default=50)
    parser.add_argument("--prefill-max-wait-ms", type=float, default=1000)
    parser.add_argument("--kv-cache-block-limit", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8448)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--attention-backend", choices=("flash", "triton"), default="flash")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "int8"), default="auto")
    parser.add_argument("--execution-mode", choices=("original", "buffered"), default="original")
    parser.add_argument("--rms-norm-backend", choices=("compiled", "triton"), default="compiled")
    parser.add_argument("--fuse-decode-qk-rope-cache", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-decode-steps", type=int, default=32)
    parser.add_argument("--compile-recompile-limit", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--capture-nsys", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.num_requests, args.arrival_rate, args.ttft_slo_ms, args.itl_slo_ms,
           args.kv_cache_block_limit, args.max_num_seqs, args.max_num_batched_tokens,
           args.compile_recompile_limit, args.timeout_s) <= 0:
        parser.error("Counts, rates, budgets and latency targets must be positive")
    if args.kv_cache_dtype == "int8" and args.attention_backend != "triton":
        parser.error("INT8 KV requires Triton Attention")
    return args


def warmup(llm, args, vocab):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.scheduled_batch import PrefillBudgetController
    # Same short/long shape families in all policies. New prompt arrivals during
    # Decode exercise mixed kernels; hashes/counters/controller are reset below.
    for length in (256, 512, 8192):
        if length + args.warmup_decode_steps + 2 > args.max_model_len:
            continue
        blocks = (length + args.warmup_decode_steps + 257) // 256
        count = min(args.max_num_seqs // 2 or 1, max(1, (args.kv_cache_block_limit - 2) // blocks), 32)
        for i in range(count):
            llm.add_request([1000 + i + length] * length,
                            SamplingParams(ignore_eos=True, max_tokens=args.warmup_decode_steps + 2))
        added = False
        while not llm.is_finished():
            _, info = llm.step_with_info()
            if not added and info.phase == "decode":
                llm.add_request([1100] * 257, SamplingParams(ignore_eos=True, max_tokens=2))
                added = True
    torch.cuda.synchronize()
    manager = llm.scheduler.block_manager
    manager.hash_to_block_id.clear()
    for block in manager.blocks:
        block.hash, block.token_ids = -1, []
    for key in llm.scheduler.stats:
        llm.scheduler.stats[key] = 0
    if llm.scheduler.controller:
        llm.scheduler.controller = PrefillBudgetController(llm.model_runner.config)


def main():
    args = parse_args()
    import torch
    from transformers import AutoConfig
    import triton
    import flash_attn
    from nanovllm import LLM, SamplingParams
    from bench import code_version
    model = os.path.expanduser(args.model)
    # Measurement harness only: heterogeneous rank/stride/shape families share
    # decorated RMSNorm functions. Prevent an implicit compiled->eager baseline.
    limit_name = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    setattr(torch._dynamo.config, limit_name, args.compile_recompile_limit)
    fail_on_limit = hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit")
    if fail_on_limit:
        torch._dynamo.config.fail_on_recompile_limit_hit = True
    vocab = AutoConfig.from_pretrained(model).vocab_size
    trace = json.loads(args.trace.read_text()) if args.trace else make_trace(
        args.workload, args.num_requests, args.arrival_rate, args.seed, vocab, args.saturated)
    if trace["vocab_size"] != vocab:
        raise ValueError("Trace vocabulary differs from model")
    requests, token_hash = materialize_trace(trace)
    if any(len(r["prompt"]) + r["output_len"] > args.max_model_len for r in requests):
        raise ValueError("Trace request exceeds configured context length")
    if args.write_trace:
        args.write_trace.parent.mkdir(parents=True, exist_ok=True)
        args.write_trace.write_text(json.dumps(trace, indent=2) + "\n")
    torch.manual_seed(args.seed)
    config_keys = ("scheduler_policy", "prefill_token_budget", "prefill_budget_min", "prefill_budget_max",
                   "ttft_slo_ms", "itl_slo_ms", "prefill_max_wait_ms", "kv_cache_block_limit",
                   "max_num_seqs", "max_num_batched_tokens", "max_model_len", "tensor_parallel_size",
                   "attention_backend", "kv_cache_dtype", "execution_mode", "rms_norm_backend",
                   "fuse_decode_qk_rope_cache", "enforce_eager")
    kwargs = {key: getattr(args, key) for key in config_keys}
    llm = LLM(model, **kwargs, profile_stages=args.capture_nsys)
    try:
        warmup(llm, args, vocab)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        if args.capture_nsys:
            torch.cuda.cudart().cudaProfilerStart()
        report = run_online(llm, requests, lambda n: SamplingParams(ignore_eos=True, max_tokens=n),
                            args.ttft_slo_ms, args.itl_slo_ms, args.timeout_s)
        torch.cuda.synchronize()
        if args.capture_nsys:
            torch.cuda.cudart().cudaProfilerStop()
        report.update(config=dict(kwargs, seed=args.seed, model=model,
                                  warmup_decode_steps=args.warmup_decode_steps,
                                  compile_recompile_limit=args.compile_recompile_limit,
                                  fail_on_recompile_limit_hit=fail_on_limit,
                                  warmup_prompt_lengths=[256, 512, 8192]),
                      trace_sha256=trace_hash(trace), tokens_sha256=token_hash,
                      workload=trace["workload"], rate=trace["rate"], saturated=trace["saturated"],
                      cache_memory=llm.model_runner.cache_memory, code=code_version(),
                      gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      triton_version=triton.__version__, flash_attn_version=flash_attn.__version__,
                      python_version=sys.version, profile_only=args.capture_nsys,
                      peak_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
                      free_blocks_after=len(llm.scheduler.block_manager.free_block_ids))
        # Include the measurement implementation, not just inference code, in its hash.
        report["measurement_sha256"] = hashlib.sha256(Path(__file__).read_bytes() +
            (Path(__file__).parent / "online.py").read_bytes()).hexdigest()
        report["slo_sensitivity"] = {f"{ttft}/{itl}": summarize_records(
            [dict(row) for row in report["requests"]], report["metrics"]["elapsed_seconds"], ttft, itl)
            for ttft in (1000, 2000, 4000) for itl in (25, 50, 100)}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(policy=args.scheduler_policy, workload=trace["workload"],
                              rate=trace["rate"], **report["metrics"])), flush=True)
        if report["metrics"]["timed_out"]:
            raise RuntimeError("Online trial timed out; incomplete requests remain SLO failures")
        if report["free_blocks_after"] != report["cache_memory"]["blocks"]:
            raise RuntimeError("Completed trial leaked KV blocks")
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()

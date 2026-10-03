"""Measure allocated capacity and actually admit an expanded 8K workload."""

import argparse
import atexit
import json
import math
import os
from pathlib import Path

import torch
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--backend", choices=("flash", "triton", "int8"), default="flash")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, help="Omit to use the cache's calculated 8K admission capacity")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    torch.manual_seed(41)
    llm = LLM(args.model, attention_backend="flash" if args.backend == "flash" else "triton",
              kv_cache_dtype="int8" if args.backend == "int8" else "auto",
              max_num_seqs=64, max_model_len=args.input_len + 4, max_num_batched_tokens=16384,
              gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=args.tensor_parallel_size)
    try:
        memory = dict(llm.model_runner.cache_memory)
        blocks_per_request = math.ceil((args.input_len + 4) / llm.model_runner.block_size)
        capacity = memory["blocks"] // blocks_per_request
        batch = args.batch_size or min(capacity, 64)
        if batch < 1 or batch > min(capacity, 64):
            raise ValueError("Requested batch exceeds measured no-preemption capacity or max_num_seqs")
        preemptions = 0
        old_preempt = llm.scheduler.preempt
        def preempt(seq):
            nonlocal preemptions
            preemptions += 1
            return old_preempt(seq)
        llm.scheduler.preempt = preempt
        for i in range(batch):
            llm.add_request([100 + i] * args.input_len, SamplingParams(ignore_eos=True, max_tokens=4))
        max_running = finished = 0
        while not llm.is_finished():
            outputs, _ = llm.step_with_info()
            finished += len(outputs)
            max_running = max(max_running, len(llm.scheduler.running))
        free_blocks = len(llm.scheduler.block_manager.free_block_ids)
        if finished != batch or preemptions or max_running != batch:
            raise AssertionError("Capacity-expanded workload was not simultaneously admitted without preemption")
        if free_blocks != memory["blocks"]:
            raise AssertionError("Capacity-expanded workload did not return all cache blocks")
        result = {"backend": args.backend, "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid), "cache_memory": memory,
            "input_len": args.input_len, "max_no_preemption_batch": capacity, "tested_batch": batch,
            "max_running": max_running, "preemptions": preemptions, "finished": finished,
            "free_blocks_after": free_blocks,
            "tensor_parallel_size": args.tensor_parallel_size,
            "bytes_per_token_all_ranks": memory["bytes_per_token_per_rank"] * args.tensor_parallel_size}
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        atexit.unregister(llm.exit)
        llm.exit()


if __name__ == "__main__":
    main()

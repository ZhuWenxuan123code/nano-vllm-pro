import argparse
import os
from time import perf_counter
from random import randint, seed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark nano-vLLM throughput.")
    parser.add_argument(
        "--model",
        default="~/huggingface/Qwen3-0.6B/",
        help="Local model directory.",
    )
    parser.add_argument("--num-prompts", type=int, default=256) #本次 benchmark 一共生成多少条请求/提示词。
    parser.add_argument(                                        #引擎调度时，单个 batch 最多同时处理多少条序列。
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
    parser.add_argument("--max-model-len", type=int, default=4096)           # 单请求最大上下文长度（prompt + output）的目标上限
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384) # 单个 batch 最多处理多少token
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graph and run decode in eager mode.",             # 是否启用 cuda graph
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
        parser.error(
            "maximum input and output lengths cannot exceed --max-model-len"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.temperature <= 1e-10:
        parser.error("--temperature must be greater than 1e-10")
    return args


def main():
    args = parse_args()

    from nanovllm import LLM, SamplingParams

    # from vllm import LLM, SamplingParams

    seed(args.seed)

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

    llm.generate(["Benchmark: "], SamplingParams()) # warmup
    start = perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = perf_counter() - start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    print(
        f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, "
        f"Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()

"""Fixed WikiText-2 continuation perplexity using nano-vLLM teacher forcing."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np


DATASET = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    # These imports are optional and never needed by the inference package.
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    data = hf_hub_download(DATASET, "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=REVISION)
    rows = parquet.read_table(data, columns=["text"]).column("text").to_pylist()
    text = "\n\n".join(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 65536:
        raise RuntimeError(f"Dataset has only {len(tokens)} tokens; at least 65536 required")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.data_dir / "tokens.npy"
    np.save(token_path, np.asarray(tokens[:65536], dtype=np.int64))
    manifest = {"dataset": DATASET, "revision": REVISION, "split": "test",
        "config": "wikitext-2-raw-v1", "join": "two newlines, original row order",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "parquet_sha256": digest(data),
        "tokens": 65536, "token_file_sha256": digest(token_path), "model": args.model,
        "tokenizer_files": {path.name: digest(path) for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
                            if (path := Path(args.model) / name).exists()}}
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def evaluate(args):
    import torch
    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import reset_context
    if args.diagnose:
        # Diagnostic only: quantize/dequantize selected tensors before storing
        # into a high-precision cache, isolating K/V quality from INT8 kernels.
        # This is NOT an inference backend or high-precision residual cache.
        import nanovllm.layers.attention as attention_module
        original_store = attention_module.store_kvcache
        def roundtrip(values, group_size=None):
            original_shape = values.shape
            grouped = values.float()
            if group_size:
                grouped = grouped.reshape(*original_shape[:-1], original_shape[-1] // group_size, group_size)
            maximum = grouped.abs().amax(-1)
            scales = torch.where(maximum == 0, 1., maximum / torch.full_like(maximum, 127.))
            data = (grouped / scales[..., None]).round().clamp(-127, 127)
            return (data * scales[..., None]).reshape(original_shape).to(values.dtype)
        def diagnostic_store(k, v, kc, vc, slots):
            if args.diagnose in ("k", "both"):
                k = roundtrip(k, args.diagnostic_k_group_size)
            if args.diagnose in ("v", "both"):
                v = roundtrip(v)
            original_store(k, v, kc, vc, slots)
        attention_module.store_kvcache = diagnostic_store
    tokens_path = args.data_dir / "tokens.npy"
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if digest(tokens_path) != manifest["token_file_sha256"]:
        raise ValueError("Token hash mismatch; regenerate the fixed data")
    for name, expected in manifest["tokenizer_files"].items():
        if digest(Path(args.model) / name) != expected:
            raise ValueError(f"Tokenizer hash mismatch: {name}")
    tokens = np.load(tokens_path, allow_pickle=False)
    windows = tokens[:65536].reshape(-1, args.window)
    if args.limit_windows:
        windows = windows[:args.limit_windows]
    prompt_len = args.window // 2
    tail_len = min(prompt_len, args.tail_tokens or prompt_len)
    torch.manual_seed(41)
    config = Config(model=args.model, attention_backend="flash" if args.backend == "flash" else "triton",
                    kv_cache_dtype="int8" if args.backend == "int8" else "auto",
                    enforce_eager=args.enforce_eager, max_num_seqs=len(windows),
                    max_model_len=args.window, max_num_batched_tokens=args.prefill_budget,
                    execution_mode="original", rms_norm_backend="compiled", fuse_decode_qk_rope_cache=False)
    runner = ModelRunner(config, 0, [])
    scheduler = Scheduler(config)
    mapping = {}
    nll = torch.zeros((), device="cuda", dtype=torch.float64)
    count = 0
    start = perf_counter()
    try:
        if args.shared_prefix:
            # Populate a real prefix, release it, then exercise the hash-based hit.
            seq = Sequence(windows[0, :prompt_len].tolist(), SamplingParams(ignore_eos=True, max_tokens=1))
            scheduler.add(seq)
            while not scheduler.is_finished():
                seqs, prefill = scheduler.schedule()
                inputs, positions = runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                with torch.inference_mode():
                    runner.run_model(inputs, positions, prefill)
                reset_context()
                scheduler.postprocess(seqs, [int(windows[0, prompt_len])] * len(seqs), prefill)
        for window in windows:
            seq = Sequence(window[:prompt_len].tolist(), SamplingParams(ignore_eos=True, max_tokens=tail_len))
            scheduler.add(seq)
            mapping[seq.seq_id] = window
        with torch.inference_mode():
            while not scheduler.is_finished():
                seqs, prefill = scheduler.schedule()
                inputs, positions = runner.prepare_prefill(seqs) if prefill else runner.prepare_decode(seqs)
                logits = runner.run_model(inputs, positions, prefill)
                targets, rows = [], []
                forced = []
                for row, seq in enumerate(seqs):
                    end = seq.num_cached_tokens + seq.num_scheduled_tokens
                    target = int(mapping[seq.seq_id][end]) if end < args.window else 0
                    forced.append(target)
                    if not prefill or end == seq.num_tokens:
                        rows.append(row)
                        targets.append(target)
                if rows:
                    selected = logits[rows].float()
                    target_tensor = torch.tensor(targets, device=logits.device, dtype=torch.long)
                    nll += (selected.logsumexp(-1) - selected.gather(1, target_tensor[:, None]).squeeze(1)).double().sum()
                    count += len(rows)
                reset_context()
                scheduler.postprocess(seqs, forced, prefill)
                if count and count % 4096 == 0:
                    print(f"{args.backend}: scored {count} tokens", flush=True)
        torch.cuda.synchronize()
        if count != len(windows) * tail_len:
            raise RuntimeError(f"Incorrect scoring count: {count}")
        result = {"backend": args.backend, "window": args.window, "prompt_tokens": prompt_len,
            "scored_tokens": count, "total_nll": nll.item(), "mean_nll": nll.item() / count,
            "continuation_perplexity": math.exp(nll.item() / count), "data": manifest,
            "seconds": perf_counter() - start, "prefill_budget": args.prefill_budget,
            "shared_prefix": args.shared_prefix, "enforce_eager": args.enforce_eager,
            "diagnostic_roundtrip": args.diagnose,
            "diagnostic_k_group_size": args.diagnostic_k_group_size if args.diagnose else None,
            "gpu": torch.cuda.get_device_name(), "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "cache_memory": runner.cache_memory, "free_blocks_after": len(scheduler.block_manager.free_block_ids)}
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("backend", "window", "scored_tokens", "continuation_perplexity", "seconds")}), flush=True)
    finally:
        reset_context()
        runner.exit()
        if args.diagnose:
            attention_module.store_kvcache = original_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/results/p4-data"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=("flash", "triton", "int8"), default="flash")
    parser.add_argument("--window", type=int, choices=(2048, 8192), default=2048)
    parser.add_argument("--limit-windows", type=int, default=0, help="Smoke test only; not the full quality gate")
    parser.add_argument("--tail-tokens", type=int, default=0, help="Smoke/regression only; 0 scores the full half-window")
    parser.add_argument("--prefill-budget", type=int, default=16384)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--diagnose", choices=("k", "v", "both"),
                        help="Diagnostic HP storage with quantization roundtrip; requires --backend triton")
    parser.add_argument("--diagnostic-k-group-size", type=int, choices=(32, 64, 128, 256), default=32,
                        help="Diagnostic only; runtime INT8 K uses fixed group size 32")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)
    if args.prefill_budget < 1 or args.limit_windows < 0 or args.tail_tokens < 0:
        parser.error("Invalid budget/window/tail count")
    if args.diagnose and args.backend != "triton":
        parser.error("--diagnose requires --backend triton")
    if args.prepare:
        prepare(args)
    else:
        if not args.output_json:
            parser.error("--output-json required for evaluation")
        evaluate(args)


if __name__ == "__main__":
    main()

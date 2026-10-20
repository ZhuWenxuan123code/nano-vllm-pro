import atexit
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.profiling import profile_range


@dataclass(slots=True)
class StepInfo:
    is_prefill: bool
    num_scheduled_tokens: int
    num_scheduled_seqs: int
    generated_seq_ids: list[int]
    finished_seq_ids: list[int]
    phase: str = ""
    prefill_tokens: int = 0
    decode_tokens: int = 0
    first_scheduled: dict[int, float] | None = None
    execution_path: str = ""
    p2_tokens: int = 0
    prefill_budget: int = 0
    budget_reason: str = "legacy"
    scheduler_stats: dict | None = None
    kv_used_blocks: int = 0


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.profile_stages = config.profile_stages
        Sequence.block_size = config.kvcache_block_size
        # 创建 spawn 多进程上下文
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step_with_info(self) -> tuple[list[tuple[int, list[int]]], StepInfo]:
        start = perf_counter()
        with profile_range("nanovllm::schedule", self.profile_stages):
            batch = self.scheduler.schedule_batch()
        seqs = batch.seqs
        used_blocks = len(self.scheduler.block_manager.used_block_ids)
        is_prefill = batch.phase != "decode"  # legacy bool means "contains Prefill"
        num_scheduled_tokens = sum(batch.query_lens)
        token_ids = self.model_runner.call("run_batch", batch)
        with profile_range("nanovllm::postprocess", self.profile_stages):
            generated_seq_ids = self.scheduler.postprocess_batch(batch, token_ids)
        visible_at = perf_counter()
        self.scheduler.observe_batch(batch, generated_seq_ids, (visible_at - start) * 1000, visible_at)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        config = self.model_runner.config
        graph = batch.phase == "decode" and not config.enforce_eager and len(seqs) <= 512
        info = StepInfo(
            is_prefill=is_prefill,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_seqs=len(seqs),
            generated_seq_ids=generated_seq_ids, #
            finished_seq_ids=[seq_id for seq_id, _ in outputs], #
            phase=batch.phase, prefill_tokens=batch.prefill_tokens, decode_tokens=batch.num_decode,
            first_scheduled={s.seq_id: s.first_scheduled_at for s in seqs},
            execution_path="graph" if graph else f"{batch.phase}_eager",
            p2_tokens=batch.num_decode if batch.phase == "decode" and config.fuse_decode_qk_rope_cache
                      and not getattr(config.hf_config, "attention_bias", True) else 0,
            prefill_budget=batch.prefill_budget, budget_reason=batch.budget_reason,
            scheduler_stats=dict(self.scheduler.stats),
            kv_used_blocks=used_blocks,
        )
        return outputs, info

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        outputs, info = self.step_with_info()
        num_tokens = info.num_scheduled_tokens if info.is_prefill else -info.num_scheduled_tokens
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, info = self.step_with_info()
            if info.phase == "prefill":
                prefill_throughput = info.prefill_tokens / (perf_counter() - t)
            elif info.phase == "decode":
                decode_throughput = info.decode_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

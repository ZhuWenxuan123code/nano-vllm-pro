from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduled_batch import ScheduledBatch, PrefillBudgetController


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.policy = getattr(config, "scheduler_policy", "legacy")
        self.controller = PrefillBudgetController(config) if self.policy == "adaptive" else None
        self.stats = dict(preemptions=0, recompute_tokens=0, prefix_hit_tokens=0,
                          admission_pauses=0, protected_prefill_steps=0)
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单个 step 内允许处理的 token 总数上限（只约束prefill）
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if not seq.num_tokens or seq.max_tokens < 1:
            raise ValueError("Requests need a nonempty prompt and positive output length")
        if seq.num_tokens + seq.max_tokens > self.config.max_model_len:
            raise ValueError("Prompt plus output length exceeds max_model_len")
        if (seq.num_tokens + seq.max_tokens - 2) // self.block_size + 1 > len(self.block_manager.blocks):
            raise ValueError("A single request cannot fit in the KV cache")
        seq.arrived_at = perf_counter()
        seq.first_scheduled_at = None
        seq.last_prefill_at = seq.arrived_at
        seq.last_output_at = None
        seq.recompute_upto = 0
        self.waiting.append(seq)

    def schedule_batch(self) -> ScheduledBatch:
        if self.policy == "legacy":
            seqs, prefill = self.schedule()
            flags = (prefill,) * len(seqs)
            for seq in seqs:
                if seq.first_scheduled_at is None:
                    seq.first_scheduled_at = perf_counter()
                if prefill:
                    seq.last_prefill_at = perf_counter()
            return ScheduledBatch(seqs, flags, tuple(s.num_scheduled_tokens for s in seqs))
        return self._schedule_mixed()

    def _schedule_mixed(self):
        selected = []
        # Only unscheduled victims may be evicted; selected cache pages are in flight.
        while self.running and len(selected) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                selected.append(seq)
        self.running.extendleft(reversed(selected))
        num_decode = len(selected)
        free = len(self.block_manager.free_block_ids)
        pressure = 1 - free / len(self.block_manager.blocks)
        budget, reason = self.config.prefill_token_budget, "fixed"
        if self.controller:
            budget, reason = self.controller.choose(bool(self.waiting), pressure)
        if not num_decode:
            budget, reason = self.max_num_batched_tokens, "no_decode"
        budget = min(budget, self.max_num_batched_tokens - num_decode)
        allowed_budget = budget
        now = perf_counter()
        resident = len(self.running) + sum(bool(s.block_table) for s in self.waiting)
        # Old resident chunks first: don't strand pages behind fresh admissions.
        candidates = sorted(self.waiting, key=lambda s: (not bool(s.block_table), s.last_prefill_at, s.seq_id))
        prefill = []
        protected = False
        for seq in candidates:
            if budget <= 0 or len(selected) + len(prefill) >= self.max_num_seqs:
                break
            if not seq.block_table:
                if resident >= self.max_num_seqs or (self.running and pressure >= 0.9):
                    self.stats["admission_pauses"] += 1
                    continue
                hits = self.block_manager.can_allocate(seq)
                if hits < 0:
                    continue
                reserve = max(2, sum(len(s) % self.block_size == 0 for s in self.running)) if self.running else 0
                cost = self.block_manager.allocation_cost(seq, hits)
                if self.running and len(seq) % self.block_size == 0:
                    reserve += 1
                if len(self.block_manager.free_block_ids) - cost < reserve:
                    self.stats["admission_pauses"] += 1
                    continue
                self.block_manager.allocate(seq, hits)
                self.stats["prefix_hit_tokens"] += hits * self.block_size
                resident += 1
            count = min(seq.num_tokens - seq.num_cached_tokens, budget)
            seq.is_prefill = True
            seq.num_scheduled_tokens = count
            seq.first_scheduled_at = seq.first_scheduled_at if seq.first_scheduled_at is not None else now
            overdue = (now - seq.last_prefill_at) * 1000 >= self.config.prefill_max_wait_ms
            protected |= overdue and not (self.controller and self.controller.over_target)
            seq.last_prefill_at = now
            prefill.append(seq)
            budget -= count
            if seq.num_cached_tokens + count == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            pressure = 1 - len(self.block_manager.free_block_ids) / len(self.block_manager.blocks)
        self.stats["protected_prefill_steps"] += int(protected)
        seqs = selected + prefill
        if not seqs:
            raise RuntimeError("Scheduler made no progress; inspect KV capacity and admission constraints")
        return ScheduledBatch(seqs, (False,) * num_decode + (True,) * len(prefill),
                              tuple(s.num_scheduled_tokens for s in seqs),
                              allowed_budget, reason)

    def observe_batch(self, batch, generated, elapsed_ms, visible_at):
        gaps = [(visible_at - s.last_output_at) * 1000 for s, flag in zip(batch.seqs, batch.prefill)
                if not flag and s.last_output_at is not None]
        if self.controller and batch.num_decode:
            self.controller.observe(max(gaps, default=elapsed_ms))
        for seq in batch.seqs:
            if seq.seq_id in generated:
                seq.last_output_at = visible_at

    def postprocess_batch(self, batch, token_ids):
        if len(token_ids) != len(batch.seqs):
            raise ValueError("Sample rows must match scheduled sequence rows")
        generated = []
        for seq, token, prefill in zip(batch.seqs, token_ids, batch.prefill):
            generated.extend(self.postprocess([seq], [token], prefill))
        return generated

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.policy != "legacy":
            raise ValueError("Use schedule_batch() for mixed/adaptive scheduling")
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens # 计算本批剩余 Token 容量
            if remaining == 0:
                break
            if not seq.block_table: #第一次进入prefill
                num_cached_blocks = self.block_manager.can_allocate(seq) # >= 0：可以分配，并且前面有这么多个 Block 可以复用
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else: # 进入过prefill，但是由于remaining不足，未能完成prefill
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table: # 没有kv cache blcok
                self.block_manager.allocate(seq, num_cached_blocks) # 真正建立 Block 映射并占用资源
                self.stats["prefix_hit_tokens"] += num_cached_blocks * self.block_size
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens: # prefill 之后进入 decode 阶段
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.stats["preemptions"] += 1
        seq.recompute_upto = max(getattr(seq, "recompute_upto", 0), seq.num_cached_tokens)
        seq.last_prefill_at = perf_counter()
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> list[int]:
        generated_seq_ids = []
        for seq, token_id in zip(seqs, token_ids): # token_id: 采样出的下一个 token
            self.stats["recompute_tokens"] += max(0, min(seq.num_cached_tokens + seq.num_scheduled_tokens,
                getattr(seq, "recompute_upto", 0)) - seq.num_cached_tokens)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # 一个长 Prompt 可能被拆成多轮计算。如果本轮只处理了部分 Prompt，就不能把模型采样结果当作正式输出，因此直接进入下一个序列
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            generated_seq_ids.append(seq.seq_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return generated_seq_ids

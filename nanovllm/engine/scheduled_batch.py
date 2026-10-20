"""CPU-owned execution metadata; token rows are Decode first, then Prefill."""

from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


def tp_shm_capacity(max_seqs, max_model_len, block_size, policy="legacy"):
    if policy == "legacy":
        return 2**20
    # Pickle's int32 opcode uses up to five bytes; leave per-sequence metadata
    # slack. A prefix hit reduces queries, not the transmitted token history.
    pages = (max_model_len + block_size - 1) // block_size
    return max(2**20, max_seqs * (5 * max_model_len + 5 * pages + 256) + 4096)


@dataclass(slots=True)
class ScheduledBatch:
    seqs: list[Sequence]
    prefill: tuple[bool, ...]
    query_lens: tuple[int, ...]
    prefill_budget: int = 0
    budget_reason: str = "legacy"

    @property
    def num_decode(self):
        return sum(not flag for flag in self.prefill)

    @property
    def prefill_tokens(self):
        return sum(n for n, flag in zip(self.query_lens, self.prefill) if flag)

    @property
    def phase(self):
        if not self.num_decode:
            return "prefill"
        return "mixed" if self.prefill_tokens else "decode"


class PrefillBudgetController:
    """AIMD with hysteresis. Feedback is CPU-visible token time, not launch time."""

    def __init__(self, config):
        self.minimum = config.prefill_budget_min
        self.maximum = config.prefill_budget_max
        self.budget = min(self.maximum, max(self.minimum, config.prefill_token_budget))
        self.target_ms = config.itl_slo_ms
        self.ema_ms = None
        self.steps = self.low_windows = 0
        self.checked_steps = 0
        self.reason = "initial"
        self.over_target = False

    def observe(self, gap_ms):
        self.ema_ms = gap_ms if self.ema_ms is None else 0.2 * gap_ms + 0.8 * self.ema_ms
        self.steps += 1
        self.over_target = gap_ms > self.target_ms
        if self.over_target:
            self._decrease("itl_exceeded")
        elif self.steps % 4 == 0 and self.ema_ms >= 0.9 * self.target_ms:
            self._decrease("ema_high")

    def _decrease(self, reason):
        self.budget = max(self.minimum, (self.budget // 2 // 64) * 64)
        self.low_windows = 0
        self.reason = reason

    def choose(self, waiting, pressure):
        if self.steps and self.steps % 4 == 0 and self.checked_steps != self.steps:
            if self.ema_ms < 0.75 * self.target_ms and waiting and pressure < 0.85:
                self.low_windows += 1
                if self.low_windows >= 2:
                    self.budget = min(self.maximum, self.budget + 64)
                    self.low_windows = 0
                    self.reason = "stable_increase"
            else:
                self.low_windows = 0
            # Consume each feedback window once, including repeated schedule calls.
            self.checked_steps = self.steps
        if not waiting:
            return 0, "no_waiting"
        return self.budget, self.reason

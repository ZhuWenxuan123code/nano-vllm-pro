"""Persistent Decode inputs. Host writes must follow the previous H2D event."""

import numpy as np
import torch


def graph_batch_sizes(max_batch_size: int) -> list[int]:
    limit = min(max_batch_size, 512)
    return sorted({x for x in (1, 2, 4, 8, *range(16, limit + 1, 16), limit)
                   if 0 < x <= limit})


class DecodeInputBuffers:

    def __init__(self, max_seqs, max_model_len, block_size, device="cuda", sample=True):
        self.max_seqs = max_seqs
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.sample = sample
        self.device = torch.device(device)
        self.max_blocks = (max_model_len + block_size - 1) // block_size
        shapes = {
            "tokens_positions": ((2, max_seqs), torch.int64),
            "slots_lengths": ((2, max_seqs), torch.int32),
            "block_tables": ((max_seqs, self.max_blocks), torch.int32),
        }
        if sample:
            shapes["temperatures"] = ((max_seqs,), torch.float32)
        self.host = {
            name: torch.zeros(shape, dtype=dtype, device="cpu",
                              pin_memory=self.device.type == "cuda")
            for name, (shape, dtype) in shapes.items()
        }
        self.arrays = {name: tensor.numpy() for name, tensor in self.host.items()}
        self.arrays["slots_lengths"][0].fill(-1)
        self.arrays["block_tables"].fill(-1)
        self.gpu = {name: tensor.to(self.device, copy=True) for name, tensor in self.host.items()}
        self.inputs = {
            "input_ids": self.gpu["tokens_positions"][0],
            "positions": self.gpu["tokens_positions"][1],
            "slot_mapping": self.gpu["slots_lengths"][0],
            "context_lens": self.gpu["slots_lengths"][1],
            "block_tables": self.gpu["block_tables"],
        }
        self.rows = [None] * max_seqs
        self.previous_bs = 0
        self.dirty_rows = None
        self.dirty_temperatures = False
        self.copy_done = torch.cuda.Event() if self.device.type == "cuda" else None
        self.copy_pending = False
        self.copies = 0
        self.copied_bytes = 0

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.gpu.values())

    def wait_for_host(self):
        if self.copy_pending:
            self.copy_done.synchronize()
            self.copy_pending = False

    def fill(self, seqs):
        """CPU only, after wait_for_host; copies snapshots, never aliases block tables."""
        bs = len(seqs)
        if not 0 < bs <= self.max_seqs:
            raise ValueError("Decode batch exceeds persistent buffer capacity")
        tokens, positions = self.arrays["tokens_positions"]
        slots, lengths = self.arrays["slots_lengths"]
        slots.fill(-1)
        lengths.fill(0)
        tokens[bs:] = 0
        positions[bs:] = 0
        dirty = []
        for row, seq in enumerate(seqs):
            length = len(seq)
            table = tuple(seq.block_table)
            if not 0 < length <= self.max_model_len or len(table) > self.max_blocks:
                raise ValueError("Decode sequence exceeds persistent buffer capacity")
            block_index = (length - 1) // self.block_size
            if block_index >= len(table):
                raise ValueError("Decode input is missing a KV block")
            tokens[row] = seq.last_token
            positions[row] = length - 1
            lengths[row] = length
            slots[row] = table[block_index] * self.block_size + (length - 1) % self.block_size
            identity = (getattr(seq, "seq_id", None), table)
            if self.rows[row] != identity:
                self.arrays["block_tables"][row].fill(-1)
                self.arrays["block_tables"][row, :len(table)] = table
                self.rows[row] = identity
                dirty.append(row)
        for row in range(bs, self.previous_bs):
            self.arrays["block_tables"][row].fill(-1)
            self.rows[row] = None
            dirty.append(row)
        self.dirty_rows = (min(dirty), max(dirty) + 1) if dirty else None
        if self.sample:
            temperatures = self.arrays["temperatures"]
            values = np.fromiter((seq.temperature for seq in seqs), dtype=np.float32, count=bs)
            self.dirty_temperatures = bs != self.previous_bs or not np.array_equal(temperatures[:bs], values)
            if self.dirty_temperatures:
                temperatures[:bs] = values
                temperatures[bs:] = 1.0
        self.previous_bs = bs

    def _copy(self, name, rows=None):
        source, target = self.host[name], self.gpu[name]
        if rows is not None:
            source, target = source[rows[0]:rows[1]], target[rows[0]:rows[1]]
        target.copy_(source, non_blocking=True)
        self.copies += 1
        self.copied_bytes += source.numel() * source.element_size()

    def upload(self):
        self._copy("tokens_positions")
        self._copy("slots_lengths")
        if self.dirty_rows is not None:
            self._copy("block_tables", self.dirty_rows)
        if self.sample and self.dirty_temperatures:
            self._copy("temperatures")
        if self.copy_done is not None:
            self.copy_done.record()
            self.copy_pending = True

    def temperatures(self, bs):
        return self.gpu["temperatures"][:bs]

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nanovllm.engine.decode_buffers import DecodeInputBuffers, graph_batch_sizes
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_sequence(length, blocks, temperature=0.6):
    seq = Sequence(list(range(length)), SamplingParams(temperature=temperature))
    seq.block_table = list(blocks)
    return seq


class DecodeBuffersTest(unittest.TestCase):

    def setUp(self):
        self.buffers = DecodeInputBuffers(4, 768, 256, device="cpu")

    def prepare(self, seqs):
        self.buffers.wait_for_host()
        self.buffers.fill(seqs)
        self.buffers.upload()

    def test_block_boundaries_and_padding(self):
        seqs = [make_sequence(255, [3]), make_sequence(256, [4]),
                make_sequence(257, [5, 9])]
        self.prepare(seqs)
        inputs = self.buffers.inputs
        self.assertEqual(inputs["slot_mapping"].tolist(), [3 * 256 + 254, 4 * 256 + 255, 9 * 256, -1])
        self.assertEqual(inputs["positions"].tolist(), [254, 255, 256, 0])
        self.assertEqual(inputs["context_lens"].tolist(), [255, 256, 257, 0])
        self.assertEqual(inputs["block_tables"][2].tolist(), [5, 9, -1])

    def test_reuse_has_two_copies_and_stable_addresses(self):
        seq = make_sequence(10, [3])
        self.prepare([seq])
        pointers = {name: t.data_ptr() for name, t in self.buffers.inputs.items()}
        copies = self.buffers.copies
        seq.append_token(123)
        self.prepare([seq])
        self.assertEqual(self.buffers.copies - copies, 2)
        self.assertEqual(self.buffers.inputs["input_ids"][0].item(), 123)
        self.assertEqual(pointers, {name: t.data_ptr() for name, t in self.buffers.inputs.items()})

    def test_reorder_shrink_and_join_clear_stale_rows(self):
        a, b = make_sequence(257, [2, 3]), make_sequence(10, [4], 0.8)
        self.prepare([a, b])
        self.prepare([b, a])
        self.assertEqual(self.buffers.inputs["block_tables"][0].tolist(), [4, -1, -1])
        self.prepare([b])
        self.assertEqual(self.buffers.inputs["block_tables"][1].tolist(), [-1, -1, -1])
        self.assertEqual(self.buffers.inputs["slot_mapping"][1].item(), -1)
        self.assertEqual(self.buffers.inputs["context_lens"][1].item(), 0)
        c = make_sequence(3, [6], 1.1)
        self.prepare([b, c])
        np.testing.assert_allclose(self.buffers.temperatures(2).numpy(), [0.8, 1.1])

    def test_block_table_snapshot_detects_in_place_change(self):
        seq = make_sequence(256, [2])
        self.prepare([seq])
        seq.append_token(8)
        seq.block_table.append(7)
        self.prepare([seq])
        self.assertEqual(self.buffers.inputs["block_tables"][0].tolist(), [2, 7, -1])
        seq.block_table[:] = [8, 9]
        self.prepare([seq])
        self.assertEqual(self.buffers.inputs["block_tables"][0].tolist(), [8, 9, -1])

    def test_temperature_change_without_batch_change(self):
        seq = make_sequence(1, [2])
        self.prepare([seq])
        copies = self.buffers.copies
        seq.temperature = 1.2
        self.prepare([seq])
        self.assertEqual(self.buffers.copies - copies, 3)
        self.assertAlmostEqual(self.buffers.temperatures(1).item(), 1.2, places=6)

    def test_tp_worker_does_not_require_temperature_or_sequence_id(self):
        seq = make_sequence(2, [3])
        worker_seq = Sequence.__new__(Sequence)
        seq.is_prefill = False
        worker_seq.__setstate__(seq.__getstate__())
        buffers = DecodeInputBuffers(2, 256, 256, device="cpu", sample=False)
        buffers.fill([worker_seq])
        buffers.upload()
        self.assertEqual(buffers.inputs["input_ids"][0].item(), seq.last_token)
        self.assertNotIn("temperatures", buffers.gpu)

    def test_host_buffer_is_not_reused_before_copy_completes(self):
        calls = []
        self.buffers.copy_done = SimpleNamespace(synchronize=lambda: calls.append("wait"))
        self.buffers.copy_pending = True
        self.buffers.wait_for_host()
        self.buffers.wait_for_host()
        self.assertEqual(calls, ["wait"])

    def test_capacity_errors(self):
        for seqs in ([], [make_sequence(1, [0])] * 5, [make_sequence(769, [1, 2, 3, 4])],
                     [make_sequence(257, [1])]):
            with self.subTest(length=len(seqs)), self.assertRaises(ValueError):
                self.buffers.fill(seqs)

    def test_graph_buckets_cover_irregular_limits(self):
        for limit in (1, 3, 7, 15, 17, 63, 64, 513):
            buckets = graph_batch_sizes(limit)
            self.assertEqual(buckets[-1], min(limit, 512))
            self.assertTrue(all(0 < bucket <= min(limit, 512) for bucket in buckets))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeBuffersCudaTest(unittest.TestCase):

    def test_graph_observes_updates_without_recapture(self):
        buffers = DecodeInputBuffers(3, 512, 256)
        graph = torch.cuda.CUDAGraph()
        inputs = buffers.inputs
        out = torch.empty(3, dtype=torch.int64, device="cuda")
        with torch.cuda.graph(graph):
            out.copy_(inputs["input_ids"] + inputs["positions"])
        for seqs in ([make_sequence(255, [3]), make_sequence(257, [4, 5])],
                     [make_sequence(1, [7])]):
            buffers.wait_for_host()
            buffers.fill(seqs)
            buffers.upload()
            graph.replay()
            expected = [seq.last_token + len(seq) - 1 for seq in seqs] + [0] * (3 - len(seqs))
            self.assertEqual(out.cpu().tolist(), expected)
            self.assertTrue(buffers.host["tokens_positions"].is_pinned())


if __name__ == "__main__":
    unittest.main()

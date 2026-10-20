"""Mixed-vs-separate teacher forcing; phase changes, paging and TP lifecycle."""

import atexit
import os
import unittest
from unittest.mock import patch

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import Context, get_context, reset_context


class P5SmallCacheRunner(ModelRunner):

    def allocate_kv_cache(self):
        self.allocate_cache_storage(12)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NANOVLLM_TEST_MODEL"),
                     "CUDA and NANOVLLM_TEST_MODEL required")
class P5IntegrationTest(unittest.TestCase):

    def setUp(self):
        torch._dynamo.reset()
        limit = torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit = 64
        self.addCleanup(setattr, torch._dynamo.config, "recompile_limit", limit)

    def trajectory(self, policy, backend="flash", quantized=False, graph=False, combined=False, separate=False):
        torch._dynamo.reset()
        cfg = Config(model=os.environ["NANOVLLM_TEST_MODEL"], scheduler_policy=policy,
            attention_backend=backend, kv_cache_dtype="int8" if quantized else "auto",
            max_num_seqs=4, max_num_batched_tokens=128, max_model_len=512,
            prefill_token_budget=64, enforce_eager=not graph, fuse_decode_qk_rope_cache=combined,
            execution_mode="buffered" if combined else "original")
        runner = P5SmallCacheRunner(cfg, 0, [])
        original_attention = Attention.forward
        def separate_attention(layer, q, k, v):
            context = get_context()
            if context.phase != "mixed":
                return original_attention(layer, q, k, v)
            count = context.num_decode
            decode_context = Context(slot_mapping=context.slot_mapping[:count],
                context_lens=context.context_lens, block_tables=context.block_tables[:count])
            with patch("nanovllm.utils.context._CONTEXT", decode_context):
                decode = original_attention(layer, q[:count], k[:count], v[:count])
            prefill_context = Context(is_prefill=True, phase="prefill", cu_seqlens_q=context.prefill_cu_q,
                cu_seqlens_k=context.prefill_cu_k, max_seqlen_q=context.prefill_max_q,
                max_seqlen_k=context.prefill_max_k, slot_mapping=context.slot_mapping[count:],
                block_tables=context.block_tables[count:] if context.prefill_uses_cache else None)
            with patch("nanovllm.utils.context._CONTEXT", prefill_context):
                prefill = original_attention(layer, q[count:], k[count:], v[count:])
            return torch.cat((decode.reshape(count, layer.num_heads, layer.head_dim), prefill))
        scheduler = Scheduler(cfg)
        records, identities, phases = {}, {}, set()
        added = False
        preempted = False
        def add(length, outputs=8):
            seq = Sequence([20 + i % 100 for i in range(length)],
                           SamplingParams(ignore_eos=True, max_tokens=outputs))
            identities[seq.seq_id] = len(identities)
            scheduler.add(seq)
        add(20, 12)
        try:
            for iteration in range(200):
                if scheduler.is_finished():
                    if len(identities) == 4:
                        add(257, 2)  # released prefix and physical page reuse
                    else:
                        break
                batch = scheduler.schedule_batch()
                phases.add(batch.phase)
                with torch.inference_mode():
                    if batch.phase == "mixed":
                        inputs, positions = runner.prepare_mixed(batch)
                    elif batch.phase == "prefill":
                        inputs, positions = runner.prepare_prefill(batch.seqs)
                    else:
                        inputs, positions = runner.prepare_decode(batch.seqs)
                    # Keep GEMM shapes and dtype casts identical. The reference
                    # independently invokes the pre-existing Attention paths;
                    # otherwise BF16 projection rounding becomes a confounder.
                    with patch.object(Attention, "forward", separate_attention if separate else original_attention):
                        logits = runner.run_model(inputs, positions, batch.phase != "decode")
                    self.assertEqual(logits.shape[0], len(batch.seqs))
                    for row, (seq, count) in enumerate(zip(batch.seqs, batch.query_lens)):
                        end = seq.num_cached_tokens + count
                        if end != seq.num_tokens:
                            continue
                        block, slot = seq.block_table[(end - 1) // 256], (end - 1) % 256
                        values = runner.kv_cache[:, :, block, slot].float()
                        if quantized:
                            ks, vs = runner.k_scales[:, block, slot], runner.v_scales[:, block, slot]
                            values = torch.stack(((values[0].reshape(*ks.shape, 32) * ks[..., None]).flatten(-2),
                                                  values[1] * vs[..., None]))
                        records[(identities[seq.seq_id], seq.num_completion_tokens)] = (
                            logits[row].cpu(), values.cpu())
                reset_context()
                scheduler.postprocess_batch(batch, [30 + seq.num_completion_tokens for seq in batch.seqs])
                if not preempted:
                    victim = next((s for s in scheduler.running
                        if identities[s.seq_id] == 1 and s.num_completion_tokens == 1), None)
                    if victim is not None:
                        scheduler.running.remove(victim)
                        scheduler.preempt(victim)
                        preempted = True
                if not added:
                    for length in (255, 256, 257):
                        add(length)
                    added = True
            else:
                self.fail("Model trajectory failed to drain")
            self.assertEqual(len(scheduler.block_manager.free_block_ids), 12)
            self.assertTrue(preempted)
            self.assertGreater(scheduler.stats["recompute_tokens"], 0)
        finally:
            reset_context()
            runner.exit()
        if policy != "legacy":
            self.assertIn("mixed", phases)
        return records

    def test_high_precision_mixed_logits_cache_and_graph_transitions(self):
        # Match chunk boundaries: INT8's first chunk uses unquantized K/V;
        # later chunks read quantized pages. Changing boundaries is NOT a
        # comparison of the mixed execution kernel with the same input.
        for policy, backend, graph, combined in (("mixed", "flash", False, False),
                ("adaptive", "flash", True, False), ("mixed", "triton", True, True)):
            with self.subTest(policy=policy, backend=backend, graph=graph, combined=combined):
                reference = self.trajectory(policy, backend, graph=graph, combined=combined, separate=True)
                result = self.trajectory(policy, backend, graph=graph, combined=combined)
                self.assertEqual(reference.keys(), result.keys())
                for key in reference:
                    delta = reference[key][0].float() - result[key][0].float()
                    self.assertLess(delta.square().mean().sqrt().item(), 0.08)
                    torch.testing.assert_close(reference[key][0], result[key][0], atol=0.4, rtol=0.03)
                    amplitude = reference[key][1].abs().amax(-1, keepdim=True).clamp_min(1)
                    self.assertLess(((reference[key][1] - result[key][1]) / amplitude).abs().max().item(), 0.06)

    def test_int8_mixed_logits_cache(self):
        reference = self.trajectory("mixed", "triton", quantized=True, graph=True, combined=True, separate=True)
        result = self.trajectory("mixed", "triton", quantized=True, graph=True, combined=True)
        self.assertEqual(reference.keys(), result.keys())
        for key in reference:
            torch.testing.assert_close(reference[key][0], result[key][0], atol=0.4, rtol=0.03)
            amplitude = reference[key][1].abs().amax(-1, keepdim=True).clamp_min(1)
            self.assertLess(((reference[key][1] - result[key][1]) / amplitude).abs().max().item(), 0.06)

    @unittest.skipUnless(os.environ.get("NANOVLLM_TEST_TP") == "2" and torch.cuda.device_count() >= 2,
                         "NANOVLLM_TEST_TP=2 and two idle GPUs required")
    def test_tp2_mixed_int8_and_cache_release(self):
        with patch("nanovllm.engine.llm_engine.ModelRunner", P5SmallCacheRunner):
            llm = LLM(os.environ["NANOVLLM_TEST_MODEL"], scheduler_policy="mixed", tensor_parallel_size=2,
                max_num_seqs=4, max_num_batched_tokens=128, max_model_len=512, prefill_token_budget=64,
                attention_backend="triton", kv_cache_dtype="int8", execution_mode="buffered",
                fuse_decode_qk_rope_cache=True)
            try:
                llm.add_request([20] * 20, SamplingParams(ignore_eos=True, max_tokens=12))
                llm.step_with_info()
                for length in (255, 256, 257):
                    llm.add_request([30] * length, SamplingParams(ignore_eos=True, max_tokens=4))
                phases, finished = set(), 0
                while not llm.is_finished():
                    outputs, info = llm.step_with_info()
                    phases.add(info.phase)
                    finished += len(outputs)
                self.assertIn("mixed", phases)
                self.assertEqual(finished, 4)
                self.assertEqual(len(llm.scheduler.block_manager.free_block_ids), 12)
                ranks = llm.model_runner.call("cache_metadata")
                self.assertEqual([r["format"] for r in ranks], ["int8-k32-vhead-v1"] * 2)
            finally:
                atexit.unregister(llm.exit)
                llm.exit()


if __name__ == "__main__":
    unittest.main()

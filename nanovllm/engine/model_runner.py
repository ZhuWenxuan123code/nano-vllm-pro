import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.decode_buffers import DecodeInputBuffers, graph_batch_sizes
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.attention import Attention
from nanovllm.layers.paged_attention import AttentionWorkspace
from nanovllm.layers.kv_quantization import K_QUANT_GROUP_SIZE, INT8_CACHE_FORMAT
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.profiling import profile_range


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.decode_buffers = None

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        for module in self.model.modules():
            if isinstance(module, RMSNorm):
                module.set_backend(config.rms_norm_backend)
            elif isinstance(module, Qwen3Attention):
                module.fuse_decode_qk_rope_cache = config.fuse_decode_qk_rope_cache
            elif isinstance(module, Attention):
                module.backend = config.attention_backend
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        if config.execution_mode == "buffered":
            self.decode_buffers = DecodeInputBuffers(
                config.max_num_seqs, config.max_model_len, self.block_size,
                device=f"cuda:{rank}", sample=rank == 0,
            )
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        torch.cuda.synchronize()
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.attention_workspace = None
        if config.attention_backend == "triton":
            self.attention_workspace = AttentionWorkspace(config.max_num_seqs,
                hf_config.num_attention_heads // self.world_size, head_dim, f"cuda:{self.rank}")
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        quantized = config.kv_cache_dtype == "int8"
        dtype = torch.int8 if quantized else hf_config.dtype
        data_block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize
        scale_block_bytes = hf_config.num_hidden_layers * self.block_size * num_kv_heads * (head_dim // K_QUANT_GROUP_SIZE + 1) * 4 if quantized else 0
        block_bytes = data_block_bytes + scale_block_bytes
        # A block is naturally 256-byte aligned; reserve allocator alignment slack
        # for data/K scales/V scales. Workspace is already included in `used`.
        budget = int(total * config.gpu_memory_utilization - used - max(0, peak - current) - 768)
        config.num_kvcache_blocks = budget // block_bytes
        if self.world_size > 1:
            # All ranks must accept the same physical block IDs, even when
            # their free memory or peak activation reservations differ.
            capacity = torch.tensor(config.num_kvcache_blocks, dtype=torch.int64,
                                    device=f"cuda:{self.rank}")
            dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(capacity.item())
        assert config.num_kvcache_blocks > 0
        self.allocate_cache_storage(config.num_kvcache_blocks)

    def allocate_cache_storage(self, num_blocks):
        """Allocate one fixed format; also used by bounded-cache correctness tests."""
        config, hf = self.config, self.config.hf_config
        config.num_kvcache_blocks = num_blocks
        heads = hf.num_key_value_heads // self.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        quantized = config.kv_cache_dtype == "int8"
        if config.attention_backend == "triton" and getattr(self, "attention_workspace", None) is None:
            self.attention_workspace = AttentionWorkspace(config.max_num_seqs,
                hf.num_attention_heads // self.world_size, dim, f"cuda:{self.rank}")
        self.kv_cache = torch.empty(2, hf.num_hidden_layers, num_blocks, self.block_size, heads, dim,
                                   dtype=torch.int8 if quantized else hf.dtype, device=f"cuda:{self.rank}")
        scale_shape = (hf.num_hidden_layers, num_blocks, self.block_size, heads)
        self.k_scales = (torch.empty(*scale_shape, dim // K_QUANT_GROUP_SIZE, dtype=torch.float32,
                                    device=f"cuda:{self.rank}") if quantized else None)
        self.v_scales = (torch.empty(scale_shape, dtype=torch.float32,
                                    device=f"cuda:{self.rank}") if quantized else None)
        data_bytes = self.kv_cache.numel() * self.kv_cache.element_size()
        k_scale_bytes = self.k_scales.numel() * 4 if quantized else 0
        v_scale_bytes = self.v_scales.numel() * 4 if quantized else 0
        scale_bytes = k_scale_bytes + v_scale_bytes
        self.cache_memory = {"data_bytes": data_bytes, "scale_bytes": scale_bytes,
            "k_scale_bytes": k_scale_bytes, "v_scale_bytes": v_scale_bytes,
            "format": INT8_CACHE_FORMAT if quantized else str(hf.dtype),
            "k_quant_group_size": K_QUANT_GROUP_SIZE if quantized else None,
            "workspace_bytes": self.attention_workspace.nbytes if getattr(self, "attention_workspace", None) else 0,
            "bytes_per_token_per_rank": (data_bytes + scale_bytes) // (num_blocks * self.block_size),
            "token_slots": num_blocks * self.block_size, "blocks": num_blocks, "allocation_alignment_reserve_bytes": 768}
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.k_scale = self.k_scales[layer_id] if quantized else None
                module.v_scale = self.v_scales[layer_id] if quantized else None
                module.workspace = getattr(self, "attention_workspace", None)
                layer_id += 1

    def cache_metadata(self):
        metadata = {"rank": self.rank, "attention_backend": self.config.attention_backend,
                    "format": self.cache_memory["format"], "blocks": self.config.num_kvcache_blocks,
                    "kv_heads": self.kv_cache.shape[-2], "k_quant_group_size": self.cache_memory["k_quant_group_size"]}
        if self.world_size == 1:
            return [metadata]
        ranks = [None] * self.world_size
        dist.all_gather_object(ranks, metadata)
        return ranks

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens # 已缓存的前缀长度
            seqlen_q = seq.num_scheduled_tokens # 本次要算多少个 token
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self.decode_buffers is not None:
            buffers = self.decode_buffers
            with profile_range("nanovllm::host_buffer_wait", self.config.profile_stages):
                buffers.wait_for_host()
            with profile_range("nanovllm::decode_metadata", self.config.profile_stages):
                buffers.fill(seqs)
            with profile_range("nanovllm::h2d", self.config.profile_stages):
                buffers.upload()
            bs = len(seqs)
            inputs = buffers.inputs
            set_context(False, slot_mapping=inputs["slot_mapping"][:bs],
                        context_lens=inputs["context_lens"][:bs],
                        block_tables=inputs["block_tables"][:bs])
            return inputs["input_ids"][:bs], inputs["positions"][:bs]
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            if self.decode_buffers is None:
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        enabled = self.config.profile_stages
        try:
            with profile_range("nanovllm::prepare_inputs", enabled):
                input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
                temperatures = None
                if self.rank == 0:
                    temperatures = (self.decode_buffers.temperatures(len(seqs))
                                    if not is_prefill and self.decode_buffers is not None
                                    else self.prepare_sample(seqs))
            with profile_range("nanovllm::model", enabled):
                logits = self.run_model(input_ids, positions, is_prefill)
            if self.rank != 0:
                return None
            with profile_range("nanovllm::sample", enabled):
                token_ids = self.sampler(logits, temperatures)
            with profile_range("nanovllm::d2h", enabled):
                return token_ids.tolist()
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self): # for decode
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        if self.decode_buffers is None:
            input_ids = torch.zeros(max_bs, dtype=torch.int64)
            positions = torch.zeros(max_bs, dtype=torch.int64)
            slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
            context_lens = torch.zeros(max_bs, dtype=torch.int32)
            block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        else:
            inputs = self.decode_buffers.inputs
            input_ids, positions = inputs["input_ids"], inputs["positions"]
            slot_mapping, context_lens = inputs["slot_mapping"], inputs["context_lens"]
            block_tables = inputs["block_tables"]
            # Valid dummy cache addresses during capture; first upload replaces them.
            slot_mapping.zero_()
            block_tables.zero_()
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = graph_batch_sizes(max_bs)
        self.graphs = {}
        self.graph_pool = None

        # 从大 batch 往小 batch 依次捕获。这样首张图创建的 memory pool 可以被后面图复用
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if self.decode_buffers is not None:
            block_tables.fill_(-1)
            slot_mapping.fill_(-1)

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    rms_norm_backend: str = "compiled"
    fuse_decode_qk_rope_cache: bool = False
    execution_mode: str = "original"
    attention_backend: str = "flash"
    kv_cache_dtype: str = "auto"
    profile_stages: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if self.attention_backend not in ("flash", "triton"):
            raise ValueError(f"unsupported attention backend: {self.attention_backend}")
        if self.kv_cache_dtype not in ("auto", "int8"):
            raise ValueError(f"unsupported KV cache dtype: {self.kv_cache_dtype}")
        if self.kv_cache_dtype == "int8" and self.attention_backend != "triton":
            raise ValueError("INT8 KV cache requires attention_backend='triton'")
        if self.execution_mode not in ("original", "buffered"):
            raise ValueError(f"unsupported execution mode: {self.execution_mode}")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.rms_norm_backend not in ("compiled", "triton"):
            raise ValueError(f"unsupported RMSNorm backend: {self.rms_norm_backend}")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.attention_backend == "triton":
            heads = self.hf_config.num_attention_heads
            kv_heads = self.hf_config.num_key_value_heads
            dim = getattr(self.hf_config, "head_dim", self.hf_config.hidden_size // heads)
            if (dim not in (64, 128, 256) or heads % kv_heads or
                    heads // kv_heads not in (1, 2, 4, 8, 16) or
                    heads % self.tensor_parallel_size or kv_heads % self.tensor_parallel_size or
                    str(self.hf_config.dtype) not in ("torch.float16", "torch.bfloat16")):
                raise ValueError("Triton Attention requires FP16/BF16, supported GQA/head dimension and divisible TP heads")

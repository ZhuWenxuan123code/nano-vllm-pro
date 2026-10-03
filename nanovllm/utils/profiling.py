"""Optional nested CPU/NVTX ranges; disabled in normal benchmarks."""

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def profile_range(name, enabled=False):
    if not enabled:
        yield
        return
    nvtx = torch.cuda.nvtx.range(name) if torch.cuda.is_available() else nullcontext()
    with torch.profiler.record_function(name), nvtx:
        yield

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from benchmarks.profile_workload import (
    output_prefix,
    parse_args,
    summarize_torch_profiler,
)


class ProfileHelpersTest(unittest.TestCase):

    def test_output_prefix_distinguishes_eager_and_cudagraph(self):
        eager = SimpleNamespace(
            phase="decode", enforce_eager=True, output_dir=Path("profiles")
        )
        graph = SimpleNamespace(
            phase="decode", enforce_eager=False, output_dir=Path("profiles")
        )
        self.assertEqual(output_prefix(eager), Path("profiles/decode-eager"))
        self.assertEqual(output_prefix(graph), Path("profiles/decode-cudagraph"))

        prefill = SimpleNamespace(
            phase="prefill", enforce_eager=False, output_dir=Path("profiles")
        )
        self.assertEqual(output_prefix(prefill), Path("profiles/prefill-eager"))

    def test_prefill_requires_one_full_batch_per_step(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(
                [
                    "--phase", "prefill",
                    "--batch-size", "8",
                    "--input-len", "1024",
                    "--max-num-batched-tokens", "4096",
                ]
            )

    def test_profiler_summary_excludes_annotations_and_memcpy(self):
        events = [
            SimpleNamespace(
                device_type="DeviceType.CUDA",
                linked_correlation_id=1,
                name="test_kernel",
                device_time_total=10.0,
            ),
            SimpleNamespace(
                device_type="DeviceType.CUDA",
                linked_correlation_id=0,
                name="nanovllm::decode_step_0",
                device_time_total=100.0,
            ),
            SimpleNamespace(
                device_type="DeviceType.CUDA",
                linked_correlation_id=2,
                name="Memcpy DtoD",
                device_time_total=2.0,
            ),
        ]
        averages = [
            SimpleNamespace(
                device_type="DeviceType.CPU",
                key="aten::mm",
                count=2,
                self_device_time_total=10.0,
            ),
            SimpleNamespace(
                device_type="DeviceType.CPU",
                key="nanovllm::decode_step_0",
                count=1,
                self_device_time_total=100.0,
            ),
            SimpleNamespace(
                device_type="DeviceType.CPU",
                key="cudaLaunchKernel",
                count=3,
                self_device_time_total=0.0,
            ),
        ]
        profiler = SimpleNamespace(
            events=lambda: events,
            key_averages=lambda: averages,
        )

        summary = summarize_torch_profiler(profiler)

        self.assertEqual(summary["cuda_activity_count"], 2)
        self.assertEqual(summary["kernel_activity_count"], 1)
        self.assertEqual(summary["cuda_launch_api_calls"], 3)
        self.assertEqual(summary["total_kernel_cuda_time_us"], 10.0)
        self.assertEqual(summary["top_kernels"][0]["name"], "test_kernel")
        self.assertEqual(summary["top_operators"][0]["name"], "aten::mm")


if __name__ == "__main__":
    unittest.main()

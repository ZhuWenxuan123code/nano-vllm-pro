import argparse
import json
import statistics
from pathlib import Path


METRICS = {
    "E2E tok/s": ("e2e_throughput",),
    "Prefill tok/s": ("prefill_throughput",),
    "Decode tok/s": ("decode_throughput",),
    "TTFT P50 ms": ("ttft_ms", "p50"),
    "TTFT P99 ms": ("ttft_ms", "p99"),
    "TPOT P50 ms": ("tpot_ms", "p50"),
    "TPOT P99 ms": ("tpot_ms", "p99"),
}


def metric_value(report, path):
    value = report["metrics"]
    for key in path:
        value = value[key]
    return value


def load_cases(result_dir):
    cases = {}
    for path in sorted(result_dir.glob("*.json")):
        case, separator, run = path.stem.rpartition("-")
        if not separator or not run.isdigit():
            continue
        cases.setdefault(case, []).append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Summarize nano-vLLM benchmark JSON files.")
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results/baseline"),
    )
    args = parser.parse_args()

    cases = load_cases(args.result_dir)
    if not cases:
        parser.error(f"no '<case>-<run>.json' files found in {args.result_dir}")

    for case, reports in cases.items():
        print(f"\n[{case}] runs={len(reports)}")
        for name, path in METRICS.items():
            values = [metric_value(report, path) for report in reports]
            values = [value for value in values if value is not None]
            if not values:
                continue
            mean = statistics.mean(values)
            median = statistics.median(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            print(
                f"{name:18} mean={mean:10.2f} "
                f"median={median:10.2f} std={stdev:8.2f}"
            )


if __name__ == "__main__":
    main()

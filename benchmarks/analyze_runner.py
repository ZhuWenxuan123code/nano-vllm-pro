"""Summarize P3 NVTX ranges and GPU intervals from Nsight SQLite exports."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def union_duration(intervals, start, end):
    total = 0
    cursor = start
    for left, right in sorted(intervals):
        left, right = max(left, start, cursor), min(right, end)
        if right > left:
            total += right - left
            cursor = right
    return total


def summarize_intervals(ranges, activities, discard_steps=5):
    steps = sorted((start, end) for start, end, name in ranges
                   if name.startswith("nanovllm::decode_step_"))
    steps = steps[discard_steps:]
    if not steps:
        raise ValueError("No Decode steps after discarding startup steps")
    start, end = steps[0][0], steps[-1][1]
    durations = defaultdict(list)
    for left, right, name in ranges:
        if left >= start and right <= end and not name.startswith("nanovllm::decode_step_"):
            durations[name].append((right - left) / 1000)
    metadata = [(left, right) for left, right, name in ranges
                if name == "nanovllm::decode_metadata" and left >= start and right <= end]
    if len(metadata) != len(steps):
        raise ValueError("Requires buffered --profile-stages, one metadata range per Decode step")
    gpu_intervals = [(left, right) for left, right, _ in activities]
    eligible_us, ratios, idle_us, step_us = [], [], [], []
    for index, (left, right) in enumerate(steps):
        # Include the inter-step host gap. Never sum nested GPU/NVTX durations.
        stop = steps[index + 1][0] if index + 1 < len(steps) else right
        elapsed = stop - left
        idle = elapsed - union_duration(gpu_intervals, left, stop)
        cpu = union_duration(metadata, left, right)
        eligible = min(cpu, idle)
        eligible_us.append(eligible / 1000)
        ratios.append(eligible / elapsed if elapsed else 0)
        idle_us.append(idle / 1000)
        step_us.append(elapsed / 1000)
    counts = defaultdict(int)
    for left, right, kind in activities:
        if left >= start and right <= end:
            counts[kind] += 1
    return {
        "steps": len(steps),
        "window_ms": (end - start) / 1e6,
        "step_median_us": median(step_us),
        "gpu_idle_median_us": median(idle_us),
        "overlap_upper_bound_median_us": median(eligible_us),
        "overlap_upper_bound_ratio": median(ratios),
        "activity_counts": dict(counts),
        "cpu_ranges": {name: {"count": len(values), "median_us": median(values),
                              "sum_us": sum(values)} for name, values in durations.items()},
    }


def read_trace(path):
    import sqlite3

    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "NVTX_EVENTS" not in tables:
            raise ValueError(f"Missing NVTX ranges: {path}")
        ranges = db.execute(
            "SELECT n.start, n.end, COALESCE(n.text, s.value, '') "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE n.end IS NOT NULL"
        ).fetchall()
        activities = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table in tables:
                activities.extend((a, b, kind) for a, b in db.execute(f"SELECT start, end FROM {table}"))
        if not any(kind == "kernel" for _, _, kind in activities):
            raise ValueError("No kernel records; capture with --cuda-graph-trace=node")
        columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        if "graphNodeId" not in columns or not db.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId > 0"
        ).fetchone()[0]:
            raise ValueError("The gate requires CUDA Graph node traces, not only graph-external kernels")
        return ranges, activities


def evaluate_gate(reports):
    if len(reports) != 3 or any(report["steps"] < 100 for report in reports):
        raise ValueError("The P3 gate requires three independent captures with >=100 retained steps each")
    estimates = [report["overlap_upper_bound_median_us"] for report in reports]
    center = median(estimates)
    mad = median(abs(value - center) for value in estimates)
    threshold_runs = sum(report["overlap_upper_bound_ratio"] >= 0.05 for report in reports)
    return {
        "passed": threshold_runs >= 2 and center > 2 * mad,
        "runs_at_least_five_percent": threshold_runs,
        "overlap_upper_bound_median_us": center,
        "run_mad_us": mad,
        "note": "Feasibility estimate, not measured speedup. Requires an otherwise idle GPU and matching captures.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs=3, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    configs = []
    for path in args.traces:
        sidecar = path.with_suffix(".run.json")
        config = json.loads(sidecar.read_text(encoding="utf-8"))["config"]
        expected = {"execution_mode": "buffered", "profile_stages": True,
                    "phase": "decode", "batch_size": 64, "input_len": 128,
                    "tensor_parallel_size": 1, "enforce_eager": False,
                    "fuse_decode_qk_rope_cache": False}
        if any(config.get(key) != value for key, value in expected.items()):
            parser.error(f"Capture configuration does not match the P3 gate: {sidecar}")
        configs.append(config)
    if any(config != configs[0] for config in configs[1:]):
        parser.error("Gate captures must use the same configuration and GPU")
    reports = [{"trace": str(path), **summarize_intervals(*read_trace(path))} for path in args.traces]
    result = {"config": configs[0], "captures": reports, "gate": evaluate_gate(reports)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()

"""Staged, resumable online A/B suite. Timing subprocesses never use a profiler."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median, stdev
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.online import make_trace, trace_hash


WORKLOADS = ("steady", "burst", "mixed", "pressure")
BUDGETS = (64, 128, 256, 512, 1024)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize(root, suite="main", expected_runs=5):
    if suite not in ("main", "combined") or expected_runs < 1:
        raise ValueError("Expected a valid suite and a positive repetition count")
    groups = {}
    for path in (root / suite).rglob("*.json"):
        report = json.loads(path.read_text())
        if report.get("profile_only") or report["metrics"]["timed_out"]:
            raise ValueError(f"Invalid timed result: {path}")
        key = (report["workload"], report["rate"], report["config"]["scheduler_policy"])
        groups.setdefault(key, []).append(report)
    expected_cells = len(WORKLOADS) * 3 * 3 if suite == "main" else 2 * 3
    coverage = dict(suite=suite, expected_runs_per_cell=expected_runs,
                    completed_runs=sum(len(reports) for reports in groups.values()),
                    expected_runs=expected_cells * expected_runs,
                    complete_cells=sum(len(reports) == expected_runs for reports in groups.values()),
                    expected_cells=expected_cells)
    coverage["complete"] = (coverage["completed_runs"] == coverage["expected_runs"] and
                            coverage["complete_cells"] == expected_cells)
    snapshot = (f"Snapshot: {coverage['completed_runs']}/{coverage['expected_runs']} runs; "
                f"{coverage['complete_cells']}/{expected_cells} cells have all {expected_runs} repetitions.")
    lines = ["# P5 online A/B", "", snapshot,
             "Complete matrix." if coverage["complete"] else "PARTIAL matrix: do not present this snapshot as final acceptance.",
             "", "TTFT ≤ 2000 ms and every ITL ≤ 50 ms. Includes drain time.", "",
             "| Workload | req/s | Policy | Runs | Good req/s | Good tok/s | Output tok/s | TTFT P99 ms | ITL P99 ms | SLO miss % |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    aggregate = []
    traces = {}
    diagnostics = ["", "## Scheduling and execution diagnostics", "",
                   "Medians per run. Recompute is divided by executed Prefill tokens; Graph is divided by all steps.", "",
                   "| Workload | req/s | Policy | Queue P99 ms | Submit lag P99 ms | Preemptions | Recompute % | KV peak % | Graph steps % | P2 Decode tokens % |",
                   "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (workload, rate, policy), reports in sorted(groups.items()):
        fingerprints = {(r["trace_sha256"], r["tokens_sha256"], r["cache_memory"]["blocks"],
                         r["code"]["source_sha256"], r["measurement_sha256"], r["gpu_uuid"]) for r in reports}
        if len(fingerprints) != 1:
            raise ValueError("Repeated timing runs have different code/trace/GPU/capacity")
        identity = next(iter(fingerprints))
        old = traces.setdefault((workload, rate), identity)
        if old != identity:
            raise ValueError("A/B policies have different code/trace/GPU/capacity")
        fields = {name: [r["metrics"][name] for r in reports] for name in (
            "goodput_requests", "goodput_tokens", "e2e_throughput", "slo_miss_fraction")}
        ttft = [r["metrics"]["ttft_ms"]["p99"] for r in reports]
        itl = [r["metrics"]["itl_ms"]["p99"] for r in reports]
        row = dict(workload=workload, rate=rate, policy=policy, runs=len(reports),
                   **{name: dict(median=median(values), std=stdev(values) if len(values) > 1 else 0)
                      for name, values in fields.items()}, ttft_p99_ms=median(ttft), itl_p99_ms=median(itl))
        for name in ("ttft_ms", "itl_ms", "queue_wait_ms", "submission_lag_ms"):
            row[name] = {q: median(r["metrics"][name][q] for r in reports) for q in ("p50", "p99")}
        diagnostic_values = {name: [] for name in (
            "preemptions", "recompute_tokens", "prefix_hit_tokens", "admission_pauses",
            "protected_prefill_steps", "recompute_fraction", "kv_peak_fraction",
            "graph_step_fraction", "mixed_step_fraction", "p2_decode_token_fraction")}
        for report in reports:
            metrics = report["metrics"]
            stats = metrics["scheduler_stats"]
            for name in ("preemptions", "recompute_tokens", "prefix_hit_tokens", "admission_pauses",
                         "protected_prefill_steps"):
                diagnostic_values[name].append(stats[name])
            steps = sum(metrics["phases"].values())
            diagnostic_values["recompute_fraction"].append(stats["recompute_tokens"] / max(1, metrics["prefill_tokens"]))
            diagnostic_values["kv_peak_fraction"].append(metrics["kv_used_blocks_peak"] / report["cache_memory"]["blocks"])
            diagnostic_values["graph_step_fraction"].append(metrics["graph_steps"] / max(1, steps))
            diagnostic_values["mixed_step_fraction"].append(metrics["mixed_eager_steps"] / max(1, steps))
            diagnostic_values["p2_decode_token_fraction"].append(metrics["p2_tokens"] / max(1, metrics["decode_tokens"]))
        row["diagnostics"] = {name: dict(median=median(values), std=stdev(values) if len(values) > 1 else 0)
                              for name, values in diagnostic_values.items()}
        aggregate.append(row)
        lines.append(f"| {workload} | {rate:.3f} | {policy} | {len(reports)} | "
                     f"{median(fields['goodput_requests']):.3f} | {median(fields['goodput_tokens']):.2f} | "
                     f"{median(fields['e2e_throughput']):.2f} | {median(ttft):.2f} | {median(itl):.2f} | "
                     f"{100 * median(fields['slo_miss_fraction']):.2f} |")
        d = row["diagnostics"]
        diagnostics.append(f"| {workload} | {rate:.3f} | {policy} | "
                           f"{row['queue_wait_ms']['p99']:.2f} | {row['submission_lag_ms']['p99']:.2f} | "
                           f"{d['preemptions']['median']:.1f} | {100 * d['recompute_fraction']['median']:.2f} | "
                           f"{100 * d['kv_peak_fraction']['median']:.2f} | {100 * d['graph_step_fraction']['median']:.2f} | "
                           f"{100 * d['p2_decode_token_fraction']['median']:.2f} |")
    prefix = "" if suite == "main" else "combined-"
    write_json(root / f"{prefix}coverage.json", coverage)
    write_json(root / f"{prefix}summary.json", aggregate)
    (root / f"{prefix}comparison.md").write_text("\n".join(lines + diagnostics) + "\n")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/results/p5-scheduling-controlled-gpu1"))
    parser.add_argument("--stage", choices=("all", "calibrate", "tune", "main", "combined", "summarize"), default="all")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--tune-requests", type=int, default=128)
    parser.add_argument("--tune-runs", type=int, default=3)
    args = parser.parse_args()
    if min(args.runs, args.requests, args.tune_requests, args.tune_runs) < 1:
        parser.error("Request/repetition counts must be positive")
    if args.stage == "summarize":
        summarize(args.root, expected_runs=args.runs)
        if (args.root / "combined").exists():
            summarize(args.root, "combined", args.runs)
        return
    from bench import code_version
    from transformers import AutoConfig
    import torch
    source = code_version()["source_sha256"]
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    measurement = hashlib.sha256((Path(__file__).parent / "bench_online.py").read_bytes() +
        (Path(__file__).parent / "online.py").read_bytes()).hexdigest()
    vocab = AutoConfig.from_pretrained(args.model).vocab_size
    def trial(workload, rate, policy, budget, path, count, seed=0, saturated=False, combined=False):
        trace = make_trace(workload, count, rate, seed, vocab, saturated)
        blocks = 128 if workload == "pressure" else 512
        if path.exists():
            report = json.loads(path.read_text())
            if (report["code"]["source_sha256"] != source or report["measurement_sha256"] != measurement or
                    report["trace_sha256"] != trace_hash(trace) or report["metrics"]["timed_out"] or
                    report["config"]["prefill_token_budget"] != budget or
                    report["config"]["scheduler_policy"] != policy or report["cache_memory"]["blocks"] != blocks or
                    report["gpu_uuid"] != gpu_uuid or report["config"]["model"] != args.model or
                    report["config"]["attention_backend"] != ("triton" if combined else "flash") or
                    report["config"]["execution_mode"] != ("buffered" if combined else "original") or
                    report["config"]["fuse_decode_qk_rope_cache"] != combined):
                raise ValueError(f"Stale/incompatible result {path}; select a new RESULT_ROOT")
            return report
        trace_path = args.root / "traces" / f"{workload}-{rate:.12g}-{seed}-{count}-{int(saturated)}.json"
        write_json(trace_path, trace)
        command = [sys.executable, "benchmarks/bench_online.py", "--model", args.model,
                   "--trace", str(trace_path), "--scheduler-policy", policy,
                   "--prefill-token-budget", str(budget), "--kv-cache-block-limit", str(blocks),
                   "--seed", str(seed), "--timeout-s", str(max(300, count * 16,
                                                             trace["requests"][-1]["arrival_s"] + 300)),
                   "--output-json", str(path)]
        if combined:
            command += ["--attention-backend", "triton", "--execution-mode", "buffered",
                        "--fuse-decode-qk-rope-cache"]
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"RUN {path}", flush=True)
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        if "torch._dynamo hit config." in path.with_suffix(".log").read_text():
            raise RuntimeError(f"Compilation fallback invalidates {path}; inspect warmup/compile limit")
        report = json.loads(path.read_text())
        print(f"DONE goodput={report['metrics']['goodput_tokens']:.1f} ITL99={report['metrics']['itl_ms']['p99']:.2f}", flush=True)
        return report
    refs_path = args.root / "reference-rates.json"
    if args.stage in ("all", "calibrate"):
        refs = {}
        for workload in WORKLOADS:
            results = [trial(workload, 1, "legacy", 256, args.root / "calibration" / f"{workload}-{i}.json",
                             args.requests, saturated=True) for i in range(1, 4)]
            refs[workload] = median(r["metrics"]["request_throughput"] for r in results)
        write_json(refs_path, dict(source_sha256=source, measurement_sha256=measurement, rates=refs))
    else:
        manifest = json.loads(refs_path.read_text())
        if manifest["source_sha256"] != source or manifest["measurement_sha256"] != measurement:
            raise ValueError("Reference rates belong to different code; use a new RESULT_ROOT")
        refs = manifest["rates"]
    selected_path = args.root / "fixed-budgets.json"
    if args.stage in ("all", "tune"):
        selected = {}
        for workload in WORKLOADS:
            scores = []
            for budget in BUDGETS:
                results = [trial(workload, 0.9 * refs[workload], "mixed", budget,
                                 args.root / "tuning" / workload / f"budget-{budget}-{i}.json",
                                 args.tune_requests, seed=1) for i in range(1, args.tune_runs + 1)]
                scores.append((median(r["metrics"]["goodput_tokens"] for r in results),
                               -median(r["metrics"]["itl_ms"]["p99"] for r in results),
                               -median(r["metrics"]["ttft_ms"]["p99"] for r in results), -budget))
            selected[workload] = -max(scores)[3]
        write_json(selected_path, dict(source_sha256=source, measurement_sha256=measurement, budgets=selected))
    elif args.stage not in ("calibrate",):
        manifest = json.loads(selected_path.read_text())
        if manifest["source_sha256"] != source or manifest["measurement_sha256"] != measurement:
            raise ValueError("Fixed budgets belong to different code; use a new RESULT_ROOT")
        selected = manifest["budgets"]
    if args.stage in ("all", "main", "combined"):
        suites = ("main", "combined") if args.stage == "all" else (args.stage,)
        for suite in suites:
            workloads = WORKLOADS if suite == "main" else ("burst", "mixed")
            factors = (0.7, 0.9, 1.1) if suite == "main" else (0.9,)
            for workload in workloads:
                for factor in factors:
                    for repeat in range(1, args.runs + 1):
                        policies = ("legacy", "mixed", "adaptive")
                        order = policies[(repeat - 1) % 3:] + policies[:(repeat - 1) % 3]
                        for policy in order:
                            trial(workload, factor * refs[workload], policy,
                                  selected[workload] if policy == "mixed" else 256,
                                  args.root / suite / workload / str(factor) / f"{policy}-{repeat}.json",
                                  args.requests, combined=suite == "combined")
        if (args.root / "main").exists():
            summarize(args.root, expected_runs=args.runs)
        if (args.root / "combined").exists():
            summarize(args.root, "combined", args.runs)


if __name__ == "__main__":
    main()

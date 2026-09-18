import argparse
import statistics
from pathlib import Path

try:
    from .summarize import METRICS, load_cases, metric_value
except ImportError:
    from summarize import METRICS, load_cases, metric_value


HIGHER_IS_BETTER = {
    "E2E tok/s",
    "Prefill tok/s",
    "Decode tok/s",
}


def median_metric(reports, path):
    values = [metric_value(report, path) for report in reports]
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def improvement_percent(baseline, candidate, higher_is_better):
    if baseline == 0:
        return None
    ratio = candidate / baseline
    return (ratio - 1 if higher_is_better else 1 / ratio - 1) * 100


def format_value(value):
    return "n/a" if value is None else f"{value:.2f}"


def format_improvement(value):
    return "n/a" if value is None else f"{value:+.2f}%"


def config_mismatches(baseline, candidate):
    baseline_config = baseline[0].get("config", {})
    candidate_config = candidate[0].get("config", {})
    keys = baseline_config.keys() | candidate_config.keys()
    return sorted(
        key
        for key in keys
        if baseline_config.get(key) != candidate_config.get(key)
    )


def compare_cases(baseline_cases, candidate_cases):
    lines = []
    common_cases = sorted(baseline_cases.keys() & candidate_cases.keys())
    missing_baseline = sorted(candidate_cases.keys() - baseline_cases.keys())
    missing_candidate = sorted(baseline_cases.keys() - candidate_cases.keys())

    if missing_baseline:
        lines.append(f"Warning: missing from baseline: {', '.join(missing_baseline)}")
    if missing_candidate:
        lines.append(f"Warning: missing from candidate: {', '.join(missing_candidate)}")
    if lines and common_cases:
        lines.append("")

    for index, case in enumerate(common_cases):
        baseline = baseline_cases[case]
        candidate = candidate_cases[case]
        mismatches = config_mismatches(baseline, candidate)
        lines.extend(
            [
                f"## {case}",
                "",
                f"Baseline runs: {len(baseline)}; candidate runs: {len(candidate)}",
            ]
        )
        if mismatches:
            lines.append(
                "Warning: configuration differs for " + ", ".join(mismatches)
            )
        lines.extend(
            [
                "",
                "| Metric | Baseline median | Candidate median | Improvement |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, path in METRICS.items():
            baseline_value = median_metric(baseline, path)
            candidate_value = median_metric(candidate, path)
            improvement = None
            if baseline_value is not None and candidate_value is not None:
                improvement = improvement_percent(
                    baseline_value,
                    candidate_value,
                    name in HIGHER_IS_BETTER,
                )
            lines.append(
                f"| {name} | {format_value(baseline_value)} | "
                f"{format_value(candidate_value)} | {format_improvement(improvement)} |"
            )
        if index < len(common_cases) - 1:
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate nano-vLLM benchmark results."
    )
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Write the Markdown table to this path.")
    args = parser.parse_args()

    baseline_cases = load_cases(args.baseline_dir)
    candidate_cases = load_cases(args.candidate_dir)
    if not baseline_cases:
        parser.error(f"no '<case>-<run>.json' files found in {args.baseline_dir}")
    if not candidate_cases:
        parser.error(f"no '<case>-<run>.json' files found in {args.candidate_dir}")

    table = compare_cases(baseline_cases, candidate_cases)
    print(table, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Fail closed unless both fixed-subset continuation perplexity gates pass."""

import argparse
import json
import math
from pathlib import Path


def check(root):
    rows = []
    for length in (2048, 8192):
        reports = {name: json.loads((root / f"{name}-{length}.json").read_text())
                   for name in ("flash", "triton", "int8")}
        reference = reports["flash"]
        if reports["int8"].get("cache_memory", {}).get("format") != "int8-k32-vhead-v1":
            raise ValueError("Quality gate requires the current K32 INT8 format; old reports cannot qualify")
        for name, report in reports.items():
            if (report["window"] != length or report["scored_tokens"] != 32768 or
                    report["data"] != reference["data"] or report["shared_prefix"] or
                    report.get("diagnostic_roundtrip") or
                    report["prefill_budget"] != reference["prefill_budget"]):
                raise ValueError(f"{name}/{length}: incomplete or incomparable quality evaluation")
            if not math.isfinite(report["continuation_perplexity"]) or report["continuation_perplexity"] <= 0:
                raise ValueError(f"{name}/{length}: invalid perplexity")
        baseline = reference["continuation_perplexity"]
        high = reports["triton"]["continuation_perplexity"]
        quant = reports["int8"]["continuation_perplexity"]
        relative = (quant / baseline - 1) * 100
        row = {"window": length, "flash_ppl": baseline, "triton_ppl": high, "int8_ppl": quant,
               "relative_increase_percent": relative, "passed": relative <= 1.0}
        rows.append(row)
    return {"passed": all(r["passed"] for r in rows), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = check(args.result_dir)
    (args.result_dir / "gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("INT8 quality gate failed. Stop and consult the user; do not change quantization format automatically.")


if __name__ == "__main__":
    main()

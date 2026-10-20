"""GPU-independent open-loop arrival driver and token-visible latency metrics."""

import hashlib
import json
from collections import Counter
from random import Random
from time import perf_counter, sleep


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    at = (len(values) - 1) * q / 100
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def distribution(values):
    return {f"p{q}": percentile(values, q) for q in (50, 95, 99)}


def make_trace(workload, count, rate, seed=0, vocab_size=151936, saturated=False):
    if workload not in ("steady", "burst", "mixed", "pressure") or count < 1 or rate <= 0:
        raise ValueError("Invalid workload, request count or arrival rate")
    shape_rng, arrival_rng = Random(seed), Random(seed + 7919)
    requests, arrival = [], 0.0
    for index in range(count):
        length = 8192 if workload == "pressure" else 512
        if workload == "mixed":
            length = 8192 if shape_rng.random() < 0.25 else 256
        if index:
            arrival += arrival_rng.expovariate(rate) if workload == "mixed" else 1 / rate
        time = 0.0 if saturated else arrival
        if workload == "burst" and not saturated:
            time = int(arrival // 5) * 5 + (arrival % 5) / 5
        requests.append(dict(request_id=index, arrival_s=time, input_len=length,
                             output_len=128, prompt_seed=(seed + 11) * 1000000 + index))
    return dict(format="nanovllm-online-v1", workload=workload, seed=seed, rate=rate,
                saturated=saturated, vocab_size=vocab_size, requests=requests)


def materialize_trace(trace):
    if trace.get("format") != "nanovllm-online-v1" or not trace.get("requests"):
        raise ValueError("Expected a nonempty nanovllm-online-v1 trace")
    result, previous, identities = [], -1.0, set()
    digest = hashlib.sha256()
    for row in trace["requests"]:
        if row["arrival_s"] < previous or row["arrival_s"] < 0 or row["request_id"] in identities:
            raise ValueError("Arrival times must be nonnegative/sorted; request IDs unique")
        if row["input_len"] <= 0 or row["output_len"] <= 0:
            raise ValueError("Lengths must be positive")
        previous = row["arrival_s"]
        identities.add(row["request_id"])
        rng = Random(row["prompt_seed"])
        prompt = [rng.randrange(trace["vocab_size"]) for _ in range(row["input_len"])]
        for token in prompt:
            digest.update(token.to_bytes(4, "little"))
        result.append(dict(row, prompt=prompt))
    return result, digest.hexdigest()


def run_online(llm, requests, params_factory, ttft_slo_ms=2000, itl_slo_ms=50,
               timeout_s=300, clock=perf_counter, sleeper=sleep):
    """All clock values use one monotonic timebase. Record events before file IO."""
    if not requests:
        raise ValueError("An online run requires requests")
    records, seq_to_record, steps = [], {}, []
    next_request = 0
    start = clock()
    timed_out = False
    while next_request < len(requests) or not llm.is_finished():
        now = clock()
        if now - start >= timeout_s:
            timed_out = True
            break
        while next_request < len(requests) and requests[next_request]["arrival_s"] <= now - start:
            row = requests[next_request]
            seq_id = llm.add_request(row["prompt"], params_factory(row["output_len"]))
            record = dict(request_id=row["request_id"], seq_id=seq_id, arrival_s=row["arrival_s"],
                          submitted_s=clock() - start, first_scheduled_s=None,
                          input_len=len(row["prompt"]), expected_output_tokens=row["output_len"],
                          token_times_s=[], finished_s=None)
            records.append(record)
            seq_to_record[seq_id] = record
            next_request += 1
        if llm.is_finished():
            if next_request < len(requests):
                sleeper(min(0.05, max(0, requests[next_request]["arrival_s"] - (clock() - start))))
            continue
        step_start = clock()
        outputs, info = llm.step_with_info()
        end = clock() - start
        for seq_id, when in (getattr(info, "first_scheduled", None) or {}).items():
            seq_to_record[seq_id]["first_scheduled_s"] = when - start
        generated = info.generated_seq_ids
        if len(generated) != len(set(generated)):
            raise RuntimeError("Duplicate token event within a step")
        for seq_id in generated:
            record = seq_to_record[seq_id]
            if record["finished_s"] is not None:
                raise RuntimeError("Token emitted after request completion")
            record["token_times_s"].append(end)
        output_sizes = {seq_id: len(tokens) for seq_id, tokens in outputs}
        for seq_id in info.finished_seq_ids:
            record = seq_to_record[seq_id]
            if record["finished_s"] is not None or output_sizes.get(seq_id) != len(record["token_times_s"]):
                raise RuntimeError("Duplicate completion or missing token events")
            if output_sizes[seq_id] != record["expected_output_tokens"]:
                raise RuntimeError("Fixed-length request completed with an unexpected output length")
            record["finished_s"] = end
        steps.append(dict(end_s=end, elapsed_ms=(clock() - step_start) * 1000,
                          phase=info.phase, prefill_tokens=info.prefill_tokens,
                          decode_tokens=info.decode_tokens, execution_path=info.execution_path,
                          p2_tokens=info.p2_tokens, prefill_budget=info.prefill_budget,
                          budget_reason=info.budget_reason, kv_used_blocks=info.kv_used_blocks,
                          scheduler_stats=info.scheduler_stats))
    elapsed = clock() - start
    # Include never-submitted arrivals in the SLO failure denominator on timeout.
    for row in requests[next_request:]:
        records.append(dict(request_id=row["request_id"], seq_id=None, arrival_s=row["arrival_s"],
                            submitted_s=None, first_scheduled_s=None, input_len=len(row["prompt"]),
                            expected_output_tokens=row["output_len"], token_times_s=[], finished_s=None))
    metrics = summarize_records(records, elapsed, ttft_slo_ms, itl_slo_ms)
    metrics.update(timed_out=timed_out, phases=dict(Counter(s["phase"] for s in steps)),
                   graph_steps=sum(s["execution_path"] == "graph" for s in steps),
                   mixed_eager_steps=sum(s["phase"] == "mixed" for s in steps),
                   p2_tokens=sum(s["p2_tokens"] for s in steps),
                   decode_tokens=sum(s["decode_tokens"] for s in steps),
                   prefill_tokens=sum(s["prefill_tokens"] for s in steps),
                   kv_used_blocks_peak=max((s["kv_used_blocks"] for s in steps), default=0),
                   scheduler_stats=steps[-1]["scheduler_stats"] if steps else {})
    return dict(metrics=metrics, requests=records, steps=steps)


def summarize_records(records, elapsed, ttft_limit, itl_limit):
    ttft, itls, queues, submit_lags, e2e = [], [], [], [], []
    completed = good = good_tokens = total_tokens = 0
    for row in records:
        tokens = row["token_times_s"]
        gaps = [(b - a) * 1000 for a, b in zip(tokens, tokens[1:])]
        first = (tokens[0] - row["arrival_s"]) * 1000 if tokens else None
        passed = row["finished_s"] is not None and first is not None and first <= ttft_limit and all(
            gap <= itl_limit for gap in gaps)
        row.update(ttft_ms=first, itl_max_ms=max(gaps, default=None), slo_pass=passed)
        total_tokens += len(tokens)
        itls.extend(gaps)
        if first is not None:
            ttft.append(first)
        if row["first_scheduled_s"] is not None:
            queues.append((row["first_scheduled_s"] - row["arrival_s"]) * 1000)
        if row["submitted_s"] is not None:
            submit_lags.append((row["submitted_s"] - row["arrival_s"]) * 1000)
        if row["finished_s"] is not None:
            completed += 1
            e2e.append((row["finished_s"] - row["arrival_s"]) * 1000)
        if passed:
            good += 1
            good_tokens += len(tokens)
    return dict(elapsed_seconds=elapsed, total_requests=len(records), completed_requests=completed,
                incomplete_requests=len(records) - completed, slo_requests=good,
                slo_miss_fraction=1 - good / len(records), total_output_tokens=total_tokens,
                e2e_throughput=total_tokens / elapsed, request_throughput=completed / elapsed,
                goodput_requests=good / elapsed, goodput_tokens=good_tokens / elapsed,
                ttft_ms=distribution(ttft), itl_ms=distribution(itls), e2e_ms=distribution(e2e),
                queue_wait_ms=distribution(queues), submission_lag_ms=distribution(submit_lags),
                ttft_slo_ms=ttft_limit, itl_slo_ms=itl_limit)


def trace_hash(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

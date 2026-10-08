"""Median, p80, p95 and mean request latency of a sequential kit run, overall and per benchmark.

    python scripts/latency_report.py runs/latency/results.jsonl [--json runs/latency/latency.json]

Reads the kit's per-request wall time (`total_wall_ms`, HTTP request to all answers). Per-benchmark
figures are shown where at least 5 rows answered, as on the board. The gate (Decision Index 0.3): median,
mean and 80th percentile each at most 1,000 ms per request on one RTX PRO 6000, measured by the maintainers
(0.2.1 gated the median only). Warm-up rows belong in a separate run and are not read here.
"""
import argparse
import json
import statistics
from collections import defaultdict


def stats(values):
    values = sorted(values)
    return {'n': len(values), 'p50': round(statistics.median(values), 1),
            'p80': round(values[min(len(values) - 1, int(0.80 * len(values)))], 1),
            'p95': round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
            'mean': round(statistics.mean(values), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results')
    parser.add_argument('--json')
    args = parser.parse_args()
    overall, by_benchmark, failed = [], defaultdict(list), 0
    for line in open(args.results):
        row = json.loads(line)
        if row.get('status') != 'ok':
            failed += 1
            continue
        ms = float(row['total_wall_ms'])
        overall.append(ms)
        by_benchmark[row['dataset']].append(ms)
    report = {'overall': stats(overall), 'failed': failed, 'gate_ms': 1000,
              'benchmarks': {b: stats(v) for b, v in sorted(by_benchmark.items()) if len(v) >= 5}}
    o = report['overall']
    report['gate'] = {k: o[k] <= 1000 for k in ('p50', 'mean', 'p80')}
    report['passes'] = all(report['gate'].values())
    print(f"{o['n']} answered, {failed} failed: p50 {o['p50']} ms, p80 {o['p80']} ms, p95 {o['p95']} ms, mean {o['mean']} ms "
          f"(gate: median, mean and p80 <= 1000 ms -> {'PASS' if report['passes'] else 'FAIL'})")
    for b, s in report['benchmarks'].items():
        print(f"  {b:32} n {s['n']:4}  p50 {s['p50']:8}  p80 {s['p80']:8}  p95 {s['p95']:8}  mean {s['mean']:8}")
    if args.json:
        from pathlib import Path
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, 'w') as stream:
            json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()

"""Row files for the pre-submission checks, drawn from the verified Decision Index suite (0.3 by default).

    python scripts/samples.py compat --out runs/compat-rows.jsonl
    python scripts/samples.py latency --out runs/latency-rows.jsonl --warmup-out runs/latency-warmup.jsonl

compat: two requests per benchmark track (the kit's compatibility-pass shape): every request
format reaches the service once before a long run.

latency: our approximation of the maintainers' latency-v1 protocol, whose exact rows are not
published: 750 rows stratified by benchmark and by prompt length (proxy-token quartiles within each
benchmark, proportional allocation with at least one row per benchmark), plus 10 different rows
as untimed warm-up. Fixed seed; no outcome is looked at. Latency only: these rows are not scored.

Needs the reproduction kit (see docs/evaluation.md) and the suite in ./suite-0.3 or --suite-dir
(--edition 0.2.1 with ./suite-0.2 for the previous edition).
"""
import argparse
import random
from collections import defaultdict


def load(suite_dir, edition):
    from decision_index.suite.io import Suite
    suite = Suite(suite_dir, edition)
    suite.verify(strict=True)
    return list(suite.rows(apply_exclusions=True))


def write(path, rows):
    from pathlib import Path
    from decision_index.suite.io import dumps
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as stream:
        for row in rows:
            stream.write(dumps(row) + '\n')
    print(f'{path}: {len(rows)} rows')


def compat(rows, rng):
    by_track = defaultdict(list)
    for row in rows:
        by_track[row['_evaluation']['track']].append(row)
    return [r for track in sorted(by_track) for r in rng.sample(by_track[track], min(2, len(by_track[track])))]


def latency(rows, rng, n=750, warmup=10):
    by_benchmark = defaultdict(list)
    for row in rows:
        by_benchmark[row['_evaluation']['dataset']].append(row)
    total = len(rows)
    quota = {b: max(1, round(n * len(v) / total)) for b, v in by_benchmark.items()}
    while sum(quota.values()) > n:  # trim rounding overshoot from the largest quotas
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(by_benchmark, key=lambda b: len(by_benchmark[b]) / quota[b])] += 1
    timed = []
    for benchmark in sorted(by_benchmark):
        pool = sorted(by_benchmark[benchmark], key=lambda r: (int(r['_evaluation']['proxy_tokens']), r['_evaluation']['run_id']))
        quartiles = [pool[i * len(pool) // 4:(i + 1) * len(pool) // 4] for i in range(4)]
        quartiles = [q for q in quartiles if q]
        picks = []
        for i in range(quota[benchmark]):  # round-robin over length quartiles
            bucket = [r for r in quartiles[i % len(quartiles)] if r not in picks]
            if bucket:
                picks.append(rng.choice(bucket))
        timed += picks
    chosen = {r['_evaluation']['run_id'] for r in timed}
    rest = [r for r in rows if r['_evaluation']['run_id'] not in chosen]
    return timed, rng.sample(rest, warmup)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=('compat', 'latency'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup-out')
    parser.add_argument('--edition', default='0.3', choices=('0.3', '0.2.1'))
    parser.add_argument('--suite-dir', help='default: suite-0.3 (0.3) or suite-0.2 (0.2.1)')
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    suite_dir = args.suite_dir or ('suite-0.3' if args.edition == '0.3' else 'suite-0.2')
    rows, rng = load(suite_dir, args.edition), random.Random(args.seed)
    if args.kind == 'compat':
        write(args.out, compat(rows, rng))
    else:
        if not args.warmup_out:
            parser.error('latency needs --warmup-out')
        timed, warm = latency(rows, rng)
        write(args.out, timed)
        write(args.warmup_out, warm)


if __name__ == '__main__':
    main()

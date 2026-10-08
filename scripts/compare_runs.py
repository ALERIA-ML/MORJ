"""Compare two Decision Index results.jsonl files on the requests both answered.

Typical use: a sample run on new hardware (candidate) against the submitted full run
(reference). Reports decision agreement, probability drift, coverage of the candidate,
and paired sequential latency, including a projection of the full-suite median.

    python scripts/compare_runs.py --reference runs/reference-sample/results.jsonl \
        --candidate runs/sample/results.jsonl --reference-median 171.0
"""
import argparse
import json
import statistics
from collections import Counter


def load(path):
    rows = {}
    with open(path) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[row['run_id']] = row
    return rows


def best(probabilities):
    return max(probabilities, key=probabilities.get)


def compare_answer(a, b):
    """(same decision, largest probability difference) for one question."""
    if a['type'] == 'noul':
        return (a['noul'] >= .5) == (b['noul'] >= .5), abs(a['noul'] - b['noul'])
    keys = a['probabilities']
    drift = max(abs(a['probabilities'][k] - b['probabilities'][k]) for k in keys)
    if a['type'] == 'score':
        return best(a['probabilities']) == best(b['probabilities']), max(drift, abs(a['score'] - b['score']))
    return a['choice'] == b['choice'], drift


def quantile(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference-median', type=float,
                        help='full-suite median latency (ms) of the reference run, to project the candidate median')
    parser.add_argument('--json', help='also write the report here')
    args = parser.parse_args()
    reference, candidate = load(args.reference), load(args.candidate)
    status = Counter(r['status'] for r in candidate.values())
    shared = [k for k, r in candidate.items() if r['status'] == 'ok' and reference.get(k, {}).get('status') == 'ok']
    missing = [k for k in candidate if k not in reference]
    questions = same = requests_same = 0
    drifts, by_type = [], Counter()
    changed = []
    for key in shared:
        a, b = reference[key]['response']['answers'], candidate[key]['response']['answers']
        if set(a) != set(b):
            raise SystemExit(f'{key}: different question keys')
        all_same = True
        for q in a:
            ok, drift = compare_answer(a[q], b[q])
            questions += 1
            same += ok
            drifts.append(drift)
            by_type[(a[q]['type'], ok)] += 1
            if not ok:
                all_same = False
                changed.append({'run_id': key, 'question': q, 'drift': round(drift, 4)})
        requests_same += all_same
    ref_ms = [reference[k]['total_wall_ms'] for k in shared]
    cand_ms = [candidate[k]['total_wall_ms'] for k in shared]
    report = {
        'candidate_status': dict(status),
        'shared_requests': len(shared),
        'candidate_ids_missing_from_reference': len(missing),
        'questions': questions,
        'same_decision': same,
        'same_decision_rate': same / questions if questions else None,
        'requests_with_every_decision_same': requests_same,
        'by_type': {f'{t}/{"same" if ok else "changed"}': n for (t, ok), n in sorted(by_type.items())},
        'probability_drift': {'p50': quantile(drifts, .5), 'p99': quantile(drifts, .99), 'max': max(drifts)} if drifts else None,
        'largest_changed': sorted(changed, key=lambda c: -c['drift'])[:10],
    }
    if shared:
        ratio = statistics.median(cand_ms) / statistics.median(ref_ms)
        report['latency_ms'] = {
            'reference': {'p50': statistics.median(ref_ms), 'p95': quantile(ref_ms, .95), 'mean': statistics.mean(ref_ms)},
            'candidate': {'p50': statistics.median(cand_ms), 'p95': quantile(cand_ms, .95), 'mean': statistics.mean(cand_ms)},
            'median_ratio': ratio,
            'median_of_paired_ratios': statistics.median(c / r for c, r in zip(cand_ms, ref_ms)),
        }
        if args.reference_median:
            report['latency_ms']['projected_full_suite_median'] = args.reference_median * ratio
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, 'w') as stream:
            stream.write(text + '\n')


if __name__ == '__main__':
    main()

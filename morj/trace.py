"""What the engine did for one request, for the playground and debugging.

A trace is assembled after the answers are final and never changes them.
Token counts and cache hits come from TensorRT-LLM's own results.
"""
from . import method

METHOD = {
    'profiles': {name: p.summary() for name, p in method.PROFILES.items()},
    'efforts': method.EFFORTS,
    'default_effort': method.DEFAULT_EFFORT,
    'rating': "nine digits on two-option questions; truth questions once, other pairs in both orders ('pairs')",
    'rating_mapping': 'p(A) = clamp(0.01 + (E[rating] / 10 - 0.1) * 0.98 / 0.8, 0.01, 0.99)',
    'temperature': 'T(n) = max({2}, {0} - {1} ln n) on unrated questions; rated letters stay at T = 1'.format(
        method.TEMPERATURE_CURVE[0], -method.TEMPERATURE_CURVE[1], method.TEMPERATURE_CURVE[2]),
}


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(builder, questions, passes, steps, gated, started, stages, graph_replays, profile=method.HIGH):
    """`passes` holds (branches, rows, stats) per engine pass; `started` is the inference start (perf_counter)."""
    tokenizer = builder.tokenizer

    def at(seconds):
        return round((seconds - started) * 1000, 3)

    branches = []
    for number, (items, rows, stats) in enumerate(passes, 1):
        seed = items[0].input_ids
        timings = stats.get('timings') or [None] * len(items)
        for index, (b, row, cached, timing) in enumerate(zip(items, rows, stats['cached_tokens'], timings, strict=True)):
            keys = method.options(questions[b.question])[0]
            head, state, body, tail = b.segments or (0, 0, len(b.input_ids), 0)
            entry = {
                'pass': number, 'seed': index == 0, 'question': b.question, 'kind': b.kind,
                'prompt_tokens': len(b.input_ids), 'cached_tokens': cached,
                'shared_prefix_tokens': shared_prefix(seed, b.input_ids),
                'segments': {'head': head, 'state': state, 'question': body, 'tail': tail},
                'question_text': tokenizer.decode(b.input_ids[head + state:head + state + body]),
                'probabilities': row,
            }
            if b.kind == 'answer':
                entry.update(order=list(b.order), labels=builder.label_names[:len(b.order)],
                             options=[keys[i] for i in b.order])
            else:
                entry.update(labels=list('123456789'), rated_option=keys[0] if b.kind == 'rating' else keys[1])
            if timing:
                queued, start, end = timing
                entry.update(queued_ms=at(queued), start_ms=at(start), end_ms=at(end))
            branches.append(entry)
    answers = {}
    for key, spec in questions.items():
        keys = method.options(spec)[0]
        step = steps.get(key, {})
        answers[key] = {
            'type': spec['type'], 'options': keys, 'gated': key in gated,
            'gate_eligible': profile.gated and len(keys) >= 3,
            'orders': step.get('orders', []), 'ratings': step.get('ratings', []),
            'letters': step.get('letters'), 'rating': step.get('rating'), 'final': step.get('final'),
        }
    prompt = sum(b['prompt_tokens'] for b in branches)
    cached = sum(b['cached_tokens'] for b in branches)
    return {
        'method': METHOD,
        'profile': profile.summary(),
        'stages_ms': stages,
        'tokens': {'prompt': prompt, 'cached': cached, 'computed': prompt - cached},
        'template': {'head': tokenizer.decode(builder.chat.head), 'tail': tokenizer.decode(builder.rating.answer_tail)},
        'graph_replays': graph_replays,
        'branches': branches,
        'questions': answers,
    }

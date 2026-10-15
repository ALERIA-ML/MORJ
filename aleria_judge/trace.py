"""What the engine did for one request, for the playground and debugging.

A trace is assembled after the answers are final and never changes them.
Token counts and cache hits come from TensorRT-LLM's own results.
"""
from . import method

METHOD = {
    'profiles': {name: p.summary() for name, p in method.PROFILES.items()},
    'default_effort': method.DEFAULT_EFFORT,
    'rating': 'nine digits on two-option questions, in both orders',
    'rating_mapping': 'p(A) = clamp(0.01 + (E[rating] / 10 - 0.1) * 0.98 / 0.8, 0.01, 0.99)',
    'temperature': 'T(n) = max({2}, {0} - {1} ln n) on unrated questions; rated letters stay at T = 1'.format(
        method.TEMPERATURE_CURVE[0], -method.TEMPERATURE_CURVE[1], method.TEMPERATURE_CURVE[2]),
}


def method_info(default_effort):
    """METHOD with the service's configured default effort (JUDGE_DEFAULT_EFFORT), not the library default."""
    return {**METHOD, 'default_effort': method.profile(default_effort).name}


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build(builder, questions, passes, steps, gated, started, stages, profile=method.HIGH):
    """`passes` holds (branches, rows, stats) per engine pass; `started` is the inference start (perf_counter)."""
    tokenizer = builder.tokenizer

    def at(seconds):
        return round((seconds - started) * 1000, 3)

    branches = []
    for number, (items, rows, stats) in enumerate(passes, 1):
        seed = items[0].input_ids
        timings = stats.get('timings') or [None] * len(items)
        for index, (b, row, tokens, cached, timing) in enumerate(zip(items, rows, stats['prompt_tokens'],
                                                                     stats['cached_tokens'], timings, strict=True)):
            keys = method.keys(questions[b.question])
            head, images, state, body, tail = b.segments or (0, 0, 0, len(b.input_ids), 0)
            # TensorRT-LLM counts each <|image|> placeholder as the image's soft tokens. The images open every branch
            # of the request, so the expansion belongs to the image segment and to the prefix shared with the seed.
            expanded = tokens - len(b.input_ids)
            start = head + images + state
            entry = {
                'pass': number, 'seed': index == 0, 'question': b.question, 'kind': b.kind,
                'prompt_tokens': tokens, 'cached_tokens': cached,
                'shared_prefix_tokens': shared_prefix(seed, b.input_ids) + expanded,
                'segments': {'head': head, 'images': images + expanded, 'state': state, 'question': body, 'tail': tail},
                'question_text': tokenizer.decode(b.input_ids[start:start + body]),
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
        keys = method.keys(spec)
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
        'branches': branches,
        'questions': answers,
    }

"""Synthetic requests for the golden test: every question type, label width and rating path."""

COLORS = {'red': None, 'green': 'grass', 'blue': {'hex': '#00f'}, 'black': None, 'white': None}
TICKET = {'customer': 'Dana', 'messages': [{'from': 'customer', 'text': f'Message {i}: the export still fails.'}
                                           for i in range(10)], 'plan': 'enterprise', 'tags': None}

REQUESTS = {
    'truth': ('The meeting moved to Thursday at 10:00.', {
        'yn': {'type': 'choice', 'instructions': 'Did the meeting move?', 'criteria': {'yes': None, 'no': None}},
        'flag': {'type': 'noul', 'instructions': 'Is the meeting on Friday?'},
        'flag_described': {'type': 'noul', 'criteria': {'true': 'It is in the morning', 'false': 'It is not'}},
    }),
    'pairs': ('Option one ships Monday; option two ships Friday but costs less.', {
        'pair': {'type': 'choice', 'instructions': 'Which ships first?', 'criteria': {'one': None, 'two': None}},
        'level': {'type': 'score', 'instructions': 'How urgent is shipping?', 'criteria': ['Not urgent', 'Urgent']},
    }),
    'multi': (TICKET, {
        'route': {'type': 'choice', 'instructions': 'Route the ticket.',
                  'criteria': {'billing': 'Payments', 'bug': 'Product defect', 'howto': 'Usage question',
                               'sales': None}},
        'severity': {'type': 'score', 'instructions': 'Rate severity.',
                     'criteria': ['Trivial', 'Minor', 'Major', 'Critical', 'Blocker']},
        'color': {'type': 'choice', 'instructions': 'Pick a color.', 'criteria': COLORS},
        'single': {'type': 'choice', 'instructions': 'Only one option.', 'criteria': {'only': 'The sole choice'}},
    }),
    'wide': ('Café 日本語 state with "quotes" and\nnewlines <turn|>', {
        'twelve': {'type': 'choice', 'instructions': 'Which number?', 'criteria': {str(i): None for i in range(12)}},
        'wide255': {'type': 'choice', 'criteria': {f'item-{i}': f'Item number {i}' for i in range(255)}},
    }),
    'many': ('Count the questions.', {f'q{i}': {'type': 'choice', 'instructions': f'Question {i}?',
                                              'criteria': {'a': None, 'b': None, 'c': None}} for i in range(130)}),
}


def readout(rng, branches):
    """Deterministic label distributions: sometimes confident, sometimes spread."""
    rows = []
    for b in branches:
        weights = [rng.random() ** 4 for _ in b.label_ids]
        if rng.random() < .3:
            weights[rng.randrange(len(weights))] += 30
        rows.append([w / sum(weights) for w in weights])
    return rows

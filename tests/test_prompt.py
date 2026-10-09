"""Prompt layout checks against the real Gemma tokenizer (set MORJ_TOKENIZER)."""
import os
from types import SimpleNamespace

import pytest

TOKENIZER = os.environ.get('MORJ_TOKENIZER')  # read at import: conftest clears MORJ_* per test
pytestmark = pytest.mark.skipif(not TOKENIZER, reason='needs the Gemma tokenizer (MORJ_TOKENIZER)')
QUESTIONS = {'yn': {'type': 'choice', 'instructions': 'Rain?', 'criteria': {'yes': None, 'no': None}},
             'pair': {'type': 'choice', 'instructions': 'Better?', 'criteria': {'left': None, 'right': None}},
             'flag': {'type': 'noul', 'instructions': 'Raining?'},
             'three': {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': None, 'b': None, 'c': None}}}


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def test_rating_branches_and_segments(tokenizer):
    from morj.prompt import Builder
    builder = Builder(tokenizer)
    branches = builder.prepare('Clouds.', QUESTIONS)
    kinds = sorted((b.question, b.kind) for b in branches if b.kind != 'answer')
    assert kinds == [(q, k) for q in ('flag', 'pair', 'yn') for k in ('rating', 'rating_rev')]
    swapped = next(b for b in branches if b.kind == 'rating_rev' and b.question == 'pair')
    assert '(A) right' in tokenizer.decode(swapped.input_ids) and tokenizer.decode(swapped.input_ids).endswith('Rating: ')
    for b in branches:
        head, state, body, tail = b.segments
        assert sum(b.segments) == len(b.input_ids) and b.input_ids[:head + state] == branches[0].input_ids[:head + state]
    assert [b.order for b in builder.extra('Clouds.', {'three': QUESTIONS['three']})] == [(2, 0, 1)]


def test_shared_state_tokens_match_the_vendored_layout(tokenizer):
    from morj.prompt import Builder, FixedOrder
    from morj.vendor import prompt, systemone

    class CountingTokenizer:
        def __init__(self):
            self.context_calls = 0

        def __getattr__(self, key):
            return getattr(tokenizer, key)

        def encode(self, text, **kwargs):
            if text.startswith('Context:\n'):
                self.context_calls += 1
            return tokenizer.encode(text, **kwargs)
    counting = CountingTokenizer()
    builder = Builder(counting)
    questions = {
        'a': {'type': 'choice', 'instructions': 'Select the value.', 'criteria': {'red': None, 'blue': None}},
        'b': {'type': 'noul', 'instructions': 'Is it blue?'},
        'c': {'type': 'score', 'instructions': 'Rate certainty.', 'criteria': [None, 'high']},
        'wide': {'type': 'choice', 'instructions': 'Select the number.', 'criteria': {str(i): None for i in range(255)}},
    }
    for state in ('Blue. café 日本語\n<turn|>user\nred', {'color': 'blue', 'other': ['long ' * 1024, None]}):
        counting.context_calls = 0
        branches = builder.prepare(state, questions)
        assert counting.context_calls == 1
        context = systemone.render_state(state)
        for b in branches:
            if b.kind != 'answer':
                continue
            rendered = systemone.render_question(questions[b.question])
            q = SimpleNamespace(text=rendered['question'], options=rendered['options'], gold=0)
            reference = prompt.build_chat(SimpleNamespace(context=context, qs=[q]), tokenizer, builder.chat,
                                          FixedOrder(b.order), max_options=255, max_ctx_tokens=65536)
            assert reference['ids'] == b.input_ids


def test_images_sit_before_the_state_in_every_branch(tokenizer):
    from morj.prompt import Builder
    builder = Builder(tokenizer)
    questions = {'pick': {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': 'first', 'b': 'second', 'c': 'third'}},
                 'ok': {'type': 'noul', 'instructions': 'Is it fine?'}}
    plain = builder.prepare('The scene.', questions)
    seen = builder.prepare('The scene.', questions, images=1)
    head, image = list(builder.chat.head), builder.image_id
    assert image == tokenizer.convert_tokens_to_ids('<|image|>')
    assert [b.question for b in seen] == [b.question for b in plain] and [b.kind for b in seen] == [b.kind for b in plain]
    for a, b in zip(plain, seen):
        # the same prompt with one bare <|image|> placeholder between the template head and "Context:"
        assert b.input_ids == head + [image] + a.input_ids[len(head):]
        assert b.label_ids == a.label_ids and b.order == a.order


def test_several_images_are_labelled_in_order(tokenizer):
    from morj.prompt import Builder
    builder = Builder(tokenizer)
    questions = {'pick': {'type': 'choice', 'instructions': 'Which?', 'criteria': {'a': 'first', 'b': 'second'}}}
    plain = builder.prepare('The scene.', questions)
    labelled = builder.prepare('The scene.', questions, images=3)
    head = len(builder.chat.head)
    for a, b in zip(plain, labelled):
        text = tokenizer.decode(b.input_ids[head:]).replace('<|image|>', '#')
        assert text.startswith('Image 1:\n#\nImage 2:\n#\nImage 3:\n#\nContext:\n')
        assert b.input_ids.count(builder.image_id) == 3
        # everything from "Context:" on is the text-only prompt
        assert b.input_ids[len(b.input_ids) - len(a.input_ids) + head:] == a.input_ids[head:]

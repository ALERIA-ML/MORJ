"""Opt-in HTTP reliability probe; never imports or starts the GPU backend."""
import argparse
import asyncio
from collections import Counter
import json
import os
import random
import time
from pathlib import Path

import httpx


def payload(value, repeat=1, questions=4):
    state = f'The authoritative recorded color is {value}. ' + 'Unrelated archival text. ' * repeat
    return {'state': state, 'questions': {f'q{i}': {
        'type': 'choice', 'instructions': 'What is the authoritative recorded color?',
        'criteria': {'red': None, 'blue': None}} for i in range(questions)}}


CODES = ['amber', 'cobalt', 'crimson', 'jade', 'onyx', 'saffron', 'teal', 'violet']


def secret_payload(rng, index):
    """One caller's unique fact, padded to a random length, with shuffled options."""
    code = rng.choice(CODES)
    padding = 'Unrelated archival text. ' * rng.choice([0, 32, 512, 2048])
    state = {'caller': f'user-{index}', 'note': padding, 'record': f'The access code for user-{index} is {code}.'}
    questions = {}
    for q in range(rng.randint(1, 6)):
        options = rng.sample(CODES, len(CODES))
        questions[f'u{index}-q{q}'] = {'type': 'choice', 'instructions': f'What is the access code for user-{index}?',
                                       'criteria': {name: None for name in options}}
    return code, {'state': state, 'questions': questions}


async def isolation_stress(client, callers=96, seed=7):
    """Concurrent callers must each receive only their own question keys and facts."""
    rng = random.Random(seed)
    cases = [secret_payload(rng, i) for i in range(callers)]
    async def call(index, code, body):
        request_id = f'isolation-{index}'
        while True:
            response = await client.post('/v1/systemone', json=body, headers={'x-request-id': request_id})
            if response.status_code != 529:
                break
            await asyncio.sleep(float(response.headers.get('retry-after', 1)))
        assert response.status_code == 200, response.text[:300]
        assert response.headers['x-request-id'] == request_id
        answers = response.json()['answers']
        assert set(answers) == set(body['questions']), (index, sorted(answers))
        return sum(answer['choice'] == code for answer in answers.values()), len(answers)
    results = await asyncio.gather(*(call(i, code, body) for i, (code, body) in enumerate(cases)))
    correct, total = map(sum, zip(*results))
    # Keys can never cross requests; the facts are trivial, so a misrouted
    # state would show up as wrong codes far beyond model error.
    assert correct == total, f'{total - correct}/{total} questions answered with another state'
    return {'callers': callers, 'questions': total, 'correct': correct}


async def validate(args):
    started = time.time()
    result = {'hardware': args.hardware, 'base_url': args.url, 'started_unix': started}
    headers = {}
    if os.getenv(args.api_key_env):
        headers['Authorization'] = f'Bearer {os.environ[args.api_key_env]}'
    async with httpx.AsyncClient(base_url=args.url, timeout=150, headers=headers) as client:
        before = (await client.get('/stats')).json()
        health = (await client.get('/health')).json()
        # Conflicting states, unique request salts, repeats and concurrent calls.
        for round in range(3):
            values = ['red', 'blue'] * 8
            responses = await asyncio.gather(*(client.post('/v1/systemone', json=payload(v, 64)) for v in values))
            for value, response in zip(values, responses):
                assert response.status_code == 200, response.text
                assert all(a['choice'] == value for a in response.json()['answers'].values())
        result['isolation_requests'] = 48
        result['isolation_questions_correct'] = 192
        result['isolation_stress'] = await isolation_stress(client)
        # Prompt overflow is a caller error, leaves capacity available, and never truncates.
        overflow = await client.post('/v1/systemone', json=payload('blue', 70000, 1))
        assert overflow.status_code == 422, overflow.text[:300]
        result['overflow_status'] = overflow.status_code
        # A disconnected client must release both native work and API admission.
        try:
            await client.post('/v1/systemone', json=payload('red', 9000, 4), timeout=.2)
            raise AssertionError('Cancellation probe unexpectedly completed before its client deadline')
        except httpx.ReadTimeout:
            pass
        for _ in range(100):
            await asyncio.sleep(.1)
            stats = (await client.get('/stats')).json()
            if stats['inflight'] == stats['waiting'] == 0:
                break
        else:
            raise AssertionError('Cancelled request retained admission for over ten seconds')
        recovered = await client.post('/v1/systemone', json=payload('blue'))
        assert recovered.status_code == 200, recovered.text
        assert all(a['choice'] == 'blue' for a in recovered.json()['answers'].values())
        result['disconnect_recovered'] = True
        # Burst exceeds the documented 4 active + 16 queued admission capacity.
        responses = await asyncio.gather(*(client.post('/v1/systemone', json=payload('blue', 768, 4)) for _ in range(40)))
        counts = Counter(r.status_code for r in responses)
        assert set(counts) <= {200, 529} and counts[200] > 0 and counts[529] > 0, counts
        assert all(r.headers.get('retry-after') for r in responses if r.status_code == 529)
        assert all(all(a['choice']=='blue' for a in r.json()['answers'].values()) for r in responses if r.status_code==200)
        result['burst_statuses'] = dict(counts)
        # Independent near-limit contexts exceed the resident KV budget when
        # combined; native admission must serialize them without truncation.
        long_bodies = [payload(value, 1, 1) for value in ('red','blue')]
        for body in long_bodies:
            body['state'] += ' x' * 64000
        responses = await asyncio.gather(*(client.post('/v1/systemone', json=body) for body in long_bodies))
        result['concurrent_long_contexts'] = []
        for response in responses:
            assert response.status_code == 200, response.text
            tokens = response.json()['usage']['input_tokens']
            # Two orders per question, plus a rating branch for two-option questions when enabled.
            branches = 3 if health.get('binary_rating') else 2
            assert 63000 * branches <= tokens < 65536 * branches, tokens
            result['concurrent_long_contexts'].append({'input_tokens':tokens,'answer':response.json()['answers']['q0']})
        after = (await client.get('/stats')).json()
        assert after['inflight'] == after['waiting'] == 0
        assert after['graphs']['replays'] > before['graphs']['replays']
        assert (await client.get('/health')).status_code == 200
        result.update(passed=True, before=before, after=after, elapsed_seconds=time.time()-started)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:30100')
    parser.add_argument('--hardware', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--api-key-env', default='MORJ_API_KEY')
    asyncio.run(validate(parser.parse_args()))

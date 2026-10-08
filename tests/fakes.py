"""GPU-free stand-ins for the prompt builder and TensorRT-LLM engine."""
import asyncio
from contextlib import asynccontextmanager
import time
from types import SimpleNamespace

import httpx

from morj.method import HIGH, Branch, gated_orders, options, orders, rating_kinds

PAYLOAD = {'state': 'A fact', 'questions': {'q': {'type': 'choice', 'criteria': {'a': None, 'b': None}}}}
DIGITS = tuple(range(100, 109))


class Builder:
    """Branches shaped like the real builder's: orders, ratings, gated orders; at most 128 questions per call."""
    label_names = [f'L{i}' for i in range(255)]

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(decode=lambda ids: ' '.join(map(str, ids)))
        self.chat = SimpleNamespace(head=[1])
        self.rating = SimpleNamespace(answer_tail=[9])

    def prepare(self, state, questions, profile=HIGH):
        return self._build(state, questions, False, profile)

    def extra(self, state, questions):
        return self._build(state, questions, True)

    def _build(self, state, questions, gated, profile=HIGH):
        assert 1 <= len(questions) <= 128
        self.calls.append(len(questions))
        if state == 'too long':
            raise ValueError('Prompt exceeds context limit; no truncation')
        branches = []
        for key, spec in questions.items():
            n = len(options(spec)[0])
            for order in gated_orders(n) if gated else orders(n, profile.orders):
                branches.append(Branch(key, [1, 2, 3], tuple(range(n)), order, 'answer', (1, 1, 1, 0)))
            for kind in [] if gated else rating_kinds(spec, n, profile):
                branches.append(Branch(key, [1, 2, 4, 5], DIGITS, tuple(range(9)), kind, (1, 1, 2, 0)))
        return branches


class Engine:
    """Answers put `peak` on the first shown label (uniform without it); ratings say 9. `pick` overrides per branch."""

    def __init__(self, delay=0, peak=None, pick=None):
        self.graph_metrics = {'captures': 1, 'replays': 0}
        self.delay, self.peak, self.pick, self.fail_on, self.cancelled = delay, peak, pick, None, 0
        self.passes = []

    async def score_request(self, branches):
        started = self.graph_metrics['replays'] > 0  # the startup readiness probe is never delayed
        try:
            await asyncio.sleep(self.delay if started else 0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail_on is not None and any(b.question == self.fail_on for b in branches):
            raise RuntimeError('Native failure')
        self.graph_metrics['replays'] += 1
        self.passes.append(len(branches))
        rows = []
        for b in branches:
            n = len(b.label_ids)
            if b.kind != 'answer':
                rows.append([0.0] * 8 + [1.0])
                continue
            top = self.pick(b) if self.pick else 0
            if n == 1:
                rows.append([1.0])
            elif self.peak or self.pick:
                peak = self.peak or .9
                rows.append([peak if i == top else (1 - peak) / (n - 1) for i in range(n)])
            else:
                rows.append([1 / n] * n)
        now = time.perf_counter()
        return rows, {'cached_tokens': [0] + [2] * (len(branches) - 1), 'timings': [(now, now, now)] * len(branches)}


@asynccontextmanager
async def serving(app):
    """Run the app's lifespan and yield an in-process async client."""
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            yield client


async def until(condition, timeout=1.0):
    for _ in range(int(timeout / .001)):
        if condition():
            return
        await asyncio.sleep(.001)
    raise AssertionError('condition not reached')

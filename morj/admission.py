"""Bound live native context demand before the KV scheduler must preempt it."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager


class TokenBudget:
    """First-come, first-served token reservations.

    Waiters are granted strictly in arrival order, so a long branch cannot be
    starved by a stream of smaller ones that would otherwise keep fitting.
    """

    def __init__(self, capacity):
        self.capacity, self.used = capacity, 0
        self.waiters = deque()

    @property
    def waiting(self):
        return sum(1 for _, future in self.waiters if not future.done())

    def _grant(self):
        while self.waiters:
            tokens, future = self.waiters[0]
            if future.done():
                self.waiters.popleft()
                continue
            if self.used + tokens > self.capacity:
                break
            self.used += tokens
            future.set_result(None)
            self.waiters.popleft()

    @asynccontextmanager
    async def reserve(self, tokens):
        if not 0 < tokens <= self.capacity:
            raise ValueError('Native context exceeds admission capacity')
        if not self.waiters and self.used + tokens <= self.capacity:
            self.used += tokens
        else:
            future = asyncio.get_running_loop().create_future()
            entry = (tokens, future)
            self.waiters.append(entry)
            try:
                await future
            except asyncio.CancelledError:
                if future.done() and not future.cancelled():
                    # Granted in the same step as cancellation: return it.
                    self.used -= tokens
                else:
                    try:
                        self.waiters.remove(entry)
                    except ValueError:
                        pass
                self._grant()
                raise
        try:
            yield
        finally:
            self.used -= tokens
            self._grant()

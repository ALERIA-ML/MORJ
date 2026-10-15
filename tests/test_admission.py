import asyncio

from aleria_judge.admission import TokenBudget


def test_long_requests_cannot_overcommit_and_cancelled_waiter_releases_nothing():
    async def run():
        budget = TokenBudget(100)
        entered = asyncio.Event()
        async def waiting():
            async with budget.reserve(70):
                entered.set()
        async with budget.reserve(70):
            task = asyncio.create_task(waiting())
            await asyncio.sleep(.01)
            assert not entered.is_set() and budget.used == 70
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert budget.used == 70
        assert budget.used == 0
        async with budget.reserve(30), budget.reserve(70):
            assert budget.used == 100
        assert budget.used == 0
    asyncio.run(run())


def test_waiters_are_served_in_arrival_order():
    async def run():
        budget, order = TokenBudget(100), []

        async def branch(name, tokens):
            async with budget.reserve(tokens):
                order.append(name)
                await asyncio.sleep(.01)
        async with budget.reserve(60):
            large = asyncio.create_task(branch('large', 90))
            await asyncio.sleep(.001)
            small = asyncio.create_task(branch('small', 10))
            await asyncio.sleep(.001)
            # The small branch would fit now but must not overtake the earlier waiter.
            assert order == [] and budget.waiting == 2
        await asyncio.gather(large, small)
        assert order == ['large', 'small'] and budget.used == 0 and not budget.waiters
    asyncio.run(run())


def test_cancelled_waiter_unblocks_followers():
    async def run():
        budget = TokenBudget(100)
        async with budget.reserve(60):
            blocked = asyncio.create_task(budget.reserve(90).__aenter__())
            await asyncio.sleep(.001)
            entered = asyncio.Event()

            async def follower():
                async with budget.reserve(30):
                    entered.set()
            task = asyncio.create_task(follower())
            await asyncio.sleep(.001)
            assert not entered.is_set()
            blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)
            await asyncio.wait_for(entered.wait(), 1)
            await task
        assert budget.used == 0
    asyncio.run(run())

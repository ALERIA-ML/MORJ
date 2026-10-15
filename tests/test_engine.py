import asyncio
from types import SimpleNamespace

import pytest

from aleria_judge.engine import Engine, answer_probabilities


def test_first_branch_seeds_the_cache_then_the_rest_run_together():
    from aleria_judge.admission import TokenBudget
    engine = object.__new__(Engine)
    engine.token_budget = TokenBudget(10_000)
    calls = []

    async def score(branches, salt, shared=0, images=()):
        calls.append(([b.question for b in branches], salt))
        return [([b.value, 1 - b.value], 0 if b.question == 'a' else 32, (0., 1., 2.), 301) for b in branches]
    engine._score = score
    prefix = list(range(300))
    branches = [SimpleNamespace(question=k, value=i / 4, input_ids=prefix + [i]) for i, k in enumerate('abc', 1)]

    async def run():
        for _ in range(2):
            rows, stats = await engine.score_request(branches)
            assert rows == [[.25, .75], [.5, .5], [.75, .25]]
            assert stats == {'prompt_tokens': [301] * 3, 'cached_tokens': [0, 32, 32], 'timings': [(0., 1., 2.)] * 3}
    asyncio.run(run())
    assert [c[0] for c in calls] == [['a'], ['b', 'c'], ['a'], ['b', 'c']]
    assert calls[0][1] == calls[1][1] != calls[2][1] == calls[3][1]  # one fresh cache salt per request


def test_answer_probabilities_read_only_the_label_tokens():
    torch = pytest.importorskip('torch')
    logits = torch.full((1, 10), -1e4)
    logits[0, 3], logits[0, 7] = 2.0, 0.0
    p = answer_probabilities(logits, [3, 7])
    assert p == pytest.approx([torch.sigmoid(torch.tensor(2.0)).item(), 1 - torch.sigmoid(torch.tensor(2.0)).item()])
    for bad in ([], [3, 3], [11]):
        with pytest.raises(ValueError):
            answer_probabilities(logits, bad)


def test_shared_budget_reserves_the_common_prefix_once():
    import asyncio
    from types import SimpleNamespace
    from aleria_judge.admission import TokenBudget
    from aleria_judge.engine import Engine, common_prefix

    assert common_prefix([1, 2, 3, 4], [1, 2, 9]) == 2
    engine = object.__new__(Engine)
    engine.token_budget = TokenBudget(500)
    peaks = []

    async def score(branches, salt, shared=0, images=()):
        peaks.append((len(branches), shared, engine.token_budget.used))
        return [([1.0], 0, (0, 0, 0), 1) for _ in branches]
    engine._score = score
    prefix = list(range(300))
    branches = [SimpleNamespace(input_ids=prefix + [1000 + i]) for i in range(5)]
    asyncio.run(engine.score_request(branches))
    # first wave alone, then four branches sharing a 300-token prefix: the prefix is held once (300 of 500)
    assert peaks == [(1, 0, 0), (4, 300, 300)]
    assert engine.token_budget.used == 0


def test_branches_without_a_useful_shared_prefix_run_in_one_wave():
    import asyncio
    from types import SimpleNamespace
    from aleria_judge.admission import TokenBudget
    from aleria_judge.engine import Engine

    engine = object.__new__(Engine)
    engine.token_budget = TokenBudget(1000)
    waves = []

    async def score(branches, salt, shared=0, images=()):
        waves.append(len(branches))
        return [([1.0], 0, (0, 0, 0), 1) for _ in branches]
    engine._score = score
    head = list(range(10))
    asyncio.run(engine.score_request([SimpleNamespace(input_ids=head + [1000 + i] * 50) for i in range(2)]))
    assert waves == [2]  # 10 shared tokens < 256: no seeding wave
    waves.clear()
    long = list(range(300))
    asyncio.run(engine.score_request([SimpleNamespace(input_ids=long + [1000 + i]) for i in range(3)]))
    assert waves == [1, 2]  # 300 shared tokens: seed first, then the rest reuse it

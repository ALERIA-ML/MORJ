"""CPU property test of morj.lean_core.Tree (run by test_lean.py in its own process) (torch/triton stubbed): every node's KV path is its sequence prefix, positions are
depths, readouts hit each sequence's last token, and the packed size equals the number of distinct prefixes."""
import os, random, sys, types
import numpy as np


def load_lean():
    tr = types.ModuleType('triton')
    tr.jit = lambda f=None, **k: f if f else (lambda g: g)
    tr.cdiv = lambda a, b: -(-a // b)
    tl = types.ModuleType('triton.language')
    tl.constexpr = int
    tm = types.ModuleType('torch')
    tm.int32, tm.int64 = 'i32', 'i64'

    class T:
        def __init__(self, a):
            self.a = np.asarray(a)

        def to(self, *a, **k):
            return self
    tm.as_tensor = lambda a, dtype=None: T(a)
    tm.inference_mode = lambda: (lambda f: f)
    sys.modules.update({'triton': tr, 'triton.language': tl, 'torch': tm})
    import importlib.util
    # the morj package that is importable here (the source tree, or /app/morj in the service image)
    found = importlib.util.find_spec('morj')
    path = os.path.join(os.path.dirname(found.origin), 'lean_core.py') if found else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'morj', 'lean_core.py')
    spec = importlib.util.spec_from_file_location('lean_core', path)
    lean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lean)
    return lean


def check(lean, seqs, passes=None, window=10**9, single_max=10**9, step_tokens=10**9, mode='none', head=()):
    """Lay out `seqs`, plan `passes` (lists of sequence indices; default one pass over all) as LeanGemma4.run does in
    `mode` ('none': nothing kept between passes; 'identity': whole pools kept; 'pinned': only what later passes read),
    and simulate the K/V pools: every segment must read exactly its path's tokens in its window, every readout must
    point at its sequence's last token, and nothing live may be overwritten."""
    tree = lean.Tree(seqs + ([head] if head else []))
    full = np.zeros(tree.num_tokens, dtype=np.int64)
    depth = np.zeros(tree.num_tokens, dtype=np.int64)
    for nid, (b, s, e) in enumerate(tree.nodes):
        full[tree.offsets[nid]:tree.offsets[nid + 1]] = tree.seqs[b][s:e]
        depth[tree.offsets[nid]:tree.offsets[nid + 1]] = np.arange(s, e)
    head_nodes = tree.paths[-1] if head else []
    head_off = np.concatenate([tree.slots(n) for n in head_nodes]) if head else np.zeros(0, np.int64)
    computed, read, tables, kept = set(head_nodes), set(), None, None
    identity = mode == 'identity'
    for pas in passes or [list(range(len(seqs)))]:
        if not pas:
            continue
        cached = computed if mode != 'none' else set(head_nodes)
        need = tree.pass_nodes(pas, cached)
        pinned = None
        if mode == 'pinned':
            future = [b for b in range(len(tree.seqs)) if b not in read and b not in set(pas)]
            pinned = tree.pinned_for(cached | set(need), future, window)
        plan = tree.plan_pass(need, pas, window, single_max, step_tokens, identity,
                              prealloc=() if identity else head_off, initial=tables, pinned=pinned)
        if identity:
            if kept is None:
                kept = {k: {int(o): int(o) for o in head_off} for k in 'gs'}
            pools = kept
        elif mode == 'pinned' and kept is not None:
            pools = kept
        else:
            pools = {k: {int(plan['slot_of'][k][o]): int(o) for o in head_off} for k in 'gs'}
        for st in plan['steps']:
            assert list(st['tokens']) == list(full[st['own']]) and list(st['positions']) == list(depth[st['own']])
            for k in 'gs':
                for slot, o in zip(st['slots'][k], st['own']):
                    pools[k][int(slot)] = int(o)
            for i, (n, a, b) in enumerate(st['segs']):
                bn, sn, en = tree.nodes[n]
                path = list(tree.seqs[bn][:b])
                for k, lo in (('g', 0), ('s', max(0, a - window + 1))):
                    idx = st['kv_indices_' + k][st['kv_indptr_' + k][i]:st['kv_indptr_' + k][i + 1]]
                    got = [pools[k][int(x)] for x in idx]
                    assert list(full[got]) == path[lo:b] and list(depth[got]) == list(range(lo, b)), (k, n, a, b)
        for b, (k, r) in zip(pas, plan['readout']):
            st = plan['steps'][k]
            assert st['tokens'][r] == seqs[b][-1] and st['positions'][r] == len(seqs[b]) - 1
        assert sum(len(st['tokens']) for st in plan['steps']) == sum(tree.nodes[n][2] - tree.nodes[n][1] for n in need)
        read |= set(pas)
        if mode != 'none':
            computed |= set(need)
        if mode == 'pinned':
            kept = pools
            tables = {}
            for kk in 'gs':
                table = np.full(tree.num_tokens, -1)
                table[pinned[kk]] = plan['slot_of'][kk][pinned[kk]]
                tables[kk] = table
    prefixes = {tuple(s[:k]) for s in seqs + ([head] if head else []) for k in range(1, len(s) + 1)}
    assert tree.num_tokens == len(prefixes), (tree.num_tokens, len(prefixes))
    return tree


if __name__ == '__main__':
    lean = load_lean()
    head = [1, 2, 3] + list(range(100, 150))
    q0, q1 = list(range(200, 230)), list(range(300, 330))
    tail, rate = [7, 8, 9], [55, 56, 57, 58, 9]
    tree = check(lean, [head + q0 + tail, head + q1 + tail, head + q0 + rate, head + q1 + rate])
    print('MORJ-like tree: nodes', tree.nodes, 'packed', tree.num_tokens, 'of', tree.prompt_tokens)
    random.seed(0)
    for _ in range(3000):
        base = [random.randint(0, 5) for _ in range(random.randint(1, 30))]
        seqs = []
        for _ in range(random.randint(1, 8)):
            cut = random.randint(0, len(base))
            seqs.append(base[:cut] + [random.randint(0, 3) for _ in range(random.randint(0 if cut else 1, 12))])
            if random.random() < .3:
                base = seqs[-1]
        split = random.randint(1, len(seqs))
        order = random.sample(range(len(seqs)), len(seqs))
        kw = dict(window=random.choice([1, 2, 4, 10**9]), single_max=random.choice([3, 8, 10**9]),
                  step_tokens=random.choice([1, 3, 7]), mode=random.choice(['none', 'identity', 'pinned']))
        if random.random() < .5:  # a shared head every sequence starts with
            hd = [9, 9, 8]
            seqs = [hd + s for s in seqs]
            kw['head'] = hd
        check(lean, seqs, **kw)
        check(lean, seqs, [order[:split], order[split:]], **kw)
    print('3,000 random trees ok (one and two passes, steps, windows, no/whole/pinned retention, shared head)')

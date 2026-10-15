import subprocess
import sys
from pathlib import Path

import pytest

from aleria_judge.config import Settings


def test_lean_is_the_default_engine_and_trt_stays_available():
    assert Settings().engine == 'lean'
    assert Settings(engine='trt').engine == 'trt'
    with pytest.raises(ValueError, match="engine must be 'lean' or 'trt'"):
        Settings(engine='vllm')


def test_lean_engine_serves_bf16_only():
    with pytest.raises(ValueError, match='the lean engine serves the bf16 checkpoint'):
        Settings(engine='lean', checkpoint='nvfp4')
    assert Settings(engine='trt', checkpoint='nvfp4').checkpoint == 'nvfp4'


def test_cpu_threads_default_and_check():
    assert Settings().cpu_threads == 8
    assert Settings.from_env({'JUDGE_CPU_THREADS': '1'}).cpu_threads == 1
    with pytest.raises(ValueError, match='cpu_threads must be positive'):
        Settings(cpu_threads=0)


def test_lean_token_tree_and_planner():
    """Thousands of random trees through the lean engine's tree layout and step planner, with simulated K/V pools
    (one and two passes, windows, retention modes, a shared head); torch and triton are stubbed in that process."""
    out = subprocess.run([sys.executable, str(Path(__file__).with_name('lean_tree_check.py'))],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert '3,000 random trees ok' in out.stdout


def test_health_reports_engine_and_cpu_threads():
    from fastapi.testclient import TestClient
    from aleria_judge.server import create_app
    from tests.fakes import Builder, Engine
    with TestClient(create_app(Settings(), engine=Engine(), builder=Builder())) as client:
        health = client.get('/health').json()
    assert (health['engine'], health['cpu_threads']) == ('lean', 8)


def bare_lean(vision, model):
    """A LeanEngine around fakes (no GPU): `vision.encode` and `model.prepare`/`run` as given."""
    import concurrent.futures
    from collections import OrderedDict
    from aleria_judge.lean_engine import LeanEngine
    import threading
    from types import SimpleNamespace
    import torch
    engine = LeanEngine.__new__(LeanEngine)
    engine.worker = concurrent.futures.ThreadPoolExecutor(1)
    engine.model, engine.vision, engine.image_soft_tokens = model, vision, 560
    engine.lock = threading.Lock()
    engine.ended = OrderedDict()
    kept = SimpleNamespace(pools=[(torch.empty(1024, dtype=torch.uint8), torch.empty(1024, dtype=torch.uint8))])
    engine.requests = OrderedDict({'earlier': (kept, {})})  # another request's kept K/V: 2 KiB
    return engine


def branch():
    from types import SimpleNamespace
    return SimpleNamespace(question='q', kind='answer', order=(0, 1), input_ids=[1, 2, 3], label_ids=(7, 8))


class OomVision:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def encode(self, images, soft_tokens):
        import torch
        self.calls += 1
        if self.calls <= self.failures:
            raise torch.OutOfMemoryError('CUDA out of memory (test)')
        return ['features']


class OneSeqModel:
    free_bytes = 10 * 2**30

    def __init__(self, run_failures=0, images=True):
        from types import SimpleNamespace
        self.last_need, self.run_failures, self.images, self.available = [0], run_failures, images, []
        self.tree = SimpleNamespace(paths=[[0]], seqs=[[1, 2, 3]], nodes=[(0, 0, 3)])

    def prepare(self, seqs, labels, retain=False, images=None, available=None):
        from types import SimpleNamespace
        assert images == (['features'] if self.images else None)
        self.available.append(available)
        return SimpleNamespace(retain=False, pinned=False, tree=self.tree, pools=None)

    def run(self, req, seqs, available=None):
        import torch
        if self.run_failures:
            self.run_failures -= 1
            raise torch.OutOfMemoryError('CUDA out of memory (test)')
        return [[0.25, 0.75]]


def test_out_of_memory_in_image_encoding_is_the_requests_memory_error():
    import asyncio
    vision = OomVision(failures=2)
    engine = bare_lean(vision, OneSeqModel())
    with pytest.raises(MemoryError, match='out of memory'):
        asyncio.run(engine.score_request([branch()], images=['image'], salt='s'))
    assert vision.calls == 2 and not engine.requests  # retried once after dropping the kept K/V


def test_the_413_holds_no_traceback_of_the_failed_pass():
    import asyncio
    engine = bare_lean(OomVision(failures=2), OneSeqModel())
    try:
        asyncio.run(engine.score_request([branch()], images=['image'], salt='s'))
    except MemoryError as error:
        assert error.__context__ is None and error.__cause__ is None  # no frames (and GPU tensors) kept alive


def test_dropping_kept_kv_lets_the_request_fit():
    import asyncio
    vision = OomVision(failures=1)
    engine = bare_lean(vision, OneSeqModel())
    probs, stats = asyncio.run(engine.score_request([branch()], images=['image'], salt='s'))
    assert probs == [[0.25, 0.75]] and stats['prompt_tokens'] == [3] and vision.calls == 2


def test_a_request_too_large_for_the_gpu_is_a_413_and_the_service_keeps_serving(monkeypatch):
    from fastapi.testclient import TestClient
    from tests.fakes import PAYLOAD
    from tests.test_server import app_with
    app = app_with()
    engine = app.state.service.engine
    with TestClient(app) as client:
        original = engine.score_request

        async def too_large(*args, **kwargs):
            raise MemoryError('request needs 90 GiB')
        monkeypatch.setattr(engine, 'score_request', too_large)
        r = client.post('/v1/systemone', json=PAYLOAD)
        assert r.status_code == 413 and 'does not fit' in r.text
        assert not app.state.service.failed
        monkeypatch.setattr(engine, 'score_request', original)
        assert client.post('/v1/systemone', json=PAYLOAD).status_code == 200


def test_a_pass_plans_against_memory_other_requests_keep():
    import asyncio
    model = OneSeqModel(images=False)
    engine = bare_lean(None, model)
    asyncio.run(engine.score_request([branch()], salt='s'))
    assert model.available == [OneSeqModel.free_bytes - 2048]


def test_out_of_memory_in_a_pass_evicts_other_requests_kept_kv_and_retries():
    import asyncio
    model = OneSeqModel(run_failures=1, images=False)
    engine = bare_lean(None, model)
    probs, _ = asyncio.run(engine.score_request([branch()], salt='s'))
    assert probs == [[0.25, 0.75]] and 'earlier' not in engine.requests
    assert model.available == [OneSeqModel.free_bytes - 2048, OneSeqModel.free_bytes]  # the retry has the whole budget


def test_release_drops_kept_kv_at_once():
    engine = bare_lean(None, OneSeqModel(images=False))
    engine.release('earlier')
    assert not engine.requests


def test_the_retry_computes_this_pass_alone_without_the_gated_orders():
    import asyncio
    from types import SimpleNamespace

    class Recording(OneSeqModel):
        def prepare(self, seqs, labels, retain=False, images=None, available=None):
            self.calls = getattr(self, 'calls', []) + [(len(seqs), retain)]
            return super().prepare(seqs, labels, retain, images, available)
    model = Recording(run_failures=1, images=False)
    engine = bare_lean(None, model)
    gated = SimpleNamespace(question='q', kind='answer', order=(1, 0), input_ids=[1, 2, 4], label_ids=(7, 8))
    asyncio.run(engine.score_request([branch()], salt='s', future=[gated]))
    assert model.calls == [(2, True), (1, False)]


def test_a_pass_of_a_request_that_already_ended_keeps_no_kv():
    import asyncio
    from types import SimpleNamespace

    class Keeping(OneSeqModel):
        def prepare(self, seqs, labels, retain=False, images=None, available=None):
            req = super().prepare(seqs, labels, retain, images, available)
            req.retain = retain
            return req
    engine = bare_lean(None, Keeping(images=False))
    engine.release('s')  # the server released it (cancelled) before its queued pass ran
    gated = SimpleNamespace(question='q', kind='answer', order=(1, 0), input_ids=[1, 2, 4], label_ids=(7, 8))
    asyncio.run(engine.score_request([branch()], salt='s', future=[gated]))
    assert 's' not in engine.requests


def test_busy_seconds_cover_a_running_pass_only():
    import asyncio
    seen = []

    class Watching(OneSeqModel):
        def run(self, req, seqs, available=None):
            seen.append(engine.busy_seconds())
            return super().run(req, seqs, available)
    engine = bare_lean(None, Watching(images=False))
    assert engine.busy_seconds() is None
    asyncio.run(engine.score_request([branch()], salt='s'))
    assert seen[0] is not None and seen[0] >= 0 and engine.busy_seconds() is None


def test_a_pass_that_does_not_fit_raises_instead_of_looping():
    """With almost no memory available, a small one-step pass must end in MemoryError (which the engine turns into
    freeing memory and a retry), never spin in the planner: that loop once froze the engine's only worker thread."""
    import threading
    from aleria_judge import lean_core
    model = lean_core.LeanGemma4.__new__(lean_core.LeanGemma4)
    model.free_bytes, model.step_tokens, model.single_max, model.act_bytes, model.window = 2**34, 8192, 16384, 10**6, 1024
    model.geom = {'sliding_attention': (8, 256), 'full_attention': (4, 512)}
    model.types = ['sliding_attention', 'full_attention']
    req = lean_core.Request(lean_core.Tree([list(range(1, 400))]), [[0]], False, [])
    outcome = []

    def run():
        try:
            model.run(req, [0], available=1)
        except MemoryError as error:
            outcome.append(error)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), 'the planner loops forever'
    assert outcome and isinstance(outcome[0], MemoryError)


def test_one_request_in_inference_by_default_on_lean():
    assert Settings().active_requests == 1
    assert Settings(engine='trt').active_requests == 4
    assert Settings(max_requests=3).active_requests == 3

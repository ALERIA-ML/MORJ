"""A bad request is that request's error: it must never mark the engine failed."""
import base64
import io
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

from tests.fakes import PAYLOAD
from tests.test_server import app_with, png


def broken_png():
    """A PNG whose image data spans two IDAT chunks, the second with a non-alphabetic chunk type: PIL raises
    SyntaxError("broken PNG file") while loading it (not OSError)."""
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), (10, 200, 30)).save(buffer, 'PNG')
    data, chunks, pos = buffer.getvalue(), [], 8
    while pos < len(data):
        length, = struct.unpack('>I', data[pos:pos + 4])
        chunks.append((data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]))
        pos += 12 + length

    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    out = data[:8]
    for kind, body in chunks:
        if kind == b'IDAT':
            out += chunk(b'IDAT', body[:len(body) // 2]) + chunk(b"\x01P'>", body[len(body) // 2:])
        else:
            out += chunk(kind, body)
    return 'data:image/png;base64,' + base64.b64encode(out).decode()


def test_broken_png_is_a_422_and_the_service_keeps_serving():
    from PIL import Image
    from morj.images import decode
    with pytest.raises(SyntaxError):  # what PIL itself raises for this file
        Image.open(io.BytesIO(base64.b64decode(broken_png().partition(',')[2]))).load()
    with pytest.raises(ValueError, match='could not be decoded'):
        decode([broken_png()])
    with TestClient(app_with()) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, 'images': [broken_png()]})
        assert r.status_code == 422 and 'could not be decoded' in r.text
        assert c.post('/v1/systemone', json={**PAYLOAD, 'images': [png('red')]}).status_code == 200
        assert c.get('/health').status_code == 200


def test_error_outside_the_engine_is_a_500_and_the_service_keeps_serving(monkeypatch):
    app = app_with()
    with TestClient(app) as c:
        def broken(*args, **kwargs):
            raise RuntimeError('a bug in combining answers')
        monkeypatch.setattr('morj.server.combine', broken)
        r = c.post('/v1/systemone', json=PAYLOAD)
        assert r.status_code == 500 and 'keeps serving' in r.text
        assert not app.state.service.failed
        monkeypatch.undo()
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 200
        assert c.get('/health').status_code == 200


def test_vision_engines_warm_up_with_two_images_before_ready():
    from tests.fakes import Engine

    class Warm(Engine):
        needs_image_warmup = True
    engine = Warm()
    with TestClient(app_with(engine=engine)) as c:
        assert len(engine.images) == 2  # the last startup call: two images
        assert c.get('/health').status_code == 200


def test_a_stalled_engine_fails_the_service(monkeypatch):
    import time
    import morj.server as server
    from tests.fakes import Engine
    monkeypatch.setattr(server, 'STALL_CHECK_SECONDS', 0.05)
    engine = Engine()
    engine.busy_seconds = lambda: 10_000.0  # one pass running far past request_timeout + grace
    app = app_with(engine=engine)
    with TestClient(app) as c:
        deadline = time.time() + 5
        while not app.state.service.failed and time.time() < deadline:
            time.sleep(0.05)
        assert app.state.service.failed
        assert c.get('/health').status_code == 503 and c.get('/live').status_code == 503


def test_a_burst_cannot_pass_the_queue_limit_together():
    """The place in the queue is taken at arrival, before anything awaits; all at once, at most capacity get in."""
    import asyncio
    from tests.fakes import PAYLOAD, Engine, serving
    from tests.test_server import idle

    async def run():
        app = app_with(engine=Engine(delay=.3), max_requests=1, max_queue=2)
        async with serving(app) as c:
            answers = await asyncio.gather(*[c.post('/v1/systemone', json=PAYLOAD) for _ in range(12)])
            codes = sorted(r.status_code for r in answers)
            assert codes.count(200) == 3 and codes.count(529) == 9
            assert idle(app) and app.state.service.gate.admitted == 0
    asyncio.run(run())


def test_a_stdout_that_stops_reading_never_blocks_requests():
    import logging
    import threading
    from morj.metrics import NonBlockingLog, access_logger, dropped_log_lines
    from tests.fakes import PAYLOAD

    class Stuck:
        def __init__(self):
            self.release = threading.Event()

        def write(self, text):
            self.release.wait()

        def flush(self):
            pass
    stuck = Stuck()
    handler = NonBlockingLog(capacity=2, stream=stuck)
    record = logging.LogRecord('x', logging.INFO, __file__, 1, 'line', None, None)
    for _ in range(10):
        handler.emit(record)  # returns at once although the writer is stuck
    assert handler.dropped >= 7
    stuck.release.set()

    logger = access_logger()
    [h for h in logger.handlers if isinstance(h, NonBlockingLog)][0].stream = blocked = Stuck()
    try:
        with TestClient(app_with(access_log=True)) as c:
            for _ in range(3):
                assert c.post('/v1/systemone', json=PAYLOAD).status_code == 200
            assert c.get('/live').status_code == 200
            assert 'morj_log_lines_dropped' in c.get('/metrics').text
        assert dropped_log_lines() >= 0
    finally:
        blocked.release.set()
        for h in logger.handlers:
            if isinstance(h, NonBlockingLog):
                h.stream = None


def test_health_and_traces_report_the_configured_default_effort():
    from tests.fakes import PAYLOAD
    with TestClient(app_with(default_effort='low')) as c:
        assert c.get('/health').json()['method']['default_effort'] == 'low'
        trace = c.post('/v1/trace', json=PAYLOAD).json()['trace']
        assert trace['method']['default_effort'] == 'low'


def test_usage_reports_cached_and_image_tokens():
    from tests.fakes import PAYLOAD
    from tests.test_server import png
    with TestClient(app_with()) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, 'images': [png('red'), png('blue')]}).json()
        assert r['usage']['image_tokens'] == r['image_tokens'] == 2 * 560
        assert r['usage']['cached_tokens'] >= 0
        text = c.post('/v1/systemone', json=PAYLOAD).json()
        assert text['usage']['image_tokens'] == 0 and 'image_tokens' not in text

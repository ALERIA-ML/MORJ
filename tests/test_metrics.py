import json
import logging

from fastapi.testclient import TestClient

from morj.config import Settings
from morj.server import create_app
from tests.fakes import PAYLOAD, Builder, Engine


def sample(text, name, labels=''):
    for line in text.splitlines():
        if line.startswith(name + labels + ' ') or line.startswith(name + labels + '{'):
            return float(line.rsplit(' ', 1)[1])
    raise AssertionError(f'{name}{labels} missing')


def test_metrics_count_outcomes_tokens_and_stages():
    with TestClient(create_app(Settings(), engine=Engine(), builder=Builder())) as c:
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 200
        assert c.post('/v1/systemone', json={**PAYLOAD, 'state': 'too long'}).status_code == 422
        assert c.post('/v1/systemone', json={**PAYLOAD, 'images': ['x']}).status_code == 422
        response = c.get('/metrics')
        assert response.status_code == 200 and response.headers['content-type'].startswith('text/plain')
        text = response.text
        assert sample(text, 'morj_requests_total', '{status="200"}') == 1
        assert sample(text, 'morj_requests_total', '{status="422"}') == 2
        assert sample(text, 'morj_questions_total') == 1
        assert sample(text, 'morj_branches_total') == 4
        assert sample(text, 'morj_input_tokens_total') == 14
        assert sample(text, 'morj_request_duration_seconds_count') == 3
        assert sample(text, 'morj_stage_duration_seconds_count', '{stage="inference"}') == 1
        assert sample(text, 'morj_ready') == 1 and sample(text, 'morj_failed') == 0
        assert sample(text, 'morj_inflight_requests') == 0
        assert 'morj_build_info' in text and sample(text, 'morj_gated_questions_total') == 0


def test_metrics_are_public_even_with_authentication():
    with TestClient(create_app(Settings(api_key='secret'), engine=Engine(), builder=Builder())) as c:
        assert c.get('/metrics').status_code == 200
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 401
        assert sample(c.get('/metrics').text, 'morj_requests_total', '{status="401"}') == 1


def test_one_structured_log_line_per_request():
    records = []
    class Collect(logging.Handler):
        def emit(self, record):
            records.append(json.loads(record.getMessage()))
    handler = Collect()
    app = create_app(engine=Engine(), builder=Builder())
    app.state.metrics.access_log.addHandler(handler)
    try:
        with TestClient(app) as c:
            c.post('/v1/systemone', json=PAYLOAD, headers={'x-request-id': 'trace-7'})
            c.get('/health')
    finally:
        app.state.metrics.access_log.removeHandler(handler)
    lines = records
    assert len(lines) == 1
    line = lines[0]
    assert line['request_id'] == 'trace-7' and line['status'] == 200 and line['questions'] == 1
    assert line['branches'] == 4 and line['input_tokens'] == 14 and set(line['stages']) >= {'prepare', 'queue', 'inference'}


def test_access_log_can_be_disabled():
    app = create_app(Settings(access_log=False), engine=Engine(), builder=Builder())
    assert app.state.metrics.access_log is None

"""Prometheus metrics and one structured log line per decision request."""
import json
import logging
import queue
import sys
import threading
import time

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .patches.scheduler import SUSPENSIONS

LATENCY_BUCKETS = (.025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300)


class NonBlockingLog(logging.Handler):
    """Formats a record on the caller's thread and hands the line to a writer thread through a bounded queue, so a
    stdout that stops reading (an attached terminal that sleeps, a dropped SSH session) can never block request
    handling, as it froze the service in the RTX PRO 6000 run of 2026-10-10. Lines that do not fit are dropped and
    counted (`judge_log_lines_dropped` in /metrics)."""

    def __init__(self, capacity=10000, stream=None):
        super().__init__()
        self.lines, self.stream, self.dropped = queue.Queue(capacity), stream, 0
        threading.Thread(target=self._write, daemon=True, name='judge-log').start()

    def emit(self, record):
        try:
            self.lines.put_nowait(self.format(record))
        except queue.Full:
            self.dropped += 1
        except Exception:
            self.handleError(record)

    def _write(self):
        while True:
            line = self.lines.get()
            stream = self.stream or sys.stdout
            try:
                stream.write(line + '\n')
                stream.flush()
            except Exception:
                pass


def access_logger():
    logger = logging.getLogger('aleria_judge.access')
    if not logger.handlers:
        handler = NonBlockingLog()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def dropped_log_lines():
    return sum(getattr(h, 'dropped', 0) for h in logging.getLogger('aleria_judge.access').handlers)


class ServiceMetrics:
    """Owns a private registry so several apps (and tests) never collide."""

    def __init__(self, settings, service, access_log=True):
        self.registry = registry = CollectorRegistry()
        self.access_log = access_logger() if access_log else None
        self.requests = Counter('judge_requests', 'Decision requests by HTTP status', ['status'], registry=registry)
        self.latency = Histogram('judge_request_duration_seconds', 'End-to-end decision request latency',
                                 buckets=LATENCY_BUCKETS, registry=registry)
        self.stages = Histogram('judge_stage_duration_seconds', 'Request latency by stage', ['stage'],
                                buckets=LATENCY_BUCKETS, registry=registry)
        self.questions = Counter('judge_questions', 'Answered questions', registry=registry)
        self.efforts = Counter('judge_effort_requests', 'Answered decision requests by effort profile', ['profile'],
                               registry=registry)
        self.branches = Counter('judge_branches', 'Scored prompt branches (orders and ratings)', registry=registry)
        self.gated = Counter('judge_gated_questions', 'Questions whose orders disagreed and got a second pass',
                             registry=registry)
        self.input_tokens = Counter('judge_input_tokens', 'Prompt tokens over all branches', registry=registry)
        self.cached_tokens = Counter('judge_cached_tokens', 'Prompt tokens served from the KV cache', registry=registry)

        def live(name, documentation, read):
            Gauge(name, documentation, registry=registry).set_function(read)

        def budget(attribute):
            return getattr(getattr(service.engine, 'token_budget', None), attribute, 0) or 0

        live('judge_inflight_requests', 'Admitted requests running inference', lambda: service.gate.inflight)
        live('judge_log_lines_dropped', 'Access log lines dropped because stdout was not reading', dropped_log_lines)
        live('judge_kv_suspensions', 'TensorRT-LLM KV suspensions since start (expected 0)', lambda: SUSPENSIONS['count'])
        live('judge_waiting_requests', 'Requests preparing or queued', lambda: service.gate.waiting)
        live('judge_ready', '1 when serving', lambda: float(service.available))
        live('judge_failed', '1 after a native failure, until restart', lambda: float(service.failed))
        live('judge_native_context_tokens', 'Resident context tokens reserved in the KV budget', lambda: budget('used'))
        live('judge_native_context_waiters', 'Branches waiting for KV budget', lambda: budget('waiting'))
        info = Gauge('judge_build_info', 'Static deployment identity',
                     ['model', 'checkpoint', 'context_length', 'tensorrt_llm'], registry=registry)
        info.labels(settings.served_name, settings.checkpoint, str(settings.context_length), runtime_version()).set(1)

    def observe(self, request_id, status, seconds, details):
        """Record one finished /v1/systemone request. `details` comes from the handler."""
        self.requests.labels(str(status)).inc()
        self.latency.observe(seconds)
        for stage, milliseconds in details.get('stages', {}).items():
            self.stages.labels(stage).observe(milliseconds / 1000)
        if status == 200:
            self.questions.inc(details.get('questions', 0))
            if details.get('effort'):
                self.efforts.labels(details['effort']).inc()
            self.branches.inc(details.get('branches', 0))
            self.gated.inc(details.get('gated_questions', 0))
            self.input_tokens.inc(details.get('input_tokens', 0))
            self.cached_tokens.inc(details.get('cached_tokens', 0))
        if self.access_log:
            self.access_log.info(json.dumps({
                'event': 'request', 'time': time.time(), 'request_id': request_id, 'status': status,
                'duration_ms': round(seconds * 1000, 2), **details}, separators=(',', ':')))

    def render(self):
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def runtime_version():
    try:
        from importlib.metadata import version
        return version('tensorrt_llm')
    except Exception:
        return 'unknown'

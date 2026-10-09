"""Prometheus metrics and one structured log line per decision request."""
import json
import logging
import sys
import time

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .patches.scheduler import SUSPENSIONS

LATENCY_BUCKETS = (.025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def access_logger():
    logger = logging.getLogger('morj.access')
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class ServiceMetrics:
    """Owns a private registry so several apps (and tests) never collide."""

    def __init__(self, settings, service, access_log=True):
        self.registry = registry = CollectorRegistry()
        self.access_log = access_logger() if access_log else None
        self.requests = Counter('morj_requests', 'Decision requests by HTTP status', ['status'], registry=registry)
        self.latency = Histogram('morj_request_duration_seconds', 'End-to-end decision request latency',
                                 buckets=LATENCY_BUCKETS, registry=registry)
        self.stages = Histogram('morj_stage_duration_seconds', 'Request latency by stage', ['stage'],
                                buckets=LATENCY_BUCKETS, registry=registry)
        self.questions = Counter('morj_questions', 'Answered questions', registry=registry)
        self.efforts = Counter('morj_effort_requests', 'Answered decision requests by effort profile', ['profile'],
                               registry=registry)
        self.branches = Counter('morj_branches', 'Scored prompt branches (orders and ratings)', registry=registry)
        self.gated = Counter('morj_gated_questions', 'Questions whose orders disagreed and got a second pass',
                             registry=registry)
        self.input_tokens = Counter('morj_input_tokens', 'Prompt tokens over all branches', registry=registry)
        self.cached_tokens = Counter('morj_cached_tokens', 'Prompt tokens served from the KV cache', registry=registry)

        def live(name, documentation, read):
            Gauge(name, documentation, registry=registry).set_function(read)

        def budget(attribute):
            return getattr(getattr(service.engine, 'token_budget', None), attribute, 0) or 0

        live('morj_inflight_requests', 'Admitted requests running inference', lambda: service.gate.inflight)
        live('morj_kv_suspensions', 'TensorRT-LLM KV suspensions since start (expected 0)', lambda: SUSPENSIONS['count'])
        live('morj_waiting_requests', 'Requests preparing or queued', lambda: service.gate.waiting)
        live('morj_ready', '1 when serving', lambda: float(service.available))
        live('morj_failed', '1 after a native failure, until restart', lambda: float(service.failed))
        live('morj_native_context_tokens', 'Resident context tokens reserved in the KV budget', lambda: budget('used'))
        live('morj_native_context_waiters', 'Branches waiting for KV budget', lambda: budget('waiting'))
        info = Gauge('morj_build_info', 'Static deployment identity',
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

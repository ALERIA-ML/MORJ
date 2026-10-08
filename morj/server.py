"""`POST /v1/systemone` over the TensorRT-LLM engine.

A request is one task: prepare (tokenize and validate), wait for an
inference slot, then score under a deadline. The request's effort profile
(`X-Decision-Effort` header or `options.effort`; default `high`) sets the
first pass (orders plus ratings) and whether questions whose orders disagree
get a second pass with two more orders. The task races the client connection;
whatever happens, its own `finally` returns the slot.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import faulthandler
from functools import partial
import hmac
import logging
import os
from pathlib import Path
import signal
import tempfile
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import trace as tracing
from .checkpoint import say
from .config import Settings
from .metrics import ServiceMetrics
from .middleware import BodyLimit, RequestContext
from . import method
from .method import combine, disagreeing
from .prompt import MAX_QUESTIONS
from .schema import EvaluationRequest

log = logging.getLogger(__name__)
CANCEL_GRACE = 5
READINESS_PROBE = ('The recorded color is blue.',
                   {'q': {'type': 'choice', 'instructions': 'What color is recorded?',
                          'criteria': {'red': None, 'blue': None}}})


class QueueTimeout(Exception):
    pass


class Gate:
    """`active` requests run inference; up to `queued` more prepare or wait."""

    def __init__(self, active, queued, timeout):
        self._slots = asyncio.Semaphore(active)
        self.capacity, self.timeout = active + queued, timeout
        self.waiting = self.inflight = 0

    def full(self):
        return self.waiting + self.inflight >= self.capacity

    @asynccontextmanager
    async def queue(self):
        self.waiting += 1
        try:
            yield
        finally:
            self.waiting -= 1

    async def acquire(self):
        try:
            await asyncio.wait_for(self._slots.acquire(), self.timeout)
        except TimeoutError:
            raise QueueTimeout from None
        self.inflight += 1
        return True

    def release(self):
        self.inflight -= 1
        self._slots.release()


@dataclass
class Counters:
    completed: int = 0
    rejected: int = 0
    cancelled: int = 0
    timed_out: int = 0
    errors: int = 0
    cached_tokens: int = 0


class Service:
    def __init__(self, settings, engine=None, builder=None):
        self.settings, self.engine, self.builder = settings, engine, builder
        self.owns_engine = engine is None
        self.gate = Gate(settings.max_requests, settings.max_queue, settings.queue_timeout)
        self.counters = Counters()
        self.ready = self.failed = False
        self.exit_task = None

    # Lifecycle ---------------------------------------------------------------

    async def start(self):
        if self.builder is None:
            from transformers import AutoTokenizer
            from .prompt import Builder
            tokenizer = await run_in_threadpool(AutoTokenizer.from_pretrained, self.settings.tokenizer_path,
                                                local_files_only=True, trust_remote_code=False)
            self.builder = Builder(tokenizer, self.settings.context_length)
        if self.engine is None:
            from .engine import Engine
            # TRT-LLM's executor must be created on the process's main thread.
            self.engine = Engine(self.settings.model_path, context_length=self.settings.context_length,
                                 batch_size=self.settings.branch_slots,
                                 max_num_tokens=self.settings.prefill_tokens,
                                 cache_fraction=self.settings.cache_fraction,
                                 nvfp4_backends=[b for b in self.settings.nvfp4_backends.split(',') if b],
                                 shared_budget=self.settings.shared_budget,
                                 seed_min_shared=self.settings.seed_min_shared,
                                 cache_avg_seq_len=self.settings.cache_avg_seq_len)
        state, questions = READINESS_PROBE
        replays = self.engine.graph_metrics['replays']
        branches = self.builder.prepare(state, questions)
        combine(questions, branches, (await self.engine.score_request(branches))[0])
        if self.settings.require_graphs and self.engine.graph_metrics['replays'] <= replays:
            raise RuntimeError('Readiness requires a replayed prefill CUDA graph')
        self.ready = True

    def stop(self):
        self.ready = False
        if self.owns_engine and self.engine is not None:
            self.engine.shutdown()

    @property
    def available(self):
        return self.ready and not self.failed

    def fail(self, request_id, reason):
        """A failed native executor cannot serve again: drain, then exit for the supervisor."""
        self.counters.errors += 1
        log.error('%s (request %s); service marked unavailable', reason, request_id)
        if self.failed:
            return
        self.failed = True
        if self.owns_engine:
            self.exit_task = asyncio.get_running_loop().create_task(self._exit_after_drain())

    async def _exit_after_drain(self):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.drain_timeout
        while (self.gate.inflight or self.gate.waiting) and loop.time() < deadline:
            await asyncio.sleep(.05)
        await asyncio.sleep(.5)  # let the last responses flush
        os._exit(1)

    # Errors ------------------------------------------------------------------

    def unavailable(self, detail='Model is not ready'):
        return HTTPException(503, detail, headers={'Retry-After': str(self.settings.retry_after)})

    def overloaded(self, detail):
        self.counters.rejected += 1
        return HTTPException(529, detail, headers={'Retry-After': str(self.settings.retry_after)})

    # Request path ------------------------------------------------------------

    async def prepare(self, state, questions, gated=False, profile=method.HIGH):
        """First-pass branches of the profile, or with `gated` the second-pass orders of these questions."""
        keys = list(questions)
        build = self.builder.extra if gated else partial(self.builder.prepare, profile=profile)
        branches = []
        try:
            for start in range(0, len(keys), MAX_QUESTIONS):  # the builder takes at most 128 per call
                chunk = {k: questions[k] for k in keys[start:start + MAX_QUESTIONS]}
                branches += await run_in_threadpool(build, state, chunk)
        except ValueError as error:
            raise HTTPException(422, f'{error}. Every question branch must fit in {self.settings.context_length} '
                                     'tokens including the state, instructions and all options; inputs are never '
                                     'truncated.') from error
        return branches

    async def decide(self, state, questions, stages, decision, trace=False, profile=method.HIGH):
        """Prepare, wait for a slot, score the profile's passes. Validation fails before any queue wait."""
        clock = time.perf_counter()
        acquired = False
        try:
            async with self.gate.queue():
                branches = await self.prepare(state, questions, profile=profile)
                stages['prepare'], clock = ms(clock), time.perf_counter()
                acquired = await self.gate.acquire()
            stages['queue'], clock = ms(clock), time.perf_counter()
            if self.failed:
                raise self.unavailable('Inference engine failed; retry after the service restarts')
            started, replays = clock, (self.engine.graph_metrics or {}).get('replays', 0)
            try:
                async with asyncio.timeout(self.settings.request_timeout):
                    rows, stats = await self.engine.score_request(branches)
                    passes = [(branches, rows, stats)]
                    gated = disagreeing(branches, rows) if profile.gated else []
                    if gated:  # the orders disagree: score the other two rotations of those questions
                        extra = await self.prepare(state, {k: questions[k] for k in gated}, gated=True)
                        if extra:
                            passes.append((extra, *await self.engine.score_request(extra)))
                            decision['gated_questions'] = len(gated)
            except TimeoutError:
                self.counters.timed_out += 1
                raise HTTPException(504, f'Inference deadline of {self.settings.request_timeout:g} seconds exceeded')
            stages['inference'] = ms(clock)
        except QueueTimeout:
            raise self.overloaded('Queue deadline exceeded') from None
        finally:
            if acquired:
                self.gate.release()
        branches = [b for p in passes for b in p[0]]
        rows = [r for p in passes for r in p[1]]
        steps = {} if trace else None
        result = combine(questions, branches, rows, steps, profile)
        result['usage'] = {'input_tokens': sum(len(b.input_ids) for b in branches), 'output_tokens': 0,
                           'effort': profile.name, 'branches': len(branches)}
        cached = sum(c for p in passes for c in p[2]['cached_tokens'])
        decision.update(branches=len(branches), input_tokens=result['usage']['input_tokens'], cached_tokens=cached)
        self.counters.cached_tokens += cached
        if trace:
            replayed = (self.engine.graph_metrics or {}).get('replays', 0) - replays
            result['trace'] = await run_in_threadpool(tracing.build, self.builder, questions, passes, steps, gated,
                                                      started, dict(stages), replayed, profile)
        return result

    def effort(self, body, request):
        """The request's profile: `X-Decision-Effort` header or `options.effort`, else the service default."""
        header = request.headers.get('x-decision-effort')
        field = body.options.effort if body.options else None
        try:
            chosen = [method.profile(e) for e in (header, field) if e]
            if len({p.name for p in chosen}) > 1:
                raise ValueError(f'X-Decision-Effort ({header!r}) and options.effort ({field!r}) disagree')
            return chosen[0] if chosen else method.profile(self.settings.default_effort)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    async def handle(self, body, request, trace=False):
        profile = self.effort(body, request)
        if not self.available:
            raise self.unavailable()
        if self.gate.full():
            raise self.overloaded('Queue is full')
        context = request.scope['state']
        questions = {key: question.model_dump() for key, question in body.questions.items()}
        context['decision'].update(questions=len(questions), effort=profile.name)
        task = asyncio.create_task(self.decide(body.state, questions, context['stages'], context['decision'], trace,
                                               profile))
        stop = asyncio.Event()
        watcher = asyncio.create_task(watch_disconnect(request, stop))
        bound = self.settings.queue_timeout + self.settings.request_timeout + 10
        try:
            await asyncio.wait((task, watcher), timeout=bound, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                result = task.result()
                self.counters.completed += 1
                return result
            if watcher.done():
                self.counters.cancelled += 1
                raise HTTPException(499, 'Client disconnected')
            self.counters.timed_out += 1
            raise HTTPException(504, 'Request exceeded its overall deadline')
        except HTTPException:
            raise
        except Exception:
            log.exception('Native inference failed')
            self.fail(context['request_id'], 'Native inference failed')
            raise self.unavailable('Inference engine failed; retry after the service restarts') from None
        finally:
            stop.set()
            await self._settle((task, watcher), context['request_id'])

    async def _settle(self, tasks, request_id):
        for task in tasks:
            task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=CANCEL_GRACE)
        for task in done:
            if not task.cancelled():
                task.exception()  # retrieved; already reported above
        if pending:
            self.fail(request_id, f'Native cancellation exceeded {CANCEL_GRACE} seconds')

    def stats(self):
        budget = getattr(self.engine, 'token_budget', None)
        slots = getattr(self.engine, 'slots', None)
        return {**asdict(self.counters), 'inflight': self.gate.inflight, 'waiting': self.gate.waiting,
                'native_context_tokens': getattr(budget, 'used', None),
                'native_context_waiters': getattr(budget, 'waiting', None),
                'free_branch_slots': getattr(slots, '_value', None),
                'graphs': getattr(self.engine, 'graph_metrics', None)}


def ms(since):
    return 1000 * (time.perf_counter() - since)


async def watch_disconnect(request, stop):
    # Starlette checks receive() inside an AnyIO cancel scope that can swallow
    # an outer cancellation, so poll with an explicit stop flag instead.
    while not stop.is_set():
        if await request.is_disconnected():
            return
        await asyncio.sleep(.1)


STACKS = []


def register_stack_dump():
    """`docker kill -s USR1 <container>` appends every thread's stack to $TMPDIR/morj-stacks.log (/tmp, mounted from
    the host cache folder). faulthandler writes from the signal handler without the GIL or stdout, so even a process
    frozen inside a blocked log write can be diagnosed. Registered once the engine is up: its native libraries
    install their own SIGUSR1 handler while it loads."""
    STACKS.append(open(Path(tempfile.gettempdir()) / 'morj-stacks.log', 'a', buffering=1))
    faulthandler.register(signal.SIGUSR1, file=STACKS[-1], all_threads=True)


def create_app(settings=None, *, engine=None, builder=None):
    settings = settings or Settings.from_env()
    service = Service(settings, engine, builder)
    created = int(time.time())

    @asynccontextmanager
    async def lifespan(app):
        try:
            await service.start()
            register_stack_dump()
            host = 'localhost' if settings.host in ('127.0.0.1', '0.0.0.0', '::') else settings.host
            say(f'ready. Playground: http://{host}:{settings.port}/   API: POST http://{host}:{settings.port}/v1/systemone')
            yield
        finally:
            service.stop()

    app = FastAPI(title='MORJ decision service', lifespan=lifespan)
    app.state.service = service
    app.state.metrics = metrics = ServiceMetrics(settings, service, access_log=settings.access_log)
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    app.add_middleware(RequestContext, on_response=metrics.observe)

    async def authenticate(authorization: str | None = Header(default=None)):
        expected = f'Bearer {settings.api_key}'.encode()
        if settings.api_key and not hmac.compare_digest((authorization or '').encode(), expected):
            raise HTTPException(401, 'Missing or invalid API key', headers={'WWW-Authenticate': 'Bearer'})

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={'detail': [
            {k: item[k] for k in ('loc', 'msg', 'type')} for item in error.errors()]})

    @app.post('/v1/systemone', dependencies=[Depends(authenticate)])
    async def systemone(body: EvaluationRequest, request: Request):
        return await service.handle(body, request)

    if settings.trace:
        @app.post('/v1/trace', dependencies=[Depends(authenticate)])
        async def trace(body: EvaluationRequest, request: Request):
            """Same answers as /v1/systemone, plus what the engine did for every branch."""
            return await service.handle(body, request, trace=True)

    @app.get('/health')
    async def health():
        if not service.available:
            raise service.unavailable()
        return {'status': 'ready', 'model': settings.served_name, 'checkpoint': settings.checkpoint,
                'checkpoint_path': settings.model_path, 'precision': settings.spec.precision,
                'modalities': ['text'], 'authentication': bool(settings.api_key),
                'context_length': settings.context_length, 'prefill_tokens': settings.prefill_tokens,
                'nvfp4_backends': settings.nvfp4_backends or 'default', 'branch_slots': settings.branch_slots,
                'shared_budget': settings.shared_budget,
                'default_effort': method.profile(settings.default_effort).name, 'method': tracing.METHOD,
                'trace': settings.trace,
                'prefill_graphs': service.engine.graph_metrics}

    @app.get('/live')
    async def live():
        if service.failed:
            raise service.unavailable('Inference engine failed; restart pending')
        return {'status': 'alive', 'ready': service.ready}

    @app.get('/metrics', include_in_schema=False)
    async def prometheus():
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)

    @app.get('/stats', dependencies=[Depends(authenticate)])
    async def stats():
        return service.stats()

    @app.get('/v1/models', dependencies=[Depends(authenticate)])
    async def models():
        """The served model, in the OpenAI model-list format vLLM uses."""
        return {'object': 'list', 'data': [{
            'id': settings.served_name, 'object': 'model', 'created': created, 'owned_by': 'morj',
            'root': settings.spec.repo, 'parent': None, 'max_model_len': settings.context_length}]}

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    async def playground():
        return Path(__file__).with_name('playground.html').read_text()

    return app

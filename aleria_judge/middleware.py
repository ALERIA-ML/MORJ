"""Pure ASGI middleware: they never consume the handler's disconnect events."""
import re
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders

REQUEST_ID = re.compile(r'[A-Za-z0-9._:-]{1,128}')


class BodyLimit:
    """Reject bodies over `limit` bytes with 413, using Content-Length when declared."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        too_large = JSONResponse({'detail': f'Request body exceeds {self.limit} bytes'}, status_code=413)
        declared = dict(scope.get('headers', ())).get(b'content-length', b'')
        if declared.isdigit() and int(declared) > self.limit:
            return await too_large(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunks.append(message.get('body', b''))
            size += len(chunks[-1])
            if size > self.limit:
                return await too_large(scope, receive, send)
            if not message.get('more_body', False):
                break
        body = {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}

        async def replay():
            nonlocal body
            if body is None:
                return await receive()
            message, body = body, None
            return message
        await self.app(scope, replay, send)


class RequestContext:
    """Request IDs, `server-timing` stages, and a callback once the status is known.

    The handler fills `scope['state']['stages']` (milliseconds per stage) and
    `scope['state']['decision']` (counts for metrics and logs).
    """

    def __init__(self, app, on_response=None, path='/v1/systemone'):
        self.app, self.on_response, self.path = app, on_response, path

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        claimed = dict(scope.get('headers', ())).get(b'x-request-id', b'').decode('latin-1')
        request_id = claimed if REQUEST_ID.fullmatch(claimed) else uuid.uuid4().hex
        state = scope.setdefault('state', {})
        state.update(request_id=request_id, stages={}, decision={})

        async def send_with_metadata(message):
            if message['type'] == 'http.response.start':
                elapsed = time.perf_counter() - started
                headers = MutableHeaders(scope=message)
                headers['x-request-id'] = request_id
                headers['server-timing'] = ', '.join(
                    [f'total;dur={elapsed * 1000:.2f}'] + [f'{k};dur={v:.2f}' for k, v in state['stages'].items()])
                if self.on_response and scope.get('path') == self.path:
                    self.on_response(request_id, message['status'], elapsed,
                                     {**state['decision'], 'stages': state['stages']})
            await send(message)
        await self.app(scope, receive, send_with_metadata)

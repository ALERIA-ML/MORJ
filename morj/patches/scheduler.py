"""KV cache manager V2 scheduler guard.

A context chunk that cannot get KV blocks (with 32 branch slots this is the small
sliding-window pool) waits inside TensorRT-LLM and is admitted on a later iteration,
once running branches free blocks. That is queueing, not an error: on
heavy index requests a branch waits one or two iterations (p50 ~0.7 s, max ~1.7 s on
H100). The first 100 such requests are logged as `context_allocation_failed`.

Anything that stops progress altogether would make the executor spin while
reporting healthy. After 100 consecutive iterations with active requests, none in
flight and none scheduled, we raise, so the service fails fast and restarts.
Upstream `main` has since added the same guard as `_detect_deadlock` (1,000 passes).

Earlier releases also dropped a failed first chunk's reused prefix and retried it
cold, and halved a chunk's token budget after an allocation failure. Under load the
cold retry fired on ordinary queueing and failed (~1,000 events), and with both
removed answers and latency were unchanged. The stall it did rescue (a first chunk
that can never resume its reused prefix) comes from TensorRT-LLM's one-block host KV
tier and is prevented by sizing that tier explicitly (`host_kv_cache_gb`,
docs/tensorrt-llm-patch.md); this guard stays as the last resort.
"""
import json

STALL_ITERATIONS = 100
# TensorRT-LLM's own suspension (`KVCacheManagerV2.suspend_request`), counted in /metrics (`morj_kv_suspensions`). This service should never
# trigger it: without speculative decoding a context chunk that cannot get blocks waits instead, and one generated
# token means no decode-step allocations to evict for. Without a host KV tier, suspended pages cannot leave the GPU
# (TensorRT-LLM warns that scheduling can then deadlock), so a suspension is logged loudly.
SUSPENSIONS = {'count': 0}


def note_suspension(host_tier):
    """Count one TensorRT-LLM suspension; the log record for the first 10 and every 1,000th, else None."""
    SUSPENSIONS['count'] += 1
    n = SUSPENSIONS['count']
    if n > 10 and n % 1000:
        return None
    record = {'event': 'kv_suspended', 'count': n, 'host_kv_tier': host_tier}
    if not host_tier:
        record['warning'] = ('TensorRT-LLM suspended a request but no host KV tier is configured, so its pages cannot '
                             'leave the GPU; set MORJ_HOST_KV_CACHE_GB (see docs/deploy.md)')
    return record


def describe(scheduler, req):
    manager = scheduler.kv_cache_manager
    stats = manager.get_kv_cache_stats()
    cache = manager.kv_cache_map.get(req.py_request_id)
    return {'request_id': req.py_request_id, 'state': req.state_value, 'first_chunk': req.is_first_context_chunk,
            'position': req.context_current_position, 'remaining': req.context_remaining_length,
            'cache_capacity': cache.capacity if cache is not None else None,
            'free_blocks_per_window': stats.num_free_blocks_per_window_size}


def install(host_tier=False):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler as Scheduler

    originals = {'allocate': Scheduler._try_allocate_context, 'schedule': Scheduler.schedule_request,
                 'suspend': KVCacheManagerV2.suspend_request}

    def suspend(manager, req):
        record = note_suspension(host_tier)
        if record is not None:
            print(json.dumps({**record, 'request_id': req.py_request_id}), flush=True)
        return originals['suspend'](manager, req)
    reported = set()

    def allocate(scheduler, req, num_tokens):
        allocated = originals['allocate'](scheduler, req, num_tokens)
        if not allocated and req.py_request_id not in reported and len(reported) < 100:
            reported.add(req.py_request_id)
            print(json.dumps({'event': 'context_allocation_failed', 'requested_tokens': num_tokens,
                              **describe(scheduler, req)}), flush=True)
        return allocated

    def schedule(scheduler, active_requests, inflight_request_ids):
        result = originals['schedule'](scheduler, active_requests, inflight_request_ids)
        stalled = bool(active_requests) and not inflight_request_ids and not result.num_fitting_requests
        scheduler._stalled = getattr(scheduler, '_stalled', 0) + 1 if stalled else 0
        if scheduler._stalled >= STALL_ITERATIONS:
            raise RuntimeError('TRT-LLM scheduler made no progress for %d iterations: %s'
                               % (STALL_ITERATIONS, json.dumps([describe(scheduler, r) for r in active_requests])))
        return result

    Scheduler._try_allocate_context, Scheduler.schedule_request = allocate, schedule
    KVCacheManagerV2.suspend_request = suspend

    def restore():
        Scheduler._try_allocate_context = originals['allocate']
        Scheduler.schedule_request = originals['schedule']
        KVCacheManagerV2.suspend_request = originals['suspend']
    return restore

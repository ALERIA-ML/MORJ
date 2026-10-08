"""KV cache manager V2 scheduler guard.

A context chunk that cannot get KV blocks (with 32 branch slots this is the small
sliding-window pool) stays suspended inside TensorRT-LLM and is admitted on a later
iteration, once running branches free blocks. That is queueing, not an error: on
heavy index requests a branch waits one or two iterations (p50 ~0.7 s, max ~1.7 s on
H100). The first 100 such requests are logged as `context_allocation_failed`.

Anything that stops progress altogether would make the executor spin while
reporting healthy. After 100 consecutive iterations with active requests, none in
flight and none scheduled, we raise, so the service fails fast and restarts.
Upstream `main` has since added the same guard as `_detect_deadlock` (1,000 passes).

Earlier releases also dropped a failed first chunk's reused prefix and retried it
cold, and halved a chunk's token budget after an allocation failure. Neither ever
helped: across about 1,000 events the cold retry never succeeded, and with both
removed, answers and latency were unchanged on every A/B.
"""
import json

STALL_ITERATIONS = 100


def describe(scheduler, req):
    manager = scheduler.kv_cache_manager
    stats = manager.get_kv_cache_stats()
    cache = manager.kv_cache_map.get(req.py_request_id)
    return {'request_id': req.py_request_id, 'state': req.state_value, 'first_chunk': req.is_first_context_chunk,
            'position': req.context_current_position, 'remaining': req.context_remaining_length,
            'cache_capacity': cache.capacity if cache is not None else None,
            'free_blocks_per_window': stats.num_free_blocks_per_window_size}


def install():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler as Scheduler

    originals = {'allocate': Scheduler._try_allocate_context, 'schedule': Scheduler.schedule_request}
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

    def restore():
        Scheduler._try_allocate_context = originals['allocate']
        Scheduler.schedule_request = originals['schedule']
    return restore

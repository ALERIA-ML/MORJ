"""Single-token answer readout on the TensorRT-LLM PyTorch backend (NVFP4 checkpoint)."""
import asyncio
import faulthandler
import json
import os
import sys
import threading
import time
import uuid

from . import patches
from .admission import TokenBudget

PREFILL_GRAPH_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
NATIVE_TIMEOUT = 600


def answer_probabilities(logits, label_ids):
    """Softmax over one generation step's logits, restricted to the question's label tokens."""
    import torch
    if logits is None or logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError('Expected exactly one generation step of vocabulary logits')
    if not label_ids or len(set(label_ids)) != len(label_ids):
        raise ValueError('Candidate IDs must be nonempty and distinct')
    if any(type(i) is not int or not 0 <= i < logits.shape[1] for i in label_ids):
        raise ValueError('Candidate ID outside vocabulary')
    selected = logits[0, list(label_ids)].float()
    if not torch.isfinite(selected).all().item():
        raise ValueError('Missing or nonfinite candidate logits')
    return torch.softmax(selected, dim=-1).cpu().tolist()


def common_prefix(a, b):
    """Length of the common token prefix of two prompts."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Engine:
    """Scores prompt branches; each branch is a prefill plus one readout step.

    The first branch of a request runs alone and populates the KV cache; the
    rest run concurrently and reuse its prefix. A random cache salt per
    request keeps KV blocks from ever being shared across requests.
    """

    shared_budget = False
    seed_min_shared = 0

    def __init__(self, model, *, context_length, batch_size, max_num_tokens, cache_fraction,
                 cache_avg_seq_len=4096, nvfp4_backends=(), shared_budget=False, seed_min_shared=0):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        self.patches = patches.install()
        self.context_length = context_length
        self.slots = asyncio.Semaphore(batch_size)
        # Resident-token admission: concurrent long branches queue here instead
        # of forcing KV preemption inside the executor.
        self.token_budget = TokenBudget(context_length)
        # With `shared_budget`, branches after a request's first reserve only their tokens beyond the prefix
        # they share with it; the shared prefix (resident once, through block reuse) is reserved once for
        # the whole second wave. Scheduling only: prompts and readouts are unchanged.
        self.shared_budget = shared_budget
        # With `seed_min_shared` > 0, branches that share fewer prompt tokens than that with the first one
        # (e.g. the two orders of a single question, whose option block precedes the state) run in one wave:
        # waiting for the first branch to seed a reusable prefix would buy nothing.
        self.seed_min_shared = seed_min_shared
        # The checkpoint's hf_quant_config.json selects NVFP4 linears and an FP8 KV cache
        # (KvCacheConfig dtype 'auto'). Backends only choose the NVFP4 GEMM kernels.
        extra = {'nvfp4_gemm_config': {'allowed_backends': list(nvfp4_backends)}} if nvfp4_backends else {}
        with patches.observe_kv_capacity() as (capacities, _):
            self.llm = LLM(
                model=model, tensor_parallel_size=1, trust_remote_code=False,
                max_seq_len=context_length, max_batch_size=batch_size, max_num_tokens=max_num_tokens,
                disable_mm_encoder=True, enable_chunked_prefill=True, disable_overlap_scheduler=False,
                enable_autotuner=True, attn_backend='FLASHINFER',
                # Scoring stops after the first token, so only prefill graphs ever run.
                cuda_graph_config=None, prefill_cuda_graph_backend='breakable',
                prefill_capture_num_tokens=[n for n in PREFILL_GRAPH_BUCKETS if n <= max_num_tokens],
                enable_early_first_token_response=True, gather_generation_logits=True,
                # SWA scratch reuse (a TensorRT-LLM prototype) changes most answers together with block reuse.
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=cache_fraction, enable_block_reuse=True,
                                              enable_swa_scratch_reuse=False, avg_seq_len=cache_avg_seq_len,
                                              dtype='auto'),
                **extra)
        if not capacities or min(capacities) < context_length:
            self.close()
            raise RuntimeError(f'KV cache supports {capacities}, below the declared context {context_length}')
        if not self.graph_metrics['captures']:
            self.close()
            raise RuntimeError('Prefill CUDA graphs were not captured')
        self.params = SamplingParams(max_tokens=1, temperature=1., top_k=0, top_p=1., repetition_penalty=1.,
                                     presence_penalty=0., frequency_penalty=0., return_generation_logits=True,
                                     return_context_logits=False, detokenize=False, add_special_tokens=False,
                                     ignore_eos=True)

    @property
    def graph_metrics(self):
        return self.patches.graph_metrics

    async def score_request(self, branches):
        """Probabilities per branch, plus per-branch reused prompt tokens and timings.

        `timings` holds (queued, started, finished) `time.perf_counter()` values:
        a branch is queued until it holds a slot and its KV token budget.
        """
        if not branches:
            raise ValueError('At least one scoring branch is required')
        salt = uuid.uuid4().hex
        if (self.seed_min_shared and len(branches) > 1
                and min(common_prefix(branches[0].input_ids, b.input_ids) for b in branches[1:]) < self.seed_min_shared):
            rows = await self._score(branches, salt)
            return [p for p, _, _ in rows], {'cached_tokens': [c for _, c, _ in rows], 'timings': [t for _, _, t in rows]}
        rows = await self._score(branches[:1], salt)
        if len(branches) > 1 and self.shared_budget:
            seed = branches[0].input_ids
            shared = min(common_prefix(seed, b.input_ids) for b in branches[1:])
            async with self.token_budget.reserve(max(shared, 1)):
                rows += await self._score(branches[1:], salt, shared)
        elif len(branches) > 1:
            rows += await self._score(branches[1:], salt)
        return [p for p, _, _ in rows], {'cached_tokens': [c for _, c, _ in rows], 'timings': [t for _, _, t in rows]}

    async def _score(self, branches, salt, shared=0):
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._branch(branch, salt, shared)) for branch in branches]
        return [task.result() for task in tasks]

    async def _branch(self, branch, salt, shared=0):
        tokens = len(branch.input_ids)
        if not tokens or tokens + 1 > self.context_length:
            raise ValueError('Prompt exceeds context limit; no truncation')
        queued = time.perf_counter()
        async with self.slots, self.token_budget.reserve(tokens - shared + 1):
            started = time.perf_counter()
            pending = self.llm.generate_async(branch.input_ids, self.params, cache_salt=salt)
            try:
                async with asyncio.timeout(NATIVE_TIMEOUT):
                    result = await pending
            except BaseException as error:
                try:
                    pending.abort()  # cancel the native request too, not only its waiter
                except Exception as abort_error:
                    error.add_note(f'native abort also failed: {abort_error!r}')
                error.add_note(f'question={branch.question} prompt_tokens={tokens} order={branch.order}')
                raise
        finished = time.perf_counter()
        if result.prompt_token_ids != branch.input_ids or len(result.outputs) != 1:
            raise ValueError('Runtime changed prompt IDs or completion count')
        return (answer_probabilities(result.outputs[0].generation_logits, branch.label_ids), int(result.cached_tokens),
                (queued, started, finished))

    def close(self):
        try:
            self.llm.shutdown()
        finally:
            self.patches.restore()

    def shutdown(self, timeout=30):
        """Close on the owning thread; force the process down if TRT-LLM hangs."""
        def expired():
            print(json.dumps({'event': 'shutdown_timeout', 'timeout_seconds': timeout}), file=sys.stderr, flush=True)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            os._exit(1)
        timer = threading.Timer(timeout, expired)
        timer.daemon = True
        timer.start()
        try:
            self.close()
        finally:
            timer.cancel()

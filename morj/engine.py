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
from .images import SOFT_TOKENS, image_tokens

NATIVE_TIMEOUT = 600
# Branches of a request that share fewer prompt tokens than this with its first branch (e.g. the two orders of a
# single question, whose option block precedes the state) run in one wave: waiting for the first branch to seed a
# reusable prefix would buy nothing. Otherwise the first branch runs alone, and the rest reserve only their tokens
# beyond the shared prefix, which is resident once through block reuse and reserved once for the whole second wave.
# Scheduling only: prompts and readouts are unchanged.
SEED_MIN_SHARED = 256
# The typical sequence TensorRT-LLM sizes the sliding-window against the global KV pools for.
CACHE_AVG_SEQ_LEN = 4096


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

    expansion_logged = False

    def __init__(self, model, *, context_length, batch_size, max_num_tokens, cache_fraction,
                 host_kv_cache_gb=0, nvfp4_backends=(), vision=True, image_soft_tokens=SOFT_TOKENS, vision_encoder_batch=2):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig

        self.patches = patches.install(vision=vision, vision_batch=vision_encoder_batch, host_tier=host_kv_cache_gb > 0)
        self.context_length = context_length
        self.slots = asyncio.Semaphore(batch_size)
        # Resident-token admission: concurrent long branches queue here instead
        # of forcing KV preemption inside the executor.
        self.token_budget = TokenBudget(context_length)
        # The checkpoint's hf_quant_config.json selects NVFP4 linears and an FP8 KV cache
        # (KvCacheConfig dtype 'auto'). Backends only choose the NVFP4 GEMM kernels.
        extra = {'nvfp4_gemm_config': {'allowed_backends': list(nvfp4_backends)}} if nvfp4_backends else {}
        with patches.observe_kv_capacity() as capacities:
            self.llm = LLM(
                model=model, tensor_parallel_size=1, trust_remote_code=False,
                max_seq_len=context_length, max_batch_size=batch_size, max_num_tokens=max_num_tokens,
                disable_mm_encoder=not vision, enable_chunked_prefill=True, disable_overlap_scheduler=False,
                enable_autotuner=True, attn_backend='FLASHINFER',
                # No CUDA graphs: scoring stops after the first token, so only prefill could use them; breakable prefill
                # graphs were a net latency loss on H100 (mean -10% without them) and TRT-LLM refuses them with images.
                cuda_graph_config=None,
                enable_early_first_token_response=True, gather_generation_logits=True,
                # The host KV tier is always sized explicitly (0 = none, the default). Left to TensorRT-LLM, its size is
                # capped at 80% of RLIMIT_MEMLOCK; under Docker's and most Kubernetes defaults (64 KB) that is a
                # one-block tier, on which the V2 KV manager stopped making progress on a long multi-branch request
                # (deterministic on H100 at a 202-block sliding-window pool). No tier or a real one (>= 0.1 GB tested)
                # does not. With one generated token and resident-token admission, nothing was ever suspended, so the
                # tier stayed empty under every load tested (docs/tensorrt-llm-patch.md).
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=cache_fraction, enable_block_reuse=True,
                                              avg_seq_len=CACHE_AVG_SEQ_LEN, dtype='auto',
                                              host_cache_size=int(host_kv_cache_gb * 2**30)),
                **extra)
        if not capacities or min(capacities) < context_length:
            self.close()
            raise RuntimeError(f'KV cache supports {capacities}, below the declared context {context_length}')
        self.vision, self.image_soft_tokens = vision, image_soft_tokens
        if vision:
            processor = json.load(open(os.path.join(model, 'processor_config.json'))).get('image_processor', {})
            if processor.get('max_soft_tokens') != image_soft_tokens:
                self.close()
                raise RuntimeError(f"image_soft_tokens={image_soft_tokens} but the checkpoint's processor uses "
                                   f"max_soft_tokens={processor.get('max_soft_tokens')}: serve a checkpoint directory whose "
                                   'processor_config.json sets the budget')
        # The <|image|> placeholder id, from the checkpoint (Gemma 4: 258880); only image requests use it.
        self.image_id = json.load(open(os.path.join(model, 'config.json'))).get('image_token_id') if vision else None
        self.params = SamplingParams(max_tokens=1, temperature=1., top_k=0, top_p=1., repetition_penalty=1.,
                                     presence_penalty=0., frequency_penalty=0., return_generation_logits=True,
                                     return_context_logits=False, detokenize=False, add_special_tokens=False,
                                     ignore_eos=True)

    async def score_request(self, branches, images=()):
        """Probabilities per branch, plus per-branch reused prompt tokens and timings.

        `timings` holds (queued, started, finished) `time.perf_counter()` values:
        a branch is queued until it holds a slot and its KV token budget.
        """
        if not branches:
            raise ValueError('At least one scoring branch is required')
        salt = uuid.uuid4().hex
        shared = min((common_prefix(branches[0].input_ids, b.input_ids) for b in branches[1:]), default=0)
        if shared < SEED_MIN_SHARED:
            rows = await self._score(branches, salt, images=images)
        else:
            rows = await self._score(branches[:1], salt, images=images)
            async with self.token_budget.reserve(shared):
                rows += await self._score(branches[1:], salt, shared, images=images)
        return [p for p, _, _ in rows], {'cached_tokens': [c for _, c, _ in rows], 'timings': [t for _, _, t in rows]}

    async def _score(self, branches, salt, shared=0, images=()):
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._branch(branch, salt, shared, images)) for branch in branches]
        return [task.result() for task in tasks]

    async def _branch(self, branch, salt, shared=0, images=()):
        # Each <|image|> placeholder becomes the image's soft tokens inside TensorRT-LLM.
        tokens = len(branch.input_ids) + (image_tokens(self.image_soft_tokens) - 1) * len(images)
        if not tokens or tokens + 1 > self.context_length:
            raise ValueError('Prompt exceeds context limit; no truncation')
        queued = time.perf_counter()
        async with self.slots, self.token_budget.reserve(tokens - shared + 1):
            started = time.perf_counter()
            # The per-image budget comes from the checkpoint's processor_config.json: TensorRT-LLM 1.3.0rc29's Gemma 4
            # input processor ignores mm_processor_kwargs (checked at startup, see __init__).
            prompt = ({'prompt_token_ids': branch.input_ids, 'multi_modal_data': {'image': list(images)}}
                      if images else branch.input_ids)
            pending = self.llm.generate_async(prompt, self.params, cache_salt=salt)
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
        if len(result.outputs) != 1:
            raise ValueError('Runtime changed the completion count')
        if images:
            # TensorRT-LLM expands the image placeholders (and, for Gemma 4, re-tokenizes the prompt text). Everything
            # after the last image must come back token for token.
            after = branch.input_ids[len(branch.input_ids) - branch.input_ids[::-1].index(self.image_id):]
            if result.prompt_token_ids[-len(after):] != after or len(result.prompt_token_ids) <= len(branch.input_ids):
                raise ValueError('Runtime changed the prompt IDs after the images')
            if not self.expansion_logged:  # once per process: what a placeholder really became
                self.expansion_logged = True
                print(json.dumps({'event': 'image_expansion', 'images': len(images), 'placeholder_ids': len(branch.input_ids),
                                  'runtime_ids': len(result.prompt_token_ids), 'max_soft_tokens': self.image_soft_tokens}), flush=True)
        elif result.prompt_token_ids != branch.input_ids:
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

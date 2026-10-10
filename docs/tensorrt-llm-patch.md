# TensorRT-LLM workarounds

MORJ runs on the TensorRT-LLM 1.3.0rc29 PyTorch backend. Four things in that release need handling: one
setting, two process-local patches and one checkpoint overlay. None of them changes an answer. The patches
install only when the TensorRT-LLM files they touch are byte-identical to the validated release
([`runtime_identity.json`](../morj/patches/runtime_identity.json)); that check runs at image build
([`Dockerfile`](../Dockerfile)) and again at startup.

## Host KV tier

**Symptom.** One long multi-branch request makes the scheduler spin: its first context chunks wait at
position 0 while every KV block is free. After 100 such iterations the stall guard (below) stops the
executor, `/health` turns unavailable and every later request gets 503 until the service restarts.

**Trigger.** A Decision Index 0.3 ToolRet request with 21 branches (200,652 prompt tokens, a short shared
prefix and ~18.8k-token branches) fails deterministically on a fresh service whose sliding-window KV pool has
202 blocks (on an H100 PCIe 80 GB, within 1.4 s). The outcome depends on the exact pool size
(149, 152, 205 and 209 blocks pass), so other requests can hit it on other GPUs or settings.

**Cause.** When `host_cache_size` is unset, TensorRT-LLM sizes its host (secondary) KV tier automatically,
capped at 80% of `RLIMIT_MEMLOCK`. Docker and most Kubernetes runtimes give containers 64 KB, so the tier is
one block per pool group. The V2 KV manager's reuse logic then reaches a state in which a first chunk claims a
reusable prefix it can never resume. Nothing is ever suspended and the tier is never filled; its one-block
presence is enough. With no host tier, or a real one, the same request runs:

| Host KV tier | Blocks per pool group | Result (same request, 202-block pool) |
| --- | --- | --- |
| automatic, memlock 64 KB | 1 / 1 | stall |
| explicit 0.004 GB, 0.03 GB | 1 / 1, 1 / 2 | stall |
| none (`host_cache_size=0`) | — | answered |
| explicit 0.1 to 16 GB | 2 / 11 to 474 / 1,810 | answered |
| automatic, `--ulimit memlock=-1` | ~11 GiB | answered |

**Fix.** MORJ always passes `host_cache_size` explicitly (`HOST_KV_CACHE_GB`, default 0: no tier); values
between 0 and 0.5 GB are refused, and TensorRT-LLM's automatic sizing is never used. MORJ generates one token
per branch and admits only what fits the GPU pool, so no request was ever suspended under any load tested
(sequential, 8 and 32 concurrent clients, 8 concurrent ~46k-token requests) and a tier stayed empty;
`/metrics` counts suspensions (`morj_kv_suspensions`), and without a tier each one is logged as a warning.

## Stall guard

[`patches/scheduler.py`](../morj/patches/scheduler.py) wraps the V2 scheduler. A context chunk that cannot
get KV blocks waits inside TensorRT-LLM and is admitted on a later iteration; that is queueing, and the first
100 such waits are logged as `context_allocation_failed`. If 100 consecutive iterations pass with active
requests and none in flight or scheduled, the guard raises, so the service fails fast and restarts instead of
spinning while reporting healthy. Upstream `main` has since added the same check (`_detect_deadlock`, 1,000
passes). The guard only ever stops a stuck executor; it never touches a request that makes progress.

## Vision encoder batching

TensorRT-LLM concatenates every image of every context request in an engine step and runs Gemma 4's vision
tower over all of them in one call. At 1,120 vision tokens an image is about 10,000 patches, and eight images
ran an H100 80 GB out of memory inside the vision attention, which stopped the executor.
[`patches/vision.py`](../morj/patches/vision.py) encodes at most two images per call (`MORJ_VISION_ENCODER_BATCH`)
and concatenates the results. The tower attends within each image only, so the embeddings are the same; the
patch refuses to install if the source of `Gemma4ForConditionalGeneration._get_image_features` changed.

## Image budget overlay

Gemma 4 reads 280 vision tokens per image by default. TensorRT-LLM 1.3.0rc29 ignores
`mm_processor_kwargs` and reads the budget only from the checkpoint's `processor_config.json`, so at startup
MORJ builds a symlinked copy of the checkpoint with that file edited (`checkpoint/image1120` in the cache
directory; no weights are copied) and refuses to start if the engine's budget differs from
`IMAGE_SOFT_TOKENS`.

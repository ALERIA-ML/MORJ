"""Service settings, read from MORJ_* environment variables, and the checkpoint check.

The decision method is fixed per effort in `method.py`. The capacity settings change speed and memory, not the
method; other values than the evaluated run's are allowed, logged at startup and reported in /health.
"""
from dataclasses import dataclass, fields
import hashlib
import os
from pathlib import Path

from .checkpoint import CHECKPOINTS
from .images import SOFT_TOKENS
from .method import profile

# The settings of the evaluated Decision Index run; they are also the defaults. cache_fraction 0.7: at 1,120 soft
# tokens per image the vision encoder needs the headroom (0.8 ran out of memory in the encoder on an H100 80 GB);
# text only (vision false) can use 0.8.
EVALUATED = {'checkpoint': 'bf16', 'default_effort': 'high', 'vision': True, 'image_soft_tokens': SOFT_TOKENS,
             'context_length': 65536, 'branch_slots': 32, 'prefill_tokens': 2048, 'cache_fraction': 0.7}
NVFP4_BACKENDS = ('cutlass', 'cublaslt', 'cuda_core', 'cutedsl', 'marlin')
TRUE, FALSE = ('1', 'true', 'yes'), ('0', 'false', 'no')


@dataclass(frozen=True)
class Settings:
    # Model
    checkpoint: str = 'bf16'  # 'bf16' (google/gemma-4-31B-it) or 'nvfp4' (nvidia/Gemma-4-31B-IT-NVFP4, Blackwell)
    model_path: str = ''      # empty: a checkpoint mounted at /model, else the pinned revision from the HF cache
    tokenizer_path: str = ''  # empty: the overlay of canonical tokenizer files is built under ~/.cache at startup
    model_name: str = ''      # empty: the name the checkpoint was evaluated under (gemma-4-31b-trt / -nvfp4-trt)
    context_length: int = 65536
    # Method: the effort profile of requests that name none. The official kit names none.
    default_effort: str = 'high'
    # TensorRT-LLM capacity
    branch_slots: int = 32
    prefill_tokens: int = 2048
    cache_fraction: float = 0.7
    # Images (the `images` field): loads Gemma 4's vision tower.
    vision: bool = True
    image_soft_tokens: int = SOFT_TOKENS  # Gemma 4 vision tokens per image: 70, 140, 280 (checkpoint default), 560 or 1120
    vision_encoder_batch: int = 2     # images per vision-tower call (bounds encoder memory; same embeddings)
    # TensorRT-LLM's host KV tier, in GB of pinned host RAM: 0 (default, no tier) or at least 0.5. Never TensorRT-LLM's
    # auto sizing: under a container's default RLIMIT_MEMLOCK (64 KB) it builds a one-block tier on which the V2 KV
    # manager can stop making progress (docs/tensorrt-llm-patch.md). This service never used a tier in testing (no suspensions).
    host_kv_cache_gb: float = 0
    # NVFP4 GEMM backends TensorRT-LLM may pick from, comma-separated; empty keeps its default.
    # Blackwell (SM100/SM120) needs nothing here. 'marlin' runs the weights on Hopper/Ada with BF16
    # activations: a functional check only, not the evaluated numerics or speed.
    nvfp4_backends: str = ''
    # Admission and deadlines
    max_requests: int = 4
    max_queue: int = 16
    queue_timeout: float = 30
    request_timeout: float = 300  # per request, inference only; the longest index requests take ~10-20 s
    drain_timeout: float = 10
    retry_after: int = 1
    max_body_bytes: int = 64 * 1024 * 1024
    # HTTP
    host: str = '127.0.0.1'
    port: int = 30104
    keep_alive: float = 75
    api_key: str = ''
    access_log: bool = True
    trace: bool = True  # POST /v1/trace: answers plus every branch's prompt, cache reuse, timing and readout

    def __post_init__(self):
        backends = [b for b in self.nvfp4_backends.split(',') if b]
        try:
            profile(self.default_effort)
            effort_ok = True
        except ValueError:
            effort_ok = False
        checks = {
            f'checkpoint must be one of {", ".join(CHECKPOINTS)}': self.checkpoint in CHECKPOINTS,
            'default_effort must be low, medium or high': effort_ok,
            'capacity settings must be positive': min(self.max_requests, self.branch_slots, self.prefill_tokens,
                                                      self.max_body_bytes) >= 1 and self.max_queue >= 0,
            'timeouts must be positive': min(self.queue_timeout, self.request_timeout, self.keep_alive) > 0,
            'context_length must be at least 2': self.context_length >= 2,
            'host_kv_cache_gb must be 0 (no host KV tier) or at least 0.5': self.host_kv_cache_gb == 0 or self.host_kv_cache_gb >= 0.5,
            'image_soft_tokens must be one of 70, 140, 280, 560, 1120': self.image_soft_tokens in (70, 140, 280, 560, 1120),
            'vision_encoder_batch must be at least 1': self.vision_encoder_batch >= 1,
            'cache_fraction must be in (0, 1)': 0 < self.cache_fraction < 1,
            'drain_timeout >= 0 and retry_after >= 1': self.drain_timeout >= 0 and self.retry_after >= 1,
            f'nvfp4_backends must be a comma-separated subset of {",".join(NVFP4_BACKENDS)}':
                all(b in NVFP4_BACKENDS for b in backends),
        }
        failed = [message for message, ok in checks.items() if not ok]
        if failed:
            raise ValueError('Invalid settings: ' + '; '.join(failed))

    @property
    def spec(self):
        return CHECKPOINTS[self.checkpoint]

    @property
    def served_name(self):
        return self.model_name or self.spec.model_name

    @classmethod
    def from_env(cls, environ=os.environ):
        values = {}
        for field in fields(cls):
            raw = environ.get('MORJ_' + field.name.upper())
            if raw is None:
                continue
            if field.type is bool:
                if raw.strip().lower() not in TRUE + FALSE:
                    raise ValueError(f'MORJ_{field.name.upper()} must be a boolean, got {raw!r}')
                values[field.name] = raw.strip().lower() in TRUE
            else:
                values[field.name] = field.type(raw)
        return cls(**values)


def differences(settings):
    """Settings that differ from the evaluated run's, as {name: [value, evaluated value]}; empty when none do."""
    return {k: [getattr(settings, k), v] for k, v in EVALUATED.items() if getattr(settings, k) != v}


def verify_checkpoint(settings):
    """The checkpoint files must match the pinned revision."""
    expected = settings.spec.identity()
    for name, digest in expected['sha256'].items():
        if hashlib.sha256((Path(settings.model_path) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} differs from the pinned {expected["model"]} checkpoint')

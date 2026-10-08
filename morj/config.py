"""Service settings, read from MORJ_* environment variables, and release checks.

The decision method is fixed per effort profile in `method.py`; nothing here changes an answer
except `default_effort`, the profile used when a request names none.
"""
from dataclasses import dataclass, fields
import hashlib
import os
from pathlib import Path

from .checkpoint import CHECKPOINTS
from .method import profile

# The runtime capacity every profile was evaluated with. Startup refuses anything else.
PROFILE = {'context_length': 65536, 'branch_slots': 32, 'prefill_tokens': 2048, 'cache_fraction': 0.8,
           'shared_budget': True, 'seed_min_shared': 256, 'cache_avg_seq_len': 4096}
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
    cache_fraction: float = 0.8
    require_graphs: bool = True
    shared_budget: bool = True        # count a request's shared prompt prefix once in KV admission (scheduling only)
    seed_min_shared: int = 256        # >0: run branches in one wave when they share fewer tokens than this (scheduling only)
    cache_avg_seq_len: int = 4096     # typical sequence TensorRT-LLM sizes the sliding-window vs global KV pools for
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
            'default_effort must be an effort or profile name (low, medium, high, high@1, ...)': effort_ok,
            'capacity settings must be positive': min(self.max_requests, self.branch_slots, self.prefill_tokens,
                                                      self.max_body_bytes) >= 1 and self.max_queue >= 0,
            'timeouts must be positive': min(self.queue_timeout, self.request_timeout, self.keep_alive) > 0,
            'context_length must be at least 2': self.context_length >= 2,
            'cache_avg_seq_len must be positive': self.cache_avg_seq_len >= 1,
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


def verify_release(settings):
    """The runtime profile and the checkpoint must match the evaluated ones."""
    drift = {k: (getattr(settings, k), v) for k, v in PROFILE.items() if getattr(settings, k) != v}
    if drift:
        raise ValueError(f'Configuration differs from the evaluated profile: {drift}')
    expected = settings.spec.identity()
    for name, digest in expected['sha256'].items():
        if hashlib.sha256((Path(settings.model_path) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} differs from the pinned {expected["model"]} checkpoint')

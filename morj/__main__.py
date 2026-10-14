"""Serve one checkpoint with every effort profile from one GPU-owning process."""
from dataclasses import replace
import os
from pathlib import Path

import uvicorn

from . import checkpoint
from .config import Settings, differences, verify_checkpoint
from .method import profile
from .server import create_app

if __name__ == '__main__':
    settings = Settings.from_env()
    if settings.engine == 'lean':  # before torch's CUDA allocator starts: the lean engine's tree buffers vary per pass
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    torch.set_num_threads(settings.cpu_threads)
    if not torch.cuda.is_available():
        raise SystemExit('morj: no GPU is visible. Install the NVIDIA Container Toolkit and pick a GPU, '
                         'for example GPU=0 docker compose up')
    spec = settings.spec
    settings = replace(settings, model_path=checkpoint.resolve(spec, settings.model_path))
    verify_checkpoint(settings)
    if differences(settings):
        checkpoint.say(f'settings differ from the evaluated run (value, evaluated): {differences(settings)}')
    major, minor = torch.cuda.get_device_capability()
    if spec.needs_fp4 and major < 10 and 'marlin' not in settings.nvfp4_backends:
        raise SystemExit(f'morj: this GPU (SM{major}{minor}) has no FP4 tensor cores for the NVFP4 '
                         'checkpoint. Use a Blackwell GPU (e.g. RTX PRO 6000, B200), MORJ_CHECKPOINT=bf16, or '
                         'MORJ_NVFP4_BACKENDS=marlin for a functional check with BF16 activations.')
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'morj'
    if not settings.tokenizer_path:
        settings = replace(settings, tokenizer_path=checkpoint.tokenizer_overlay(
            settings.model_path, cache / 'tokenizer', spec.identity()['tokenizer']['sha256']))
    if settings.vision:
        if settings.checkpoint != 'bf16':
            raise SystemExit('morj: images are validated on the BF16 checkpoint only; '
                             'use MORJ_CHECKPOINT=bf16 or MORJ_VISION=false')
        # Fingerprints were checked above on the checkpoint itself; the overlay only changes the image budget.
        settings = replace(settings, model_path=checkpoint.image_budget_overlay(
            settings.model_path, cache / 'checkpoint', settings.image_soft_tokens))
    loading = f' and the vision tower ({settings.image_soft_tokens} soft tokens per image)' if settings.vision else ''
    checkpoint.say(f'serving {spec.repo} ({spec.precision}) from {settings.model_path}; default effort '
                   f'{profile(settings.default_effort).name}. Loading weights, sizing the KV cache{loading} '
                   'takes a few minutes on a first start')
    # Idle keep-alive outlives common client and proxy pools, so pooled clients
    # never race a server-side close. Shutdown waits for admitted requests.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1,
                access_log=False, log_level='warning', timeout_keep_alive=int(settings.keep_alive),
                timeout_graceful_shutdown=int(settings.queue_timeout + settings.request_timeout + 10))

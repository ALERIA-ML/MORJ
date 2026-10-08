"""Process-local TensorRT-LLM patches, valid only for the pinned runtime.

`install` refuses to run unless every TRT-LLM source file we patch is
byte-identical to the validated 1.3.0rc29 release (`runtime_identity.json`).
Each patch can be restored, so tests and shutdown leave the library intact.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

IDENTITY = json.loads(Path(__file__).with_name('runtime_identity.json').read_text())


def fingerprint(package_dir):
    """SHA-256 of the patched TRT-LLM sources under an installed `tensorrt_llm` directory."""
    return {name: hashlib.sha256((Path(package_dir) / name).read_bytes()).hexdigest()
            for name in IDENTITY['sha256']}


def verify_runtime():
    """Check the installed TRT-LLM without importing it, so it also runs at image build time."""
    from importlib.metadata import version
    from importlib.util import find_spec
    package = Path(find_spec('tensorrt_llm').submodule_search_locations[0])
    if version('tensorrt_llm') != IDENTITY['version'] or fingerprint(package) != IDENTITY['sha256']:
        raise RuntimeError(f'TensorRT-LLM differs from the validated {IDENTITY["version"]} sources; '
                           'revalidate every patch before upgrading')
    return IDENTITY


@dataclass
class Patches:
    graph_metrics: dict
    restores: list = field(default_factory=list)

    def restore(self):
        while self.restores:
            self.restores.pop()()


def install():
    """Verify the runtime, then install every patch. Returns a restorable handle."""
    from . import attention, graph_metrics, scheduler
    verify_runtime()
    metrics, restore_metrics = graph_metrics.install()
    patches = Patches(metrics, [restore_metrics, scheduler.install(), attention.install()])
    print(json.dumps({'event': 'trtllm_patches_installed', 'version': IDENTITY['version'],
                      'patches': ['scheduler', 'attention', 'graph_metrics']}), flush=True)
    return patches


@contextmanager
def observe_kv_capacity():
    """Record the context length and free blocks per attention window of each KV cache manager TRT-LLM builds."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    original, capacities, pools = KVCacheManagerV2.__init__, [], []

    def observed(manager, *args, **kwargs):
        original(manager, *args, **kwargs)
        capacities.append(int(manager.max_seq_len))
        stats = manager.get_kv_cache_stats()
        pools.append(({int(w): int(n) for w, n in stats.num_free_blocks_per_window_size.items()}, int(stats.tokens_per_block)))
        print(json.dumps({'event': 'kv_pool', 'context_length': capacities[-1],
                          'free_blocks_per_window': stats.num_free_blocks_per_window_size,
                          'tokens_per_block': stats.tokens_per_block, 'max_blocks': stats.max_num_blocks}), flush=True)

    KVCacheManagerV2.__init__ = observed
    try:
        yield capacities, pools
    finally:
        KVCacheManagerV2.__init__ = original

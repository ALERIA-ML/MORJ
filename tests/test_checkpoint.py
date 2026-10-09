"""Where the served checkpoint comes from: local directory, /model mount, or the Hugging Face cache."""
import json

import httpx
import pytest

from morj import checkpoint

SPEC = checkpoint.CHECKPOINTS['nvfp4']

hub = pytest.importorskip('huggingface_hub')


def make(path, shards=('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'), missing=()):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': dict(enumerate(shards))}))
    for shard in shards:
        if shard not in missing:
            (path / shard).write_bytes(b'weights')
    return path


@pytest.fixture
def hub_calls(tmp_path, monkeypatch):
    """Replaces snapshot_download; each call runs the next step of `script` (a path, a callable or an error)."""
    monkeypatch.setattr(checkpoint, 'MOUNT', tmp_path / 'no-mount')
    calls, script = [], []

    def snapshot_download(repo, revision, allow_patterns, local_files_only=False):
        assert (repo, revision) == (SPEC.repo, SPEC.revision)
        calls.append('cache' if local_files_only else 'download')
        outcome = script.pop(0)
        outcome = outcome() if callable(outcome) else outcome
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)
    monkeypatch.setattr(hub, 'snapshot_download', snapshot_download)
    return calls, script


def test_explicit_path_must_be_a_complete_checkpoint(tmp_path):
    with pytest.raises(SystemExit, match='not a complete checkpoint'):
        checkpoint.resolve(SPEC, str(make(tmp_path / 'partial', missing=('model-00002-of-00002.safetensors',))))
    assert checkpoint.resolve(SPEC, str(make(tmp_path / 'full'))) == str(tmp_path / 'full')


def test_a_mounted_checkpoint_is_served_without_touching_the_hub(tmp_path, monkeypatch, hub_calls):
    monkeypatch.setattr(checkpoint, 'MOUNT', make(tmp_path / 'mount'))
    assert checkpoint.resolve(SPEC) == str(tmp_path / 'mount') and hub_calls[0] == []


def test_a_complete_cached_revision_is_reused(tmp_path, hub_calls):
    calls, script = hub_calls
    script.append(make(tmp_path / 'snapshot'))
    assert checkpoint.resolve(SPEC) == str(tmp_path / 'snapshot') and calls == ['cache']


def test_an_incomplete_cache_resumes_the_download(tmp_path, hub_calls):
    calls, script = hub_calls
    partial = make(tmp_path / 'snapshot', missing=('model-00002-of-00002.safetensors',))
    script += [partial, lambda: make(tmp_path / 'snapshot')]
    assert checkpoint.resolve(SPEC) == str(partial) and calls == ['cache', 'download']


def test_first_start_downloads_and_a_refusal_explains_access(tmp_path, hub_calls):
    calls, script = hub_calls
    script += [hub.errors.LocalEntryNotFoundError('not cached'), lambda: make(tmp_path / 'snapshot')]
    assert checkpoint.resolve(SPEC) == str(tmp_path / 'snapshot') and calls == ['cache', 'download']
    denied = httpx.Response(403, request=httpx.Request('GET', 'https://huggingface.co'))
    script += [hub.errors.LocalEntryNotFoundError('not cached'), hub.errors.GatedRepoError('gated', response=denied)]
    with pytest.raises(SystemExit, match='refused access'):
        checkpoint.resolve(SPEC)


def test_tokenizer_overlay_uses_the_canonical_files_and_checks_hashes(tmp_path):
    import hashlib
    expected = SPEC.identity()['tokenizer']['sha256']
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'tokenizer.json').write_text('{"not": "the gemma tokenizer"}')
    (model / 'chat_template.jinja').write_text('older template, never used')
    with pytest.raises(ValueError, match='tokenizer.json'):
        checkpoint.tokenizer_overlay(model, tmp_path / 'overlay', expected)
    fake = {**expected, 'tokenizer.json': hashlib.sha256((model / 'tokenizer.json').read_bytes()).hexdigest()}
    overlay = checkpoint.tokenizer_overlay(model, tmp_path / 'overlay', fake)
    for name in ('tokenizer_config.json', 'chat_template.jinja'):
        assert hashlib.sha256((tmp_path / 'overlay' / name).read_bytes()).hexdigest() == expected[name]
    assert (tmp_path / 'overlay' / 'chat_template.jinja').read_text() != 'older template, never used'
    assert overlay == str(tmp_path / 'overlay')


def test_both_checkpoints_share_the_canonical_tokenizer_files():
    bf16, nvfp4 = (checkpoint.CHECKPOINTS[name].identity() for name in ('bf16', 'nvfp4'))
    assert bf16['tokenizer'] == nvfp4['tokenizer']
    assert bf16['sha256']['tokenizer.json'] == nvfp4['sha256']['tokenizer.json']
    for spec in checkpoint.CHECKPOINTS.values():
        identity = spec.identity()
        assert (identity['model'], identity['revision']) == (spec.repo, spec.revision)


def test_image_budget_overlay_links_the_weights_and_edits_only_the_processor_config(tmp_path):
    source = make(tmp_path / 'ckpt')
    processor = {'image_seq_length': 280, 'image_processor': {'max_soft_tokens': 280, 'image_seq_length': 280, 'x': 1}}
    (source / 'processor_config.json').write_text(json.dumps(processor))
    assert checkpoint.image_budget_overlay(source, tmp_path / 'cache', 280) == str(source)  # budget already right
    target = checkpoint.Path(checkpoint.image_budget_overlay(source, tmp_path / 'cache', 1120))
    assert target != source and (target / 'model-00001-of-00002.safetensors').is_symlink()
    assert (target / 'model-00001-of-00002.safetensors').read_bytes() == b'weights'
    edited = json.loads((target / 'processor_config.json').read_text())
    assert edited == {'image_seq_length': 1120, 'image_processor': {'max_soft_tokens': 1120, 'image_seq_length': 1120, 'x': 1}}
    assert json.loads((source / 'processor_config.json').read_text()) == processor  # source untouched
    assert checkpoint.image_budget_overlay(source, tmp_path / 'cache', 1120) == str(target)  # rebuilding is idempotent

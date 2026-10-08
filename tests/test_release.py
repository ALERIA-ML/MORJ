"""Settings, release checks and the request contract; no GPU."""
import pytest
from pydantic import ValidationError

from morj.checkpoint import CHECKPOINTS
from morj.config import Settings, verify_release
from morj.schema import EvaluationRequest


def write_checkpoint(directory, content=None, checkpoint='bf16'):
    for name in CHECKPOINTS[checkpoint].identity()['sha256']:
        (directory / name).write_text(content or '{}')


def test_settings_from_environment():
    settings = Settings.from_env({'MORJ_ACCESS_LOG': 'false', 'MORJ_PORT': '8000', 'MORJ_TRACE': 'no',
                                  'MORJ_API_KEY': 'secret'})
    assert (settings.access_log, settings.port, settings.trace, settings.api_key) == (False, 8000, False, 'secret')
    with pytest.raises(ValueError, match='boolean'):
        Settings.from_env({'MORJ_REQUIRE_GRAPHS': 'maybe'})
    with pytest.raises(ValueError, match='cache_fraction'):
        Settings(cache_fraction=1)


def test_release_rejects_profile_drift_and_other_checkpoints(tmp_path):
    for name, value in dict(context_length=32768, branch_slots=8, prefill_tokens=4096, cache_fraction=.7,
                            shared_budget=False, seed_min_shared=0, cache_avg_seq_len=16384).items():
        with pytest.raises(ValueError, match='evaluated profile'):
            verify_release(Settings(**{name: value}))
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='differs from the pinned'):
        verify_release(Settings(model_path=str(tmp_path)))


def test_request_contract():
    EvaluationRequest(state='x', questions={str(i): {'type': 'noul'} for i in range(300)})  # no question cap
    for bad in ({'questions': {}}, {'questions': {'s': {'type': 'score', 'criteria': ['low', None]}}},
                {'questions': {'q': {'type': 'noul'}}, 'images': ['data:image/png;base64,abc']}):
        with pytest.raises(ValidationError):
            EvaluationRequest(**{'state': 'x', 'questions': {'q': {'type': 'noul'}}, **bad})


def test_nvfp4_backends_setting():
    assert Settings.from_env({'MORJ_NVFP4_BACKENDS': 'marlin'}).nvfp4_backends == 'marlin'
    assert Settings(nvfp4_backends='cutlass,cublaslt').nvfp4_backends == 'cutlass,cublaslt'
    with pytest.raises(ValueError, match='nvfp4_backends'):
        Settings(nvfp4_backends='cutlass,fp16')


def test_checkpoint_and_default_effort_settings():
    assert Settings().spec.repo == 'google/gemma-4-31B-it' and Settings().served_name == 'gemma-4-31b-trt'
    nvfp4 = Settings.from_env({'MORJ_CHECKPOINT': 'nvfp4', 'MORJ_DEFAULT_EFFORT': 'medium'})
    assert (nvfp4.served_name, nvfp4.default_effort) == ('gemma-4-31b-nvfp4-trt', 'medium')
    assert Settings(model_name='custom').served_name == 'custom'
    with pytest.raises(ValueError, match='checkpoint'):
        Settings(checkpoint='fp8')
    with pytest.raises(ValueError, match='default_effort'):
        Settings(default_effort='maximal')


def test_shipped_tokenizer_files_match_the_pinned_hashes():
    import hashlib
    from morj import checkpoint
    for spec in CHECKPOINTS.values():
        pinned = spec.identity()
        for name in ('tokenizer_config.json', 'chat_template.jinja'):
            shipped = hashlib.sha256((checkpoint.TOKENIZER_FILES / name).read_bytes()).hexdigest()
            assert shipped == pinned['tokenizer']['sha256'][name]
    assert 'hf_quant_config.json' in CHECKPOINTS['nvfp4'].identity()['sha256']

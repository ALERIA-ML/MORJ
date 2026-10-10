"""Settings, release checks and the request contract; no GPU."""
import pytest
from pydantic import ValidationError

from morj.checkpoint import CHECKPOINTS
from morj.config import Settings, differences, verify_checkpoint
from morj.schema import EvaluationRequest


def write_checkpoint(directory, content=None, checkpoint='bf16'):
    for name in CHECKPOINTS[checkpoint].identity()['sha256']:
        (directory / name).write_text(content or '{}')


def test_settings_from_environment():
    settings = Settings.from_env({'MORJ_ACCESS_LOG': 'false', 'MORJ_PORT': '8000', 'MORJ_TRACE': 'no',
                                  'MORJ_API_KEY': 'secret'})
    assert (settings.access_log, settings.port, settings.trace, settings.api_key) == (False, 8000, False, 'secret')
    with pytest.raises(ValueError, match='boolean'):
        Settings.from_env({'MORJ_VISION': 'maybe'})
    with pytest.raises(ValueError, match='cache_fraction'):
        Settings(cache_fraction=1)


def test_other_settings_are_reported_and_other_checkpoints_refused(tmp_path):
    assert differences(Settings()) == {}
    changed = dict(context_length=32768, branch_slots=8, prefill_tokens=4096, cache_fraction=.8, vision=False)
    assert differences(Settings(**changed)) == {'vision': [False, True], 'context_length': [32768, 65536],
                                                'branch_slots': [8, 32], 'prefill_tokens': [4096, 2048],
                                                'cache_fraction': [.8, .7]}
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match='differs from the pinned'):
        verify_checkpoint(Settings(model_path=str(tmp_path)))


def test_request_contract():
    EvaluationRequest(state='x', questions={str(i): {'type': 'noul'} for i in range(300)})  # no question cap
    for bad in ({'questions': {}}, {'questions': {'s': {'type': 'score', 'criteria': ['low', None]}}},
                {'images': ['https://example.com/a.png']}, {'images': ['data:image/bmp;base64,AAAA']},
                {'images': ['data:image/png;base64,AAAA'] * 9}):
        with pytest.raises(ValidationError):
            EvaluationRequest(**{'state': 'x', 'questions': {'q': {'type': 'noul'}}, **bad})


def test_request_accepts_up_to_eight_data_url_images():
    request = EvaluationRequest(state='x', questions={'q': {'type': 'noul'}}, images=['data:image/png;base64,AAAA'] * 8)
    assert len(request.images) == 8


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


def test_host_kv_tier_is_explicit_and_never_degenerate():
    assert Settings().host_kv_cache_gb == 0                          # no host tier by default
    assert Settings(host_kv_cache_gb=16).host_kv_cache_gb == 16
    assert Settings(host_kv_cache_gb=0.5).host_kv_cache_gb == 0.5
    for bad in (0.1, -1):  # a few blocks (the stalling one-block case) or TensorRT-LLM's auto sizing
        with pytest.raises(ValueError, match='host_kv_cache_gb'):
            Settings(host_kv_cache_gb=bad)

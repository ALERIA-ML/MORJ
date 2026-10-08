from types import SimpleNamespace

import pytest

from morj.patches import IDENTITY


def test_attention_padding_preserves_live_rows_and_rejects_other_mismatches():
    torch = pytest.importorskip('torch')
    from morj.patches.attention import pad_attention_output
    output, hidden = torch.randn(2, 16), torch.randn(32, 16)
    padded = pad_attention_output(output, hidden, SimpleNamespace(padded_num_tokens=32, num_tokens=2))
    assert padded.shape == hidden.shape and torch.count_nonzero(padded[2:]) == 0
    torch.testing.assert_close(padded[:2], output)
    assert pad_attention_output(output, output, SimpleNamespace()) is output
    with pytest.raises(RuntimeError, match='shape mismatch'):
        pad_attention_output(output, hidden, SimpleNamespace(padded_num_tokens=None, num_tokens=2))


def test_runtime_identity_pins_every_patched_source():
    assert IDENTITY['version'] == '1.3.0rc29' and len(IDENTITY['sha256']) == 6

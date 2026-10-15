import pytest

from aleria_judge.patches import IDENTITY


def test_runtime_identity_pins_every_patched_source():
    assert IDENTITY['version'] == '1.3.0rc29' and len(IDENTITY['sha256']) == 3


def test_vision_slices_give_the_same_embeddings_as_one_call():
    torch = pytest.importorskip('torch')
    from aleria_judge.patches.vision import encode_in_slices, slices
    assert slices(5, 2) == [(0, 2), (2, 4), (4, 5)] and slices(2, 2) == [(0, 2)]
    calls = []

    def tower(model, pixel_values, image_position_ids=None, image_seq_lens=None):
        """Per-image encoder: each image's first `seq_len` rows, scaled by its position ids, flattened in order."""
        calls.append(pixel_values.shape[0])
        rows = [pixel_values[i, :n] * image_position_ids[i, :n, :1] for i, n in enumerate(image_seq_lens)]
        return torch.cat(rows)

    pixels, positions = torch.randn(5, 7, 3), torch.randn(5, 7, 2)
    lens = [7, 3, 5, 1, 6]
    whole = tower(None, pixels, positions, lens)
    calls.clear()
    sliced = encode_in_slices(tower, None, pixels, positions, lens, batch=2)
    torch.testing.assert_close(sliced, whole)
    assert calls == [2, 2, 1]


def test_suspensions_are_counted_and_warned_about_without_a_host_tier():
    from aleria_judge.patches import scheduler
    scheduler.SUSPENSIONS['count'] = 0
    first = scheduler.note_suspension(host_tier=False)
    assert first['count'] == 1 and 'warning' in first and first['host_kv_tier'] is False
    assert 'warning' not in scheduler.note_suspension(host_tier=True)
    logged = [scheduler.note_suspension(host_tier=False) for _ in range(2000)]
    assert sum(r is not None for r in logged) == 8 + 2  # counts 3..10, then 1,000 and 2,000
    assert scheduler.SUSPENSIONS['count'] == 2002
    scheduler.SUSPENSIONS['count'] = 0

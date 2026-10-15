"""Run Gemma 4's vision tower over at most `batch` images per call.

TensorRT-LLM concatenates every image of every context request in an engine step and encodes them
in one vision-tower call. At 1,120 soft tokens an image is about 10,000 patches, so eight images
need about 80,000 patches of activations at once: on an H100 80 GB with the KV cache at 70% that
ran out of memory inside the vision attention and stopped the executor. Gemma 4 opts into neither
TensorRT-LLM's encoder cache nor its encoder item scheduling, which would bound this.

The vision tower attends within each image only (per-image `cu_seqlens`), and its output is the
images' valid tokens in order, so encoding consecutive slices and concatenating the results gives
the same embeddings with a peak bounded by `batch` images.
"""
import hashlib
import inspect

GET_IMAGE_FEATURES_SHA256 = '311e139db4b0f3b0de3b5cf68a4f0bfca00a68aabf72bc74721bc3319d85ab5d'


def slices(count, batch):
    """[start, stop) ranges of at most `batch` items covering `count` items in order."""
    return [(start, min(start + batch, count)) for start in range(0, count, batch)]


def encode_in_slices(original, model, pixel_values, image_position_ids=None, image_seq_lens=None, *, batch):
    import torch
    count = pixel_values.shape[0]
    if count <= batch:
        return original(model, pixel_values=pixel_values, image_position_ids=image_position_ids,
                        image_seq_lens=image_seq_lens)
    parts = [original(model, pixel_values=pixel_values[start:stop],
                      image_position_ids=None if image_position_ids is None else image_position_ids[start:stop],
                      image_seq_lens=None if image_seq_lens is None else list(image_seq_lens[start:stop]))
             for start, stop in slices(count, batch)]
    return torch.cat(parts).contiguous()


def install(batch):
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration as Model
    original = Model._get_image_features
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != GET_IMAGE_FEATURES_SHA256:
        raise RuntimeError('Gemma4ForConditionalGeneration._get_image_features changed; revalidate the vision patch')

    def get_image_features(model, pixel_values, image_position_ids=None, image_seq_lens=None):
        return encode_in_slices(original, model, pixel_values, image_position_ids, image_seq_lens, batch=batch)

    Model._get_image_features = get_image_features
    return lambda: setattr(Model, '_get_image_features', original)

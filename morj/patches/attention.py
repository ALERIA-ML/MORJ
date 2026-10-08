"""Pad Gemma 4 attention output to the CUDA-graph bucket during prefill warmup.

With breakable prefill CUDA graphs, eager attention returns only the live rows
while the residual stream is padded to the capture bucket, so the add fails.
The tail is zero-padded; live rows are never touched.
"""
import hashlib
import inspect

FORWARD_SHA256 = '11ef92815d07e4b6e1e9be0278579abdfed72066058a5eee5fa9b2f30744bed5'


def pad_attention_output(output, hidden_states, metadata):
    if output.shape[0] == hidden_states.shape[0]:
        return output
    if (getattr(metadata, 'padded_num_tokens', None) != hidden_states.shape[0]
            or output.shape[0] != metadata.num_tokens or output.shape[0] > hidden_states.shape[0]):
        raise RuntimeError('Unexpected Gemma attention shape mismatch outside padded prefill')
    import torch.nn.functional as F
    return F.pad(output, (0, 0, 0, hidden_states.shape[0] - output.shape[0]))


def install():
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4Attention
    original = Gemma4Attention.forward
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != FORWARD_SHA256:
        raise RuntimeError('Gemma4Attention.forward changed; revalidate the padding patch')

    def forward(layer, position_ids, hidden_states, attn_metadata, **kwargs):
        output = original(layer, position_ids=position_ids, hidden_states=hidden_states,
                          attn_metadata=attn_metadata, **kwargs)
        return pad_attention_output(output, hidden_states, attn_metadata)

    Gemma4Attention.forward = forward
    return lambda: setattr(Gemma4Attention, 'forward', original)

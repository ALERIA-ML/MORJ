# TensorRT-LLM patch: Gemma 4 attention under prefill CUDA graphs

MORJ applies one patch to TensorRT-LLM 1.3.0rc29 that the service cannot start without:
[`morj/patches/attention.py`](../morj/patches/attention.py) (unchanged since release 1.0). While the runtime
warms up a prefill CUDA graph, it pads Gemma 4's attention output with zero rows. This note explains the bug
it works around and why the padding cannot change an answer.

**The prefill graphs themselves may not be worth keeping.** TensorRT-LLM refuses breakable prefill graphs
for multimodal models, so our vision-language build (in development, not in this repository) runs without
them, and on 650 heavy text requests it
turned out faster than this service on the same GPU (median 2,174 against 2,389 ms). A controlled A/B on the
text service then isolated the graphs: 1,200 stratified Decision Index requests on three H100 PCIe GPUs, with
graphs turned on then off, off then on, and on twice as a noise floor. Without prefill graphs, p80 latency
fell 21% and mean and p95 latency 10%; the median did not move. Requests of 1k–16k characters and requests
with 10 or more questions got 13–26% faster at the median; only short single questions (under 1k
characters) got slower, by about 6 ms. 99.88% of answers were identical: the graphs change the numerics
slightly, while a repeat with graphs on changed none. The evaluated configuration and the service default
still use prefill graphs, so the patch below is still required. Turning the graphs off would make it
unnecessary, because the bug only occurs while a graph bucket warms up, but that has not been validated on
the RTX PRO 6000 or in a full run.

The other files in `patches/` fix no bugs: `scheduler.py` stops an engine that makes no progress,
`graph_metrics.py` counts graph captures and replays, and `__init__.py` refuses to run on TensorRT-LLM sources
other than the ones these patches were checked against. Their docstrings explain them.

TensorRT-LLM links below point at the `v1.3.0rc29` tag. The SHA-256 of `modeling_gemma4.py` and
`breakable_cuda_graph_runner.py` at that tag match
[`runtime_identity.json`](../morj/patches/runtime_identity.json), so this is
the code the service runs.

## Prefill graphs are on

The service only scores: each branch is one prompt prefill plus one output token whose logits are read.
[`engine.py`](../morj/engine.py#L87-L88) therefore turns decode graphs off
(`cuda_graph_config=None`) and prefill graphs on, with TensorRT-LLM's breakable backend
(`prefill_cuda_graph_backend='breakable'`), captured at powers of two from 32 tokens up to `max_num_tokens`
(2,048 by default). A prefill batch is padded up to the next captured size (its bucket) and replayed as a
graph. Attention runs outside the graph: the backend ends the captured segment before each attention call,
runs attention eagerly and resumes with the next segment.

The service requires these graphs. Startup fails if none was captured
([`engine.py:97`](../morj/engine.py#L97)), and with `require_graphs` (on by
default) the service is not ready until its readiness request replays one
([`server.py:128`](../morj/server.py#L128)).

## The bug

Before capturing a bucket, the graph runner runs the whole forward pass eagerly twice at the bucket size
([`warmup`](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph_runner.py#L57-L67)).
The batch is padded: `attn_metadata.padded_num_tokens` is the bucket size and `attn_metadata.num_tokens` the
real token count. Two pieces of code then disagree on the number of rows:

- **The decoder layer keeps the padding.** Gemma 4's layer saves `residual = hidden_states` (bucket rows) and
  adds the attention output back to it
  ([`modeling_gemma4.py` L932-L968](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/models/modeling_gemma4.py#L932-L968)).
  The model drops the padding only after the last layer
  ([L1527-L1528](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/models/modeling_gemma4.py#L1527-L1528)).
- **Eager attention drops it.** The shared `Attention` base class slices q, k and v to `num_tokens`
  ([`attention.py` L921-L927](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/attention/attention.py#L921-L927)),
  so it returns only the live rows.

Capture and replay take a different attention path. Inside the breakable-graph context, attention allocates its
output from the unsliced q, with the bucket's row count
([`attention.py` L1063-L1071](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/attention/attention.py#L1063-L1071),
[`interface.py` L1119-L1125](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/attention/backends/interface.py#L1119-L1125)),
and writes the live rows into it in place, so the shapes match. The runner enters that context only for
[capture](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph_runner.py#L118-L129)
and [replay](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc29/tensorrt_llm/_torch/pyexecutor/breakable_cuda_graph_runner.py#L196-L200),
not for warmup. Warmup therefore takes the eager path, and the residual add receives `[bucket, hidden]` plus
`[live, hidden]` and fails on the shape mismatch. No graph is captured, and the service does not start.

The slicing is in the shared base class, not in Gemma code. We have only hit, patched and tested it on Gemma 4.

## The fix

The patch wraps `Gemma4Attention.forward`. When the attention output has fewer rows than the residual, it
appends zero rows until they match:

```python
if output.shape[0] == hidden_states.shape[0]:
    return output
if (getattr(metadata, 'padded_num_tokens', None) != hidden_states.shape[0]
        or output.shape[0] != metadata.num_tokens or output.shape[0] > hidden_states.shape[0]):
    raise RuntimeError('Unexpected Gemma attention shape mismatch outside padded prefill')
return F.pad(output, (0, 0, 0, hidden_states.shape[0] - output.shape[0]))
```

## Why it cannot change an answer

- **It pads one exact case.** The residual must have `padded_num_tokens` rows and attention must have returned
  exactly `num_tokens`. Any other mismatch raises instead of being padded over.
- **Live rows are untouched.** Zeros go after them; [`tests/test_patches.py`](../tests/test_patches.py)
  checks that the live rows are unchanged, the tail is zero and other mismatches raise.
- **Padded rows never reach a logit.** They belong to padding tokens, which the model slices off before the LM
  head.
- **It only acts during warmup**, whose outputs the runner discards. During capture, and in any eager pass without
  padding, the shapes already match and the wrapper returns the output unchanged. On replay the decoder body is
  replaced by the graph, so the wrapper does not run at all.
- **It cannot silently apply to other code.** It refuses to install if the source of `Gemma4Attention.forward`
  changed, and the service refuses to run unless the patched TensorRT-LLM files match `runtime_identity.json`.
  That check runs at image build ([`Dockerfile`](../Dockerfile#L9)) and again at
  startup.

## When it can go

As of 2026-10-07, upstream TensorRT-LLM `main` still has the bug. After an upgrade, re-pin
`runtime_identity.json` and start the service without the patch: if startup succeeds and `/health` reports
nonzero `prefill_graphs` captures and replays, the patch can go.

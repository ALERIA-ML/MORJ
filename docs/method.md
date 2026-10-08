# Method

MORJ answers a structured question without generating text. Each question becomes a few prompts
(*branches*); the engine prefills each branch once, reads the next-token distribution over the option
labels, and combines the branches into one answer. The default effort, `high@1`, combines four views of the
same question. This page describes them, what each adds, how the engine runs them, and how the method was
chosen.

## A branch

All prompts use the state-first chat layout of [Mapika/decider](https://github.com/Mapika/decider),
vendored under Apache-2.0 in `morj/vendor/`. A prompt contains the rendered state, one question, its options
with single-token labels (`(A)`, `(B)`, …, then two-letter labels up to 255 options) and `Answer: (`. The
service reads the logits of the label tokens at that position and applies a softmax. That is one prefill
per branch and no decoding.

## The four views of `high@1`

1. **Two candidate orders.** Each question is scored with its options as given and rotated by n // 2. Each
   order's distribution is mapped back to the original options and the two are averaged. This cancels much
   of the model's position bias.
2. **Nine-bin rating.** A two-option question gets one more branch: the same question, then *"Rate how
   likely option (A) is the correct answer on a scale from 1 to 9 … Return only the digit."* and
   `Rating: `. The expected digit E maps to `p(A) = clamp(0.01 + (E/10 − 0.1) · 0.98/0.8, 0.01, 0.99)`,
   following [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Truth questions (Noul, or a
   Choice whose options are exactly yes/no) are rated once in the original order. Other two-option
   questions are rated in both orders, and `p = (p_forward + 1 − p_swapped) / 2`. The letter average and
   the rating are averaged. At T = 1, Gemma puts more than 0.99 on over 95% of its yes/no answers, yet only
   87% of those are right. The rating spreads that probability, which matters for Brier scores and for
   ranking by p(yes).
3. **Gated orders.** If a question has three or more options and its two orders pick different options, a
   second pass scores the rotations by n // 4 and 3n // 4, and all orders are averaged (with three options
   the rotation by n // 4 is the original order, so only one is added). On a 10,000-request balanced
   sample of the index, the gate fired on 9.7% of requests (18.8% of those with a question of three or more
   options).
4. **Option-count temperature.** Questions without a rating soften each order's distribution with
   `T(n) = max(0.05, 6.9 − 0.7 ln n)` before averaging. This never changes an order's argmax, but it
   changes how disagreeing orders combine and how confident the answer is. Rated questions keep their
   letters at T = 1. See [Disclosures](#disclosures) for how the curve was fitted.

The response follows the shared API ([api.md](api.md)): Choice returns `choice`, `probabilities` over the
original keys and `confidence = (p_best − 1/n) / (1 − 1/n)`. Noul returns P(true). Score returns the
expected level, the distribution, a legend and a distance-based confidence.

## What each view adds

Full Decision Index 0.2.1 runs with the official reproduction kit and scorer, one sequential HTTP client
per server, every request answered. Each row adds one change to the row above it.

| Configuration | Index | Δ | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| One candidate order (Decider prompt, T = 1) | 56.98 | | 89 | 610 |
| + second order, rotated by n/2 | 57.68 | +0.70 | 180 | 1,008 |
| + nine-bin rating on truth questions, option-count temperature elsewhere | 58.37 | +0.69 | 144 | 847 |
| + rating in both orders on other two-option questions | 58.42 | +0.05 | 151 | 951 |
| **+ gated orders (`high@1`)** | **58.68** | **+0.26** | **178** | **1,249** |

Every run was on one H100 PCIe 80 GB per server; the last one was split into two halves on two H100s.
Latencies come from successive serving revisions, so compare them only roughly. For reference, the public
board (0.2.1 snapshot of 2026-09-28) listed Jev 1.13.0 at 57.91, Surogate Rune 26B-A4B v3 at 57.44 and
upstream Decider chat · Gemma-4-31B at 57.33; upstream Decider served on our H100 with its published
temperature scored 57.32.

Where the gain over one order comes from, in native benchmark metrics:

| Benchmark | One order | `high@1` | Main contributor |
| --- | --- | --- | --- |
| ForecastBench (Brier ↓) | 0.262 | 0.187 | rating: a spread probability instead of 0/1 |
| BRIGHT (nDCG@10) | 0.436 | 0.472 | rating: ranking by p(yes) is no longer tied |
| ToolRet (nDCG@10) | 0.629 | 0.651 | rating |
| GSM8K | 0.825 | 0.867 | orders and gated orders |
| MMLU-Pro | 0.695 | 0.712 | orders and gated orders |
| iSarcasmEval (F1) | 0.511 | 0.547 | orders and rating |
| BPoMP | 0.903 | 0.978 | second order |
| HLE | 0.154 | 0.128 | regression (501 questions) |

### Order averaging on a fine-tuned model

Position bias is not only a stock-model problem. On [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
(a Qwen3.8-27B fine-tune that reads answers from a joint schema head), averaging the original option order
with its rotation by n/2 raised the full Decision Index 0.2.1 from 61.96 (Cloudflare's server, one order) to
62.94; gated orders added 0.07 more (63.01). Only the orders transfer: clef has its own answer head, so the
rating and temperature do not apply. The two-order runs used our TensorRT-LLM port of clef, whose answers
matched clef's PyTorch code with two orders on 812 of 812 questions, so the gain is the second order's. These
were internal runs; the port is not part of this repository.

## Effort profiles

The same service runs cheaper subsets of the views when a request asks for them
([api.md](api.md#effort)):

| `effort` | Views per question | Branches / question | Index 0.2.1 | p50 ms, one question | p50 ms, all requests |
| --- | --- | --- | --- | --- | --- |
| `low@1` | one order, T = 1 | 1.0 | 56.98 | 59 | 92 |
| `medium@1` | two orders; rating on yes/no questions, temperature elsewhere | 2.8 | 58.37 | 109 (1.8x) | 174 (1.9x) |
| `high@1` | the four views above | 3.0 | 58.71 | 135 (2.3x) | 201 (2.2x) |

The extra views cost prefill only, and the branches of each pass reuse the KV cache of the request's shared
prefix (template and state), so three branches per question cost about 2.3 times one at the median.
Latency: BF16 on one H100 PCIe 80 GB, the same 1,000 index requests sent one at a time at each effort (755
of them single-question). `low` and `medium` were timed with the earlier 4-slot scheduling, which is slower
on multi-question requests but gives the same single-question medians, so the one-question column is the
like-for-like comparison: `high@1` adds 76 ms at the median for 1.7 index points. One H100 sustains about
1.25 requests/s on this mix at any concurrency (prefill-bound): add GPUs for throughput.

`high@1` builds the same prompts and answers as release 1.0, which produced the 0.2.1 runs
(`tests/test_golden.py`, which anyone can run). `low@1` and `medium@1` were checked against the earlier,
unpublished serving code of their full runs: token-identical prompts and bit-identical probabilities on the
golden requests (141 and 284 branches).

## Engine

The service uses the TensorRT-LLM 1.3.0rc29 PyTorch backend through its `LLM` API, the same layer
`trtllm-serve` wraps, on one GPU.

- **Shared prefix.** All branches of a request begin with the same template and state. The first branch
  (the *seed*) runs alone and fills the KV cache; the remaining branches then run concurrently in up to 32
  branch slots and reuse that prefix through TensorRT-LLM's block reuse. KV admission counts a request's
  shared prefix once. When the branches share fewer than 256 prompt tokens, waiting for a seed buys
  nothing, so they all run in one wave. A random `cache_salt` per pass keeps requests from ever sharing KV
  state.
- **Prefill only.** Each branch requests one token and reads its generation logits. Breakable prefill CUDA
  graphs are captured at startup for 32–2,048-token chunks, with FlashInfer attention and 2,048-token
  chunked prefill. At startup the service sends itself one probe request; `/health` turns ready only after
  it has replayed a captured graph.
- **KV pools.** Gemma 4 mixes full-attention and sliding-window layers. The pools are sized for a
  4,096-token typical sequence, so long many-option requests (several ~7k-token branches) prefill side by
  side. The cache holds at least one maximum-length 65,536-token branch; startup refuses less.
- **Admission.** Branches are admitted first-come, first-served against a budget of resident context
  tokens, so long branches queue instead of forcing KV preemption. Above 4 running and 16 waiting requests
  the service answers 529 with `Retry-After`. Per-request deadlines cancel native work. A native failure
  drains in-flight requests and exits for the supervisor to restart.
- **Patches.** Three process-local patches live in `morj/patches/`: one pads Gemma 4 attention output to
  the CUDA-graph bucket during warmup (without it the service cannot start; see
  [tensorrt-llm-patch.md](tensorrt-llm-patch.md)), one stops an engine that makes no progress, and one
  counts graph captures and replays. They install only when the patched TensorRT-LLM files match the
  pinned SHA-256 values.

BF16 weights take 57.2 GiB on the GPU (text tower only). The context length (65,536 tokens), 2,048-token
prefill chunks and 0.8 KV cache fraction are fixed: startup refuses other values.

The scheduling above is release 1.2. Release 1.0 ran the same method with 4 branch slots and
without shared-prefix admission; the two gave the same decisions on 99.91% of answers (58.71 against 58.68
on 0.2.1), with requests of 10 or more questions 1.9 to 2.4 times faster at the median.

## Disclosures

- **Evaluated code.** `tests/golden.json` was produced by the exact source of release 1.0.
  `tests/test_golden.py` checks that this code builds identical token branches, gate decisions and answers.
  Before release, the same check passed on 10,000 Decision Index requests: 219,467 first-pass branches and
  9,961 gated branches, all identical, with bit-identical answers on the same machine. Across machines, the
  C math library can change the last bit of a probability (at most 6 × 10⁻¹⁷ in the golden cases), never a
  decision.
- **Temperature fit data.** The option-count temperature curve was fitted by negative log-likelihood on 512
  examples from the training splits of BANKING77, CLINC150, WANLI and ARC-Challenge, with any example that
  appears in the Decision Index corpus removed. BANKING77 and CLINC150 are also sources of index
  benchmarks. No index data or labels were used for the fit.
- **Method selection.** The index was also our development benchmark. The nine-bin rating was frozen as the
  primary change before a full-suite run that applied it to every two-option question (58.17). Restricting
  the single rating to truth questions was decided after that run showed losses on other pairs (58.35 on the
  same stored outputs). Rating the other pairs in both orders and the disagreement gate were then accepted
  by a pre-registered rule on a fixed 10,000-request balanced sample (at least +0.20 index, at most 1.5×
  median latency) and confirmed on the full suite. The rating text, its mapping and all constants were
  never tuned on index labels. Temperature stays at 1 everywhere except the documented curve. The serving
  defaults of release 1.2 were also selected on index runs.
- **Prior work.** The prompt layout comes from Mapika/decider, and the rating readout and mapping from
  simple-jev v1. Order averaging and rating are well-known debiasing and calibration ideas. Our contribution
  is their combination, the disagreement gate, and the TensorRT-LLM serving path.

## Limitations

- Text only; images are rejected.
- BF16 is validated on H100 PCIe 80 GB and RTX PRO 6000 Blackwell Server Edition; NVFP4 on RTX PRO 6000.
  Other GPUs need their own bring-up and quality check ([deploy.md](deploy.md#moving-to-another-gpu)).
- Depends on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- The gated pass uses a fresh cache salt, so it recomputes the state. Reusing the first pass's prefix is an
  open speed-up.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Probabilities are only partly calibrated: the rating and the temperature curve reduce Gemma's
  overconfidence, but a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on
  `confidence` or a probability threshold, tune that threshold on your own labelled data.

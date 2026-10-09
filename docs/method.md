# Method

MORJ answers a structured question without generating text. Each question becomes a few prompts
(*branches*); the engine prefills each branch once, reads the next-token distribution over the option
labels, and combines the branches into one answer. The default effort, `high`, combines four views of the
same question. This page describes them, what each adds, how the engine runs them, and how the method was
chosen.

## A branch

All prompts use the state-first chat layout of [Mapika/decider](https://github.com/Mapika/decider),
vendored under Apache-2.0 in `morj/vendor/`. A prompt contains the rendered state, one question, its options
with single-token labels (`(A)`, `(B)`, …, then two-letter labels up to 255 options) and `Answer: (`. Each
option is shown as `name: description` (its name alone without a description), except that a name repeating one
of the question's labels exactly (`A`, or `(A)` as shown) is dropped when the option has a description: rotated,
`(A) B: no` would contradict its label. The
service reads the logits of the label tokens at that position and applies a softmax. That is one prefill
per branch and no decoding.

## The four views of `high`

1. **Two candidate orders.** Each question is scored with its options as given and rotated by n // 2. Each
   order's distribution is mapped back to the original options and the two are averaged. This cancels much
   of the model's position bias.
2. **Nine-bin rating.** A two-option question gets one more branch: the same question, then *"Rate how
   likely option (A) is the correct answer on a scale from 1 to 9 … Return only the digit."* and
   `Rating: `. The expected digit E maps to `p(A) = clamp(0.01 + (E/10 − 0.1) · 0.98/0.8, 0.01, 0.99)`,
   following [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Every two-option question (Noul,
   a two-option Choice or a two-level Score) is rated in both orders, and `p = (p_forward + 1 − p_swapped) / 2`.
   The letter average and the rating are averaged. At T = 1, Gemma puts more than 0.99 on over 95% of its yes/no answers, yet only
   87% of those are right. The rating spreads that probability, which matters for Brier scores and for
   ranking by p(yes).
3. **Gated orders.** If a question has three or more options and its two orders pick different options, a
   second pass scores the rotations by n // 4 and 3n // 4, and all orders are averaged (with three options
   the rotation by n // 4 is the original order, so only one is added). On a 10,000-request balanced
   sample of the index, the gate fired on 9.7% of requests (18.8% of those with a question of three or more
   options).
4. **Option-count temperature.** Questions with three or more options soften each order's distribution with
   `T(n) = max(0.05, 6.9 − 0.7 ln n)` before averaging. This never changes an order's argmax, but it
   changes how disagreeing orders combine and how confident the answer is. Rated questions keep their
   letters at T = 1: tempering them too (T(2) = 6.4) left the rating to order candidates by p(yes) and cost
   about 1.2 nDCG@10 on BRIGHT and ToolRet (full Decision Index 0.3: 59.10 against 59.16). See
   [Disclosures](#disclosures) for how the curve was fitted.

The response follows the shared API ([api.md](api.md)): Choice returns `choice`, `probabilities` over the
original keys and `confidence = (p_best − 1/n) / (1 − 1/n)`. Noul returns P(true). Score returns the
expected level, the distribution, a legend and a distance-based confidence.

## What each view adds

Full Decision Index 0.2.1 runs with the official reproduction kit and scorer, one sequential HTTP client
per server, every request answered. Each row adds one change to the row above it. In these runs truth
questions (Noul, or a Choice whose options are exactly yes/no) were rated in one order only; the last row,
on Decision Index 0.3, rates them in both orders like every other two-option question.

| Configuration | Index | Δ | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| One candidate order (Decider prompt, T = 1) | 56.98 | | 89 | 610 |
| + second order, rotated by n/2 | 57.68 | +0.70 | 180 | 1,008 |
| + nine-bin rating on truth questions, option-count temperature elsewhere | 58.37 | +0.69 | 144 | 847 |
| + rating in both orders on other two-option questions | 58.42 | +0.05 | 151 | 951 |
| **+ gated orders** | **58.68** | **+0.26** | **178** | **1,249** |
| *Decision Index 0.3:* the same (one RTX PRO 6000) | 59.09 | | 171 | 1,396 |
| **+ every two-option question rated in both orders (H100)** | **59.16** | **+0.07** | **163** | **1,115** |

The 0.2.1 runs were on one H100 PCIe 80 GB per server; the last one was split into two halves on two H100s.
The last run was split into six shards on six H100 PCIe 80 GB, one sequential client each; its latency is
not the RTX PRO 6000 figure. 99.36% of its decisions equal the previous row's. Latencies come from successive
serving revisions, so compare them only roughly.

Where the gain over one order comes from, in native benchmark metrics:

| Benchmark | One order | Four views | Main contributor |
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

| `effort` | Views per question | Branches / question | Full index |
| --- | --- | --- | --- |
| `low` | one order, T = 1 | 1.0 | 56.98 (0.2.1) |
| `medium` | two orders; rating of every two-option question in both orders; temperature on the rest | 3.34 | no full run |
| `high` | the four views above | 3.43 | 59.16 (0.3) |

Branch counts are averages over the Decision Index 0.3 questions. The extra views cost prefill only, and the
branches of each pass reuse the KV cache of the request's shared prefix (template and state), so three
branches per question cost about 2.3 times one at the median (one question, BF16 on one H100 PCIe 80 GB:
one order 59 ms, the four views 135 ms with truth questions rated in one order). One H100 sustains about
1.25 requests/s on the index mix at any concurrency (prefill-bound): add GPUs for throughput.

## Engine

The service uses the TensorRT-LLM 1.3.0rc29 PyTorch backend through its `LLM` API, the same layer
`trtllm-serve` wraps, on one GPU.

- **Shared prefix.** All branches of a request begin with the same template and state. The first branch
  (the *seed*) runs alone and fills the KV cache; the remaining branches then run concurrently in up to 32
  branch slots and reuse that prefix through TensorRT-LLM's block reuse. KV admission counts a request's
  shared prefix once. When the branches share fewer than 256 prompt tokens, waiting for a seed buys
  nothing, so they all run in one wave. A random `cache_salt` per request, shared by its passes, keeps
  requests from ever sharing KV state while the gated pass reuses the first pass's template, images and state.
- **Prefill only.** Each branch requests one token and reads its generation logits, with FlashInfer
  attention and 2,048-token chunked prefill. No CUDA graphs: on text, breakable prefill graphs were a net
  latency loss (mean −10% and p80 −21% without them on H100, 99.88% of answers equal), and TensorRT-LLM
  refuses them for multimodal models. At startup the service sends itself one readiness request; `/health`
  turns ready after it is answered.
- **Images.** Gemma 4's vision tower encodes a request's images (1,120 vision tokens each) once; they sit
  before the state, so every branch reuses their KV blocks. The tower encodes at most two images per call:
  TensorRT-LLM otherwise encodes every image of an engine step at once, and eight images ran an H100 out of
  memory. Each image is encoded on its own either way, so the embeddings are the same. With two or more
  images each is introduced as `Image k:`; without labels the model lost the middle images of eight (BLINK,
  927 rows: 66.0 against 64.7 skill, multi-image rows 49.8 against 46.9).
- **KV pools.** Gemma 4 mixes full-attention and sliding-window layers. The pools are sized for a
  4,096-token typical sequence, so long many-option requests (several ~7k-token branches) prefill side by
  side. The cache holds at least one branch of the full context length; startup refuses less.
- **Admission.** Branches are admitted first-come, first-served against a budget of resident context
  tokens, so long branches queue instead of forcing KV preemption. Above 4 running and 16 waiting requests
  the service answers 529 with `Retry-After`. Per-request deadlines cancel native work. A native failure
  drains in-flight requests and exits for the supervisor to restart.
- **Host KV tier.** TensorRT-LLM's host (CPU) KV tier is always sized explicitly; the default is none.
  Left to TensorRT-LLM, a container's default memlock limit gives a one-block tier on which the KV manager
  can stop making progress on a long multi-branch request ([tensorrt-llm-patch.md](tensorrt-llm-patch.md)).
- **Patches.** Two process-local patches live in `morj/patches/`: one stops an engine that makes no progress,
  and one encodes images in slices of two. They install only when the patched TensorRT-LLM files match the
  pinned SHA-256 values ([tensorrt-llm-patch.md](tensorrt-llm-patch.md)).

BF16 weights take 57.2 GiB on the GPU for the text tower, plus the vision tower when images are on. The
evaluated run used a 65,536-token context, 32 branch slots, 2,048-token prefill chunks and a 0.7 KV cache
fraction (headroom for the vision encoder); these are the defaults. They change speed and memory, not the
method: other values are allowed ([deploy.md](deploy.md#settings)), logged at startup and reported in
`/health`. In earlier runs, 4 against 32 branch slots gave the same decisions on 99.91% of answers.

## Disclosures

- **Evaluated code.** The Decision Index 0.3 run was served by the pre-release source of this code. Before
  release this code was checked against it: identical token branches, gate decisions and answers on fixed
  synthetic requests, and the same decisions on 300 random requests of the edition (760 of 760 questions).
- **Temperature fit data.** The option-count temperature curve was fitted by negative log-likelihood on 512
  examples from the training splits of BANKING77, CLINC150, WANLI and ARC-Challenge. BANKING77, CLINC150 and
  ARC are also sources of index benchmarks; the index uses their test splits. Examples equal to an index item
  were meant to be excluded, but the check compared whole strings and missed utterances that the index embeds
  in a longer instruction: one fit example duplicates a CLINC150 test item up to punctuation, and 7 more
  differ from one by a word or two. Refitting without the duplicate gives the same constants; without every
  near match the fit moves one grid step (6.8 − 0.675 ln n).
- **Method selection.** The index was also our development benchmark. The nine-bin rating was frozen as the
  primary change before a full-suite run that applied it to every two-option question (58.17). Restricting
  the single rating to truth questions was decided after that run showed losses on other pairs (58.35 on the
  same stored outputs). Rating the other pairs in both orders and the disagreement gate were then accepted
  by a pre-registered rule on a fixed 10,000-request balanced sample (at least +0.20 index, at most 1.5×
  median latency) and confirmed on the full suite. The rule on option names was later replaced by one rule,
  every two-option question rated in both orders: 59.16 on a full 0.3 run against 59.09, a difference within
  noise, with one rule fewer. Applying the temperature to rated letters as well was tested
  on the full suite and rejected (59.10). The rating text, its mapping and all constants were never tuned on
  index labels. Temperature stays at 1 everywhere except the documented curve. The serving defaults were
  also selected on index runs.
- **Prior work.** The prompt layout comes from Mapika/decider, and the rating readout and mapping from
  simple-jev v1. Order averaging and rating are well-known debiasing and calibration ideas. Our contribution
  is their combination, the disagreement gate, and the TensorRT-LLM serving path.

## Limitations

- Images need the BF16 checkpoint; NVFP4 serves text only. At most 8 images per request.
- BF16 is validated on H100 PCIe 80 GB and RTX PRO 6000 Blackwell Server Edition; NVFP4 on RTX PRO 6000.
  This code has run the full Decision Index on H100 only so far.
  Other GPUs need their own bring-up and quality check ([deploy.md](deploy.md#moving-to-another-gpu)).
- Depends on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Probabilities are only partly calibrated: the rating and the temperature curve reduce Gemma's
  overconfidence, but a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on
  `confidence` or a probability threshold, tune that threshold on your own labelled data.

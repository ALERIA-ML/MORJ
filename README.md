# MORJ: Multi-Order Readout Judge

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Decision Index](https://img.shields.io/badge/Decision_Index-60.24-2ea44f)](https://huggingface.co/spaces/multimodalart/jev-decision-index)

Zero-shot structured decisions from stock **Gemma 4 31B** on one GPU. Give MORJ a state (a ticket, a document, a conversation, any JSON, up to eight images) and typed questions: pick one of up to 255 options, yes or no, a score on an ordinal scale. It answers each with probabilities.

There is no fine-tuning and no generated text. MORJ reads the model's next-token distribution over the option labels, for several rotations of the option order, and combines them. It runs on TensorRT-LLM behind the Decision Index `POST /v1/systemone` API and ships a playground that shows what the engine did for any request.

MORJ is built by [Aleria AI](https://aleria.ai). Aleria builds sovereign AI systems that run entirely inside your environment, on-premise or air-gapped, with no external API calls. Systems like that need reliable typed decisions, and we would rather ship them as an open method on a stock model than as opaque weights behind someone else's endpoint: everything here runs on your own GPU.

**[Decision Index](https://github.com/apolinario/decision-index) 0.3, public part: 60.24** (BF16 on one RTX PRO 6000 Blackwell Server Edition; all 140,620 requests answered, one at a time, through the HTTP API; median 173 ms and p80 475 ms per request). This is our run scored with the official kit, not a maintainer-reviewed board entry; the board measures its latency gate (median, mean and p80 at most 1,000 ms on one RTX PRO 6000) itself.

## Quickstart

You need Linux with one NVIDIA GPU with 80 GB or more (BF16; validated on H100 PCIe 80 GB and RTX PRO 6000 96 GB), Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and about 100 GB of disk for the TensorRT-LLM image and the checkpoint.

```bash
docker compose up -d
docker compose logs -f
```

Keep the server detached and follow its log with `logs -f`, which you can stop and start again at any time. An attached `docker compose up` whose terminal stops reading (a sleeping laptop, a dropped SSH session) blocks the server's log writes, and requests wait until the terminal reads again.

The first start builds the image on NVIDIA's TensorRT-LLM 1.3.0rc29 release image, downloads [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) at a pinned revision into `~/.cache/huggingface`, loads it with Gemma 4's vision tower (a few minutes). When the log says `ready`, open <http://localhost:30104/> for the playground, or call the API:

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}, "team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

Images go in `images` as base64 data URLs (PNG, JPEG, WebP or GIF; up to eight, 5 MB each; remote URLs are never fetched). The body is piped in, as an encoded image is too long for a command-line argument:

```bash
printf '{"state": "A customer sent this photo with a return request.", "images": ["data:image/jpeg;base64,%s"], "questions": {"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}, "category": {"type": "choice", "criteria": {"electronics": null, "clothing": null, "furniture": null, "other": null}}}}' "$(base64 < photo.jpg | tr -d '\n')" | curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' --data-binary @-
```

On a Blackwell GPU, NVIDIA's 4-bit checkpoint (`nvidia/Gemma-4-31B-IT-NVFP4`, 33 GB) is faster and scores about 1 point lower; it serves text only:

```bash
CHECKPOINT=nvfp4 VISION=false docker compose up -d
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT` | `bf16` | `bf16` or `nvfp4` |
| `DEFAULT_EFFORT` | `high` | Effort of requests that name none: `low`, `medium` or `high` |
| `GPU` | `0` | Physical GPU ordinal |
| `PORT` | `30104` | Playground and API port |
| `HOST` | `127.0.0.1` | `0.0.0.0` accepts remote clients; set `API_KEY` too |
| `API_KEY` | empty | Bearer key for `/v1/*`; empty disables authentication |
| `MODEL_DIR` | unset | Serve a checkpoint already on disk instead of downloading it |
| `VISION` | `true` | Accept `images` (BF16 only); `false` serves text only |
| `HOST_KV_CACHE_GB` | `0` | TensorRT-LLM's host KV tier: `0` (none) or at least 0.5; never left to TensorRT-LLM ([why](docs/tensorrt-llm-patch.md#host-kv-tier)) |

All settings, checks and operations: [docs/deploy.md](docs/deploy.md).

## How it works

Each question becomes a few prompts (*branches*) in the state-first layout of [Mapika/decider](https://github.com/Mapika/decider): the state, one question, its options with single-token labels, and `Answer: (`. The engine prefills each branch once and reads the softmax over the label tokens. For a four-option question:

```
           labels (A)(B)(C)(D)
order 1     A   B   C   D       options as given
order 2     C   D   A   B       rotated by n/2, mapped back and averaged with order 1
order 3     B   C   D   A       ┐ only when orders 1 and 2 pick different options
order 4     D   A   B   C       ┘ (rotations by n/4 and 3n/4; about 10% of requests)
```

1. **Two orders.** Averaging the original and the half-rotated order cancels much of the model's position bias.
2. **Nine-digit rating.** Every two-option question also asks, in both orders, how likely option (A) is on a 1–9 scale, and the expected digit becomes a probability. Gemma is near-certain on almost every yes/no answer; the rating spreads that probability, which matters for Brier scores and for ranking by p(yes).
3. **Gated orders.** When the two orders disagree on a question with three or more options, the rotations by n/4 and 3n/4 are scored too (with three options, the one rotation left) and all orders are averaged.
4. **Option-count temperature.** Questions with three or more options soften each order with `T(n) = max(0.05, 6.9 − 0.7 ln n)` before averaging, fitted on training data ([disclosures](docs/method.md#disclosures)).

The branches of a request share the KV cache of its common prefix, so three branches per question cost about 2.3 times one. Images go before the state, so every branch reuses them too. Formulas, per-benchmark gains, the engine and how the method was chosen: [docs/method.md](docs/method.md). Order averaging helps fine-tuned models too: it added 0.98 points to Cloudflare's clef on Decision Index 0.2.1 ([method.md](docs/method.md#order-averaging-on-a-fine-tuned-model)).

## Effort

Each request can trade accuracy for latency with a header (`X-Decision-Effort: low`) or a field (`"options": {"effort": "low"}`); clients that send neither, including the Decision Index kit, get `high`.

| `effort` | What each question gets | Branches / question | Decision Index 0.3 |
| --- | --- | --- | --- |
| `low` | one order | 1.0 | no full run |
| `medium` | two orders; rating of every two-option question in both orders; temperature on the rest | 3.34 | no full run |
| `high` (default) | all four views above | 3.43 | 60.24 |

The index is the full public part, BF16 on one RTX PRO 6000; `low` and `medium` have no full run. Every response reports the effort that ran in `usage.effort`. Request and response format, status codes and the playground: [docs/api.md](docs/api.md).

## Repository layout

```
morj/       the service; the method itself is morj/method.py
tests/      offline tests, no GPU needed
docs/       method, API, deployment, evaluation runbook, TensorRT-LLM patch note
```

| Document | Contents |
| --- | --- |
| [docs/method.md](docs/method.md) | The four views, what each adds, effort profiles, the engine, disclosures and limitations |
| [docs/api.md](docs/api.md) | Endpoints, request and response format, effort, status codes, playground |
| [docs/deploy.md](docs/deploy.md) | Requirements, settings, tests, operations, other GPUs |
| [docs/evaluation.md](docs/evaluation.md) | Reproducing a Decision Index number and submitting a run |
| [docs/tensorrt-llm-patch.md](docs/tensorrt-llm-patch.md) | The TensorRT-LLM issues MORJ works around (host KV tier, stall guard, vision encoder memory, image budget) |

## Tests

The offline tests need no GPU:

```bash
uv run --extra test python -m pytest
```

With the Gemma tokenizer files (a `google/gemma-4-31B-it` download, or the overlay the service builds in `~/.cache/morj/bf16/morj/tokenizer`), the prompt tests also check the real token layout:

```bash
MORJ_TOKENIZER=/path/to/gemma-4-31B-it uv run --extra test python -m pytest tests/test_prompt.py
```

## Limitations

- Images need the BF16 checkpoint (NVFP4 serves text only); at most eight per request, 1,122 prompt tokens each.
- One process serves one GPU and one checkpoint. Concurrency adds latency rather than throughput (one H100 sustains about 1.25 requests/s on the index mix): add GPUs for throughput.
- Depends on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Probabilities are only partly calibrated: a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on `confidence` or a probability threshold, tune it on your own labelled data.

## Acknowledgements

The prompt layout comes from [Mapika/decider](https://github.com/Mapika/decider) (vendored under Apache-2.0 in `morj/vendor/`), and the rating readout and mapping from [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Order averaging and rating are well-known debiasing and calibration ideas; MORJ's contribution is their combination, the disagreement gate and the TensorRT-LLM serving path. Scores come from the [Decision Index](https://github.com/apolinario/decision-index) kit. NVIDIA supported the index runs with $1,000 of credits on [brev.nvidia.com](https://brev.nvidia.com/). Thank you.

## License

Apache-2.0 ([LICENSE](LICENSE)); third-party notices in [NOTICE](NOTICE). Gemma 4 is released by Google under Apache-2.0.

# MORJ: Multi-Order Rotation Judge

Zero-shot structured decisions from stock **Gemma 4 31B** on one GPU. Give MORJ a state (a ticket, a
document, a conversation, any JSON) and typed questions: pick one of up to 255 options, yes or no, a score
on an ordinal scale. It answers each with probabilities.

There is no fine-tuning and no generated text. MORJ reads the model's next-token distribution over the
option labels, for several rotations of the option order, and combines them. It runs on TensorRT-LLM behind
the Decision Index `POST /v1/systemone` API and ships a playground that shows what the engine did for any request.

**[Decision Index](https://github.com/apolinario/decision-index) 0.3, public part: 59.09** on one RTX PRO
6000 Blackwell. All 140,620 requests were answered, one at a time through the HTTP API, with a median of
171 ms, p80 587 ms and mean 437 ms: within the board's latency gate (median, mean and p80 at most 1,000 ms
on that card). This is our local run scored with the official kit, not a maintainer-reviewed board entry;
the [scorer files](results/morj-high-bf16/README.md) list every benchmark.

## Quickstart

You need Linux with one NVIDIA GPU with 80 GB or more (BF16; validated on H100 PCIe and RTX PRO 6000), Docker
with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and about 100 GB of disk for the TensorRT-LLM image and the checkpoint.

```bash
docker compose up
```

The first start builds the image on NVIDIA's TensorRT-LLM 1.3.0rc29 release image, downloads
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) at a pinned revision into
`~/.cache/huggingface`, loads it and captures CUDA graphs (a few minutes). When the log says `ready`, open
<http://localhost:30104/> for the playground, or call the API:

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}, "team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

On a Blackwell GPU, NVIDIA's 4-bit checkpoint (`nvidia/Gemma-4-31B-IT-NVFP4`, 33 GB) is faster and scores
about 1 point lower:

```bash
CHECKPOINT=nvfp4 docker compose up
```

| Variable | Default | |
| --- | --- | --- |
| `CHECKPOINT` | `bf16` | `bf16` or `nvfp4` |
| `DEFAULT_EFFORT` | `high` | Effort of requests that name none: `low`, `medium` or `high` |
| `GPU` | `0` | Physical GPU ordinal |
| `PORT` | `30104` | Playground and API port |
| `HOST` | `127.0.0.1` | `0.0.0.0` accepts remote clients; set `API_KEY` too |
| `API_KEY` | empty | Bearer key for `/v1/*`; empty disables authentication |
| `MODEL_DIR` | — | Serve a checkpoint already on disk instead of downloading it |

All settings, checks and operations: [docs/deploy.md](docs/deploy.md).

## How it works

Each question becomes a few prompts (*branches*) in the state-first layout of
[Mapika/decider](https://github.com/Mapika/decider): the state, one question, its options with
single-token labels, and `Answer: (`. The engine prefills each branch once and reads the softmax over the
label tokens. For a four-option question:

```
           labels (A)(B)(C)(D)
order 1     A   B   C   D       options as given
order 2     C   D   A   B       rotated by n/2, mapped back and averaged with order 1
order 3     B   C   D   A       ┐ only when orders 1 and 2 pick different options
order 4     D   A   B   C       ┘ (rotations by n/4 and 3n/4; about 10% of requests)
```

1. **Two orders.** Averaging the original and the half-rotated order cancels much of the model's position
   bias.
2. **Nine-digit rating.** A two-option question also asks how likely option (A) is on a 1–9 scale, and the
   expected digit becomes a probability. Gemma is near-certain on almost every yes/no answer; the rating
   spreads that probability, which matters for Brier scores and for ranking by p(yes).
3. **Gated orders.** When the two orders disagree on a question with three or more options, the rotations
   by n/4 and 3n/4 are scored too (with three options, the one rotation left) and all orders are averaged.
4. **Option-count temperature.** Unrated questions soften each order with `T(n) = max(0.05, 6.9 − 0.7 ln n)`
   before averaging, fitted on training splits with every index item removed.

On Decision Index 0.2.1, one order scores 56.98; the second order adds 0.70, the rating and temperature
0.69, rating other pairs in both orders 0.05 and the gated orders 0.26, for 58.68. The branches of a
request share the KV cache of its common prefix, so three branches per question cost about 2.3 times one.
Formulas, per-benchmark gains, the engine and how the method was chosen: [docs/method.md](docs/method.md).
Order averaging helps fine-tuned models too: it added 0.98 points to Cloudflare's clef on Decision Index
0.2.1 ([method.md](docs/method.md#order-averaging-on-a-fine-tuned-model)).

## Effort

Each request can trade accuracy for latency with a header (`X-Decision-Effort: low`) or a field
(`"options": {"effort": "low"}`); clients that send neither, including the Decision Index kit, get `high`.

| `effort` | What each question gets | Index 0.2.1 | p50, one question |
| --- | --- | --- | --- |
| `low` | one order | 56.98 | 59 ms |
| `medium` | two orders; rating on yes/no questions, temperature elsewhere | 58.37 | 109 ms |
| `high` (default) | all four views above | 58.71 | 135 ms |

BF16 on one H100 PCIe; `high` scored 58.68 with release 1.0's scheduling and 58.71 with this release's. Every
response reports the profile that ran in `usage.effort`. Request and response
format, status codes and the playground: [docs/api.md](docs/api.md).

## Repository layout

```
morj/       the service: method.py (the method), prompt.py (token branches), engine.py (TensorRT-LLM),
            server.py (HTTP), trace.py and playground.html, checkpoint.py (pinned weights and tokenizer),
            patches/ (TensorRT-LLM patches), vendor/ (Decider prompt code), tools/ (live reliability probe)
tests/      offline tests, no GPU needed; the golden test pins prompts and answers to the evaluated runs
scripts/    Decision Index helpers: samples, latency report, run comparison, hang watcher
docs/       method, API, deployment, evaluation runbook, TensorRT-LLM patch note
results/    scorer files of the Decision Index 0.3 run
```

| Document | |
| --- | --- |
| [docs/method.md](docs/method.md) | The four views, what each adds, effort profiles, the engine, disclosures and limitations |
| [docs/api.md](docs/api.md) | Endpoints, request and response format, effort, status codes, playground |
| [docs/deploy.md](docs/deploy.md) | Requirements, settings, tests, operations, other GPUs |
| [docs/evaluation.md](docs/evaluation.md) | Reproducing a Decision Index number and submitting a run |
| [docs/tensorrt-llm-patch.md](docs/tensorrt-llm-patch.md) | The one TensorRT-LLM bug MORJ works around, and why it cannot change an answer |
| [results/](results/morj-high-bf16/README.md) | The Decision Index 0.3 run: every benchmark, hardware, latency, file hashes |

## Tests

The offline tests need no GPU:

```bash
uv run --extra test python -m pytest
```

With the Gemma tokenizer files (a `google/gemma-4-31B-it` download, or the overlay the service builds in
`~/.cache/morj/bf16/morj/tokenizer`), the golden test also checks that this code builds the same prompts,
gate decisions and answers as the evaluated runs:

```bash
MORJ_TOKENIZER=/path/to/gemma-4-31B-it uv run --extra test python -m pytest tests/test_golden.py tests/test_prompt.py
```

## Limitations

- Text only; images are rejected.
- One process serves one GPU and one checkpoint. Concurrency adds latency rather than throughput (one H100
  sustains about 1.25 requests/s on the index mix): add GPUs for throughput.
- Depends on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Probabilities are only partly calibrated: a stated 0.9 is not guaranteed to be right 90% of the time on
  your task. If you act on `confidence` or a probability threshold, tune it on your own labelled data.

## Acknowledgements

The prompt layout comes from [Mapika/decider](https://github.com/Mapika/decider) (vendored under Apache-2.0
in `morj/vendor/`), and the rating readout and mapping from
[simple-jev](https://github.com/featherless-ai/simple-jev) v1. Order averaging and rating are well-known
debiasing and calibration ideas; MORJ's contribution is their combination, the disagreement gate and the
TensorRT-LLM serving path. Scores come from the [Decision Index](https://github.com/apolinario/decision-index)
kit.

## License

Apache-2.0 ([LICENSE](LICENSE)); third-party notices in [NOTICE](NOTICE). Gemma 4 is released by Google
under Apache-2.0.

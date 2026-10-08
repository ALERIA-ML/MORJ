# Running the service

One process owns one GPU. It serves the playground at `/`, `POST /v1/systemone` and
`POST /v1/trace` on a single port, bound to loopback by default.

## Requirements

- Linux with one NVIDIA GPU. `CHECKPOINT=bf16` (default): at least 80 GB; BF16 weights take
  57.2 GiB; validated on H100 PCIe 80 GB and RTX PRO 6000. `CHECKPOINT=nvfp4`: a Blackwell GPU (FP4 tensor cores,
  SM100 or SM120), target RTX PRO 6000 96 GB; the checkpoint is about 33 GB with an FP8 KV cache.
- Docker with the NVIDIA Container Toolkit, and a driver that runs NVIDIA's
  `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29` image (CUDA 13).
- Free disk: 35 GB for the image, 62 GB (bf16) or 33 GB (nvfp4) for the checkpoint, a few GB of caches.

## Start

```bash
docker compose up
```

That is the whole setup. On the first start the log shows, in order:

1. The image build. The only step is checking that the TensorRT-LLM sources match the ones our
   patches were validated against.
2. `downloading google/gemma-4-31B-it at revision 842da37 …` (or `nvidia/Gemma-4-31B-IT-NVFP4 at
   revision 4135a98 …`): the checkpoint goes into
   `~/.cache/huggingface` in the standard Hugging Face cache layout. An interrupted download resumes,
   and a copy that other tools already put in that cache is reused.
3. `serving … default effort high@1`: the tokenizer overlay with Google's canonical chat template
   is assembled and checked. Then `trtllm_patches_installed`, `kv_pool` events and CUDA-graph
   capture: 3–4 minutes for bf16 on a first start, longer for nvfp4 (GEMM autotuning).
4. `ready. Playground: http://localhost:30104/`.

Use `docker compose up -d` to run in the background and `docker compose logs -f` to follow it.
Stop it with `docker compose down`.

The service fails fast instead of starting degraded. It stops when no GPU is visible, the GPU has
no FP4 tensor cores (unless `NVFP4_BACKENDS=marlin`), the checkpoint or tokenizer files differ from
the pinned ones, the KV cache cannot hold a 65,536-token branch, or no prefill CUDA graph was captured.

## Settings

Every setting has a default. Pass overrides in the shell, for example
`GPU=3 PORT=8080 docker compose up`, or put them in a `.env` file next to `docker-compose.yml`.

| Variable | Default | |
| --- | --- | --- |
| `GPU` | `0` | Physical GPU ordinal (as listed by `nvidia-smi`) |
| `PORT` | `30104` | Port of the playground and the API |
| `HOST` | `127.0.0.1` | Listening address; `0.0.0.0` accepts remote clients |
| `API_KEY` | empty | Bearer key for `/v1/*` and `/stats`; set it before using `HOST=0.0.0.0` |
| `MODEL_DIR` | — | Serve a checkpoint already on disk (a directory with `config.json` and the safetensors shards) |
| `HF_CACHE` | `~/.cache/huggingface` | Hugging Face cache that receives the download |
| `CHECKPOINT` | `bf16` | `bf16` (google/gemma-4-31B-it) or `nvfp4` (nvidia/Gemma-4-31B-IT-NVFP4, Blackwell) |
| `DEFAULT_EFFORT` | `high` | Effort profile of requests that name none; the official kit names none |
| `CACHE_DIR` | `~/.cache/morj` | Compilation caches (one subfolder per checkpoint), the tokenizer overlay and temporary files |
| `HF_TOKEN` | empty | Not needed for these Apache-2.0 checkpoints; a saved `hf auth login` token is also used |
| `NVFP4_BACKENDS` | empty | `nvfp4` only: GEMM backends (comma-separated); empty keeps TensorRT-LLM's Blackwell defaults |

The method and the runtime profile are fixed to the evaluated configuration: context 65,536
tokens, 32 branch slots, 2,048-token prefill chunks, a 0.8 KV cache fraction, shared-prefix KV
admission, one-wave branches below 256 shared tokens and KV pools sized for a 4,096-token typical
sequence. Startup refuses other values.

Admission limits do not affect answers and can be tuned with `MORJ_MAX_REQUESTS` (4), `MORJ_MAX_QUEUE`
(16), `MORJ_QUEUE_TIMEOUT` (30 s) and `MORJ_REQUEST_TIMEOUT` (300 s); pass them through the `environment`
section of `docker-compose.yml`. A request is answered (or refused with 504) within
`queue_timeout + request_timeout + 10` = 340 s, so give clients a timeout above that (the runbook uses 400 s).

With `MODEL_DIR`, the directory must hold the same revision; startup compares the hashes of its
config, weight index, quantization config and tokenizer with `morj/provenance/identity-<checkpoint>.json`.

## Use it

On the GPU machine, open <http://localhost:30104/>. From a laptop, tunnel the port and open the
same URL:

```bash
ssh -N -L 30104:127.0.0.1:30104 gpu-server
```

In the playground, pick a preset or edit the state and questions, then press
**Run on the engine** (Cmd/Ctrl+Enter). Click a branch to see its prompt. **Save trace** downloads
the full record, and **Open a saved trace…** shows it later, even from `playground.html` opened
straight from disk without a server. Two traces recorded with the BF16 release on an H100 are in
[`traces/`](traces/). The request format, effort selection and status codes are in [api.md](api.md).

The API takes the same JSON:

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}, "team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

`/health` returns 503 while the service starts. Before reporting ready, the service sends itself one probe
request and checks that it replayed a captured prefill CUDA graph; `/health` then returns 200 with the
method, checkpoint and graph counters.

## Tests

Offline tests need no GPU:

```bash
pip install -e '.[test]'
```

```bash
python -m pytest
```

With the tokenizer overlay the service builds (or a `google/gemma-4-31B-it` directory, whose
tokenizer files are identical), the golden test also proves that this code builds the same prompts,
gate decisions and answers as the evaluated BF16 run:

```bash
MORJ_TOKENIZER=~/.cache/morj/bf16/morj/tokenizer python -m pytest tests/test_golden.py tests/test_prompt.py
```

Against a running instance, this probe checks request isolation, overload and oversize handling:

```bash
docker compose exec server python -m morj.tools.validate_live --url http://127.0.0.1:30104 --hardware "1x RTX PRO 6000 Blackwell Server Edition 96 GB" --output /tmp/validate-live.json
```

## Operations

- **Monitoring.** `/metrics` exposes request counts by status, end-to-end and per-stage latency,
  branches, gated questions, prompt and cached tokens, queue depth, KV budget and graph replays.
  Every request writes one JSON log line with its ID, stage timings and token counts.
- **Overload.** Above 4 running and 16 waiting requests, the service answers 529 with
  `Retry-After` instead of growing queues. Concurrency adds latency rather than throughput,
  because prefill saturates the GPU.
- **Failures.** If native inference fails, the service stops admitting requests, lets in-flight
  ones finish or fail with 503, then exits; compose restarts it with a fresh CUDA context.
- **Shutdown** waits up to `queue_timeout + request_timeout + 10` seconds for admitted requests.
- **Files.** The container runs as root, so files it creates in the cache directories belong to root.

## Functional check without a Blackwell GPU

TensorRT-LLM can run the NVFP4 weights on Hopper or Ada through Marlin kernels with BF16
activations. It checks loading, the FP8 KV cache, prompts, graphs and the API before renting a
Blackwell card; its decisions and timings are not the release's:

```bash
CHECKPOINT=nvfp4 NVFP4_BACKENDS=marlin docker compose up
```

## Moving to another GPU

Run the same sequence on the new card and stop at the first failure: start-up log events, KV
capacity of at least 65,536 tokens, graph capture, `/health`, then the sample comparison in
[evaluation.md](evaluation.md#3-decisions-against-the-reference-run). Decisions should match
the BF16 reference on most questions (NVFP4 changes the weights, so not everywhere). Scattered
near-ties are expected; a systematic change on a benchmark or question type is not.

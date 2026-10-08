# Evaluation on the Decision Index

Every Decision Index number for this service is produced **through the production endpoint**: the
service from [deploy.md](deploy.md) runs as deployed, and the official
[reproduction kit](https://github.com/apolinario/decision-index) at commit `9eb2dbe2` (edition 0.3) sends
each request's `state` and `questions` to `POST /v1/systemone` exactly as stored, one request at a time,
with its HTTP engine and scorer unchanged. Nothing calls the engine in-process. The kit's own HTTP timeout
(600 s) is kept: the service answers or returns 504 within its 340 s request bound.

Run the steps in order on the card you evaluate and stop at the first failure. The board measures
latency on one **RTX PRO 6000 Blackwell Server Edition** 96 GB and adds a model only if its median,
mean and 80th-percentile request latency are each at most 1,000 ms. This runbook is for edition 0.3, the
board edition since 2026-10-06; [edition 0.2.1](#edition-021) differs only in the kit commit and the suite.
Scores of the two editions are not comparable.

| | 0.2.1 | 0.3 |
| --- | --- | --- |
| Public suite | 150,317 scoreable requests, 44 benchmarks, 38 in the index | the same rows with GSM8K rebuilt (2,638 requests with new distractors) and ForecastBench retired: 140,178 scoreable, 43 benchmarks, 37 in the index |
| Board ranking | the public index | Full score: 20% public index, 50% private tests of the same skills, 30% private new-domain tasks (the private parts are run by the maintainers only) |
| Latency gate | median at most 1,000 ms on one RTX PRO 6000 | median, mean and p80 each at most 1,000 ms on one RTX PRO 6000 |
| Kit commit | [`87d4650`](https://github.com/apolinario/decision-index/tree/87d4650b42b377c0291a89c1f1a879f9b31082bf) | [`9eb2dbe`](https://github.com/apolinario/decision-index/tree/9eb2dbe2a358004c8782c66e40a83ac07b953fec) |

What you need: a Linux machine with one NVIDIA GPU and Docker (see [deploy.md](deploy.md#requirements)), a
CPU-only Python 3.11+ environment for the kit, and time: a full run takes about 17 hours (BF16) or 16 hours
(NVFP4) on one RTX PRO 6000 and about 18 hours on one H100. Split the suite across GPUs to finish sooner
([step 5](#5-full-run)).

**Effort.** The kit sends requests without `options`, so every request gets the service default, `high@1`:
the evaluated method. `/health` shows `default_effort`, and every result row records `usage.effort`.

## 0. Install the kit and get the 0.3 suite

A CPU-only Python 3.11+ environment next to the service (on hosts with an older Python, `uv venv -p 3.12`):

```bash
pip install "decision-index[rebuild] @ https://github.com/apolinario/decision-index/archive/9eb2dbe2a358004c8782c66e40a83ac07b953fec.tar.gz"
```

Edition 0.3 is the 0.2.1 row files plus `gsm8k-rows.jsonl.gz` (2,638 rebuilt GSM8K requests). Either
rebuild everything (about 7 GB of downloads; `HF_TOKEN` with accepted
[HLE access](https://huggingface.co/datasets/cais/hle)):

```bash
python -m decision_index suite rebuild --edition 0.3 --work work
```

or import row files you already have (`gsm8k-rows.jsonl.gz` is built from the pinned GSM8K test file
and the 0.2.1 rows by the kit's `decision_index.suite.build.gsm8k_v3`):

```bash
python -m decision_index suite import --edition 0.3 --dir suite-0.3 --rows selected-rows.jsonl.gz --added-rows added-rows.jsonl.gz --gsm8k-rows gsm8k-rows.jsonl.gz
```

A pip-installed kit has no `hub/` folder, so `suite import` copies only the row files. Add the exclusion
list and the 0.3 manifest from the kit's repository, then verify the folder (`"match": true`):

```bash
curl -fsSL https://raw.githubusercontent.com/apolinario/decision-index/9eb2dbe2a358004c8782c66e40a83ac07b953fec/hub/excluded-questions.json -o suite-0.3/excluded-questions.json && curl -fsSL https://raw.githubusercontent.com/apolinario/decision-index/9eb2dbe2a358004c8782c66e40a83ac07b953fec/hub/0.3/manifest.json -o suite-0.3/manifest.json
```

```bash
python -m decision_index suite verify --edition 0.3 --dir suite-0.3
```

`suite import` refuses files whose hashes differ from the published ones. Expected uncompressed hashes:
`b2b56d6f…0792d5` (base), `7429f3c9…4ce89` (added), `75933489…9f4738a` (GSM8K); exclusions
`331df32d…155e81`. A full 0.3 run has 140,620 requests, 140,178 scoreable. The scripts below read
`./suite-0.3` (pass `--suite-dir` otherwise).

## 1. Start the service and record the card

```bash
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv
```

Workstation and Max-Q editions of the RTX PRO 6000 have other power limits and clocks; record what you
ran on. Start the service (see [deploy.md](deploy.md) for `CHECKPOINT`, `GPU`, `PORT`, `MODEL_DIR` and the
caches) and wait for `/health`:

```bash
docker compose up -d --build
```

```bash
until curl -fsS http://127.0.0.1:30104/health; do sleep 10; done
```

The examples assume `http://127.0.0.1:30104` and the served model name in `$MODEL`
(`gemma-4-31b-trt` for `CHECKPOINT=bf16`, `gemma-4-31b-nvfp4-trt` for `nvfp4`; `/health` shows it).
The kit and these scripts do not create missing output folders for row files, so create `runs/` first:

```bash
export MODEL=$(curl -fsS http://127.0.0.1:30104/health | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])") && mkdir -p runs
```

If authentication is on, export the key as `DECISION_INDEX_API_KEY`: the kit's HTTP engine sends it as a bearer token.

## 2. Compatibility pass

Two requests from every benchmark track; every request must come back `ok`:

```bash
python scripts/samples.py compat --out runs/compat-rows.jsonl
```

```bash
python -m decision_index run --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --rows runs/compat-rows.jsonl --out runs/compat --compact
```

## 3. Decisions against the reference run

A balanced 2,000-row sample, compared question by question with the same rows answered by a reference
deployment: this service with `CHECKPOINT=bf16` at the default effort on a validated GPU (H100 PCIe or RTX
PRO 6000), e.g. `runs/reference-sample` produced with the same two commands there. Its full-suite result
is [59.09 on one RTX PRO 6000](../results/morj-high-bf16/README.md) (median 171.0 ms):

```bash
python -m decision_index suite sample --edition 0.3 --dir suite-0.3 --n 2000 --seed 20261002 --out runs/sample-rows.jsonl
```

```bash
python -m decision_index run --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --rows runs/sample-rows.jsonl --out runs/sample --compact
```

```bash
python scripts/compare_runs.py --reference runs/reference-sample/results.jsonl --candidate runs/sample/results.jsonl --reference-median 171.0 --json runs/sample/compare.json
```

The reference is `high@1` on BF16, so compare runs served with the default effort. With
`CHECKPOINT=bf16` on another GPU, decisions should match almost everywhere (kernel differences move
near-ties). With `nvfp4` the weights change, so `same_decision_rate` will be clearly below 1; NVFP4 scored
1.01 index below BF16 on 0.2.1. Read `largest_changed`: scattered near-ties are
expected; a whole benchmark or question type flipping points to a serving fault (template, KV cache,
kernels). `--reference-median` is the full-suite median on one RTX PRO 6000: keep it only for a reference on
that card; `projected_full_suite_median` scales it and is only indicative.

## 4. Latency check (the submission gate)

The maintainers time a stratified sample sent one at a time to one process, after untimed warm-up rows.
Their rows are not published; `samples.py latency` draws an equivalent 750-row sample with a fixed seed:

```bash
python scripts/samples.py latency --out runs/latency-rows.jsonl --warmup-out runs/latency-warmup.jsonl
```

```bash
python -m decision_index run --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --rows runs/latency-warmup.jsonl --out runs/latency-warmup --compact
```

```bash
python -m decision_index run --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --rows runs/latency-rows.jsonl --out runs/latency --compact
```

```bash
python scripts/latency_report.py runs/latency/results.jsonl --json runs/latency/latency.json
```

The report prints median, p80, p95 and mean overall and per benchmark and whether median, mean and p80
are all within 1,000 ms. Keep the GPU otherwise idle while timing. For reference, release 1.2 on one RTX
PRO 6000 passed this check with p50 171 / p80 533 / mean 421 ms, and its full 0.3 run gave p50 171 / p80 587 /
mean 437 ms.

## 5. Full run

The complete 0.3 suite, sequentially, through the same endpoint:

```bash
python -m decision_index pipeline --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --out runs/full --compact
```

`pipeline` runs, scores and writes `scores.json` with `complete: true` when every request has an answer.
Rerunning the same command resumes: answered rows are kept and errors are retried. At about 0.44 s per
request this is about 17 hours on one RTX PRO 6000. Copy `runs/full` off the machine regularly; cloud
instances lose their disk when deleted.

To finish sooner, split the suite into group-disjoint row files and run one service plus one sequential
runner per GPU with `run --rows`, then concatenate the `results.jsonl` files and score them together:

```bash
python -m decision_index score --edition 0.3 --suite-dir suite-0.3 --results runs/merged/results.jsonl --engine http --out runs/merged
```

## 6. Submit

Follow the kit's [instructions](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).
The run must be complete (`scores.json` says `"complete": true`) and untouched: the maintainers re-score
`results.jsonl`. Upload the run's files unchanged to `runs/<name>/` of a Hugging Face dataset:

```bash
hf upload <you>/<dataset> runs/full runs/<name> --repo-type dataset --include results.jsonl --include scores.json --include index.json --include benchmark-summary.json --include environment.json --include status.json
```

The maintainers must be able to open it: a private dataset gives them a 404. `results.jsonl` contains
benchmark text, including gated HLE items, so check the source datasets' terms before choosing public or
gated access.

Then open a pull request to the kit adding a row to `submissions/README.md`: the model name, the model repo
(`google/gemma-4-31B-it` @ `842da37`, or NVIDIA's NVFP4 quantization of it), the results link
(`runs/<name>/scores.json`), the kit commit and engine (`http` → this repository at its commit), the
hardware, and the settings (`docker compose up`: `CHECKPOINT=bf16`, default effort `high`). After review, the
maintainers start this service themselves, run the private parts of the Full score on it, compare answers on
their own sample and measure latency on one RTX PRO 6000.

Declare the capacity limits: text only; up to 65,536 tokens per branch; 1–255 Choice options and 2–10 Score
levels; inputs never truncated and options never filtered.

Disclose (see [method.md](method.md#disclosures)): an inference technique over a stock checkpoint; the
option-count temperature was fitted on the training splits of WANLI, ARC-Challenge, BANKING77 and CLINC150
with every Decision Index item removed (other splits of two index datasets); the method and the serving
defaults were selected on Decision Index runs (development subsets and full runs).

## Reading the scores

`scores.json` holds the headline `decision_index` (the 0.3 public index, 0–100), area skills and each
benchmark's native metric and chance-corrected skill, plus the run's latency percentiles (`latency_ms`).
Compare runs only on the same edition and the same complete set of requests.

## Edition 0.2.1

The 0.2.1 numbers in [method.md](method.md) came from the same steps with the kit at commit
`87d4650b42b377c0291a89c1f1a879f9b31082bf`:

```bash
pip install "decision-index[rebuild] @ https://github.com/apolinario/decision-index/archive/87d4650b42b377c0291a89c1f1a879f9b31082bf.tar.gz"
```

That kit reads the suite from `./suite-0.2`: build it with `python -m decision_index suite rebuild` (or
`suite import --rows selected-rows.jsonl.gz --added-rows added-rows.jsonl.gz`), check it with
`suite verify`, and drop `--edition 0.3 --suite-dir suite-0.3` from the kit commands. A full 0.2.1 run has
150,317 scoreable requests. `scripts/samples.py` and `scripts/latency_report.py` follow 0.3.

## Cheaper checks

- **Without a GPU.** `MORJ_TOKENIZER=<google/gemma-4-31B-it directory> python -m pytest tests/test_golden.py
  tests/test_prompt.py` checks that this code builds the same token branches, gate decisions and answers as
  the evaluated runs ([deploy.md](deploy.md#tests)).
- **On a sample.** Answer the same 2,000 rows from `decision_index suite sample` on your deployment and on a
  reference deployment, then compare them question by question with `scripts/compare_runs.py`
  ([step 3](#3-decisions-against-the-reference-run)).

## What to expect

- **Same code, same GPU type:** the same decisions; the C math library can change the last bit of a
  probability across machines, never a decision.
- **Another GPU:** release 1.0 scored 58.68 on two H100s and 58.67 on one RTX PRO 6000 (0.2.1, timeouts
  included). On 2,001 sampled 0.3 requests, an H100 gave the same decisions as the published RTX PRO 6000
  run on 99.81% of questions.
- **Scheduling changes:** release 1.2's scheduling (32 branch slots, shared-prefix admission, resized KV
  pools) gave the same decisions as release 1.0 on 99.91% of answers (58.71 against 58.68).
- **NVFP4 against BF16:** 1.01 lower, mostly Knowledge, Language and Tools.
- **Cloud stalls:** release 1.0's runs on a cloud RTX PRO 6000 saw the service stop answering for 8 to 105
  minutes and then continue without an error; release 1.2's run there had none. Keep the client timeout and
  rerun `pipeline` to retry errors. `scripts/hang_watch.sh <container> <url> <out dir>` records every
  thread's stack if the service stops answering.

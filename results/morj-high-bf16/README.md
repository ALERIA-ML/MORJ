# MORJ on one RTX PRO 6000 (Decision Index 0.3)

Decision Index **0.3**, public part (the board edition since 2026-10-06; the board ranks by a Full score of 20% this public part, 50% private tests of the same skills and 30% private new-domain tasks, which only the maintainers run), scored with the official kit (`decision_index score`) at commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`.
Release 1.2 served with `docker compose up` on the board's latency card and scored through `POST /v1/systemone` by the official kit, one request at a time: a fresh run of all 140,620 requests of the edition, following [docs/evaluation.md](../../docs/evaluation.md) after its compatibility pass (180 of 180 ok) and latency check. This is the submitted run, published as `runs/morj-high-bf16`. The code in this repository is the same release with the package renamed from `gemma_decisions` to `morj`; prompts, method and answers are unchanged.

| | |
| --- | --- |
| Edition | 0.3 (`decision-index-0.3`) |
| Decision Index | **59.09** |
| Raw index / breadth skill | 68.88 / 58.27 |
| Requests | 140,178 scoreable, 140,178 ok, `complete: true` (140,620 sent, 0 errors) |
| Corpus | base `b2b56d6fb636…`, added `7429f3c9cddd…`, rebuilt GSM8K `759334894858…`, 442 official exclusions |
| Sequential HTTP latency | p50 171.0 ms, p80 586.9 ms, p95 1,395.5 ms, mean 437.3 ms over 140,620 answered requests ([`latency.json`](latency.json)); slowest request 15.9 s (ToolRet) |
| Hardware | One NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (600 W power limit, driver 580.126.09, CUDA 13.0) on a cloud instance; one server and one sequential client |
| Served | Release 1.2 before the rename (source folder `gemma-cd-bf16`, image `gemma-cd-bf16:1.2`), served model name `gemma-4-31b-trt`, default effort `high@1` |
| Checkpoint | `google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16 |
| Client | kit HTTP engine, runner `f4c9f36c4c55…`, kit default timeout |
| Ran | 2026-10-07T10:52Z to 2026-10-08T03:57Z (17.1 hours) |
| Scored | 2026-10-08T03:57:49+00:00 |

No stalls: `scripts/hang_watch.sh` ran for the whole run and captured nothing after the service was ready.

**Same answers from this repository's code.** On 2,001 requests sampled from this edition
(`decision_index suite sample --n 2000 --seed 20261002`), MORJ 1.2.1 and the release 1.2 image that served
this run, each on one H100 PCIe 80 GB, gave the same decision on all 17,066 questions. Each agrees with this
run on 99.81% of them: the remaining 33 are numerics of H100 against Blackwell, the same for both.

## Latency check

Before the full run, the same service answered the runbook's latency sample: 750 rows stratified by benchmark and prompt length (`samples.py latency`, fixed seed) after 10 untimed warm-up rows, one at a time. 750 of 750 answered: p50 171.3 ms, p80 532.8 ms, p95 1,403.4 ms, mean 421.3 ms. That passes the 0.3 gate (median, mean and p80 each at most 1,000 ms). The full run above passes it too. The maintainers time their own unpublished sample. Files: [`latency-check/`](latency-check/).

## Areas

| Area | Benchmarks | Skill |
| --- | --- | --- |
| Knowledge & Reasoning | 11 | 46.35 |
| Language Understanding | 9 | 60.45 |
| Retrieval & Classification | 6 | 63.99 |
| Tools & Automation | 5 | 76.34 |
| Arts & Human Taste | 6 | 47.18 |

## Benchmarks

Native metric on its own scale and the chance-corrected skill that enters the index.

| ID | Benchmark | Metric | Requests | Score | Skill | In index | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BFCL | case exact accuracy | 1,694 | 0.9787 | 97.13 | yes | 350.8 |
| 2 | ToolRet | nDCG@10 | 685 | 0.6534 | 59.97 | yes | 2,567.7 |
| 3 | API-Bank | accuracy | 508 | 0.8543 | 85.15 | yes | 1,753.2 |
| 4 | BANKING77 | macro-F1 | 3,080 | 0.7927 | 79.00 | yes | 698.0 |
| 5 | CLINC150+OOS | macro-F1 | 5,500 | 0.9195 | 91.90 | yes | 713.5 |
| 6 | RouterBench | selected quality (quality objective) | 10,000 | 0.8002 | 53.15 | no | 779.7 |
| 9 | Home appliance simulator | case exact accuracy | 88 | 0.6364 | 63.64 | yes | 1,523.8 |
| 10 | SGD/SGD-X | macro-F1 | 2,500 | 0.0312 | 0.00 | no | 405.8 |
| 11 | ContractNLI | macro-F1 | 123 | 0.7382 | 62.15 | yes | 1,074.2 |
| 12 | ANLI | macro-F1 | 3,200 | 0.7340 | 60.16 | yes | 110.0 |
| 20 | BPoMP | accuracy | 5,000 | 0.9790 | 95.97 | yes | 170.3 |
| 21 | Humicroedit | accuracy | 2,628 | 0.6579 | 31.58 | yes | 155.2 |
| 22 | POP909-CL | accuracy | 2,000 | 0.2810 | 27.07 | yes | 4,460.7 |
| 23 | cfcolor | accuracy | 5,000 | 0.6502 | 29.52 | yes | 287.9 |
| 24 | MMLU | accuracy | 14,033 | 0.8671 | 82.28 | no | 106.0 |
| 25 | GPQA Diamond | accuracy | 196 | 0.5000 | 33.33 | yes | 151.0 |
| 26 | ARC-Easy | accuracy | 2,376 | 0.9899 | 98.65 | no | 103.2 |
| 27 | ARC-Challenge | accuracy | 1,172 | 0.9770 | 96.93 | no | 103.2 |
| 28 | WinoGrande | accuracy | 1,267 | 0.8390 | 67.80 | yes | 112.6 |
| 29 | HellaSwag | accuracy | 10,042 | 0.9294 | 90.59 | yes | 155.9 |
| 30 | GSM8K | accuracy | 2,638 | 0.7472 | 69.84 | yes | 114.3 |
| 31 | ChessBench | accuracy | 5,000 | 0.2402 | 17.25 | yes | 700.2 |
| 32 | MuSR | accuracy | 752 | 0.6449 | 43.55 | yes | 397.3 |
| 33 | SATA-Bench | case exact accuracy | 1,650 | 0.2891 | 27.97 | yes | 473.2 |
| 34 | SimpleBench | accuracy | 10 | 0.5000 | 40.00 | no | 114.0 |
| 36 | BRIGHT | nDCG@10 | 220 | 0.4720 | 40.27 | yes | 3,220.9 |
| 37 | Amazon ESCI | macro-F1 | 5,000 | 0.5346 | 41.62 | yes | 149.2 |
| 38 | ACOS | per-review F1 | 1,565 | 0.1959 | 17.02 | yes | 2,938.4 |
| 39 | FinEntity | macro-F1 | 979 | 0.9109 | 86.90 | yes | 162.3 |
| 40 | iSarcasmEval | Sarcasm F1 · track A, English | 4,600 | 0.5469 | 41.70 | yes | 106.2 |
| 41 | VAST | macro-F1 | 3,006 | 0.7342 | 60.14 | yes | 110.2 |
| 42 | NLI4CT | macro-F1 | 5,500 | 0.8254 | 66.03 | yes | 286.1 |
| 43 | CRUXEval | accuracy | 570 | 0.8000 | 68.27 | yes | 125.2 |
| 44 | CLadder | accuracy | 5,000 | 0.7448 | 48.96 | yes | 122.0 |
| 45 | HLE | accuracy | 501 | 0.1198 | 0.00 | yes | 205.6 |
| 50 | Habermas Machine | accuracy | 1,676 | 0.4994 | 27.35 | yes | 342.6 |
| 56 | PhishNChips phishing decisions | accuracy | 2,000 | 0.8765 | 75.30 | yes | 476.6 |
| 57 | MMLU-Pro | accuracy | 12,032 | 0.7123 | 67.64 | yes | 153.7 |
| 58 | BBH fixed-option tasks | accuracy | 5,507 | 0.7574 | 64.84 | yes | 114.7 |
| 59 | RAGTruth response-level hallucination | F1 on hallucinated class | 2,700 | 0.7751 | 53.37 | yes | 284.1 |
| 61 | HoVer claim verification | accuracy | 4,000 | 0.7600 | 52.00 | yes | 200.7 |
| 62 | When2Call MCQ | accuracy | 3,652 | 0.7741 | 69.88 | yes | 249.6 |
| 64 | New Yorker caption matching | accuracy | 528 | 0.7727 | 71.59 | yes | 114.6 |

## Files

The JSON files are the kit's original output; `latency.json` is derived from the predictions with [`scripts/latency_report.py`](../../scripts/latency_report.py).

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `benchmark-summary.json` | 35,234 | `96636af182d88121edbe498f78e703f734e8373282e810d1b88707d41518538b` |
| `environment.json` | 981 | `ecc83f8f5d8cd827c2214d88e116023232c3c62783c4b7fe28d96e2d388f3774` |
| `index.json` | 10,027 | `aa99983b49bf4c441171453a210c1caac90faebabe35a4b175ca540f39a3fa15` |
| `latency-check/environment.json` | 845 | `2bef67e4da9465277019d3fd8a010c6905aa27342576bdae1ef3153846364151` |
| `latency-check/latency.json` | 4,320 | `eaa6f2b3423b3a01d50f8b904b3819c1f1c99352094fab6546cbc6601fc649d2` |
| `latency-check/status.json` | 151 | `c3b593c2ae9a83dba4d149236b94384fde0e2bf673349ff311c11fc854677c12` |
| `latency.json` | 5,658 | `bfdc29590d622f2a56b8b4f1b87af6096f26113b8095fa9e9011d18f5e109298` |
| `scores.json` | 32,085 | `8aa14706f4fe3b3201a5a59ca27bb95f6b069612193aad82522e47c89c73c9df` |
| `status.json` | 159 | `7c92d2388b1ecae37c1f8889d5665c1e6ff84969a4234c8db141f7dc0c50f709` |
| `results.jsonl` (predictions; not in git because it contains benchmark text) | 205,291,799 | `9a35a1ae9bec3a7eb238b922ca16bae5372fc7849c626f246c1b42fc9dd21e32` |

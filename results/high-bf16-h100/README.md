# MORJ 1.0.0 on H100 PCIe (Decision Index 0.3)

Decision Index **0.3**, public part (the board edition since 2026-10-06; the board ranks by a Full score of 20% this public part, 50% private tests of the same skills and 30% private new-domain tasks, which only the maintainers run), scored with the official kit (`decision_index score`) at commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`.
This repository at release commit `33b5501` (MORJ 1.0.0) with every setting at its default (BF16, default effort `high`; only `MORJ_PORT` set per shard), scored through `POST /v1/systemone` by the official kit: all 140,620 requests of the edition in eight shards, one server and one sequential client per GPU. It is a verification run on H100s, not on the board's latency card: the RTX PRO 6000 run of this release is still to come.

| | |
| --- | --- |
| Edition | 0.3 (`decision-index-0.3`) |
| Decision Index | **59.15** |
| Raw index / breadth skill | 68.93 / 58.35 |
| Requests | 140,178 scoreable, 140,178 ok, `complete: true` (140,620 sent, 0 errors) |
| Corpus | base `b2b56d6fb636…`, added `7429f3c9cddd…`, rebuilt GSM8K `759334894858…`, 442 official exclusions |
| Sequential HTTP latency | p50 161.5 ms, p80 441.3 ms, p95 1,094.2 ms, mean 408.4 ms over 140,620 answered requests, eight shards in parallel ([`latency.json`](latency.json)) |
| Hardware | Eight NVIDIA H100 PCIe 80 GB in one server (350 W power limit, driver 595.91.07, CUDA 13.2); one server process and one sequential client per GPU |
| Served | `git archive` of commit `33b55010b03c811ac9a4e2ea32f50a548a219069`, built as image `sha256:38374c1f142a…` and started with no settings but the port; served model name `gemma-4-31b-trt`, default effort `high`, no host KV tier |
| Checkpoint | `google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16 |
| Client | kit HTTP engine, runner `f4c9f36c4c55…`, kit default timeout; rows split round-robin into eight shards of 17,577 or 17,578 |
| Ran | 2026-10-09T08:57Z to 2026-10-09T11:27Z (2.5 hours on eight GPUs) |
| Scored | 2026-10-09T11:36:15+00:00 |

No stalls: every shard answered all of its rows in one pass, and the scheduler's stall guard never fired.

**Same answers as the pre-release source.** The pre-release source of this code (package `gemma_decisions`) scored 59.16 on the same edition and GPU type the day before (six H100 shards). Against that run, request by request, this release gave the same decision on 323,254 of 323,259 questions (99.998%), and 139,643 of 140,620 responses (99.31%) were byte-identical; the others differ in probability digits. Every benchmark score is the same except BRIGHT (nDCG@10 0.4636 against 0.4643, one decision) and ToolRet (0.6536 against 0.6535), which together make the 0.01 between the two indexes.

**Latency.** These are H100s, with the eight shards running at the same time. The board's gate (median, mean and p80 each at most 1,000 ms) is measured on one RTX PRO 6000 and has not been run for this release yet.

## Areas

| Area | Benchmarks | Skill |
| --- | --- | --- |
| Knowledge & Reasoning | 11 | 46.56 |
| Language Understanding | 9 | 60.56 |
| Retrieval & Classification | 6 | 63.88 |
| Tools & Automation | 5 | 76.33 |
| Arts & Human Taste | 6 | 47.24 |

## Benchmarks

Native metric on its own scale and the chance-corrected skill that enters the index.

| ID | Benchmark | Metric | Requests | Score | Skill | In index | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BFCL | case exact accuracy | 1,694 | 0.9787 | 97.13 | yes | 284.3 |
| 2 | ToolRet | nDCG@10 | 685 | 0.6536 | 59.99 | yes | 2,382.0 |
| 3 | API-Bank | accuracy | 508 | 0.8543 | 85.15 | yes | 1,613.5 |
| 4 | BANKING77 | macro-F1 | 3,080 | 0.7929 | 79.02 | yes | 396.9 |
| 5 | CLINC150+OOS | macro-F1 | 5,500 | 0.9195 | 91.90 | yes | 596.3 |
| 6 | RouterBench | selected quality (quality objective) | 10,000 | 0.8003 | 53.18 | no | 655.1 |
| 9 | Home appliance simulator | case exact accuracy | 88 | 0.6364 | 63.64 | yes | 1,145.0 |
| 10 | SGD/SGD-X | macro-F1 | 2,500 | 0.0293 | 0.00 | no | 343.7 |
| 11 | ContractNLI | macro-F1 | 123 | 0.7379 | 62.10 | yes | 899.7 |
| 12 | ANLI | macro-F1 | 3,200 | 0.7359 | 60.44 | yes | 107.5 |
| 20 | BPoMP | accuracy | 5,000 | 0.9790 | 95.97 | yes | 159.5 |
| 21 | Humicroedit | accuracy | 2,628 | 0.6594 | 31.89 | yes | 159.1 |
| 22 | POP909-CL | accuracy | 2,000 | 0.2790 | 26.59 | yes | 4,983.2 |
| 23 | cfcolor | accuracy | 5,000 | 0.6502 | 29.59 | yes | 207.9 |
| 24 | MMLU | accuracy | 14,033 | 0.8672 | 82.29 | no | 107.7 |
| 25 | GPQA Diamond | accuracy | 196 | 0.5204 | 36.05 | yes | 118.2 |
| 26 | ARC-Easy | accuracy | 2,376 | 0.9899 | 98.65 | no | 107.2 |
| 27 | ARC-Challenge | accuracy | 1,172 | 0.9770 | 96.93 | no | 107.6 |
| 28 | WinoGrande | accuracy | 1,267 | 0.8374 | 67.48 | yes | 108.9 |
| 29 | HellaSwag | accuracy | 10,042 | 0.9296 | 90.61 | yes | 112.1 |
| 30 | GSM8K | accuracy | 2,638 | 0.7502 | 70.23 | yes | 109.5 |
| 31 | ChessBench | accuracy | 5,000 | 0.2410 | 17.33 | yes | 444.2 |
| 32 | MuSR | accuracy | 752 | 0.6436 | 43.34 | yes | 256.1 |
| 33 | SATA-Bench | case exact accuracy | 1,650 | 0.2867 | 27.72 | yes | 424.7 |
| 34 | SimpleBench | accuracy | 10 | 0.5000 | 40.00 | no | 111.0 |
| 36 | BRIGHT | nDCG@10 | 220 | 0.4636 | 39.32 | yes | 3,110.6 |
| 37 | Amazon ESCI | macro-F1 | 5,000 | 0.5367 | 41.89 | yes | 128.1 |
| 38 | ACOS | per-review F1 | 1,565 | 0.1944 | 16.86 | yes | 2,855.0 |
| 39 | FinEntity | macro-F1 | 979 | 0.9109 | 86.90 | yes | 160.1 |
| 40 | iSarcasmEval | Sarcasm F1 · track A, English | 4,600 | 0.5466 | 41.67 | yes | 112.2 |
| 41 | VAST | macro-F1 | 3,006 | 0.7349 | 60.24 | yes | 108.7 |
| 42 | NLI4CT | macro-F1 | 5,500 | 0.8251 | 65.97 | yes | 224.1 |
| 43 | CRUXEval | accuracy | 570 | 0.7930 | 67.15 | yes | 110.4 |
| 44 | CLadder | accuracy | 5,000 | 0.7464 | 49.28 | yes | 110.6 |
| 45 | HLE | accuracy | 501 | 0.1257 | 0.00 | yes | 164.5 |
| 50 | Habermas Machine | accuracy | 1,676 | 0.4994 | 27.35 | yes | 276.1 |
| 56 | PhishNChips phishing decisions | accuracy | 2,000 | 0.8765 | 75.30 | yes | 446.5 |
| 57 | MMLU-Pro | accuracy | 12,032 | 0.7128 | 67.70 | yes | 113.3 |
| 58 | BBH fixed-option tasks | accuracy | 5,507 | 0.7596 | 65.15 | yes | 110.0 |
| 59 | RAGTruth response-level hallucination | F1 on hallucinated class | 2,700 | 0.7794 | 54.26 | yes | 251.1 |
| 61 | HoVer claim verification | accuracy | 4,000 | 0.7605 | 52.10 | yes | 189.4 |
| 62 | When2Call MCQ | accuracy | 3,652 | 0.7735 | 69.80 | yes | 223.9 |
| 64 | New Yorker caption matching | accuracy | 528 | 0.7765 | 72.06 | yes | 108.0 |

## Files

The JSON files are the kit's original output: `scores.json`, `index.json` and `benchmark-summary.json` from scoring the eight shards' `results.jsonl` concatenated in shard order, and each shard's `environment.json` and `status.json` under `shards/`. `latency.json` is derived from the predictions (per-request `total_wall_ms`: median, p80, p95 and mean, overall and per benchmark).

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `benchmark-summary.json` | 35,335 | `2210b48fd66f4df428824eed872041eab15d243e3d2c2c2abaf783573d243faf` |
| `index.json` | 10,037 | `7133e5da6c7081262cd4a675d4f5b1d6e5fa202b91b495b5ccdfe56ab4caf2b9` |
| `latency.json` | 5,654 | `72ecf2f6e537da7fe1d4c56ae5e76db94987b47c33b4a115119511c88aa4a597` |
| `scores.json` | 32,114 | `8c096f940c35e406f48550fd7dca41866c836c44edc300f50681019a8bd1b2e1` |
| `shards/s0/environment.json` | 872 | `5d414a81168d02c5bcc059416e472da156ac7d4e2e8632aa146b3cab312b6d5f` |
| `shards/s0/status.json` | 156 | `853debe66010c3fccda0c78114226d4e96ed2ea481ebf7f7080a9794bd4d6c36` |
| `shards/s1/environment.json` | 871 | `664af08a5b57f7dd4d0dfeacad75cf16bcf85181d8c71e9e6e6700168026d6ed` |
| `shards/s1/status.json` | 156 | `affb5289676863915a9655dc3c60941f5099643feb89c35616981faa0e305d47` |
| `shards/s2/environment.json` | 871 | `301d91152312aff7207408c7ffbd4df84f33e932a896023f978dca46f06b76cc` |
| `shards/s2/status.json` | 156 | `e7097994ad8c22e7a0bba1a444fdfb3ed163036a5ca6497eb64e60a187457a81` |
| `shards/s3/environment.json` | 871 | `b54fb747e32bec56d64b94ef1f6b109d3f0b5d30ce254a6baa7fe7a087521146` |
| `shards/s3/status.json` | 156 | `71c9f58d93667c26e2f050d4be1386e4be1955d64127101b3ac09a5a64b7fc58` |
| `shards/s4/environment.json` | 872 | `5cdeede3740fb17415b31135f239e0ace82bce3ff1267fb9d382b88c73c70feb` |
| `shards/s4/status.json` | 156 | `0bce5d8d8d02f304821172ebe4bc47c52b6eb7f43234f412ba81c344ea069275` |
| `shards/s5/environment.json` | 871 | `fd758383836f325b49ae9b46938b156a086d4f763b2b83a640b0736c0a0afcc7` |
| `shards/s5/status.json` | 156 | `21819f3b9a4625510815f96c6783be9b89126868564fca90e170c9342d56ad8f` |
| `shards/s6/environment.json` | 872 | `199fb2a86a8de8b77088214a98416473f7bdfde848cbac892cdcff9e824608f3` |
| `shards/s6/status.json` | 156 | `792e5e1ab4684e88a6127ef1d0f505ab3504c45ecf89b8daa3eadd2c2077b272` |
| `shards/s7/environment.json` | 871 | `9ba10eb6ef831e84f3b8bf7de27f92cbffba9db64e8b02d925e5d21e87aa9872` |
| `shards/s7/status.json` | 156 | `e2d34d0a65d419b5e73688baac290263f90aa3a24c2ac39274dffc906b73db8a` |
| `results.jsonl` (predictions; not in git because it contains benchmark text) | 206,567,636 | `bb96239311dfb68e40686894d749b8840304f6948a182bc5838cde6eee2138d8` |

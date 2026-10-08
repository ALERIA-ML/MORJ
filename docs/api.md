# API

MORJ speaks the `POST /v1/systemone` decision API of the [Decision Index](https://github.com/apolinario/decision-index):
one request carries a `state` (any text or JSON) and one or more typed `questions`, and every question gets
an answer with probabilities. The kit's `http` engine works unchanged.

## Endpoints

| Endpoint | |
| --- | --- |
| `POST /v1/systemone` | Decisions: `state` plus typed `questions` |
| `POST /v1/trace` | The same input; the same answers plus the engine's record (see [Playground](#playground)). Off with `MORJ_TRACE=false` |
| `GET /` | The playground |
| `GET /health` | 503 while starting (the service's own probe request must replay a prefill CUDA graph first) or draining; then the method, checkpoint, default effort and graph counters |
| `GET /live` | Liveness |
| `GET /metrics` | Prometheus metrics (`morj_*`) |
| `GET /stats` | JSON counters, including the KV budget, free branch slots and graph replays |
| `GET /v1/models` | The served model in vLLM's OpenAI list format: `id`, `root` (the checkpoint) and `max_model_len` (tokens per branch) |

With `API_KEY` set, `/v1/*` and `/stats` need `Authorization: Bearer <key>`.

## Request

```json
{
  "state": {
    "customer": "Dana Ortiz",
    "plan": "enterprise",
    "messages": [
      {"from": "customer", "text": "Our nightly CSV export has failed three nights in a row with \"timeout after 300s\". Finance needs the file by 9am for month-end close."},
      {"from": "agent", "text": "Thanks, I can see the job retried twice. Can you confirm the export size?"},
      {"from": "customer", "text": "About 4 GB. It worked last month. If this is not fixed today we will have to escalate to our account manager."}
    ]
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should own this ticket?",
      "criteria": {
        "billing": "Invoices, payments, refunds",
        "bug": "A product defect or regression",
        "howto": "Usage question with a documented answer",
        "infra": "Capacity, timeouts or performance of hosted jobs",
        "sales": "Plan changes or commercial terms"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the ticket?",
      "criteria": ["Can wait a week", "This week", "Today", "Within the hour"]
    },
    "escalate": {
      "type": "noul",
      "instructions": "Should a human manager be looped in now?"
    }
  }
}
```

| Question type | `criteria` | Answer |
| --- | --- | --- |
| `choice` | 1–255 named options, each with an optional description | `choice`, `probabilities` over the option names, `confidence = (p_best − 1/n) / (1 − 1/n)` |
| `score` | 2–10 ordered levels | `score` (the expected level), `probabilities` per level, a `legend` and a distance-based `confidence` |
| `noul` | optional descriptions of `true` and `false` | `noul`: P(true) |

`instructions` is optional text for the question. Inputs are never truncated and options are never dropped:
a question whose prompt would exceed 65,536 tokens is rejected with 422. Images are rejected (text only).

## Response

The answers to the request above, as recorded on one H100 (the full record is
[`traces/support-ticket.json`](traces/support-ticket.json)):

```json
{
  "answers": {
    "route": {
      "type": "choice",
      "choice": "infra",
      "probabilities": {"billing": 0.0133, "bug": 0.0189, "howto": 0.0167, "infra": 0.9357, "sales": 0.0154},
      "confidence": 0.9197
    },
    "urgency": {
      "type": "score",
      "score": 1.956,
      "probabilities": {"0": 0.0271, "1": 0.0309, "2": 0.9008, "3": 0.0412},
      "legend": {"0": "Can wait a week", "1": "This week", "2": "Today", "3": "Within the hour"},
      "confidence": 0.8737
    },
    "escalate": {"type": "noul", "noul": 0.9381}
  },
  "usage": {"input_tokens": 1393, "output_tokens": 0, "effort": "high@1", "branches": 7}
}
```

Probabilities are rounded here. `output_tokens` is always 0: nothing is generated. `usage.effort` names the
profile that ran and `usage.branches` the number of prompts scored (the recorded trace predates these two
fields; the values shown are what the request gets today).

## Effort

Each request can choose how many views of each question the engine scores:

| `effort` | Profile | What each question gets | Branches / question |
| --- | --- | --- | --- |
| `low` | `low@1` | one candidate order, T = 1 | 1.0 |
| `medium` | `medium@1` | two orders; nine-digit rating on truth questions; option-count temperature elsewhere | 2.8 |
| `high` (default) | `high@1` | `medium` + rating of other two-option questions in both orders + two more orders where the first two disagree | 3.0 |

Branch counts are averages over the Decision Index. Choose the effort with a header or a field; a request
with neither gets the service default (`DEFAULT_EFFORT`, `high` unless changed):

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -H 'X-Decision-Effort: low' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}}}'
```

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "options": {"effort": "medium"}, "questions": {"team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

- `low`, `medium` and `high` name the current version of each level; a versioned name (`high@1`) pins one.
  Versioned profiles never change behavior; a better method becomes `high@2`.
- Unknown names, and a header and field that disagree, are rejected with 422 before queueing.
- `options` is MORJ's extension of the request; the Decision Index kit omits it and gets the default.

Index score and latency of each profile: [method.md](method.md#effort-profiles).

## Status codes

| Code | Meaning |
| --- | --- |
| 200 | Answered |
| 401 | Missing or invalid API key |
| 413 | Body over 64 MiB |
| 422 | Invalid input, an unknown effort, or a question branch over 65,536 tokens |
| 499 | Client disconnected |
| 503 | Not ready yet, or draining after a native failure (`Retry-After`) |
| 504 | Deadline exceeded |
| 529 | Queue full (`Retry-After`) |

A request is answered, or refused with 504, within `queue_timeout + request_timeout + 10` = 340 s by
default; give clients a timeout above that.

## Playground

`GET /` serves a playground that runs your request through `POST /v1/trace`. Use the same JSON as
`/v1/systemone`; you get the same answers plus the engine's record:

- **Branches, prefix reuse and timeline.** Every branch's token bar is split into template, state,
  question and tail, with the part TensorRT-LLM served from cache hatched over it. A timeline shows the seed
  running alone, the concurrent branches and the gated second pass. Click a branch to see the prompt it adds
  after the shared state: options in that order, or the rating instruction.
- **Per question.** Each order's distribution mapped back to the original options (raw and softened), the
  gate verdict, the nine-bin histograms with E[rating] → p(A), and how the letters and the rating combine
  into the final answer.

**Save trace** downloads the full record and **Open a saved trace…** shows it later, even from
`morj/playground.html` opened straight from disk without a server. [`traces/`](traces/) holds two traces
recorded on an H100, one of them with a gated second pass.

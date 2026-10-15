# Vendored from Decider (https://github.com/Mapika/decider), Apache-2.0: see LICENSE in this folder.
# Modified for Aleria Judge: only the state and question rendering Judge uses are kept (Decider's
# isolated Score-level rows, answer assembly and confidence formatting are removed; Judge combines answers in
# method.py), and render_question returns only the question text and its options. An option named like one of its
# labels drops that name. JSON states are indented and passed as sent: Decider's "_index" annotation of long arrays
# is removed.
"""Jev-shaped requests on top of the decider prompt format (same wire format as TypeSafe's POST /v1/systemone).

    state      str | dict | list            JSON state is indented by two spaces, as sent; questions may name a part by path (`ticket.messages[0].text`)
    questions  {id: {"type": "choice", "instructions": ..., "criteria": {name: description | {...} | [...] | None}}      up to 255 options
                     {"type": "score",  "instructions": ..., "criteria": [level 0 description, level 1 description, ...]}  2..10 levels
                     {"type": "noul",   "instructions": ... (optional), "criteria": {"true": ..., "false": ...} (optional)}}
    a noul question needs instructions or at least one true/false description.
    ids are never shown to the model.  `instructions` and every description may be a string or any JSON value.
"""
import json

MAX_CHOICE, MAX_LEVELS = 255, 10


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_state(state):
    """A text state as is; a JSON state indented by two spaces, exactly as sent (no fields added)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_question(spec, labels=()):
    """-> dict(question=str, options=[str]): the question text and its rendered options, in request order.

    `labels`: the label names the question is shown with (A, B, ...). An option whose name repeats one of them
    exactly ("B" or "(B)") is shown by its description alone, when it has one: with the options rotated the name
    would contradict its label ("(A) B: no"). A description that is an exact copy of its name is shown once
    ("Neutral", not "Neutral: Neutral"). If either shortening makes two options look alike, both are shown in full
    as "name: description". Every other name is kept."""
    t = spec.get("type", "choice"); crit = spec.get("criteria", spec.get("options"))
    raw = spec.get("instructions", spec.get("question", ""))
    if t in ("noul", "bool") and raw in (None, ""):          # the criteria carry the question (NOUL_WITHOUT_INSTRUCTIONS)
        ins = NOUL_WITHOUT_INSTRUCTIONS
        described = isinstance(crit, dict) and any(d not in (None, "") for d in (crit.get("true", crit.get(True)), crit.get("false", crit.get(False))))
        if not described and (crit is None or isinstance(crit, dict)):     # a non-map is rejected below with the criteria message
            raise ValueError("noul question without instructions: criteria must describe true or false")
    else:
        ins = _txt(raw)
    if not ins:
        raise ValueError("question without instructions")
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE:
            raise ValueError(f"choice criteria: a map of 2..{MAX_CHOICE} options")
        as_label = set(labels) | {f"({x})" for x in labels}
        same = lambda n: crit[n] == n                       # an exact copy only: {"Neutral": "Neutral"}
        opts = [n if crit[n] in (None, "") or same(n) else _txt(crit[n]) if n in as_label else f"{n}: {_txt(crit[n])}"
                for n in crit]
        repeated = {o for o in opts if opts.count(o) > 1}   # shortened options that now look alike: show them in full
        opts = [f"{n}: {_txt(crit[n])}" if o in repeated and crit[n] not in (None, "") else o for n, o in zip(crit, opts)]
    elif t == "score":
        if isinstance(crit, dict):                                   # legend form {"0": "...", "1": "..."}
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions")
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
    elif t in ("noul", "bool"):
        if crit is not None and not isinstance(crit, dict):
            raise ValueError("noul criteria: a map of optional true/false descriptions")
        c = crit if crit is not None else {}
        f, tr = c.get("false", c.get(False)), c.get("true", c.get(True))
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}", "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
    else:
        raise ValueError(f"unknown question type {t!r}")
    return dict(question=ins, options=opts)


# A noul question may omit `instructions` (TypeSafe's OpenAPI file marks it optional).  The question id is never shown to the
# model, so the question text is this fixed sentence and the true/false descriptions, rendered as the options "no: ..." and
# "yes: ...", say what is being asked.  A request that gives instructions is rendered exactly as before.
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"

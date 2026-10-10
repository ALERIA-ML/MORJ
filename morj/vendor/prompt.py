# Vendored from Decider (https://github.com/Mapika/decider), Apache-2.0: see LICENSE in this folder.
# Modified for MORJ: build_chat accepts precomputed context_ids (the state is tokenized once per request), the
# option-token cache is capped at 4,096 entries, options are never filtered, and only the state-first chat layout
# MORJ uses is kept (the plain and schema-first layouts, Decider's config loading and helpers are removed).
"""Prompt construction in Decider's state-first chat layout: the state, the question blocks with single-token
option labels, then one "Answer: (" slot per question, whose next-token logits are read over the labels."""
import random

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)      # <= NARROW options: the original "(A) .. (J)" rendering, tokenized as a string (unchanged since v1)
MAX_OPTIONS = 255          # width of the label head. > NARROW options: "wide" rendering, one label token per option:
                           # A..Z then the first 229 two-letter upper-case strings that are single tokens (AA, AB, ...)


_LABELS = {}
_OPT_CACHE = {}


def _enc_opt(tok, text):
    """Token ids of ") <option text>" (cached: fixed label sets repeat the same strings millions of times)."""
    key = (id(tok), text)
    v = _OPT_CACHE.get(key)
    if v is None:
        v = tok.encode(f") {text}", add_special_tokens=False)
        if len(_OPT_CACHE) < 4096:
            _OPT_CACHE[key] = v
    return v


def label_table(tok):
    """(label strings, label token ids), MAX_OPTIONS entries; the first NARROW are A..J so narrow questions are unchanged."""
    key = id(tok)
    if key not in _LABELS:
        import string
        U = string.ascii_uppercase
        names = list(U) + [a + b for a in U for b in U]
        out = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                out.append((n, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        assert len(out) == MAX_OPTIONS and len({i for _, i in out}) == MAX_OPTIONS
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out],
                        tok.encode("\n(", add_special_tokens=False))
    return _LABELS[key]


def _select(q, rng, max_options):
    opts = list(range(len(q.options)))
    if len(opts) > max_options:
        raise ValueError(f'{len(opts)} options exceed the {max_options}-label head')
    rng.shuffle(opts)
    return opts


def _options_ids(tok, q, opts):
    if len(opts) <= NARROW:
        return tok.encode("".join(f"\n({LETTERS[j]}) {q.options[oi]}" for j, oi in enumerate(opts)), add_special_tokens=False)
    _, lab_ids, open_ids = label_table(tok); out = []
    for j, oi in enumerate(opts):
        out += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])
    return out


# ---- chat layout -------------------------------------------------------------------------------------------------------
# Every prompt is wrapped in the tokenizer's chat template: one user turn holding "Context:\n<state>" and the question block
# ("\n\nQuestion: <q>\nOptions:\n(A) ..\n(B) .."), thinking switched off, then "Answer: (" in the assistant turn. Every
# part is tokenized on its own (head, context, question header, option block, tail, answer piece); the label is read at the
# final " (" token.

SYSTEM_LINE = ("You are a decision function. You will be given a context and one question with labelled options. "
               "Reply as \"Answer: (X)\" with the label X of the best option or, when asked for a rating, as "
               "\"Rating: N\" with one digit N. No other words, no explanation.")


class ChatTemplate:
    """Token ids of the tokenizer's chat template around one user turn, and the answer pieces of the assistant turn.

    head/tail come from tok.apply_chat_template([user turn], add_generation_prompt=True) with thinking disabled
    (enable_thinking=False where the template knows the switch; a thinking block the template leaves open is closed here),
    with SYSTEM_LINE as the system turn.  This is the computation of the chat-layout research server the chat models were trained
    against, plus that system turn."""
    SENTINEL = "@@DECIDER_USER_CONTENT@@"

    def __init__(self, tok, answer="Answer"):
        if not getattr(tok, "chat_template", None):
            raise ValueError('the model is configured for the chat layout ("layout": "chat") but its tokenizer has no chat template')
        msgs = [{"role": "system", "content": SYSTEM_LINE}, {"role": "user", "content": self.SENTINEL}]
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        head, tail = s.split(self.SENTINEL)
        for opened, closed in (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>")):
            if tail.endswith(opened):
                tail += closed
        self.head_text, self.tail_text = head, tail
        self.head = tok.encode(head, add_special_tokens=False)
        self.tail = tok.encode(tail, add_special_tokens=False)
        self.answer = answer
        names, ids, _ = label_table(tok)
        pre = tok.encode(f"{answer}: (", add_special_tokens=False)
        for n, i in list(zip(names, ids))[:NARROW]:
            if tok.encode(f"{answer}: ({n}", add_special_tokens=False) != pre + [i]:
                raise ValueError(f"chat layout: the label {n!r} does not tokenize as one token after {answer!r}: (")
        if tok.encode(tail + f"{answer}: (", add_special_tokens=False) != self.tail + pre:
            raise ValueError("chat layout: the chat template tail merges with the answer prefix")

    def answer_ids(self, tok, k, multi):
        """Token ids of answer piece k: "Answer: (" (or "Answer k: (" with several questions), later pieces after a newline."""
        return tok.encode(f"{'' if k == 0 else chr(10)}{self.answer}{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)


_CHAT = {}


def chat_template(tok):
    """The ChatTemplate of a tokenizer, cached per tokenizer object."""
    key = id(tok)
    if key not in _CHAT:
        _CHAT[key] = (tok, ChatTemplate(tok))           # keep tok alive so its id is not reused
    return _CHAT[key][1]


def build_chat(example, tok, chat, rng=None, max_options=NARROW, max_ctx_tokens=1536, context_ids=None):
    """State-first chat layout (see the comment above LAYOUTS).  Same return fields as `build`.  max_ctx_tokens caps the
    tokens of "Context:\\n" + state, as in the plain layout; the template head and tail are not counted."""
    rng = rng or random
    qs = example.qs; multi = len(qs) > 1
    context_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False) if context_ids is None else context_ids
    ids = list(chat.head) + context_ids[:max_ctx_tokens]
    golds, nopts, perms = [], [], []
    for k, q in enumerate(qs):
        opts = _select(q, rng, max_options)
        ids += tok.encode(f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False) + _options_ids(tok, q, opts)
        golds.append(opts.index(q.gold) if q.gold in opts else -1); nopts.append(len(opts)); perms.append(opts)
    ids += chat.tail
    slots = []
    for k in range(len(qs)):
        ids += chat.answer_ids(tok, k, multi); slots.append(len(ids) - 1)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)



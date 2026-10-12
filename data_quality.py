"""
data_quality.py - one shared set of quality rules, used by the Dataset Builder (while collecting) and the Data Doctor (when analysing
or cleaning a finished dataset).

check(example)        -> (score 0-100, [flag, ...])        example = {"messages":[...]} | {"text":...} | {prompt,chosen,rejected}
grounded(ans, chunk)  -> bool                              is a generated answer supported by its source passage?
FLAGS                 -> {flag: (penalty, label, drop_by_default)}
"""

import re
import zlib
from collections import Counter

FLAGS = {
    "empty": (100, "Empty content", True),
    "placeholder": (100, "Placeholder answer (N/A, TODO, lorem ipsum…)", True),
    "pair_identical": (100, "Chosen and rejected answers are identical", True),
    "repetition": (70, "Repetitive / looping text", True),
    "echo": (60, "Answer just repeats the question", True),
    "mojibake": (60, "Garbled characters (encoding errors)", True),
    "tiny_response": (40, "Very short answer", False),
    "symbols": (40, "Mostly symbols or numbers", False),
    "language_mismatch": (40, "Question and answer in different languages", False),
    "ai_boilerplate": (35, "AI refusal / 'as an AI language model' boilerplate", True),
    "truncated": (30, "Answer cut off mid-sentence", True),
    "url_heavy": (30, "Mostly links", False),
    "code_fence": (20, "Unclosed code block", False),
    "short_prompt": (20, "Very short question", False),
    "pii": (15, "Contains emails, phone numbers or keys", False),
}

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_SECRET = re.compile(r"\b(?:sk|pk|ghp|gho|xox[bap]|AKIA)[-_A-Za-z0-9]{16,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")
_AI = re.compile(r"\b(as an ai( language model)?|as a large language model|i(?:'m| am) (?:just )?an ai|i cannot (?:assist|help|fulfill)|"
                 r"i can(?:'|no)t (?:assist|help) with|i(?:'m| am) sorry, but i (?:can(?:'|no)t|am unable)|my knowledge cutoff|"
                 r"i do(?: not|n't) have (?:personal|real-time|access to))\b", re.I)
_PLACEHOLDER = re.compile(r"^\W*(n/?a|none|null|nan|todo|tbd|undefined|lorem ipsum.*|\.{2,}|-+|\?+|no (answer|response)|"
                          r"(your )?(answer|response) (goes )?here)\W*$", re.I | re.S)
_MOJI = re.compile("(Ã[\x80-\xbf]|â€[™œ\x9d\x9c\x93\x94\xa6]|�|ï¿½|Ð[\x80-\xbf]{1,2}Ð)")
_STOP_END = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "with", "is", "that", "which", "by", "on", "as", "at", "from", "are"}
_WORD = re.compile(r"[^\W\d_]{3,}", re.U)


def script_of(text):
    """Dominant writing system of a text: latin, cjk, cyrillic, arabic, hangul, devanagari, greek, other (or None if no letters)."""
    c = Counter()
    for ch in text[:2000]:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250 or 0x1E00 <= o < 0x1F00:
            c["latin"] += 1
        elif 0x400 <= o < 0x530:
            c["cyrillic"] += 1
        elif 0x370 <= o < 0x400:
            c["greek"] += 1
        elif 0x600 <= o < 0x700:
            c["arabic"] += 1
        elif 0x900 <= o < 0x980:
            c["devanagari"] += 1
        elif 0xAC00 <= o < 0xD7B0:
            c["hangul"] += 1
        elif 0x3040 <= o < 0x3100 or 0x4E00 <= o < 0xA000:
            c["cjk"] += 1
        else:
            c["other"] += 1
    return c.most_common(1)[0][0] if c else None


def parts(ex):
    """Normalise any example to {kind, prompt, response, text, rejected}."""
    if not isinstance(ex, dict):
        return {"kind": "text", "text": str(ex)}
    if "messages" in ex and isinstance(ex["messages"], list):
        users = [m.get("content", "") for m in ex["messages"] if m.get("role") == "user"]
        bots = [m.get("content", "") for m in ex["messages"] if m.get("role") == "assistant"]
        return {"kind": "sft", "prompt": users[-1] if users else "", "response": bots[-1] if bots else "",
                "all_user": "\n".join(users)}
    if "chosen" in ex and "rejected" in ex:
        def t(v):
            return v[-1].get("content", "") if isinstance(v, list) and v and isinstance(v[-1], dict) else str(v or "")
        p = ex.get("prompt", "")
        return {"kind": "dpo", "prompt": p if isinstance(p, str) else t(p), "response": t(ex["chosen"]), "rejected": t(ex["rejected"])}
    if "prompt" in ex and "completion" in ex:
        return {"kind": "sft", "prompt": str(ex["prompt"]), "response": str(ex["completion"])}
    if "text" in ex:
        return {"kind": "text", "text": str(ex["text"])}
    return {"kind": "text", "text": " ".join(str(v) for v in ex.values() if isinstance(v, str))}


def _repetitive(text):
    if len(text) >= 300:
        raw = text.encode("utf-8", "ignore")
        if len(zlib.compress(raw)) / max(len(raw), 1) < 0.16:
            return True
    words = text.lower().split()
    if len(words) >= 40:
        tri = Counter(zip(words, words[1:], words[2:]))
        if tri and tri.most_common(1)[0][1] / max(len(words) - 2, 1) > 0.18:
            return True
    return False


def _truncated(resp):
    r = resp.rstrip()
    if len(r) < 120 or "```" in r:
        return False
    if r.endswith(("...", "…")):
        return True
    last = r.rsplit(None, 1)[-1].lower().strip(".,;:!?") if r.split() else ""
    return r[-1] in ",;:" or (r[-1].isalpha() and last in _STOP_END)


def _text_flags(text, is_resp=False):
    f = []
    if not text.strip():
        return ["empty"]
    if _MOJI.search(text):
        f.append("mojibake")
    if _repetitive(text):
        f.append("repetition")
    n = len(text)
    if "```" in text and text.count("```") % 2:
        f.append("code_fence")
    if "```" not in text and n >= 40:
        letters = sum(c.isalpha() for c in text)
        if letters / n < 0.4:
            f.append("symbols")
    urls = sum(len(u) for u in _URL.findall(text))
    if n and urls / n > 0.5:
        f.append("url_heavy")
    if _EMAIL.search(text) or _PHONE.search(text) or _SECRET.search(text):
        f.append("pii")
    return f


def check(ex):
    p = parts(ex)
    flags = []
    if p["kind"] == "text":
        flags += _text_flags(p["text"])
        if _AI.search(p["text"][:400]) and len(p["text"]) < 800:
            flags.append("ai_boilerplate")
    else:
        prompt, resp = p.get("prompt", ""), p.get("response", "")
        if not prompt.strip() or not resp.strip():
            flags.append("empty")
        else:
            flags += [x for x in _text_flags(resp, True) if x != "pii"]
            flags += [x for x in _text_flags(prompt) if x in ("mojibake", "pii")]
            if _PLACEHOLDER.match(resp.strip()):
                flags.append("placeholder")
            elif len(resp.strip()) < 15 and p["kind"] == "sft":
                flags.append("tiny_response")
            if len(prompt.strip()) < 5:
                flags.append("short_prompt")
            if resp.strip().lower() == prompt.strip().lower() or (len(prompt) > 40 and prompt.strip().lower() in resp.lower()
                                                                  and len(resp) < len(prompt) * 1.3):
                flags.append("echo")
            if _AI.search(resp[:300]):
                flags.append("ai_boilerplate")
            if _truncated(resp):
                flags.append("truncated")
            sp, sr = script_of(prompt), script_of(resp)
            if sp and sr and sp != sr and len(resp) > 60 and len(prompt) > 20:
                flags.append("language_mismatch")
            if p["kind"] == "dpo" and p.get("rejected", "").strip() == resp.strip():
                flags.append("pair_identical")
    flags = list(dict.fromkeys(flags))
    score = max(0, 100 - sum(FLAGS[x][0] for x in flags))
    return score, flags


# ----------------------------------------------------------------------------------------- synthetic Q&A grounding
_NUM = re.compile(r"\d[\d,.]*")


def _content_words(s):
    return {w.lower() for w in _WORD.findall(s) if len(w) > 3}


def grounded(question, answer, chunk, min_overlap=0.35):
    """True if the answer is plausibly supported by the passage it was generated from.
    Rejects answers whose key words mostly don't occur in the passage, and answers that invent numbers."""
    aw = _content_words(answer)
    if not aw:
        return False
    cw = _content_words(chunk) | _content_words(question)
    if len(aw & cw) / len(aw) < min_overlap:
        return False
    nums = {n.strip(".,") for n in _NUM.findall(answer) if len(n.strip(".,")) >= 2}
    if nums:
        have = {n.strip(".,") for n in _NUM.findall(chunk + " " + question)}
        if any(n not in have for n in nums):
            return False
    return True

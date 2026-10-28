"""Claim validator: schema, lexicon and the claim-forms.md / value-forms.md text rules.

Conservative on purpose: it rejects anything the referee might read differently.
validate_claim(...) returns a list of problems; an empty list means valid.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from app.generate.render import attr_words, value_text
from app.kb.store import Lexicon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACTS = os.path.join(ROOT, "contracts")
ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TYPES = {"fact", "compare", "change", "at_event", "relation"}

NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without", "cannot"}
MONTH_WORDS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
OTHER_DATE_WORDS = {"spring", "summer", "autumn", "winter", "monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday", "january", "february", "march", "april", "june", "july",
                    "august", "september", "october", "november", "december"}
TIME_WORDS = ["ago", "yesterday", "tomorrow", "once", "formerly", "previously", "originally", "initially",
              "recently", "lately", "soon", "eventually", "then", "before", "after", "since", "until", "till",
              "earlier", "later", "used to", "last week", "last month", "last quarter", "last year",
              "next week", "next month", "next quarter", "next year", "in the past", "in the future",
              "at the time", "at that time", "now", "today", "currently", "still", "yet"]
UNITS_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
              "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
NUMBER_WORDS = set(UNITS_ONES) | set(TENS) | {"hundred", "dozen", "thousand", "million", "billion", "bn"}
APPROX = ["about", "around", "approximately", "approx", "roughly", "nearly", "almost", "circa", "ca",
          "some", "estimated", "est", "close to", "more or less", "or so", "give or take"]
CMP_WORDS = ["over", "more than", "greater than", "higher than", "above", "exceeding", "in excess of",
             "upwards of", "under", "less than", "fewer than", "lower than", "below", "up to", "at most",
             "maximum", "max", "within", "at least", "minimum", "or more", "or less", "or fewer", "prior to",
             "no later than", "no earlier than", "by"]
AMOUNT_PHRASES = {
    "gt": ["more … than", "higher … than", "greater … than", "larger … than", "bigger … than", "longer … than"],
    "lt": ["less … than", "fewer … than", "lower … than", "smaller … than", "shorter … than", "cheaper … than"],
    "eq": ["the same … as", "as much … as", "as many … as", "as long as", "as high as", "equal to"],
}
DATE_PHRASES = {"gt": ["after", "later than"], "lt": ["before", "earlier than"],
                "eq": ["on the same day as", "the same day as"]}
EVENT_PHRASES = ["at the time of", "on the day of", "during", "at"]


def _phrase_re(p: str) -> str:
    parts = [re.escape(x.strip()) for x in p.split("…")]
    return r"\b" + r"(?:\s+\S+){0,3}\s+".join(parts) + r"\b" if len(parts) > 1 else r"\b" + parts[0] + r"\b"


ALL_COMPARE_PHRASES = [(op, p, "amount") for op, ps in AMOUNT_PHRASES.items() for p in ps] + \
                      [(op, p, "date") for op, ps in DATE_PHRASES.items() for p in ps]


def norm_text(t: str) -> str:
    return " ".join((t or "").lower().split()).rstrip(".!? ")


# ---- schema ------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    with open(os.path.join(CONTRACTS, "entry.schema.json"), encoding="utf-8") as f:
        entry = json.load(f)
    with open(os.path.join(CONTRACTS, "claimed_value.schema.json"), encoding="utf-8") as f:
        value = json.load(f)
    reg = Registry().with_resources([(entry["$id"], Resource.from_contents(entry)),
                                     (value["$id"], Resource.from_contents(value))])
    return Draft202012Validator(entry, registry=reg)


def schema_problems(entry: dict) -> list[str]:
    return [f"schema: {e.message[:120]}" for e in _validator().iter_errors(entry)]


# ---- names -------------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _name_patterns(lex_key: str, names: tuple[tuple[str, str], ...]):
    pats = []
    for folded, canon in sorted(names, key=lambda x: -len(x[0])):
        words = [re.escape(w) for w in folded.split()]
        pats.append((re.compile(r"(?<![\w/])" + r"\s*".join(words) + r"(?![\w/])", re.I), canon))
    return pats


def named_entities(text: str, lex: Lexicon) -> list[tuple[int, int, str]]:
    """Spans of lexicon names in text, longest names first, non-overlapping."""
    pats = _name_patterns(str(id(lex)), tuple(sorted(lex.alias.items())))
    taken: list[tuple[int, int, str]] = []
    for pat, canon in pats:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e, _ in taken):
                continue
            taken.append((m.start(), m.end(), canon))
    return sorted(taken)


def _mask(text: str, spans: list[tuple[int, int]]) -> str:
    out = list(text)
    for s, e in spans:
        for i in range(s, e):
            out[i] = " "
    return "".join(out)


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[a-z]+)?", s)


def _has_negation(s: str) -> bool:
    return any(w.lower() in NEGATIONS or w.lower().endswith("n't") for w in _words(s))


def _has_word_phrase(s: str, phrases) -> list[str]:
    low = " " + " ".join(re.findall(r"[a-z']+", s.lower())) + " "
    return [p for p in phrases if f" {p} " in low]


def _date_words(s: str) -> list[str]:
    found = [w for w in _words(s) if w in MONTH_WORDS and w != "May"]
    found += [w for w in _words(s) if w == "May"]
    found += [w for w in _words(s) if w.lower() in OTHER_DATE_WORDS and w not in found]
    return found


def _numbers(s: str) -> list[str]:
    nums = re.findall(r"(?<![A-Za-z])\d[\d,.]*", s)
    nums += [w for w in _words(s) if w.lower() in NUMBER_WORDS or re.fullmatch(r"(%s)-(%s)" % (
        "|".join(TENS), "|".join(UNITS_ONES[1:10])), w.lower())]
    return nums


def attr_spans(attribute: str, text: str) -> list[tuple[int, int]]:
    """Where the attribute is written in words: any case, space/hyphen/nothing between words."""
    words = re.findall(r"[a-z]+|\d+", attribute.lower().replace("_", " "))
    pat = r"[\s-]?".join(re.escape(w) for w in words)
    return [(m.start(), m.end()) for m in re.finditer(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", text, re.I)]


def attr_in_text(attribute: str, text: str) -> bool:
    return bool(attr_spans(attribute, text))


# ---- main --------------------------------------------------------------------------------------
def validate_claim(claim, lexicon, input_texts: set[str] | None = None) -> list[str]:
    lex = lexicon if isinstance(lexicon, Lexicon) else Lexicon(lexicon)
    p: list[str] = []
    text, form, ctype = claim.text or "", claim.form or {}, claim.type
    if not ID.match(claim.id or ""):
        p.append("id")
    if ctype not in TYPES:
        return p + ["type"]
    if not 0.5 <= float(claim.confidence) <= 1:
        p.append("confidence")
    if not text.strip() or len(text) > 300:
        p.append("text length")
    p += schema_problems({"id": claim.id, "type": ctype, "text": text, "form": form, "tag": "true",
                          "confidence": float(claim.confidence)})
    if p:
        return p
    if input_texts is not None and norm_text(text) in input_texts:
        p.append("copy of an input row")
    if re.search(r"[.!?]\s", text.strip()[:-1]) or "\n" in text:
        p.append("more than one sentence")
    if re.search(r"[​-‏⁠­﻿]", text):
        p.append("invisible character")
    p += _lexicon_problems(ctype, form, lex)
    if p:
        return p
    spans = named_entities(text, lex)
    named = [c for _, _, c in spans]
    if ctype == "compare":
        between = (getattr(claim, "meta", None) or {}).get("between")
        return p + _check_compare(text, form, lex, spans, named, between)
    checker = {"fact": _check_fact, "change": _check_change, "at_event": _check_at_event,
               "relation": _check_relation}[ctype]
    p += checker(text, form, lex, spans, named)
    return p


def _lexicon_problems(ctype, form, lex: Lexicon) -> list[str]:
    p = []

    def need(subject, attribute, what):
        if not lex.canonical(subject):
            p.append(f"{what}: unknown subject {subject!r}")
        elif not lex.has_attribute(subject, attribute):
            p.append(f"{what}: {subject!r} has no attribute {attribute!r}")

    if ctype == "compare":
        need(form["left"]["subject"], form["left"]["attribute"], "left")
        need(form["right"]["subject"], form["right"]["attribute"], "right")
    elif ctype == "relation":
        need(form["subject"], form["link"], "link")
        need(form["target"], form["attribute"], "target")
    else:
        need(form["subject"], form["attribute"], "fact")
    if ctype == "at_event":
        ev = lex.canonical(form["event"])
        if not ev or ev not in lex.event_set:
            p.append(f"unknown event {form['event']!r}")
    # The referee resolves names through the lexicon's names and aliases (value-forms.md, Name):
    # a name value it cannot resolve may read as nothing, so only lexicon names are claimed.
    for k in ("value", "from", "to"):
        v = form.get(k)
        if isinstance(v, dict) and v.get("kind") == "name" and not lex.canonical(v.get("value", "")):
            p.append(f"name value not in the lexicon: {v.get('value')!r}")
    return p


def _subject_spans(spans, subject_canon):
    return [(s, e) for s, e, c in spans if c == subject_canon]


def _common_rest_checks(rest: str, allow_negation: bool) -> list[str]:
    p = []
    if _numbers(rest):
        p.append(f"other number {_numbers(rest)[:3]}")
    if _date_words(rest):
        p.append(f"date word {_date_words(rest)[:3]}")
    tw = _has_word_phrase(rest, TIME_WORDS)
    if tw:
        p.append(f"time word {tw[:3]}")
    if re.search(r"[~≈±]", rest):
        p.append("approximation symbol")
    if not allow_negation and _has_negation(rest):
        p.append("negation")
    if re.search(r"[%€$£]", rest):
        p.append("stray value symbol")
    return p


def _value_and_polarity(text, rest, form_value, attribute, mask_spans) -> tuple[list[str], list]:
    """Checks the value is stated; returns problems and spans of the value to mask."""
    p = []
    if form_value.get("kind") == "boolean":
        if not attr_in_text(attribute, rest):
            p.append("yes/no without the attribute's words")
        neg = _has_negation(rest)
        if neg != (form_value["value"] is False):
            p.append("polarity does not match the yes/no value")
        return p, attr_spans(attribute, text)
    vt = value_text(form_value)
    if vt is None:
        return ["value cannot be written"], []
    found = _find_all(text, vt)
    if len(found) != 1:
        p.append(f"value {vt!r} stated {len(found)} times")
    for s, e in found:
        p += _qualifier_problems(text[:s], text[e:])
    return p, found


def _qualifier_problems(before: str, after: str) -> list[str]:
    """Comparison or approximation words right around a value make it another value."""
    b = " " + " ".join(re.findall(r"[a-z']+", before.lower())[-4:]) + " "
    a = " " + " ".join(re.findall(r"[a-z']+", after.lower())[:3]) + " "
    p = [f"qualifier before value: {w}" for w in CMP_WORDS + APPROX if b.endswith(f" {w} ")]
    p += [f"qualifier after value: {w}" for w in ["or more", "or less", "or fewer", "or so", "give or take",
                                                   "or under", "or below", "or lower", "or higher", "or above",
                                                   "or longer", "or shorter", "or over"] if a.startswith(f" {w} ")]
    if re.match(r"\s*(\+|-ish)", after):
        p.append("qualifier symbol after value")
    return p


def _extra_entities(spans, allowed: set[str], value: dict | None, lex: Lexicon) -> list[str]:
    extra = {c for _, _, c in spans if c not in allowed}
    if value and value.get("kind") == "name":
        vc = lex.canonical(value["value"])
        extra.discard(vc or "")
    return [f"another entity named: {e}" for e in sorted(extra)]


def _check_fact(text, form, lex, spans, named):
    subj = lex.canonical(form["subject"])
    p = []
    if subj not in named:
        p.append("subject not named")
    p += _extra_entities(spans, {subj}, form["value"], lex)
    vp, vspans = _value_and_polarity(text, _mask(text, _subject_spans(spans, subj)), form["value"],
                                     form["attribute"], [])
    p += vp
    rest = _mask(text, [(s, e) for s, e, _ in spans] + vspans)
    p += _common_rest_checks(rest, allow_negation=form["value"].get("kind") == "boolean")
    return p


def _check_at_event(text, form, lex, spans, named):
    subj, ev = lex.canonical(form["subject"]), lex.canonical(form["event"])
    p = []
    if subj not in named:
        p.append("subject not named")
    ev_spans = [(s, e) for s, e, c in spans if c == ev]
    if not ev_spans:
        p.append("event not named")
    phrase_spans = []
    for s, _ in ev_spans:
        before = text[:s]
        m = re.search(r"(?i)\b(at the time of|on the day of|during|at)\s+(the\s+)?$", before)
        if m:
            phrase_spans.append((m.start(), s))
    if ev_spans and not phrase_spans:
        p.append("no event phrase right before the event")
    p += _extra_entities(spans, {subj, ev}, form["value"], lex)
    vp, vspans = _value_and_polarity(text, _mask(text, [(s, e) for s, e, _ in spans]), form["value"],
                                     form["attribute"], [])
    p += vp
    rest = _mask(text, [(s, e) for s, e, _ in spans] + vspans + phrase_spans)
    p += _common_rest_checks(rest, allow_negation=form["value"].get("kind") == "boolean")
    return p


def _check_relation(text, form, lex, spans, named):
    subj, target = lex.canonical(form["subject"]), lex.canonical(form["target"])
    p = []
    if subj not in named:
        p.append("link subject not named")
    if not attr_in_text(form["link"], _mask(text, [(s, e) for s, e, _ in spans])):
        p.append("no relation words")
    if target in named:
        p.append("target named (the relation words must carry it)")
    p += _extra_entities(spans, {subj}, form["value"], lex)
    vp, vspans = _value_and_polarity(text, _mask(text, [(s, e) for s, e, _ in spans]), form["value"],
                                     form["attribute"], [])
    p += vp
    rest = _mask(text, [(s, e) for s, e, _ in spans] + vspans)
    p += _common_rest_checks(rest, allow_negation=form["value"].get("kind") == "boolean")
    return p


def _check_change(text, form, lex, spans, named):
    subj = lex.canonical(form["subject"])
    p = []
    if subj not in named:
        p.append("subject not named")
    if form["from"].get("kind") == "boolean":
        return p + ["yes/no change not generated"]
    a, b = value_text(form["from"]), value_text(form["to"])
    if a is None or b is None:
        return p + ["value cannot be written"]
    if a == b:
        p.append("from and to read the same")
    m = re.search(r"\bfrom\s+" + re.escape(a) + r"\s+to\s+" + re.escape(b) + r"(?![\w])", text)
    if not m:
        p.append("no 'from … to …' change phrase")
    p += _extra_entities(spans, {subj}, None, lex) if form["from"].get("kind") != "name" else \
        [x for x in _extra_entities(spans, {subj}, None, lex)
         if x.split(": ", 1)[1] not in {lex.canonical(a), lex.canonical(b)}]
    rest = _mask(text, [(s, e) for s, e, _ in spans] + ([(m.start(), m.end())] if m else []))
    p += _common_rest_checks(rest, allow_negation=False)
    if _has_word_phrase(rest, ["up", "down"]):
        p.append("up/down word")
    return p


def _check_compare(text, form, lex, spans, named, kind_between=None):
    left, right = lex.canonical(form["left"]["subject"]), lex.canonical(form["right"]["subject"])
    p = []
    rest_mask = _mask(text, [(s, e) for s, e, _ in spans])
    hits = []
    for op, phrase, between in ALL_COMPARE_PHRASES:
        for m in re.finditer(_phrase_re(phrase), rest_mask, re.I):
            hits.append((m.start(), m.end(), op, phrase, between))
    # "the same day as" is inside "on the same day as": keep the longest non-overlapping hits
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept = []
    for h in hits:
        if not any(h[0] < k[1] and k[0] < h[1] for k in kept):
            kept.append(h)
    if len(kept) != 1:
        return p + [f"{len(kept)} compare phrases"]
    s0, e0, op, phrase, between = kept[0]
    if op != form["op"]:
        p.append(f"phrase states {op}, form says {form['op']}")
    lpos = [s for s, e, c in spans if c == left and e <= s0]
    rpos = [s for s, e, c in spans if c == right and s >= e0]
    if not lpos:
        p.append("left subject not named before the phrase")
    if not rpos:
        p.append("right subject not named after the phrase")
    p += _extra_entities(spans, {left, right}, None, lex)
    rest = _mask(rest_mask, [(s0, e0)])
    if _numbers(rest):
        p.append("a number in a compare")
    if _date_words(rest):
        p.append("a date word in a compare")
    if _has_word_phrase(rest, TIME_WORDS):
        p.append("a time word in a compare")
    if _has_negation(rest):
        p.append("negation")
    if re.search(r"[%€$£]", rest):
        p.append("a value symbol in a compare")
    if kind_between and kind_between != between:
        p.append(f"{between} phrase between {kind_between}s")
    return p

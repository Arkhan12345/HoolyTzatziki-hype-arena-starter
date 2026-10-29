"""X's simple extractor (ported from eda/extract_simple.py, research by X): a second, independent reading.

The pipeline extracts with Arckan's ranker (app/extract/extractor.py); this extractor only votes: a fact both
read gets more weight in the classifier's KB (both agree -> precision 0.95 vs 0.88 on the holdout pages).
  parse_values(text)   value parser from contracts/value-forms.md
  Gazetteer            lexicon names + aliases, longest match
  AttributePicker      TF-IDF nearest neighbour on value-masked text, among the subject's attributes
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

from app.kb.normalize import key as value_key, singular  # noqa: F401

# ---- numbers
UNITS_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
               "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
               "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
              "ninety": 90}
WORD_NUM = (r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:-(?:one|two|three|four|five|six|seven|"
            r"eight|nine))?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
            r"fifteen|sixteen|seventeen|eighteen|nineteen|a hundred|(?:two|three|four|five|six|seven|eight|nine) "
            r"hundred|a dozen)")
DIGITS = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
NUM = rf"(?:{DIGITS}|{WORD_NUM})"
SCALE = r"(?:\s?(?:thousand|million|billion|bn|M|k)\b)?"


def word_to_num(w):
    w = w.lower().strip()
    if w == "a dozen":
        return 12
    if w.endswith("hundred"):
        head = w.split()[0]
        return 100 * (1 if head == "a" else UNITS_WORDS.get(head, 0))
    if "-" in w:
        a, b = w.split("-")
        return TENS_WORDS[a] + UNITS_WORDS[b]
    return UNITS_WORDS.get(w, TENS_WORDS.get(w))


def to_number(s, scale=None, price=False, pct=False):
    """Number text -> float, or None when the referee would read it as nothing."""
    s = s.strip()
    if re.fullmatch(WORD_NUM, s, re.I):
        v = float(word_to_num(s))
    else:
        if price and re.fullmatch(r"\d+,\d{2}", s):          # decimal comma only in prices: €49,99
            v = float(s.replace(",", "."))
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s):
            if re.fullmatch(r"\d+\.\d{3}", s) and not pct:      # 1.500 is ambiguous (a percentage may have 3)
                return None
            v = float(s.replace(",", ""))
        else:
            return None
    sc = (scale or "").strip()
    mult = {"thousand": 1e3, "k": 1e3, "million": 1e6, "M": 1e6, "billion": 1e9, "bn": 1e9}.get(sc, 1)
    v = v * mult
    return int(v) if float(v).is_integer() else round(v, 6)


# ------------------------------------------------------------------------------------------- vocab
PHYS = {  # spelling -> canonical unit (symbols are case-sensitive, words are not)
    "B": "B", "KB": "KB", "kB": "KB", "kb": "KB", "MB": "MB", "mb": "MB", "GB": "GB", "gb": "GB", "TB": "TB",
    "tb": "TB", "mm": "mm", "cm": "cm", "m": "m", "km": "km", "in": "in", "g": "g", "kg": "kg", "mAh": "mAh",
    "Hz": "Hz", "kHz": "kHz", "MHz": "MHz", "GHz": "GHz", "mW": "mW", "W": "W", "kW": "kW",
}
PHYS_WORDS = {"byte": "B", "kilobyte": "KB", "megabyte": "MB", "gigabyte": "GB", "gig": "GB", "terabyte": "TB",
              "millimetre": "mm", "millimeter": "mm", "centimetre": "cm", "centimeter": "cm", "metre": "m",
              "meter": "m", "kilometre": "km", "kilometer": "km", "inch": "in", "inche": "in", "gram": "g",
              "gramme": "g", "kilogram": "kg", "kilo": "kg", "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz",
              "gigahertz": "GHz", "milliwatt": "mW", "watt": "W", "kilowatt": "kW"}
DUR = {"ms": "ms", "millisecond": "ms", "s": "s", "sec": "s", "second": "s", "min": "min", "minute": "min",
       "h": "h", "hr": "h", "hour": "h", "d": "day", "day": "day", "wk": "week", "week": "week", "mo": "month",
       "mth": "month", "month": "month", "yr": "year", "year": "year"}
CURRENCY_SYM = {"€": "EUR", "$": "USD", "£": "GBP", "US$": "USD"}
ISO = "EUR|USD|GBP|CHF|RON|PLN|CZK|HUF|SEK|NOK|DKK|JPY|CNY|INR|CAD|AUD"
CURRENCY_WORD = {"euro": "EUR", "euros": "EUR", "dollar": "USD", "dollars": "USD", "lei": "RON"}
PERIOD = (r"(?:\s*(?:per user|/user|per seat|/seat))?\s*(?:/\s?(?P<p1>month|mo|day|week|year|yr)\b|"
          r"\s(?:a|per)\s(?P<p2>month|day|week|year|annum)\b|\s(?P<p3>monthly|daily|weekly|yearly|annually))")
PERIOD_MAP = {"mo": "month", "yr": "year", "annum": "year", "monthly": "month", "daily": "day", "weekly": "week",
              "yearly": "year", "annually": "year"}
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                            "august", "september", "october", "november", "december"])}
MON_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
            "oct": 10, "nov": 11, "dec": 12}
MON = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"

OPS_BEFORE = [("le", r"up to|at most|no more than|not more than|maximum|max|within"),
              ("ge", r"at least|no less than|not less than|no fewer than|minimum"),
              ("gt", r"over|more than|greater than|higher than|above|exceeding|in excess of|upwards of"),
              ("lt", r"under|less than|fewer than|lower than|below")]
OPS_AFTER = [("le", r"or less|or fewer|or under|or below|or lower|or shorter"),
             ("ge", r"or more|or higher|or above|or longer|or over")]
DATE_OPS = [("lt", r"before|prior to"), ("gt", r"after"), ("le", r"by|no later than"), ("ge", r"no earlier than")]
APPROX_BEFORE = r"(?:about|around|approximately|approx\.?|roughly|nearly|almost|circa|ca\.?|some|estimated|est\.?|close to|more or less)\s+$|[~≈±]\s*$"
APPROX_AFTER = r"^\s*(?:or so|give or take|-ish)\b"
NEG = r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b"
FUNCTION_WORDS = {"of", "our", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "from", "by",
                  "its", "their", "his", "her", "so", "far", "more", "than", "per", "each", "every", "this", "that",
                  "is", "are", "was", "were", "has", "have", "had", "it", "we", "they", "all", "now", "today",
                  "still", "already", "made", "who", "which", "while", "but", "as", "into", "across", "over",
                  "under", "up", "out", "off", "one", "about", "around"}
CLAUSE_END = r"[.,;:!?•·|]"


def clause_before(text, start):
    """Text of the clause before position `start` (a clause ends at . , ; : ! ? • · |)."""
    head = text[:start]
    m = list(re.finditer(CLAUSE_END + r"(?=\s|$)|[;:!?•·|]", head))
    return head[m[-1].end():] if m else head


def op_before(text, start, ops=OPS_BEFORE):
    before = clause_before(text, start).lower()
    for op, words in ops:
        if re.search(rf"\b(?:{words})\s*$", before):
            return op
    return None


def is_approx_or_negated(text, start, end):
    before = clause_before(text, start)
    if re.search(APPROX_BEFORE, before, re.I):
        return True
    if re.search(APPROX_AFTER, text[end:], re.I):
        return True
    b = before.lower()
    # comparison phrases containing "no"/"not" are ops, not negations
    b = re.sub(r"no more than|not more than|no less than|not less than|no fewer than|no later than|no earlier than",
               " ", b)
    return bool(re.search(NEG, b))


def in_range(text, start, end):
    after = text[end:end + 30]
    before = text[max(0, start - 30):start]
    if re.match(rf"\s*(?:[-–]|to|or)\s*[€$£]?\s*{DIGITS}", after) and not re.match(r"\s*or\s+[€$£]", after):
        return True
    if re.search(rf"{DIGITS}\s*\w*\s*(?:[-–]|to|or)\s*[€$£]?\s*$", before) and not re.search(r"[€$£]\S+\s+\w+\s+\w*\s*or\s+[€$£]?$", before):
        return True
    if re.search(r"\bbetween\s+[€$£]?\s*$", before, re.I) or \
            re.search(r"\bbetween\s+\S+(?:\s+\w+)?\s+and\s*[€$£]?$", before, re.I):
        return True
    return False


# ------------------------------------------------------------------------------------------- parser
def _add(found, taken, start, end, val):
    if any(not (end <= s or start >= e) for s, e in taken):
        return
    taken.append((start, end))
    found.append((start, end, val))


def parse_values(text, version_names=True):
    """All values the text states plainly: list of (start, end, value_dict), in text order."""
    found, taken = [], []
    t = text

    # ---- dates first (so a year never becomes a count)
    date_pats = [
        (rf"\b(\d{{4}})-(\d{{2}})-(\d{{2}})\b", "ymd_iso"),
        (rf"\b(\d{{4}})-(\d{{2}})\b(?!-)", "ym_iso"),
        (rf"\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s*({MON})\s+(\d{{4}})\b", "dmy"),
        (rf"\b({MON})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", "mdy"),
        (rf"\b({MON})\s+(\d{{4}})\b", "my"),
        (r"(?<![\d,.$€£])(?<![QH]\d )\b((?:19|20)\d{2})\b(?![\d,.]|\s?%|\s*(?:GB|MB|TB|mAh|W|Hz)\b)", "y"),
    ]
    for pat, kind in date_pats:
        for m in re.finditer(pat, t, re.I):
            g = m.groups()
            try:
                if kind == "ymd_iso":
                    v = f"{g[0]}-{g[1]}-{g[2]}"
                elif kind == "ym_iso":
                    v = f"{g[0]}-{g[1]}"
                elif kind == "dmy":
                    mo = MONTHS.get(g[1].lower().rstrip("."), MON_ABBR.get(g[1].lower().rstrip(".")))
                    v = f"{g[2]}-{mo:02d}-{int(g[0]):02d}"
                elif kind == "mdy":
                    mo = MONTHS.get(g[0].lower().rstrip("."), MON_ABBR.get(g[0].lower().rstrip(".")))
                    v = f"{g[2]}-{mo:02d}-{int(g[1]):02d}"
                elif kind == "my":
                    mo = MONTHS.get(g[0].lower().rstrip("."), MON_ABBR.get(g[0].lower().rstrip(".")))
                    v = f"{g[1]}-{mo:02d}"
                else:
                    v = g[0]
            except (TypeError, ValueError):
                continue
            if is_approx_or_negated(t, m.start(), m.end()):
                _add(found, taken, m.start(), m.end(), None)
                continue
            d = {"kind": "date", "value": v}
            op = op_before(t, m.start(), DATE_OPS)
            if op:
                d["op"] = op
            _add(found, taken, m.start(), m.end(), d)

    # ---- versions
    for m in re.finditer(r"\b(?:v|version\s|ver\.\s?)(\d+(?:\.\d+)*)\b", t, re.I):
        _add(found, taken, m.start(), m.end(), {"kind": "version", "value": m.group(1)})
    for m in re.finditer(r"(?<![\d.])\b(\d+\.\d+\.\d+)\b", t):
        _add(found, taken, m.start(), m.end(), {"kind": "version", "value": m.group(1)})
    if version_names:
        unit_after = r"%|percent|per cent|KB|MB|GB|TB|kB|mAh|[kMG]?Hz|[mk]?W|kg|g|mm|cm|km|m|in|ms|s|sec|min|h|hr|" \
                     r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|million|billion|thousand|bn|stars?|inch"
        for m in re.finditer(rf"\b([A-Z][\w]*)\s+(\d+\.\d+)\b(?!\s?(?:{unit_after})\b)", t):
            _add(found, taken, m.start(2), m.end(2), {"kind": "version", "value": m.group(2)})

    # ---- money
    cur_before = rf"(?P<cur>US\$|[€$£]|\b(?:{ISO})\s?)"
    for m in re.finditer(rf"{cur_before}(?P<num>\d+,\d{{2}}(?![\d])|{DIGITS})(?P<scale>{SCALE})(?:{PERIOD})?", t):
        _money(t, m, m.group("cur").strip(), found, taken)
    for m in re.finditer(rf"\b(?P<num>{DIGITS})(?P<scale>{SCALE})\s?(?P<cur>{ISO}|euros?|dollars?|lei)\b(?:{PERIOD})?",
                         t, re.I):
        _money(t, m, m.group("cur"), found, taken)

    # ---- percent
    for m in re.finditer(rf"(?<![\d.,])\b(?P<num>{NUM})\s?(?:%|percent\b|per cent\b)", t, re.I):
        v = to_number(m.group("num"), pct=True)
        _scalar(t, m, v, {"kind": "percent"}, found, taken)

    # ---- duration
    dur_unit = r"(?P<u>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|fortnights?|months?|mos?|mths?|years?|yrs?|decades?)"
    for m in re.finditer(rf"(?<![\d.,\w])(?P<num>{NUM}|a|an)(?:\s|-)?{dur_unit}\b", t, re.I):
        u = m.group("u").lower()
        if len(u) == 1 and m.group("u") != u:  # single-letter unit must be lower case
            continue
        num = m.group("num")
        if num.lower() in ("a", "an"):
            before = clause_before(t, m.start()).lower()
            if not re.search(r"\b(?:under|within|in|for|over|than|at least|at most|to|only|just|about|around)\s*$", before):
                continue
            v = 1
        else:
            v = to_number(num)
        if v is None:
            continue
        if u.startswith("fortnight"):
            v, unit = v * 2, "week"
        elif u.startswith("decade"):
            v, unit = v * 10, "year"
        else:
            unit = DUR.get(u if u in DUR else singular(u).rstrip("s") if u not in DUR else u)
            if unit is None:
                unit = DUR.get(singular(u))
        if unit is None:
            continue
        # a count followed by "a day" is a rate, not a duration
        if num.lower() in ("a", "an") and re.search(rf"{DIGITS}\s+\w+\s*$", t[:m.start()]):
            continue
        _scalar(t, m, v, {"kind": "duration", "unit": unit}, found, taken)

    # ---- physical quantities
    sym = "|".join(sorted((re.escape(k) for k in PHYS), key=len, reverse=True))
    for m in re.finditer(rf"(?<![\d.,\w])(?P<num>{DIGITS})(?P<scale>\s?(?:thousand|million|billion)\b)?(?:\s|-)?(?P<u>{sym})(?![\w])", t):
        if m.group("u") == "in" and not re.match(r"-?in", t[m.end("num"):m.end("num") + 3]):
            continue
        v = to_number(m.group("num"), m.group("scale"))
        _scalar(t, m, v, {"kind": "quantity", "unit": PHYS[m.group("u")]}, found, taken)
    for m in re.finditer(rf"(?<![\d.,\w])(?P<num>{DIGITS})(?:\s?[\"″])", t):
        v = to_number(m.group("num"))
        _scalar(t, m, v, {"kind": "quantity", "unit": "in"}, found, taken)
    words = "|".join(sorted(PHYS_WORDS, key=len, reverse=True))
    for m in re.finditer(rf"(?<![\d.,\w])(?P<num>{NUM})(?:\s|-)(?P<u>(?:{words})s?)\b", t, re.I):
        u = m.group("u").lower()
        u = u[:-1] if u.endswith("s") and u[:-1] in PHYS_WORDS else u
        v = to_number(m.group("num"))
        _scalar(t, m, v, {"kind": "quantity", "unit": PHYS_WORDS.get(u, u)}, found, taken)

    # ---- counted nouns
    for m in re.finditer(rf"(?<![\d.,\w$€£])(?P<num>{NUM})(?P<scale>{SCALE})(?P<plus>\+)?\s+(?P<rest>[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){{0,2}})", t, re.I):
        num, scale = m.group("num"), (m.group("scale") or "").strip()
        if re.fullmatch(r"(?:19|20)\d{2}", num) and not scale:
            continue  # a plain year-like number is a year, never a count
        if scale and scale not in ("thousand", "million", "billion", "bn", "M", "k"):
            continue
        if scale == "m":
            continue
        v = to_number(num, scale)
        if v is None:
            continue
        noun, noun_end = None, None
        toks = list(re.finditer(r"[A-Za-z][\w-]*", m.group("rest")))
        cands = []
        for tk in toks:
            w = tk.group(0)
            if w.lower() in FUNCTION_WORDS:
                if cands:
                    break
                continue
            cands.append(tk)
            if len(cands) == 2:
                break
        if not cands:
            continue
        pick = cands[-1] if len(cands) > 1 and cands[-1].group(0).lower().endswith("s") else cands[0]
        noun = pick.group(0)
        noun_end = m.start("rest") + pick.end()
        if noun.lower() in MONTHS or noun.lower() in MON_ABBR:
            continue
        d = {"kind": "quantity", "value": v, "unit": singular(noun)}
        start, end = m.start(), noun_end
        if is_approx_or_negated(t, start, end) or in_range(t, start, m.end("num")):
            _add(found, taken, start, end, None)
            continue
        op = "ge" if m.group("plus") else op_before(t, start)
        for o, w in OPS_AFTER:
            if re.match(rf"\s*(?:{w})\b", t[end:], re.I):
                op = o
        if op:
            d["op"] = op
        _add(found, taken, start, end, d)

    out = sorted([f for f in found if f[2] is not None], key=lambda x: x[0])
    return out


def _scalar(t, m, v, base, found, taken):
    if v is None:
        return
    start, end = m.start(), m.end()
    if is_approx_or_negated(t, start, end) or in_range(t, start, end):
        _add(found, taken, start, end, None)
        return
    d = dict(base)
    d["value"] = v
    op = op_before(t, start)
    for o, w in OPS_AFTER:
        if re.match(rf"\s*(?:{w})\b", t[end:], re.I):
            op = o
    if op:
        d["op"] = op
    # canonical key order like the gold forms
    order = ["kind", "value", "unit", "currency", "per", "op"]
    _add(found, taken, start, end, {k: d[k] for k in order if k in d})


def _money(t, m, cur, found, taken):
    v = to_number(m.group("num"), m.group("scale"), price=True)
    if v is None:
        return
    c = CURRENCY_SYM.get(cur, CURRENCY_WORD.get(cur.lower(), cur.upper()))
    per = None
    for g in ("p1", "p2", "p3"):
        if m.groupdict().get(g):
            per = PERIOD_MAP.get(m.group(g).lower(), m.group(g).lower())
    start, end = m.start(), m.end()
    if is_approx_or_negated(t, start, end) or in_range(t, start, end):
        _add(found, taken, start, end, None)
        return
    d = {"kind": "money", "value": v, "currency": c}
    if per:
        d["per"] = per
    op = op_before(t, start)
    if op:
        d["op"] = op
    _add(found, taken, start, end, d)


def parse_boolean(text):
    return {"kind": "boolean", "value": not bool(re.search(NEG, text, re.I))}


# ------------------------------------------------------------------------------------------- gazetteer
class Gazetteer:
    def __init__(self, lexicon):
        names = {}
        subs = lexicon["subjects"]
        for s in subs:
            names[s["name"]] = s["name"]
            for a in s.get("aliases", []):
                names[a] = s["name"]
        for e in lexicon.get("events", []):
            n = e["name"] if isinstance(e, dict) else e
            names.setdefault(n, n)
            short = re.sub(r"\s+\d{4}$", "", n)
            if short != n:
                names.setdefault(short, n)
        # "Hooly Gemini" -> "Gemini" when that short form names nothing else
        shorts = {}
        for s in subs:
            if s["name"].startswith("Hooly "):
                shorts.setdefault(s["name"][6:], []).append(s["name"])
        for sh, full in shorts.items():
            if len(full) == 1 and len(sh) >= 4 and sh not in names:
                names[sh] = full[0]
        self.names = names
        self.attrs = {s["name"]: set(s.get("attributes", [])) for s in subs}
        pats = sorted(names, key=len, reverse=True)
        self.rx = re.compile(r"\b(" + "|".join(re.escape(p) for p in pats) + r")\b", re.I)
        self.lower = {k.lower(): v for k, v in names.items()}

    def mentions(self, text):
        out = []
        for m in self.rx.finditer(text):
            out.append((m.start(), m.end(), self.lower[m.group(1).lower()]))
        return out


# ------------------------------------------------------------------------------------------- attributes
def mask(text, spans):
    s = text
    for a, b, v in sorted(spans, key=lambda x: -x[0]):
        s = s[:a] + f" VAL_{v['kind'].upper()} " + s[b:]
    return s


def fact_doc(masked, v):
    toks = [masked, f"KIND_{v['kind']}"]
    if v.get("unit"):
        toks.append(f"UNIT_{str(v['unit']).replace(' ', '_')}")
    if v.get("per"):
        toks.append(f"PER_{v['per']}")
    return " ".join(toks)


class AttributePicker:
    """Nearest training fact (cosine on TF-IDF of value-masked text) among the subject's own attributes."""

    def __init__(self, gaz):
        self.gaz = gaz

    def fit(self, train_facts):
        self.tr = train_facts.reset_index(drop=True)
        self.vec = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                              TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
        self.M = self.vec.fit_transform(self.tr["doc"])
        self.attr_kind = self.tr.groupby("attribute")["kind"].agg(lambda s: s.mode().iat[0]).to_dict()
        self.attr_units = self.tr.groupby("attribute")["unit"].agg(lambda s: set(s.dropna())).to_dict()
        self.name_values = set(self.tr.loc[self.tr.kind == "name", "value_raw"].astype(str))
        return self

    def pick(self, subject, v, doc, text):
        allowed = self.gaz.attrs.get(subject, set())
        cand = self.tr[(self.tr.kind == v["kind"]) & (self.tr.attribute.isin(allowed))] if allowed else \
            self.tr[self.tr.kind == v["kind"]]
        if len(cand):
            sims = (self.M[cand.index] @ self.vec.transform([doc]).T).toarray().ravel()
            if v.get("unit"):
                sims = sims + 0.3 * (cand["unit"].values == v["unit"])
            best = int(np.argmax(sims))
            return cand["attribute"].iat[best], float(sims[best])
        # unseen (subject, kind): attribute of the subject whose name shares words with the sentence
        words = set(re.findall(r"[a-z]+", text.lower())) | {singular(w) for w in re.findall(r"[a-z]+", text.lower())}
        best, score = None, -1
        for a in allowed:
            k = self.attr_kind.get(a)
            if k and k != v["kind"]:
                continue
            s = len(set(a.split("_")) & words)
            if s > score:
                best, score = a, s
        return best, 0.0


# ------------------------------------------------------------------------------------------- extraction
def candidate_values(text, gaz, picker_name_values, subject_spans):
    vals = parse_values(text)
    # name values: known name values or lexicon names that are not the fact's subject
    for m in re.finditer(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", text):
        span = m.group(0)
        for nv in picker_name_values:
            if nv and re.search(rf"\b{re.escape(nv)}\b", span) and not any(a <= m.start() < b for a, b, _ in vals):
                st = m.start() + span.find(nv)
                vals.append((st, st + len(nv), {"kind": "name", "value": nv}))
    return sorted(vals, key=lambda x: x[0])


def extract_rows(rows, gaz, picker):
    """rows: DataFrame (id, text, date, page) in id order. Returns extracted facts (list of dicts)."""
    out = []
    carry = {}
    for r in rows.itertuples():
        text = r.text if isinstance(r.text, str) else ""
        ments = gaz.mentions(text)
        vals = candidate_values(text, gaz, picker.name_values, ments)
        if not vals:
            vals = [(0, len(text), parse_boolean(text))]
        masked = mask(text, [(a, b, v) for a, b, v in vals if v["kind"] != "boolean"])
        seen = set()
        for a, b, v in vals:
            # subject = nearest mention before the value, else after, else the page's last subject
            before = [mm for mm in ments if mm[1] <= a and mm[2] != v.get("value")]
            after = [mm for mm in ments if mm[0] >= b and mm[2] != v.get("value")]
            if before:
                subj = before[-1][2]
            elif after:
                subj = after[0][2]
            else:
                subj = carry.get(r.page)
            if subj is None:
                continue
            if v["kind"] == "name" and v["value"] == subj:
                continue
            attr, sim = picker.pick(subj, v, fact_doc(masked, v), text)
            if attr is None:
                continue
            if v["kind"] == "name" and sim < 0.25:
                continue
            k = (subj, attr, value_key(v))
            if k in seen:
                continue
            seen.add(k)
            out.append({"id": r.id, "subject": subj, "attribute": attr, "value": v, "sim": sim,
                        "subject_mentioned": bool(before or after)})
        if ments:
            carry[r.page] = ments[-1][2]
    return out



def gold_facts(R):
    rows = []
    for r in R.itertuples():
        for j, f in enumerate(json.loads(r.form)):
            rows.append({"id": r.id, "fact_idx": j, "subject": f["subject"], "attribute": f["attribute"],
                         "value": f["value"], "kind": f["value"]["kind"]})
    return pd.DataFrame(rows)


def training_docs(R_train, G_train, gaz):
    """Gold training facts with their value-masked sentence doc."""
    docs = []
    by_id = R_train.set_index("id")["text"].to_dict()
    for g in G_train.itertuples():
        text = by_id[g.id]
        spans = parse_values(text)
        docs.append({"attribute": g.attribute, "kind": g.kind, "unit": g.value.get("unit"),
                     "value_raw": g.value.get("value"), "doc": fact_doc(mask(text, spans), g.value)})
    return pd.DataFrame(docs)



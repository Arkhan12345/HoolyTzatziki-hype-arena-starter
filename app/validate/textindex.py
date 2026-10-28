"""Text index over every input sentence, labelled or not: which names and which values each one states.

Built once per run (one pass per row). It answers one narrow question for the claim validator: "does
some input sentence already say this claim?" without extracting facts in general (that is P3's job).

Nothing here is fitted to the practice data. Names come from the run's lexicon (subjects, aliases,
events). Values are parsed by the general rules of contracts/value-forms.md: any number (digits with
thousands commas, decimals, number words, scale words), any counted noun, every currency, unit and
date format that page lists. Each value becomes comparable keys:
    ("number", n)              any count, whatever the noun after it
    ("money", "EUR", n)        any currency, any billing period (the period is ignored: stricter)
    ("percent", n)
    ("time", seconds) / ("calendar", months)   durations, converted
    (dimension, base)          physical units, converted (data, length, mass, ...)
    ("date", "YYYY[-MM[-DD]]") ("version", "x.y")
Recall is favoured over precision (a decimal is also indexed as a version, a year as a number): a
claim wrongly flagged as a restatement is replaced by another one, a missed restatement is not.
"""
from __future__ import annotations

import re
from collections import defaultdict

from app.kb.normalize import DURATION, PHYSICAL

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}
MONTH_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
              "oct": 10, "nov": 11, "dec": 12}
MONTH_RE = r"(" + "|".join(sorted(list(MONTHS) + list(MONTH_ABBR), key=len, reverse=True)) + r")\.?"
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_NUM = {w: i for i, w in enumerate(ONES)} | TENS | {"a dozen": 12, "dozen": 12, "a hundred": 100, "hundred": 100}
for _t, _v in TENS.items():
    for _i in range(1, 10):
        WORD_NUM[f"{_t}-{ONES[_i]}"] = _v + _i
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}
CURRENCY = {"€": "EUR", "$": "USD", "£": "GBP", "us$": "USD", "euro": "EUR", "euros": "EUR", "dollar": "USD",
            "dollars": "USD", "lei": "RON"}
ISO_CODES = ["EUR", "USD", "GBP", "CHF", "RON", "PLN", "CZK", "HUF", "SEK", "NOK", "DKK", "JPY", "CNY", "INR", "CAD", "AUD"]
DUR_WORDS = {"ms": "ms", "millisecond": "ms", "milliseconds": "ms", "s": "s", "sec": "s", "secs": "s",
             "second": "s", "seconds": "s", "min": "min", "mins": "min", "minute": "min", "minutes": "min",
             "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "d": "day", "day": "day", "days": "day",
             "wk": "week", "wks": "week", "week": "week", "weeks": "week", "mo": "month", "mos": "month",
             "mth": "month", "mths": "month", "month": "month", "months": "month", "yr": "year", "yrs": "year",
             "year": "year", "years": "year"}
PHYS_WORDS = {"byte": "B", "bytes": "B", "kilobyte": "KB", "kilobytes": "KB", "megabyte": "MB", "megabytes": "MB",
              "gigabyte": "GB", "gigabytes": "GB", "gig": "GB", "gigs": "GB", "terabyte": "TB", "terabytes": "TB",
              "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB", "millimetre": "mm", "millimeter": "mm",
              "millimetres": "mm", "millimeters": "mm", "centimetre": "cm", "centimeter": "cm", "centimetres": "cm",
              "centimeters": "cm", "metre": "m", "meter": "m", "metres": "m", "meters": "m", "kilometre": "km",
              "kilometer": "km", "kilometres": "km", "kilometers": "km", "inch": "in", "inches": "in", '"': "in",
              "″": "in", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g", "kilogram": "kg",
              "kilograms": "kg", "kilo": "kg", "kilos": "kg", "hertz": "Hz", "kilohertz": "kHz", "megahertz": "MHz",
              "gigahertz": "GHz", "milliwatt": "mW", "milliwatts": "mW", "watt": "W", "watts": "W",
              "kilowatt": "kW", "kilowatts": "kW"}

_NUM = r"(?<![\w.,])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
_WORDNUM = r"\b(" + "|".join(sorted(map(re.escape, WORD_NUM), key=len, reverse=True)) + r")\b"
_SCALE = r"(?:\s*(thousand|million|billion|bn|k|M)\b)?"
_TAIL = r"[\s-]?([A-Za-z\"″]+|%)?"
NUMBER_RE = re.compile(r"(?:(US\$|[€$£])\s?|\b(" + "|".join(ISO_CODES) + r")\s?)?" + _NUM + _SCALE
                       + r"(?:\s*(%|percent\b|per cent\b))?" + _TAIL)
WORDNUM_RE = re.compile(_WORDNUM + r"(?:[\s-]+([A-Za-z]+))?", re.I)
DATE_RES = [
    (re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b"), "iso"),
    (re.compile(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s*" + MONTH_RE + r",?\s+(\d{4})\b", re.I), "dmy"),
    (re.compile(r"\b" + MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I), "mdy"),
    (re.compile(r"\b" + MONTH_RE + r"\s+(\d{4})\b", re.I), "my"),
    (re.compile(r"(?<![\d,.])\b((?:19|20)\d{2})\b(?![,.]\d)"), "y"),
]
VERSION_RES = [re.compile(r"\b(?:v|version|ver\.)\s?(\d+(?:\.\d+){0,3})\b", re.I),
               re.compile(r"(?<![\d.,])(\d+\.\d+(?:\.\d+){0,2})(?![\d,])")]
COMPARE_WORDS = re.compile(r"\b(more|less|fewer|higher|lower|greater|larger|bigger|smaller|longer|shorter|cheaper|"
                           r"pricier|than|same as|as much as|as many as|equal to|before|after|earlier|later)\b", re.I)


def _to_float(num: str) -> float | None:
    s = num.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", num) else num.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _month(word: str) -> int | None:
    w = word.lower().rstrip(".")
    return MONTHS.get(w) or MONTH_ABBR.get(w)


def value_keys(text: str) -> list[tuple[tuple, int]]:
    """Every (key, char position) a sentence states, by the general value-forms rules."""
    out: list[tuple[tuple, int]] = []
    for m in NUMBER_RE.finditer(text):
        sym, code, num, scale, pct, tail = m.groups()
        n = _to_float(num)
        if n is None:
            continue
        if scale:
            n *= SCALE.get(scale.lower() if scale != "M" else "m", 1)
        pos = m.start()
        out.append((("number", round(n, 6)), pos))
        cur = (CURRENCY.get((sym or "").lower()) or code or "").upper()
        t = (tail or "")
        if not cur and t.lower() in CURRENCY:
            cur = CURRENCY[t.lower()]
        if not cur and t.upper() in ISO_CODES:
            cur = t.upper()
        if cur:
            out.append((("money", cur, round(n, 6)), pos))
        if pct or t == "%":
            out.append((("percent", round(n, 6)), pos))
        unit = t if t in PHYSICAL else PHYS_WORDS.get(t.lower()) if t else None
        if unit and unit in PHYSICAL:
            dim, f = PHYSICAL[unit]
            out.append(((dim, round(n * f, 6)), pos))
        dur = DUR_WORDS.get(t.lower()) if t and not (t == "m") else None
        if t.lower() in ("fortnight", "fortnights"):
            out.append((("time", round(n * 14 * 86400, 6)), pos))
        if dur:
            kind, f = DURATION[dur]
            out.append(((kind, round(n * f, 6)), pos))
        if "." in num and "," not in num:
            out.append((("version", num), pos))
    for m in WORDNUM_RE.finditer(text):
        n = WORD_NUM.get(m.group(1).lower())
        if n is None:
            continue
        out.append((("number", float(n)), m.start()))
        t = (m.group(2) or "").lower()
        if t in DUR_WORDS:
            kind, f = DURATION[DUR_WORDS[t]]
            out.append(((kind, round(n * f, 6)), m.start()))
        if t in ("fortnight", "fortnights"):
            out.append((("time", round(n * 14 * 86400, 6)), m.start()))
        if t in ("percent",):
            out.append((("percent", float(n)), m.start()))
    for m in re.finditer(r"\b(a|one)\s+fortnight\b", text, re.I):
        out.append((("time", 14 * 86400.0), m.start()))
    for rx, kind in DATE_RES:
        for m in rx.finditer(text):
            g = m.groups()
            if kind == "iso":
                d = g[0] + "-" + g[1] + ("-" + g[2] if g[2] else "")
            elif kind == "dmy":
                mo = _month(g[1])
                d = f"{g[2]}-{mo:02d}-{int(g[0]):02d}" if mo else None
            elif kind == "mdy":
                mo = _month(g[0])
                d = f"{g[2]}-{mo:02d}-{int(g[1]):02d}" if mo else None
            elif kind == "my":
                mo = _month(g[0])
                d = f"{g[1]}-{mo:02d}" if mo else None
            else:
                d = g[0]
            if d:
                out.append((("date", d), m.start()))
    for rx in VERSION_RES:
        for m in rx.finditer(text):
            out.append((("version", m.group(1)), m.start()))
    return out


class RowTextIndex:
    """Names and value keys of every input row, with an inverted index for fast lookups."""

    def __init__(self, rows: dict[str, dict], lexicon):
        self.lex = lexicon
        self.text = {rid: (r.get("text") or "") for rid, r in rows.items()}
        pairs = sorted({(n, c) for n, c in lexicon.alias.items()} |
                       {(lexicon.fold(e), lexicon.canonical(e) or e) for e in lexicon.events if e},
                       key=lambda x: -len(x[0]))
        self._canon = {}
        alts = []
        for i, (folded, canon) in enumerate(pairs):
            self._canon[f"n{i}"] = canon
            alts.append(f"(?P<n{i}>" + r"\s*".join(map(re.escape, folded.split())) + ")")
        self.names_re = re.compile(r"(?<![\w/])(?:" + "|".join(alts) + r")(?![\w/])", re.I) if alts else None
        self.names: dict[str, list[tuple[str, int]]] = {}
        self.values: dict[str, list[tuple[tuple, int]]] = {}
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_value: dict[tuple, set[str]] = defaultdict(set)
        self.by_year: dict[str, set[str]] = defaultdict(set)
        for rid, text in self.text.items():
            nm = [(self._canon[m.lastgroup], m.start()) for m in self.names_re.finditer(text)] if self.names_re else []
            vk = value_keys(text)
            self.names[rid], self.values[rid] = nm, vk
            for c, _ in nm:
                self.by_name[c].add(rid)
            for k, _ in vk:
                self.by_value[k].add(rid)
                if k[0] == "date":
                    self.by_year[k[1][:4]].add(rid)

    # ---- lookups -------------------------------------------------------------------------------
    def rows_with_value(self, key: tuple) -> set[str]:
        if key[0] != "date":
            return self.by_value.get(key, set())
        d = key[1]
        return {r for r in self.by_year.get(d[:4], set())
                if any(k[0] == "date" and (k[1].startswith(d) or d.startswith(k[1])) for k, _ in self.values[r])}

    def name_pos(self, rid: str, canon: str) -> list[int]:
        return [p for c, p in self.names[rid] if c == canon]

    def value_pos(self, rid: str, key: tuple) -> list[int]:
        if key[0] == "date":
            d = key[1]
            return [p for k, p in self.values[rid] if k[0] == "date" and (k[1].startswith(d) or d.startswith(k[1]))]
        return [p for k, p in self.values[rid] if k == key]


def claim_value_keys(value: dict) -> list[tuple]:
    """Keys a claim's value would show up under in some sentence (any of them counts)."""
    k, v = value.get("kind"), value.get("value")
    if k == "quantity":
        u = value.get("unit", "")
        if u in PHYSICAL:
            dim, f = PHYSICAL[u]
            return [(dim, round(float(v) * f, 6))]
        return [("number", round(float(v), 6))]
    if k == "money":
        return [("money", value.get("currency", ""), round(float(v), 6))]
    if k == "percent":
        return [("percent", round(float(v), 6))]
    if k == "duration":
        kind, f = DURATION.get(value.get("unit"), ("?", 1))
        return [(kind, round(float(v) * f, 6))]
    if k == "date":
        return [("date", str(v))]
    if k == "version":
        return [("version", str(v))]
    return []

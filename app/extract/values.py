"""Regex value parser, written from contracts/value-forms.md.

`parse_values(text, names)` returns every value mention in a sentence with its
character span. `canonical(value)` turns a parsed or gold value dict into a
hashable key so that equal values compare equal (14 days == 2 weeks,
1 GB == 1,000 MB). Counted nouns ("4 engineers") are compared by number only:
the attribute already says what is counted.

Booleans are not parsed here; the extractor reads them per sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- numbers

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "a dozen": 12, "a hundred": 100,
}
_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_UNITS = "one|two|three|four|five|six|seven|eight|nine"
WORD_NUM_RE = (
    rf"(?:(?:{_TENS})-(?:{_UNITS})|a dozen|a hundred|(?:{_UNITS}) hundred|"
    + "|".join(sorted((w for w in WORD_NUMBERS if " " not in w), key=len, reverse=True))
    + ")"
)
DIGIT_NUM_RE = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SCALE_RE = r"(?:\s?(?:thousand|million|billion|bn|M|k)\b)?"
NUM_RE = rf"(?:{DIGIT_NUM_RE}|{WORD_NUM_RE})"

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}


def word_to_number(word: str) -> float | None:
    w = word.lower().strip()
    if w in WORD_NUMBERS:
        return float(WORD_NUMBERS[w])
    if "-" in w:
        tens, unit = w.split("-", 1)
        if tens in WORD_NUMBERS and unit in WORD_NUMBERS:
            return float(WORD_NUMBERS[tens] + WORD_NUMBERS[unit])
    if w.endswith(" hundred") and w.split()[0] in WORD_NUMBERS:
        return float(WORD_NUMBERS[w.split()[0]] * 100)
    return None


def to_number(raw: str, scale: str | None = None) -> float | None:
    raw = raw.strip()
    if re.fullmatch(DIGIT_NUM_RE, raw):
        value = float(raw.replace(",", ""))
    else:
        value = word_to_number(raw)
        if value is None:
            return None
    if scale:
        s = scale.strip()
        value *= SCALES["m" if s == "M" else s.lower()]
    return value


def clean_number(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


def _sig(value: float) -> float:
    """Round to 6 significant digits so float noise never splits equal values."""
    return float(f"{float(value):.6g}")


# ---------------------------------------------------------------- units

DATA_UNITS = {
    "b": "B", "byte": "B", "bytes": "B",
    "kb": "KB", "kilobyte": "KB", "kilobytes": "KB",
    "mb": "MB", "megabyte": "MB", "megabytes": "MB",
    "gb": "GB", "gig": "GB", "gigs": "GB", "gigabyte": "GB", "gigabytes": "GB",
    "tb": "TB", "terabyte": "TB", "terabytes": "TB",
}
DATA_FACTOR = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
OTHER_UNITS = {
    "mm": ("length", 1e-3), "millimetre": ("length", 1e-3), "millimeter": ("length", 1e-3),
    "cm": ("length", 1e-2), "centimetre": ("length", 1e-2), "centimeter": ("length", 1e-2),
    "m": ("length", 1.0), "metre": ("length", 1.0), "meter": ("length", 1.0),
    "km": ("length", 1e3), "kilometre": ("length", 1e3), "kilometer": ("length", 1e3),
    "in": ("length", 0.0254), "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    '"': ("length", 0.0254), "″": ("length", 0.0254),
    "g": ("mass", 1.0), "gram": ("mass", 1.0), "gramme": ("mass", 1.0),
    "kg": ("mass", 1e3), "kilogram": ("mass", 1e3), "kilo": ("mass", 1e3),
    "mah": ("charge", 1.0),
    "hz": ("freq", 1.0), "hertz": ("freq", 1.0), "khz": ("freq", 1e3), "kilohertz": ("freq", 1e3),
    "mhz": ("freq", 1e6), "megahertz": ("freq", 1e6), "ghz": ("freq", 1e9), "gigahertz": ("freq", 1e9),
    "mw": ("power", 1e-3), "milliwatt": ("power", 1e-3), "w": ("power", 1.0), "watt": ("power", 1.0),
    "kw": ("power", 1e3), "kilowatt": ("power", 1e3),
}
# Gold forms write these units; map them to (dimension, factor) too.
GOLD_UNIT_ALIASES = {"inch": "in", "\"": "in"}

DURATION_UNITS = {
    "ms": ("s", 1e-3), "millisecond": ("s", 1e-3),
    "s": ("s", 1.0), "sec": ("s", 1.0), "secs": ("s", 1.0), "second": ("s", 1.0),
    "min": ("s", 60.0), "mins": ("s", 60.0), "minute": ("s", 60.0),
    "h": ("s", 3600.0), "hr": ("s", 3600.0), "hrs": ("s", 3600.0), "hour": ("s", 3600.0),
    "d": ("s", 86400.0), "day": ("s", 86400.0),
    "wk": ("s", 604800.0), "week": ("s", 604800.0), "fortnight": ("s", 1209600.0),
    "mo": ("month", 1.0), "mth": ("month", 1.0), "month": ("month", 1.0),
    "yr": ("month", 12.0), "year": ("month", 12.0), "decade": ("month", 120.0),
}
DURATION_WORD_RE = (
    r"(?:milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|h|days?|d|"
    r"weeks?|wks?|fortnights?|months?|mos?|mths?|years?|yrs?|decades?)"
)
DURATION_CANON = {
    "millisecond": "ms", "ms": "ms", "second": "s", "sec": "s", "s": "s",
    "minute": "min", "min": "min", "hour": "h", "hr": "h", "h": "h",
    "day": "day", "d": "day", "week": "week", "wk": "week", "fortnight": "week",
    "month": "month", "mo": "month", "mth": "month", "year": "year", "yr": "year",
    "decade": "year",
}

CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP", "us$": "USD"}
CURRENCY_CODES = "EUR|USD|GBP|CHF|RON|PLN|CZK|HUF|SEK|NOK|DKK|JPY|CNY|INR|CAD|AUD"
CURRENCY_WORDS = {"euro": "EUR", "euros": "EUR", "dollar": "USD", "dollars": "USD", "lei": "RON"}
PERIODS = {"day": "day", "week": "week", "month": "month", "mo": "month", "year": "year",
           "yr": "year", "annum": "year", "annually": "year", "monthly": "month",
           "weekly": "week", "daily": "day", "yearly": "year"}
PERIOD_RE = (
    r"(?:\s*(?:per user|per seat|/user|/seat|per (?!(?:day|week|month|mo|year|yr|annum)\b)[a-z]+(?= (?:a|an|per|each) )))?"
    r"(?:\s*(?:/|a |an |per |each )\s*(day|week|month|mo|year|yr|annum)\b|\s+(monthly|weekly|daily|yearly|annually)\b)?"
)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
)

OPS_BEFORE = [
    (r"more than|greater than|higher than|in excess of|upwards of|exceeding|over|above", "gt"),
    (r"fewer than|less than|lower than|under|below", "lt"),
    (r"no more than|not more than|up to|at most|maximum|max", "le"),
    (r"no less than|not less than|no fewer than|at least|minimum", "ge"),
]
OPS_AFTER = [
    (r"or (?:less|fewer|under|below|lower|shorter)", "le"),
    (r"or (?:more|higher|above|longer|over)", "ge"),
]
APPROX_BEFORE = (
    r"(?:about|around|approximately|approx\.?|roughly|nearly|almost|circa|ca\.?|some|"
    r"estimated|est\.?|close to|more or less|~|≈|±)\s*$"
)
DATE_OPS = [(r"before|prior to", "lt"), (r"after", "gt"), (r"by|no later than", "le"),
            (r"no earlier than", "ge")]


@dataclass
class ValueMention:
    """One value read from a sentence."""

    kind: str
    value: dict
    start: int
    end: int
    approx: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return canonical(self.value)


# ---------------------------------------------------------------- canonical key

def _singular(word: str) -> str:
    w = word.lower()
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x")):
        if w.endswith(suffix) and len(w) > len(suffix) + 1:
            return w[: -len(suffix)] + repl
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def canonical(value: dict) -> tuple:
    """Hashable comparison key for a value dict (parsed or gold)."""
    kind = value.get("kind")
    raw = value.get("value")
    op = value.get("op") or "eq"
    if kind == "quantity":
        unit = str(value.get("unit") or "")
        u = GOLD_UNIT_ALIASES.get(unit, unit)
        if u in DATA_FACTOR:
            return ("quantity", "data", _sig(float(raw) * DATA_FACTOR[u]), op)
        if u.lower() in OTHER_UNITS:
            dim, factor = OTHER_UNITS[u.lower()]
            return ("quantity", dim, _sig(float(raw) * factor), op)
        return ("quantity", "count", _sig(float(raw)), op)
    if kind == "duration":
        unit = DURATION_CANON.get(_singular(str(value.get("unit") or "")), str(value.get("unit")))
        dim, factor = DURATION_UNITS.get(unit, ("?", 1.0))
        return ("duration", dim, _sig(float(raw) * factor), op)
    if kind == "money":
        return ("money", _sig(float(raw)), value.get("currency"), value.get("per"), op)
    if kind == "percent":
        return ("percent", _sig(float(raw)), op)
    if kind == "date":
        return ("date", str(raw), op)
    if kind == "version":
        return ("version", str(raw), op)
    if kind == "name":
        return ("name", re.sub(r"\s+", "", str(raw).lower()), op)
    if kind == "boolean":
        return ("boolean", bool(raw))
    return (kind, str(raw), op)


# ---------------------------------------------------------------- helpers

def _op_before(text: str, start: int, table=OPS_BEFORE) -> str | None:
    window = text[max(0, start - 22): start].lower()
    for pattern, op in table:
        if re.search(rf"\b(?:{pattern})\s*$", window):
            return op
    return None


def _op_after(text: str, end: int) -> str | None:
    window = text[end: end + 14].lower()
    if window.startswith("+"):
        return "ge"
    for pattern, op in OPS_AFTER:
        if re.match(rf"\s*(?:{pattern})\b", window):
            return op
    return None


def _approx(text: str, start: int, end: int) -> bool:
    if re.search(APPROX_BEFORE, text[max(0, start - 20): start], re.I):
        return True
    return bool(re.match(r"\s*(?:or so|give or take|-ish)\b", text[end: end + 14], re.I))


def _in_range(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 12): start].lower()
    after = text[end: end + 14].lower()
    if re.search(r"between\s*$", before) or re.match(r"\s*(?:-|–|to)\s*[€$£]?\d", after):
        return True
    return bool(re.search(r"\d\s*(?:-|–|to)\s*[€$£]?$", before))


def _free(taken: list[tuple[int, int]], start: int, end: int) -> bool:
    return all(end <= s or start >= e for s, e in taken)


# ---------------------------------------------------------------- parsers

def _dates(text: str, taken, out):
    patterns = [
        (rf"\b(\d{{4}})-(\d{{2}})-(\d{{2}})\b", "ymd"),
        (rf"\b(?:the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s*({MONTH_RE})\s+(\d{{4}})\b", "dmy"),
        (rf"\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", "mdy"),
        (rf"\b(\d{{4}})-(\d{{2}})\b(?!-)", "ym"),
        (rf"\b({MONTH_RE})\s+(\d{{4}})\b", "my"),
    ]
    for pattern, shape in patterns:
        for m in re.finditer(pattern, text, re.I):
            if not _free(taken, m.start(), m.end()):
                continue
            g = m.groups()
            try:
                if shape == "ymd":
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                elif shape == "dmy":
                    d, mo, y = int(g[0]), MONTHS[g[1].lower().rstrip(".")], int(g[2])
                elif shape == "mdy":
                    mo, d, y = MONTHS[g[0].lower().rstrip(".")], int(g[1]), int(g[2])
                elif shape == "ym":
                    y, mo, d = int(g[0]), int(g[1]), None
                else:
                    mo, y, d = MONTHS[g[0].lower().rstrip(".")], int(g[1]), None
            except (KeyError, ValueError):
                continue
            if not (1 <= mo <= 12) or (d is not None and not 1 <= d <= 31):
                continue
            value = f"{y:04d}-{mo:02d}" + (f"-{d:02d}" if d else "")
            v = {"kind": "date", "value": value}
            op = _op_before(text, m.start(), DATE_OPS)
            if op:
                v["op"] = op
            taken.append((m.start(), m.end()))
            out.append(ValueMention("date", v, m.start(), m.end()))


def _money(text: str, taken, out):
    sym = r"(?:US\$|€|\$|£)"
    patterns = [
        rf"({sym}|\b(?:{CURRENCY_CODES})\s?)({DIGIT_NUM_RE}(?:,\d{{2}})?)({SCALE_RE}){PERIOD_RE}",
        rf"\b({DIGIT_NUM_RE}(?:,\d{{2}})?)({SCALE_RE})\s?(€|\b(?:{CURRENCY_CODES})\b|euros?\b|dollars?\b|lei\b){PERIOD_RE}",
    ]
    for i, pattern in enumerate(patterns):
        for m in re.finditer(pattern, text):
            if not _free(taken, m.start(), m.end()):
                continue
            if i == 0:
                cur_raw, num_raw, scale, p1, p2 = m.groups()
            else:
                num_raw, scale, cur_raw, p1, p2 = m.groups()
            cur_raw = cur_raw.strip()
            currency = (CURRENCY_SYMBOLS.get(cur_raw.lower()) or CURRENCY_WORDS.get(cur_raw.lower())
                        or cur_raw.upper())
            if re.fullmatch(r"\d+,\d{2}", num_raw):
                num_raw = num_raw.replace(",", ".")
            number = to_number(num_raw, scale.strip() or None)
            if number is None:
                continue
            v = {"kind": "money", "value": clean_number(number), "currency": currency}
            period = p1 or p2
            if period:
                v["per"] = PERIODS[period.lower()]
            op = _op_before(text, m.start())
            if op:
                v["op"] = op
            taken.append((m.start(), m.end()))
            out.append(ValueMention("money", v, m.start(), m.end(),
                                    approx=_approx(text, m.start(), m.end())))


def _percent(text: str, taken, out):
    for m in re.finditer(rf"\b({NUM_RE})\s?(%|percent\b|per cent\b)", text, re.I):
        if not _free(taken, m.start(), m.end()):
            continue
        number = to_number(m.group(1))
        if number is None:
            continue
        v = {"kind": "percent", "value": clean_number(number)}
        op = _op_before(text, m.start()) or _op_after(text, m.end())
        if op:
            v["op"] = op
        taken.append((m.start(), m.end()))
        out.append(ValueMention("percent", v, m.start(), m.end(),
                                approx=_approx(text, m.start(), m.end())))


def _versions(text: str, taken, out):
    patterns = [
        (r"\b(?:v|version\s|ver\.\s?)(\d+(?:\.\d+)*)\b", re.I),
        (r"\b(\d+\.\d+\.\d+(?:\.\d+)*)\b", 0),
        # one decimal point straight after a capitalised name: "Hooly Gemini 2.2"
        (r"\b[A-Z][a-zA-Z/]{1,20} (\d+\.\d+)\b(?![.,]\d)(?!\s?(?:%|percent|GB|MB|TB|kg|g\b|s\b|ms|h\b|min|GHz|inch|stars?\b|\"|″))", 0),
    ]
    for pattern, flags in patterns:
        for m in re.finditer(pattern, text, flags):
            start, end = m.start(1), m.end(1)
            if not _free(taken, start, end):
                continue
            v = {"kind": "version", "value": m.group(1)}
            taken.append((start, end))
            out.append(ValueMention("version", v, start, end))


def _durations(text: str, taken, out):
    pattern = rf"\b({NUM_RE}|an?)(?:\s?|-)({DURATION_WORD_RE})\b(?!\s?/)"
    for m in re.finditer(pattern, text, re.I):
        if not _free(taken, m.start(), m.end()):
            continue
        num_raw, unit_raw = m.group(1), m.group(2)
        unit_l = unit_raw.lower()
        if len(unit_l) == 1 and unit_raw != unit_l:
            continue  # single-letter units are lower case
        if unit_l in ("s", "d", "h") and not re.match(r"\d", num_raw):
            continue
        if num_raw.lower() in ("a", "an"):
            if not re.search(
                r"\b(?:under|within|in|for|over|than|at least|at most|to|only|just|about|around)\s*$",
                text[max(0, m.start() - 14): m.start()], re.I,
            ):
                continue
            number = 1.0
        else:
            number = to_number(num_raw)
        if number is None:
            continue
        unit = DURATION_CANON.get(_singular(unit_l), DURATION_CANON.get(unit_l))
        if unit is None:
            continue
        if unit_l.startswith("fortnight"):
            number, unit = number * 2, "week"
        if unit_l.startswith("decade"):
            number, unit = number * 10, "year"
        v = {"kind": "duration", "value": clean_number(number), "unit": unit}
        op = _op_before(text, m.start())
        if op is None and re.search(r"\bwithin\s*$", text[max(0, m.start() - 10): m.start()], re.I):
            op = "le"
        op = op or _op_after(text, m.end())
        if op:
            v["op"] = op
        taken.append((m.start(), m.end()))
        out.append(ValueMention("duration", v, m.start(), m.end(),
                                approx=_approx(text, m.start(), m.end())))


_UNIT_SYMBOLS = (
    r"KB|kB|kb|MB|mb|GB|gb|TB|tb|B|bytes?|kilobytes?|megabytes?|gigabytes?|gigs?|terabytes?|"
    r"mm|cm|km|m|in|inch(?:es)?|millimet(?:re|er)s?|centimet(?:re|er)s?|kilomet(?:re|er)s?|met(?:re|er)s?|"
    r"kg|g|grams?|grammes?|kilograms?|kilos?|mAh|kHz|MHz|GHz|Hz|hertz|kilohertz|megahertz|gigahertz|"
    r"mW|kW|W|milliwatts?|watts?|kilowatts?"
)
_COUNT_STOP = {
    "of", "the", "a", "an", "and", "or", "to", "in", "on", "at", "for", "per", "with", "by",
    "from", "is", "are", "was", "were", "has", "have", "had", "so", "far", "more", "times", "x",
}


def _quantities(text: str, taken, out):
    unit_pattern = rf"\b({NUM_RE})({SCALE_RE})(?:\s?|-)({_UNIT_SYMBOLS})(?![A-Za-z])|\b({DIGIT_NUM_RE})(\"|″)"
    for m in re.finditer(unit_pattern, text):
        if not _free(taken, m.start(), m.end()):
            continue
        if m.group(4):
            number, unit_raw = to_number(m.group(4)), m.group(5)
        else:
            number = to_number(m.group(1), m.group(2).strip() or None)
            unit_raw = m.group(3)
        if number is None:
            continue
        u = unit_raw.lower()
        if u in DATA_UNITS:
            unit = DATA_UNITS[u]
        elif unit_raw in ("mAh",):
            unit = "mAh"
        else:
            key = _singular(u) if u not in ("in", "m", "g", "mm", "cm", "km", "kg", "w", "mw", "kw") else u
            if key not in OTHER_UNITS and u not in OTHER_UNITS:
                continue
            unit = "in" if key in ("inch", "in", '"', "″") else unit_raw
            if u == "m" and unit_raw == "M":
                continue
        v = {"kind": "quantity", "value": clean_number(number), "unit": unit}
        op = _op_before(text, m.start()) or _op_after(text, m.end())
        if op:
            v["op"] = op
        taken.append((m.start(), m.end()))
        out.append(ValueMention("quantity", v, m.start(), m.end(),
                                approx=_approx(text, m.start(), m.end())))

    count_pattern = rf"\b({NUM_RE})({SCALE_RE})(\+?)((?:\s+[A-Za-z][\w-]*){{1,3}})"
    for m in re.finditer(count_pattern, text, re.I):
        num_raw = m.group(1)
        start, end_num = m.start(), m.start(1) + len(num_raw) + len(m.group(2)) + len(m.group(3))
        if not _free(taken, start, end_num):
            continue
        if re.fullmatch(r"(19|20)\d\d", num_raw):
            continue  # a plain four-digit number is a year
        if re.match(r"\d", num_raw) and re.fullmatch(r"\d+\.\d{3}", num_raw):
            continue
        if text[max(0, start - 1): start] in (".", "/", "-") and start > 0 and text[start - 2: start - 1].isdigit():
            continue
        number = to_number(num_raw, m.group(2).strip() or None)
        if number is None:
            continue
        words = m.group(4).split()
        noun = None
        for w in words:
            wl = w.lower().strip(",.;:")
            if wl in _COUNT_STOP:
                if noun is None and wl not in ("of",):
                    break
                continue
            noun = _singular(wl)
            break
        if noun is None:
            continue
        v = {"kind": "quantity", "value": clean_number(number), "unit": noun}
        op = _op_before(text, start) or ("ge" if m.group(3) else None) or _op_after(text, end_num)
        if op:
            v["op"] = op
        taken.append((start, end_num))
        out.append(ValueMention("quantity", v, start, end_num,
                                approx=_approx(text, start, end_num), extra={"noun": noun}))


def _names(text: str, names: list[tuple[str, str]], taken, out):
    """`names` = (surface form, canonical name), longest first; whole words, any case."""
    lowered = text.lower()
    for surface, canon_name in names:
        for m in re.finditer(rf"(?<![\w-]){re.escape(surface.lower())}(?![\w-])", lowered):
            if not _free(taken, m.start(), m.end()):
                continue
            taken.append((m.start(), m.end()))
            out.append(ValueMention("name", {"kind": "name", "value": canon_name}, m.start(), m.end()))


def parse_values(text: str, names: list[tuple[str, str]] | None = None) -> list[ValueMention]:
    """Every value mention in `text`, in reading order. Ranges are dropped."""
    taken: list[tuple[int, int]] = []
    out: list[ValueMention] = []
    _dates(text, taken, out)
    _money(text, taken, out)
    _percent(text, taken, out)
    _versions(text, taken, out)
    _durations(text, taken, out)
    _quantities(text, taken, out)
    if names:
        _names(text, names, taken, out)
    out = [m for m in out if not _in_range(text, m.start, m.end)]
    return sorted(out, key=lambda m: m.start)

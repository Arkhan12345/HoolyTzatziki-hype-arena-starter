"""Text-only features (label-free), ported from X's research (eda/features.py text_flags)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
NUM_WORDS = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred)\b"

PATTERNS = {
    "first_person": r"\b(?:I|I'm|I've|me|my|mine)\b",
    "we_our": r"\b(?:we|our|us)\b",
    "quote": r"[\"“”]|\bsaid\b|\bsays\b|\btold\b",
    "hearsay": r"\b(?:I have been told|told me|reportedly|apparently|rumou?r|I hear|they say|word is)\b",
    "hedge": r"\b(?:about|around|nearly|almost|roughly|over|more than|up to|under|less than|at least|at most|some)\b",
    "superlative": r"\b(?:\w+est|most|best|first|only|record|ever|any competitor|more than any|all humans|nobody|everyone|flawless|bargain|vertical)\b",
    "now_today": r"\b(?:now|today|from today|this week|as of)\b",
    "still": r"\bstill\b",
    "future": r"\b(?:will|is due|are due|plans? to|next|soon|coming|upcoming|from \d)\b",
    "negation": r"\b(?:not|no|never|none|without|n't)\b",
    "change_verb": r"\b(?:grows?|rises?|drops?|falls?|cuts?|raises?|increases?|doubles?|moves?|switch(?:es)?|replaces?|up from|down from|was|went from)\b",
    "money_sym": r"[€$£]|\bEUR\b|\bUSD\b",
    "number_words": NUM_WORDS,
    "compare_word": r"\b(?:more|less|fewer|higher|lower|greater|larger|bigger|smaller|longer|shorter|cheaper|than|before|after|earlier|later)\b",
}
JOKE_TAIL = re.compile(
    r"(,| and | while | which | but ).*\b(I|me|my|nobody|everyone|all humans|curve|tired|lunch|coffee|"
    r"personal best|poetry|flawless|learning|impossible|higher|vertical|bargain)\b", re.I)
_COMPILED = {k: re.compile(p, 0 if k == "first_person" else re.I) for k, p in PATTERNS.items()}
_KV = re.compile(r"^[^.]{2,40}: ")
_PRON = re.compile(r"^(He|She|It|Its|They|Their|His|Her|This|That)\b")
_NUM = re.compile(r"\d[\d,.]*")


def text_dates(text: str):
    t = text.lower()
    out = [(int(y), MONTHS[m], int(d)) for d, m, y in re.findall(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", t)]
    out += [(int(y), MONTHS[m], int(d)) for m, d, y in re.findall(r"\b(" + "|".join(MONTHS) + r") (\d{1,2}), (\d{4})", t)]
    if not out:
        out = [(int(y), MONTHS[m], None) for m, y in re.findall(r"\b(" + "|".join(MONTHS) + r") (\d{4})", t)]
    return out


def flags(text: str, date: str) -> dict:
    s = text or ""
    out = {k: int(bool(rx.search(s))) for k, rx in _COMPILED.items()}
    out["joke_tail"] = int(bool(JOKE_TAIL.search(s)))
    out["no_period"] = int(not s.strip().endswith((".", "!", "?", '"', "”")))
    out["kv_line"] = int(bool(_KV.search(s)))
    out["n_chars"] = len(s)
    out["n_words"] = len(s.split())
    out["n_numbers"] = len(_NUM.findall(s))
    out["n_commas"] = s.count(",")
    out["n_semicolons"] = s.count(";")
    out["starts_pronoun"] = int(bool(_PRON.search(s)))
    out["hype_score"] = out["first_person"] + out["superlative"] + out["joke_tail"] + out["hearsay"]
    delta = np.nan
    ds = text_dates(s)
    if ds:
        y, m, dd = ds[0]
        try:
            d0 = pd.Timestamp(str(date)[:10])
            td = pd.Timestamp(y, m, dd or 1)
            delta = (td - d0).days if dd else (td.to_period("M") - d0.to_period("M")).n * 30
        except (ValueError, TypeError):
            delta = np.nan
    out["text_date_delta_days"] = delta
    out["has_text_date"] = int(not np.isnan(delta))
    out["text_date_in_past"] = int(not np.isnan(delta) and delta < -1)
    out["text_date_in_future"] = int(not np.isnan(delta) and delta > 1)
    return out


from app.generate.restate import is_table_line  # noqa: E402,F401  (one definition, shared)

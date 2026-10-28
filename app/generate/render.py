"""Render values exactly as contracts/value-forms.md reads them."""
from __future__ import annotations

import re

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}
PHYSICAL_TEXT = {"in": "-inch"}  # 6.1-inch reads; a bare "in" is ambiguous in prose
DURATION_WORD = {"ms": ("ms", "ms"), "s": ("second", "seconds"), "min": ("minute", "minutes"),
                 "h": ("hour", "hours"), "day": ("day", "days"), "week": ("week", "weeks"),
                 "month": ("month", "months"), "year": ("year", "years")}
IRREGULAR_PLURAL = {"person": "people", "child": "children", "man": "men", "woman": "women"}
PHYSICAL_UNITS = {"B", "KB", "MB", "GB", "TB", "mm", "cm", "m", "km", "in", "g", "kg", "mAh",
                  "Hz", "kHz", "MHz", "GHz", "mW", "W", "kW"}


def number(x, percent: bool = False, money: bool = False) -> str | None:
    """Digits with thousands commas and a decimal point; None when it would read ambiguously."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if float(x).is_integer():
        return f"{int(x):,}"
    s = f"{x:,.6f}".rstrip("0").rstrip(".")
    decimals = s.split(".")[1] if "." in s else ""
    if len(decimals) == 3 and not percent:
        return None  # 1.500 could be fifteen hundred
    if money and len(decimals) > 2:
        return None
    return s


def plural(noun: str) -> str:
    n = noun.lower()
    if n in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[n]
    if re.search(r"[^aeiou]y$", n):
        return noun[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", n):
        return noun + "es"
    return noun + "s"


def date_text(v: str) -> str | None:
    m = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", v or "")
    if not m:
        return None
    y, mo, d = m.groups()
    if not mo:
        return y
    if not 1 <= int(mo) <= 12:
        return None
    if not d:
        return f"{MONTHS[int(mo) - 1]} {y}"
    return f"{int(d)} {MONTHS[int(mo) - 1]} {y}"


def value_text(value: dict) -> str | None:
    """The written form of a claim value, or None if it cannot be written unambiguously."""
    if not isinstance(value, dict) or "op" in value:
        return None
    k, v = value.get("kind"), value.get("value")
    if k == "quantity":
        n, unit = number(v), value.get("unit", "")
        if n is None or not unit:
            return None
        if unit in PHYSICAL_UNITS:
            return f"{n}{PHYSICAL_TEXT[unit]}" if unit in PHYSICAL_TEXT else f"{n} {unit}"
        if not re.fullmatch(r"[A-Za-z]+", unit):
            return None
        return f"{n} {unit if v == 1 else plural(unit)}"
    if k == "money":
        n, cur = number(v, money=True), value.get("currency", "")
        if n is None or not re.fullmatch(r"[A-Z]{3}", cur):
            return None
        amount = f"{SYMBOL[cur]}{n}" if cur in SYMBOL else f"{n} {cur}"
        return amount + (f" per {value['per']}" if value.get("per") else "")
    if k == "percent":
        n = number(v, percent=True)
        return None if n is None else f"{n}%"
    if k == "date":
        return date_text(str(v))
    if k == "version":
        return f"version {v}" if re.fullmatch(r"\d+(\.\d+){0,3}", str(v)) else None
    if k == "duration":
        n, unit = number(v), value.get("unit")
        if n is None or unit not in DURATION_WORD:
            return None
        one, many = DURATION_WORD[unit]
        return f"{n} {one if v == 1 else many}"
    if k == "name":
        s = " ".join(str(v).split())
        return s or None
    return None  # booleans are written as the sentence's polarity, not as a value


def attr_words(attribute: str) -> str:
    return attribute.replace("_", " ")

"""Value normalization: one comparable key per value, units converted (value-forms.md)."""
from __future__ import annotations

import json

# Physical units: unit -> (dimension, factor to the dimension's base unit).
PHYSICAL = {
    "B": ("data", 1), "KB": ("data", 1e3), "MB": ("data", 1e6), "GB": ("data", 1e9), "TB": ("data", 1e12),
    "mm": ("length", 1e-3), "cm": ("length", 1e-2), "m": ("length", 1), "km": ("length", 1e3), "in": ("length", 0.0254),
    "g": ("mass", 1), "kg": ("mass", 1e3),
    "mAh": ("charge", 1),
    "Hz": ("frequency", 1), "kHz": ("frequency", 1e3), "MHz": ("frequency", 1e6), "GHz": ("frequency", 1e9),
    "mW": ("power", 1e-3), "W": ("power", 1), "kW": ("power", 1e3),
}
# Durations: days and smaller convert to seconds; months and years to months; the two never mix.
DURATION = {
    "ms": ("time", 1e-3), "s": ("time", 1), "min": ("time", 60), "h": ("time", 3600),
    "day": ("time", 86400), "week": ("time", 604800),
    "month": ("calendar", 1), "year": ("calendar", 12),
}

IRREGULAR_SINGULAR = {"people": "person", "children": "child", "men": "man", "women": "woman"}


def singular(noun: str) -> str:
    n = noun.strip().lower()
    if n in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[n]
    if n.endswith("ies") and len(n) > 4:
        return n[:-3] + "y"
    if n.endswith(("ches", "shes", "sses", "xes", "zes")):
        return n[:-2]
    if n.endswith("s") and not n.endswith("ss") and len(n) > 3:
        return n[:-1]
    return n


def _num(x) -> float:
    return round(float(x), 9)


def dimension(value: dict) -> tuple | None:
    """What a value can be compared with: same dimension tuple = comparable."""
    k = value.get("kind")
    if k == "quantity":
        u = value.get("unit", "")
        if u in PHYSICAL:
            return ("quantity", PHYSICAL[u][0])
        return ("quantity", "count", singular(u))
    if k == "duration":
        return ("duration", DURATION.get(value.get("unit"), ("?", 1))[0])
    if k == "money":
        return ("money", value.get("currency"), value.get("per") or "once")
    if k in ("percent", "date", "version"):
        return (k,)
    return None  # names and booleans do not compare


def magnitude(value: dict):
    """A number (or date/version tuple) to order comparable values by."""
    k = value.get("kind")
    if k == "quantity":
        u = value.get("unit", "")
        return _num(value["value"] * (PHYSICAL[u][1] if u in PHYSICAL else 1))
    if k == "duration":
        return _num(value["value"] * DURATION.get(value.get("unit"), ("?", 1))[1])
    if k in ("money", "percent"):
        return _num(value["value"])
    if k == "version":
        return tuple(int(p) for p in str(value["value"]).split("."))
    if k == "date":
        return str(value["value"])
    return None


def key(value: dict) -> str:
    """Canonical equality key: two values with the same key state the same thing."""
    k = value.get("kind")
    op = value.get("op", "eq")
    if k in ("quantity", "duration", "money", "percent"):
        return json.dumps([k, dimension(value), magnitude(value), op])
    if k == "name":
        return json.dumps([k, " ".join(str(value.get("value", "")).lower().split()), op])
    return json.dumps([k, value.get("value"), op])


def same(a: dict, b: dict) -> bool:
    return key(a) == key(b)


def date_prefix_related(a: dict, b: dict) -> bool:
    """True when one date is a less precise form of the other (2026 vs 2026-06-02)."""
    if a.get("kind") != "date" or b.get("kind") != "date":
        return False
    x, y = str(a["value"]), str(b["value"])
    return x != y and (x.startswith(y) or y.startswith(x))


def compare(a: dict, b: dict) -> str | None:
    """'gt' / 'lt' / 'eq' between comparable values, else None.

    Dates of different precision compare only when they don't overlap.
    """
    if dimension(a) is None or dimension(a) != dimension(b):
        return None
    if a.get("kind") == "date":
        x, y = str(a["value"]), str(b["value"])
        if x == y:
            return "eq" if len(x) == 10 else None
        if x.startswith(y) or y.startswith(x):
            return None
        n = min(len(x), len(y))
        return "gt" if x[:n] > y[:n] else "lt" if x[:n] < y[:n] else None
    ma, mb = magnitude(a), magnitude(b)
    if ma is None or mb is None:
        return None
    if a.get("kind") == "version" and str(a["value"]).count(".") != str(b["value"]).count("."):
        return None
    return "gt" if ma > mb else "lt" if ma < mb else "eq"

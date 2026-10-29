"""Referent checks for v2: confusable attribute pairs, and value plausibility within an attribute across subjects.

Two values only contradict each other when they talk about the same thing. The attribute reader sometimes picks a
neighbour (median vs p99 response time, venue vs city, cash vs revenue); `confusable` names those neighbours, from the
attribute confusions measured on practice (eda/out/confusable_attributes.csv) plus a small fixed list of qualifier
pairs. `plausibility` places a value among the true values of the same attribute over every subject in the reader's
knowledge base, so a row about an unknown company still gets a signal ("a refund window of 400 days is odd").
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

_FIXED = {("median_response_time", "p99_response_time"), ("venue", "city"), ("cash", "revenue"),
          ("refund_window", "free_trial"), ("stable_release_date", "launched"), ("api_version", "version"),
          ("max_output_length", "context_window"), ("users", "monthly_active_users")}
_PAIRS: set | None = None


def _pairs() -> set:
    global _PAIRS
    if _PAIRS is None:
        pairs = {tuple(sorted(p)) for p in _FIXED}
        f = Path(__file__).resolve().parents[2] / "out" / "confusable_attributes.csv"
        try:
            for r in pd.read_csv(f).itertuples():
                if int(r.count) >= 2:
                    pairs.add(tuple(sorted((str(r.gold_attribute), str(r.predicted_attribute)))))
        except Exception:   # the table is optional at inference time
            pass
        _PAIRS = pairs
    return _PAIRS


def confusable(a: str, b: str) -> bool:
    return tuple(sorted((a, b))) in _pairs()


def _key_num(key) -> float | None:
    return next((x for x in key[1:] if isinstance(x, (int, float)) and not isinstance(x, bool)), None)


def _magnitude(value: dict) -> float:
    """log10 of the value on the same canonical (unit-scaled) scale the KB stores."""
    from .values import canonical
    try:
        x = _key_num(canonical(value))
    except Exception:
        return float("nan")
    return math.log10(x) if x is not None and x > 0 else float("nan")


def _attr_values(reader) -> dict:
    """attribute -> log10 magnitudes of every TRUE value in the reader's KB (cached on the reader)."""
    cache = getattr(reader, "_v2_attr_vals", None)
    if cache is None:
        cache = {}
        for (_, attr), tl in reader.kb.timeline.items():
            for _, key, lab in tl:
                if lab != "true":
                    continue
                num = _key_num(key)
                if num is not None and num > 0:
                    cache.setdefault(attr, []).append(math.log10(num))
        reader._v2_attr_vals = cache
    return cache


def plausibility(reader, attr: str, value: dict) -> tuple[float, int]:
    """Robust z of the value's log magnitude among the attribute's true values (NaN when fewer than 3)."""
    vals = _attr_values(reader).get(attr, [])
    x = _magnitude(value)
    if len(vals) < 3 or x != x:
        return float("nan"), len(vals)
    arr = np.asarray(vals)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826 or 0.25
    return float((x - med) / mad), len(vals)

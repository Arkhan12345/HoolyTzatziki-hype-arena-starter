"""Feature helpers P3's classifier can call per extracted fact."""
from __future__ import annotations

from app.kb import normalize as nz
from app.kb.store import KB


def fact_features(kb: KB, subject: str, attribute: str, value: dict, date: str) -> dict:
    v, support, hist = kb.lookup(subject, attribute, date)
    k = nz.key(value)
    gold = [m for m in hist if m.source == "gold"]
    wrong = [m for m in hist if m.source == "wrong"]
    agree = [m for m in hist if m.key == k and m.source in ("gold", "extracted")]
    disagree = [m for m in hist if m.key != k and m.source in ("gold", "extracted")]
    return {
        "kb_known": v is not None,
        "agrees_with_value_in_force": v is not None and nz.key(v) == k,
        "contradicts_value_in_force": v is not None and nz.key(v) != k,
        "support_in_force": support,
        "n_supporting_sources": len(agree),
        "n_disagreeing_sources": len(disagree),
        "gold_same_value_any_date": any(m.key == k for m in gold),
        "gold_same_value_same_date": any(m.key == k and m.date == date for m in gold),
        "known_wrong_value": any(m.key == k and m.date <= date for m in wrong),
        "value_changed_after": any(m.key != k and m.date > date for m in gold),
        "value_seen_later": any(m.key == k and m.date > date for m in gold),
        "days_since_last_gold": _days(max((m.date for m in gold if m.date <= date), default=""), date),
        "page_trust": kb.page_trust(date),
    }


def _days(a: str, b: str) -> float:
    from datetime import date as d

    try:
        return float((d.fromisoformat(b[:10]) - d.fromisoformat(a[:10])).days)
    except ValueError:
        return -1.0

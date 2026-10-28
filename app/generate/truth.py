"""Semantic truth check: re-derives each candidate's truth condition from the inventory alone.

Independent of the factories (it only reads the form), so a bug in a factory cannot slip a
claim through: the form must hold on the inventory's trusted evidence under claim-forms.md's rules.
Returns a list of problems (empty = holds).
"""
from __future__ import annotations

from app.generate.candidates import current, event_days, value_at
from app.kb import normalize as nz
from app.kb.inventory import FactInventory


def truth_problems(ctype: str, form: dict, inv: FactInventory) -> list[str]:
    try:
        return {"fact": _fact, "compare": _compare, "change": _change, "at_event": _at_event,
                "relation": _relation}[ctype](form, inv)
    except Exception as e:  # noqa: BLE001
        return [f"truth check error: {e!r}"[:120]]


def _fact(f, inv):
    v, _, _ = current(inv, f["subject"], f["attribute"])
    return [] if v is not None and nz.same(v, f["value"]) else ["fact: not the value in force on the as-of date"]


def _compare(f, inv):
    lv, _, _ = current(inv, f["left"]["subject"], f["left"]["attribute"])
    rv, _, _ = current(inv, f["right"]["subject"], f["right"]["attribute"])
    if lv is None or rv is None:
        return ["compare: a side has no current value"]
    if (f["left"]["subject"], f["left"]["attribute"]) == (f["right"]["subject"], f["right"]["attribute"]):
        return ["compare: a fact with itself"]
    op = nz.compare(lv, rv)
    return [] if op == f["op"] else [f"compare: values compare {op}, form says {f['op']}"]


def _change(f, inv):
    ev = inv.trusted(f["subject"], f["attribute"])
    a = [e.date for e in ev if nz.same(e.value, f["from"])]
    b = [e.date for e in ev if nz.same(e.value, f["to"])]
    if not a or not b:
        return ["change: a value is not in the trusted timeline"]
    if nz.same(f["from"], f["to"]):
        return ["change: from equals to"]
    return [] if min(a) < max(b) else ["change: 'to' is not later than 'from'"]


def _at_event(f, inv):
    days = event_days(inv, f["event"])
    if not days:
        return ["at_event: event has no trusted day"]
    for d in days:
        v, _, _ = value_at(inv, f["subject"], f["attribute"], d)
        if v is None or not nz.same(v, f["value"]):
            return [f"at_event: value not held on {d}"]
    return []


def _relation(f, inv):
    links = [e for e in inv.trusted(f["subject"], f["link"])
             if e.value.get("kind") == "name" and inv.lex.canonical(e.value["value"]) == inv.lex.canonical(f["target"])]
    if not links:
        return ["relation: the link never names the target"]
    lo, hi = min(e.date for e in links), max(e.date for e in links)
    tev = inv.trusted(f["target"], f["attribute"])
    keys = {e.key for e in tev}
    if len(keys) == 1 and nz.same(tev[0].value, f["value"]):
        return []
    inside = [e for e in tev if lo <= e.date <= hi]
    if inside and {e.key for e in inside} == {nz.key(f["value"])}:
        return []
    return ["relation: value not held while the link named the target"]

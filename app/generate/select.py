"""Claim generation: candidates from the KB's firm facts, validated, diversified, exactly n.

Truth comes only from gold mentions (practice rows labelled true: every value of such a row
held on its date). Extracted facts are used only to veto a value that looks superseded.
"""
from __future__ import annotations

import itertools
import json
import time
import zlib
from collections import Counter, defaultdict

from app.generate import templates as T
from app.generate.render import value_text
from app.io.types import Claim
from app.kb import normalize as nz
from app.kb.store import AS_OF, KB
from app.validate.rules import norm_text, validate_claim

QUOTA = {"change": 330, "fact": 300, "compare": 220, "at_event": 90, "relation": 60}
PER_PAIR_CAP = 6
STABLE_ATTRS = ("joined", "founded", "launched", "date", "previous_employer", "founders", "born",
                "started", "opened", "headquarters")


class _Cand:
    __slots__ = ("type", "text", "form", "conf", "pairs", "meta")

    def __init__(self, type_, text, form, conf, pairs, meta=None):
        self.type, self.text, self.form, self.conf, self.pairs, self.meta = type_, text, form, conf, pairs, meta or {}


def _gold_values(kb: KB, subject: str, attribute: str):
    """Gold mentions with exact values, sorted by date."""
    tl = kb.timeline(subject, attribute)
    if not tl:
        return []
    return [m for m in tl.gold() if "op" not in m.value]


def _distinct_runs(ms):
    """Consecutive runs of the same value: [(key, value, first_date, last_date)]."""
    runs = []
    for m in ms:
        if runs and runs[-1][0] == m.key:
            runs[-1][3] = m.date
        else:
            runs.append([m.key, m.value, m.date, m.date])
    return runs


def _superseded(kb: KB, subject: str, attribute: str, value: dict, after: str) -> bool:
    """A later wrong-labelled mention of this value, or later extracted mentions that disagree."""
    tl = kb.timeline(subject, attribute)
    if not tl:
        return False
    k = nz.key(value)
    later = [m for m in tl.mentions if m.date > after]
    if any(m.source == "wrong" and m.key == k for m in later):
        return True
    against = sum(m.weight for m in later if m.source == "extracted" and m.key != k)
    agree = sum(m.weight for m in later if m.source == "extracted" and m.key == k)
    return against >= 1.5 and against > 2 * agree


def _current(kb: KB, subject: str, attribute: str):
    """(value, confidence) on the as-of date from gold, or (None, 0)."""
    ms = _gold_values(kb, subject, attribute)
    if not ms:
        return None, 0.0
    last = ms[-1]
    same_day = {m.key for m in ms if m.date == last.date}
    if len(same_day) > 1:
        return None, 0.0
    if _superseded(kb, subject, attribute, last.value, last.date):
        return None, 0.0
    # Recency is what predicts a value still holds (tests/runtime: subsampled practice,
    # last mention from Sep on ~96% right, Jun-Aug ~75%, earlier ~50%).
    kind = last.value.get("kind")
    steady = attribute in STABLE_ATTRS or kind in ("date", "name")
    if last.date >= "2026-09-01":
        conf = 0.93
    elif last.date >= "2026-06-01":
        conf = 0.8 if steady else 0.72
    else:
        conf = 0.7 if steady else 0.5
    conf += min(0.04, 0.01 * (sum(1 for m in ms if m.key == last.key) - 1))
    return last.value, min(conf, 0.97)


# ---- candidates per type -----------------------------------------------------------------------
def fact_candidates(kb: KB):
    for (s, a) in kb.pairs():
        if not kb.lex.has_attribute(s, a):
            continue
        v, conf = _current(kb, s, a)
        if v is None or conf < 0.6:
            continue
        form = {"subject": s, "attribute": a, "value": v}
        for i, make in enumerate((T.fact_text, T.fact_text_alt)):
            text = make(s, a, v)
            if text:
                yield _Cand("fact", text, form, conf - 0.05 * i, [(s, a)], {"variant": i})


def change_candidates(kb: KB):
    for (s, a) in kb.pairs():
        if not kb.lex.has_attribute(s, a):
            continue
        runs = _distinct_runs(_gold_values(kb, s, a))
        seen = set()
        for i, j in itertools.combinations(range(len(runs)), 2):
            ki, vi, _, last_i = runs[i]
            kj, vj, first_j, _ = runs[j]
            if ki == kj or (ki, kj) in seen or first_j <= last_i:
                continue
            if vi.get("kind") == "boolean" or nz.date_prefix_related(vi, vj):
                continue
            if value_text(vi) == value_text(vj):
                continue
            seen.add((ki, kj))
            text = T.change_text(s, a, vi, vj)
            if text:
                conf = 0.95 if j == i + 1 else 0.9
                yield _Cand("change", text, {"subject": s, "attribute": a, "from": vi, "to": vj}, conf, [(s, a)])


def _event_dates(kb: KB, event: str) -> list[str]:
    ms = _gold_values(kb, event, "date")
    days = sorted({m.value["value"] for m in ms if len(str(m.value["value"])) == 10})
    return days


def _value_at(kb: KB, s: str, a: str, day: str):
    ms = _gold_values(kb, s, a)
    exact = {m.key: m for m in ms if m.date == day}
    if len(exact) == 1:
        return next(iter(exact.values())).value, 0.95
    if len(exact) > 1:
        return None, 0.0
    prev = [m for m in ms if m.date < day]
    nxt = [m for m in ms if m.date > day]
    if prev and nxt and prev[-1].key == nxt[0].key:
        return prev[-1].value, 0.88
    if prev and not nxt and len({m.key for m in ms}) == 1:
        return prev[-1].value, 0.8
    return None, 0.0


def at_event_candidates(kb: KB):
    for event in sorted(kb.lex.event_set):
        days = _event_dates(kb, event)
        if not days:
            continue
        for (s, a) in kb.pairs():
            if s == event or not kb.lex.has_attribute(s, a):
                continue
            vals = [_value_at(kb, s, a, d) for d in days]
            if any(v is None for v, _ in vals) or len({nz.key(v) for v, _ in vals}) != 1:
                continue
            v, conf = vals[0][0], min(c for _, c in vals) - (0.03 if len(days) > 1 else 0)
            text = T.at_event_text(s, a, event, v)
            if text:
                yield _Cand("at_event", text, {"subject": s, "attribute": a, "event": event, "value": v},
                            conf, [(s, a)])


def relation_candidates(kb: KB):
    for (s, link) in kb.pairs():
        if not kb.lex.has_attribute(s, link):
            continue
        ms = [m for m in _gold_values(kb, s, link) if m.value.get("kind") == "name"]
        by_target = defaultdict(list)
        for m in ms:
            t = kb.lex.canonical(m.value["value"])
            if t and t != s:
                by_target[t].append(m.date)
        for target, dates in by_target.items():
            lo, hi = min(dates), max(dates)
            for (ts, ta) in kb.pairs():
                if ts != target or not kb.lex.has_attribute(ts, ta):
                    continue
                tms = _gold_values(kb, ts, ta)
                if not tms:
                    continue
                keys = {m.key for m in tms}
                if len(keys) == 1:
                    v, conf = tms[-1].value, 0.9
                else:
                    inside = [m for m in tms if lo <= m.date <= hi]
                    if not inside or len({m.key for m in inside}) != 1:
                        continue
                    v, conf = inside[-1].value, 0.8
                text = T.relation_text(s, link, ta, v)
                if text:
                    yield _Cand("relation", text, {"subject": s, "link": link, "target": target, "attribute": ta,
                                                   "value": v}, conf, [(s, link), (ts, ta)])


def compare_candidates(kb: KB):
    facts = []
    for (s, a) in kb.pairs():
        if not kb.lex.has_attribute(s, a):
            continue
        cur, conf = _current(kb, s, a)
        if cur is None or nz.dimension(cur) is None:
            continue
        history = [m.value for m in _gold_values(kb, s, a)]
        facts.append((s, a, cur, conf, history))
    by_dim = defaultdict(list)
    for f in facts:
        by_dim[nz.dimension(f[2])].append(f)
    for dim, group in by_dim.items():
        between = "date" if dim == ("date",) else "amount"
        for (ls, la, lv, lc, lh), (rs, ra, rv, rc, rh) in itertools.combinations(group, 2):
            if (ls, la) == (rs, ra):
                continue
            if between == "amount" and la != ra:
                continue  # different amounts of one dimension compare, but read oddly; keep it simple
            op = nz.compare(lv, rv)
            if op is None:
                continue
            ops = {nz.compare(x, y) for x in lh for y in rh}
            robust = ops == {op}
            if op == "eq" and not robust:
                continue
            conf = 0.93 if robust else min(lc, rc) - 0.05
            if conf < 0.6:
                continue
            # alternate orientation for variety: flip half the pairs
            if (zlib.crc32(f"{ls}|{rs}|{la}".encode()) & 1) and op != "eq":
                (ls, la, rs, ra), op = (rs, ra, ls, la), {"gt": "lt", "lt": "gt"}[op]
            text = T.compare_text((ls, la), (rs, ra), op, between)
            form = {"left": {"subject": ls, "attribute": la}, "right": {"subject": rs, "attribute": ra}, "op": op}
            yield _Cand("compare", text, form, conf, [(ls, la), (rs, ra)], {"between": between})


# ---- selection ---------------------------------------------------------------------------------
def candidates(kb: KB):
    for gen in (change_candidates, fact_candidates, compare_candidates, at_event_candidates, relation_candidates):
        try:
            yield from gen(kb)
        except Exception as e:  # noqa: BLE001 - one broken generator must not stop the others
            import sys

            print(f"[generate] {gen.__name__} failed: {e!r}", file=sys.stderr)


def generate(kb: KB, lexicon: dict, input_texts: set[str], n: int = 1000, report: dict | None = None) -> list[Claim]:
    seen_text, seen_form = set(input_texts), set()
    valid: list[_Cand] = []
    rejected = Counter()
    n_cand, t_validate = 0, 0.0
    for c in candidates(kb):
        n_cand += 1
        nt = norm_text(c.text)
        fk = json.dumps([c.type, c.form], sort_keys=True)
        if nt in seen_text:
            rejected["duplicate text"] += 1
            continue
        probe = Claim(id="tz-probe", type=c.type, text=c.text, form=c.form, confidence=min(1, max(0.5, c.conf)))
        probe.meta = c.meta  # type: ignore[attr-defined]
        t = time.perf_counter()
        problems = validate_claim(probe, kb.lex, None)
        t_validate += time.perf_counter() - t
        if problems:
            rejected[problems[0].split(":")[0]] += 1
            continue
        seen_text.add(nt)
        valid.append(c)
        c.meta["form_key"] = fk
        c.meta["dup_form"] = fk in seen_form
        seen_form.add(fk)

    valid.sort(key=lambda c: (c.meta.get("dup_form", False), -c.conf, c.type, c.text))
    chosen, used = [], Counter()
    per_pair = Counter()
    chosen_ids = set()

    def take(c):
        chosen.append(c)
        chosen_ids.add(id(c))
        used[c.type] += 1
        for p in c.pairs:
            per_pair[p] += 1

    for c in valid:  # pass 1: quotas and spread
        if len(chosen) >= n:
            break
        if used[c.type] >= QUOTA.get(c.type, n) or c.meta.get("dup_form"):
            continue
        if any(per_pair[p] >= PER_PAIR_CAP for p in c.pairs):
            continue
        take(c)
    for c in valid:  # pass 2: anything valid, best first
        if len(chosen) >= n:
            break
        if id(c) not in chosen_ids:
            take(c)

    claims = [Claim(id=f"tz-{i + 1:04d}", type=c.type, text=c.text, form=c.form,
                    confidence=round(min(0.99, max(0.5, c.conf)), 3)) for i, c in enumerate(chosen[:n])]
    if report is not None:
        report.update({"candidates": n_cand, "candidates_valid": len(valid), "validate_s": t_validate, "rejected": dict(rejected.most_common()),
                       "by_type": dict(Counter(c.type for c in claims)), "n": len(claims)})
    return claims

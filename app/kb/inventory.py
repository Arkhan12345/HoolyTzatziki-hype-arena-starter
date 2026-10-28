"""Trusted fact inventory: the generator's whole view of the world.

The claim generator never sees rows, sentences or the extractor. It sees, per (subject, attribute),
the dated evidence the KB holds, already normalized (`key` = app.kb.normalize.key), plus the lexicon:

    inv.pairs()                         -> [(subject, attribute), ...] the lexicon knows
    inv.trusted(subject, attribute)     -> [Evidence] exact values that held on their date, by date
    inv.later(subject, attribute, d)    -> [Evidence] every mention (any source) after date d
    inv.lex                             -> app.kb.store.Lexicon

Build it from the KB (`FactInventory.from_kb(kb)`) in the pipeline, or from plain records in tests
(`FactInventory.from_records([...], lexicon)`), so the generator is tested without any extractor.

Trusted = gold (a practice row labelled true states it). With `trust_classified=True`, also the facts of
rows the sentence classifier labelled true with high confidence (KB.add_classified). With `trust_extracted=True`, an extracted
value also counts when >= 2 rows state it on one date and no gold or extracted mention that date
disagrees; off by default until P3's extraction precision is measured.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.kb import normalize as nz
from app.kb.store import KB, Lexicon


@dataclass(frozen=True)
class Evidence:
    date: str
    value: dict
    key: str
    source: str  # gold | consensus | extracted | wrong
    row_ids: tuple[str, ...]
    weight: float = 1.0


class FactInventory:
    def __init__(self, timelines: dict[tuple[str, str], list[Evidence]], lexicon: Lexicon,
                 rows: dict[str, dict] | None = None, trusted_sources=("gold", "consensus")):
        self.lex = lexicon
        self.rows = rows or {}  # row id -> {text, date, label, origin}: where each claim comes from
        self.trusted_sources = tuple(trusted_sources)
        self._all = {p: sorted(ev, key=lambda e: (e.date, e.row_ids)) for p, ev in timelines.items()}
        self._trusted = {p: [e for e in ev if e.source in self.trusted_sources and "op" not in e.value]
                         for p, ev in self._all.items()}
        self.gold_rows = {r for ev in self._all.values() for e in ev if e.source == "gold" for r in e.row_ids}

    # ---- construction --------------------------------------------------------------------------
    @classmethod
    def from_kb(cls, kb: KB, trust_extracted: bool = False, trust_classified: bool = False) -> "FactInventory":
        timelines: dict[tuple[str, str], list[Evidence]] = {}
        for (s, a) in kb.pairs():
            tl = kb.timeline(s, a)
            ev = [Evidence(m.date, m.value, m.key, m.source, (m.row_id,), m.weight) for m in tl.sorted()]
            if trust_extracted:
                ev += _consensus(ev)
            timelines[(s, a)] = ev
        trusted = ("gold", "consensus") + (("classified",) if trust_classified else ())
        return cls(timelines, kb.lex, kb.rows, trusted)

    @classmethod
    def from_records(cls, records: list[dict], lexicon: dict | Lexicon) -> "FactInventory":
        """records: {subject, attribute, value, date, source='gold', row_id='r<i>'} (test fixtures)."""
        lex = lexicon if isinstance(lexicon, Lexicon) else Lexicon(lexicon)
        timelines: dict[tuple[str, str], list[Evidence]] = defaultdict(list)
        rows = {}
        for i, r in enumerate(records):
            s = lex.canonical(r["subject"]) or r["subject"]
            rid = r.get("row_id", f"r{i}")
            timelines[(s, r["attribute"])].append(Evidence(
                r["date"], r["value"], nz.key(r["value"]), r.get("source", "gold"), (rid,)))
            rows[rid] = {"text": r.get("text", ""), "date": r["date"], "label": "true", "origin": "fixture"}
        return cls(dict(timelines), lex, rows)

    def sources(self, row_ids, limit: int = 8) -> list[dict]:
        """The sentences behind a claim, newest first (internal debug only, never in output.csv)."""
        seen, out = set(), []
        for rid in sorted(set(row_ids), key=lambda r: (self.rows.get(r, {}).get("date", ""), r), reverse=True):
            if rid in seen:
                continue
            seen.add(rid)
            info = self.rows.get(rid, {})
            out.append({"row_id": rid, "date": info.get("date"), "text": info.get("text"),
                        "label": info.get("label"), "origin": info.get("origin"),
                        **({"classified": info["classified"]} if "classified" in info else {})})
            if len(out) >= limit:
                break
        return out

    # ---- queries -------------------------------------------------------------------------------
    def pairs(self) -> list[tuple[str, str]]:
        return [p for p in self._all if self.lex.has_attribute(*p)]

    def trusted(self, subject: str, attribute: str) -> list[Evidence]:
        return self._trusted.get((self.lex.canonical(subject) or subject, attribute), [])

    def later(self, subject: str, attribute: str, after: str) -> list[Evidence]:
        return [e for e in self._all.get((self.lex.canonical(subject) or subject, attribute), []) if e.date > after]

    def stats(self) -> dict:
        n_trusted = sum(len(v) for v in self._trusted.values())
        return {"pairs": len(self._all), "pairs_in_lexicon": len(self.pairs()),
                "pairs_with_trusted": sum(1 for p in self.pairs() if self._trusted.get(p)),
                "trusted_mentions": n_trusted,
                "trusted_distinct_values": sum(len({e.key for e in v}) for v in self._trusted.values())}


def _consensus(ev: list[Evidence]) -> list[Evidence]:
    by_day: dict[str, list[Evidence]] = defaultdict(list)
    for e in ev:
        by_day[e.date].append(e)
    out = []
    for day, es in by_day.items():
        extracted = [e for e in es if e.source == "extracted"]
        keys = {e.key for e in es if e.source in ("gold", "extracted")}
        rows = {r for e in extracted for r in e.row_ids}
        if len(keys) == 1 and len(rows) >= 2 and not any(e.source == "gold" for e in es):
            e0 = extracted[0]
            out.append(Evidence(day, e0.value, e0.key, "consensus", tuple(sorted(rows)), min(e.weight for e in extracted)))
    return out

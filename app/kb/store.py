"""Temporal knowledge base: per (subject, attribute), a timeline of values by date.

Sources, strongest first:
  gold      facts of practice rows labelled true (the value held on the row's date)
  wrong     facts of practice rows labelled false with a single fact (that value was wrong then)
  extracted facts P3's extractor read from any universe row, with its confidence
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.kb import normalize as nz

AS_OF = "2026-10-29"
GOLD_WEIGHT = 1.0


@dataclass
class Mention:
    date: str
    value: dict
    key: str
    source: str  # gold | wrong | extracted
    weight: float
    row_id: str


@dataclass
class Timeline:
    subject: str
    attribute: str
    mentions: list[Mention] = field(default_factory=list)

    def sorted(self) -> list[Mention]:
        return sorted(self.mentions, key=lambda m: (m.date, nz.row_order(m.row_id)))

    def gold(self) -> list[Mention]:
        return [m for m in self.sorted() if m.source == "gold"]


class Lexicon:
    """Name/alias resolution and attribute lookup for the run's lexicon."""

    def __init__(self, lexicon: dict):
        self.raw = lexicon
        self.subjects: dict[str, dict] = {}
        self.alias: dict[str, str] = {}
        for s in lexicon.get("subjects", []):
            name = s["name"]
            self.subjects[name] = s
            for n in [name, *s.get("aliases", [])]:
                self.alias[self.fold(n)] = name
        self.events = [e if isinstance(e, str) else e.get("name", "") for e in lexicon.get("events", [])]
        self.event_set = {self.canonical(e) or e for e in self.events}

    @staticmethod
    def fold(name: str) -> str:
        return " ".join(str(name).lower().split())

    def canonical(self, name: str) -> str | None:
        return self.alias.get(self.fold(name))

    def names_of(self, subject: str) -> list[str]:
        s = self.subjects.get(subject, {})
        return [subject, *s.get("aliases", [])]

    def has_attribute(self, subject: str, attribute: str) -> bool:
        s = self.subjects.get(self.canonical(subject) or "")
        return bool(s) and attribute in s.get("attributes", [])

    def all_names(self) -> list[str]:
        return list(self.alias.keys())


class KB:
    def __init__(self, lexicon: dict | None = None):
        self.lex = Lexicon(lexicon or {"subjects": [], "events": []})
        self._tl: dict[tuple[str, str], Timeline] = {}
        self.row_dates: dict[str, str] = {}
        self.page_votes: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # date -> [agree, disagree]
        self.ingest: dict = {}  # adapter counters: extracted_in, extracted_dropped
        self.rows: dict[str, dict] = {}  # row id -> {text, date, label, origin}: provenance for claims

    # ---- build -------------------------------------------------------------------------------
    def _add(self, subject, attribute, value, date, source, weight, row_id):
        canon = self.lex.canonical(subject) or subject
        if not isinstance(value, dict) or "kind" not in value or not date:
            return
        tl = self._tl.setdefault((canon, attribute), Timeline(canon, attribute))
        tl.mentions.append(Mention(date, value, nz.key(value), source, weight, row_id))

    def add_gold(self, gold_rows: list[dict]) -> None:
        for r in gold_rows:
            label, form, date = r.get("label"), r.get("form") or [], r.get("date", "")
            self.row_dates[r["id"]] = date
            # the form lets the claim screening spot a fact that only rewords this row (app/generate/restate.py)
            self.rows[r["id"]] = {"text": r.get("text", ""), "date": date, "label": label, "origin": "practice",
                                  "form": form}
            if label == "true":
                for f in form:
                    self._add(f["subject"], f["attribute"], f["value"], date, "gold", GOLD_WEIGHT, r["id"])
            elif label == "false" and len(form) == 1:
                f = form[0]
                self._add(f["subject"], f["attribute"], f["value"], date, "wrong", GOLD_WEIGHT, r["id"])

    def add_extracted(self, facts, rows: list[dict] | None = None) -> None:
        """P3's facts (dataclass, object or dict; see app.kb.adapter), dated by their row."""
        from app.kb.adapter import to_records

        self._remember(rows)
        for f in to_records(facts, self.ingest):
            date = self.row_dates.get(f.row_id, "")
            self._add(f.subject, f.attribute, f.value, date, "extracted", f.conf, f.row_id)

    def add_classified(self, labels, facts, rows: list[dict] | None = None, min_conf: float = 0.9) -> int:
        """Structured output of the sentence classification (X/Arckan): the facts of every row the
        classifier labelled true with confidence >= min_conf become `classified` evidence for that
        row's date. Labels: objects or dicts with id, label, confidence. Returns how many were added."""
        from app.kb.adapter import _get, to_records

        self._remember(rows)
        sure = {}
        for lb in labels or []:
            rid, label, conf = _get(lb, "id"), _get(lb, "label"), _get(lb, "confidence")
            if rid and label == "true" and conf is not None and float(conf) >= min_conf:
                sure[str(rid)] = float(conf)
        n = 0
        for f in to_records(facts):
            if f.row_id in sure and self.row_dates.get(f.row_id):
                self._add(f.subject, f.attribute, f.value, self.row_dates[f.row_id], "classified",
                          min(sure[f.row_id], f.conf if f.conf > 0 else 1.0), f.row_id)
                self.rows.setdefault(f.row_id, {}).update(classified={"label": "true", "confidence": sure[f.row_id]})
                n += 1
        self.ingest["classified_facts"] = self.ingest.get("classified_facts", 0) + n
        return n

    def _remember(self, rows) -> None:
        for r in rows or []:
            self.row_dates.setdefault(r["id"], r.get("date", ""))
            self.rows.setdefault(r["id"], {"text": r.get("text", ""), "date": r.get("date", ""),
                                           "label": r.get("label") or None, "origin": "universe"})

    def build(self, facts, gold_rows: list[dict], rows: list[dict] | None = None, labels=None) -> "KB":
        self.add_gold(gold_rows)
        self.add_extracted(facts, rows)
        if labels is not None:
            self.add_classified(labels, facts, rows)
        self._page_trust()
        return self

    def _page_trust(self) -> None:
        for tl in self._tl.values():
            for m in tl.mentions:
                if m.source != "extracted":
                    continue
                v, support, _ = self.lookup(tl.subject, tl.attribute, m.date, sources=("gold",))
                if v is None:
                    continue
                self.page_votes[m.date][0 if nz.key(v) == m.key else 1] += 1

    # ---- query -------------------------------------------------------------------------------
    def timeline(self, subject: str, attribute: str) -> Timeline | None:
        return self._tl.get((self.lex.canonical(subject) or subject, attribute))

    def lookup(self, subject: str, attr: str, date: str, sources=("gold", "extracted")):
        """Value in force on `date`: the latest prior mention, gold first.

        Returns (value | None, support, history) where support is the summed weight of
        mentions agreeing with the returned value and history the sorted mentions.
        """
        tl = self.timeline(subject, attr)
        if not tl:
            return None, 0.0, []
        hist = tl.sorted()
        for src in sources:
            prior = [m for m in hist if m.source == src and m.date <= date]
            if not prior:
                continue
            last = prior[-1]
            support = sum(m.weight for m in hist if m.key == last.key and m.source in ("gold", "extracted"))
            return last.value, support, hist
        return None, 0.0, hist

    def current_value(self, subject: str, attr: str, as_of: str = AS_OF):
        return self.lookup(subject, attr, as_of)

    def pairs(self) -> list[tuple[str, str]]:
        return list(self._tl)

    def timelines(self) -> dict[tuple[str, str], list[dict]]:
        """Truth-discovery hook (P3's EM): per pair, its mentions by date.

        Each mention: date, value, page_id (the page = its date), row_id, from_gold, source, weight.
        """
        return {pair: [{"date": m.date, "value": m.value, "page_id": m.date, "row_id": m.row_id,
                        "from_gold": m.source == "gold", "source": m.source, "weight": m.weight}
                       for m in tl.sorted()]
                for pair, tl in self._tl.items()}

    def set_reliability(self, reliability: dict[str, float]) -> None:
        """Feed back page reliability (page_id -> 0..1) from P3's EM; page_trust() then returns it."""
        self.reliability = {str(k): float(v) for k, v in (reliability or {}).items()}

    def page_trust(self, date: str) -> float:
        rel = getattr(self, "reliability", None)
        if rel and date in rel:
            return rel[date]
        a, d = self.page_votes.get(date, (0, 0))
        return (a + 1) / (a + d + 2)

    def stats(self) -> dict:
        n = defaultdict(int)
        for tl in self._tl.values():
            for m in tl.mentions:
                n[m.source] += 1
        return {"pairs": len(self._tl), **n, **self.ingest}

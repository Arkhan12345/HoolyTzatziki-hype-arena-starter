"""Independent validator for generated claims: PASS / FAIL / UNVERIFIED, with reasons and evidence.

It shares no truth logic with the generator (app/generate): it reads the inventory's trusted evidence
and recomputes every truth condition of claim-forms.md on its own. Checks, in order:

  contract    entry schema, lexicon, value forms, claim forms, text <-> form   (app/validate/rules.py)
  truth       the form holds on the KB's dated evidence (as-of 29 Oct 2026 for fact/compare)
  provenance  identifiable source rows that state the claim's facts
  novelty     not an input sentence, and not a reformulation of one (see below)
  uniqueness  no other selected claim states the same information          (check_set)

Official rule (mentor clarification, Hype Arena, 28 Oct 2026, on top of contracts/): the same
information may appear only once, however it is worded: "A costs more than B." and "B costs less
than A." are one claim, and so are paraphrases and the same subject under an alias. A claim must also
not be a reformulation of an input sentence: when one input sentence names the claim's subjects and
already states every fact a fact, change, at_event or relation claim states, it is a reformulation; a
compare is one when a single sentence is about exactly its two facts and compares. A claim that ties a
value to a subject no sentence names next to it (a pronoun, a table row, "the hall") connects what
the universe says (README: "new claims ... that connect what the universe says") and is new.

UNVERIFIED = the evidence can neither confirm nor refute it (it is not selected). FAIL = refuted or invalid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.kb import normalize as nz
from app.kb.inventory import FactInventory
from app.validate.rules import norm_text, validate_claim

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"
FLIP = {"gt": "lt", "lt": "gt", "eq": "eq"}
COMPARISON_WORDS = re.compile(r"\b(more|less|fewer|higher|lower|greater|larger|bigger|smaller|longer|shorter|"
                              r"cheaper|than|same as|as much as|as many as|before|after|earlier|later)\b", re.I)


@dataclass
class Verdict:
    verdict: str
    checks: dict = field(default_factory=dict)  # check -> PASS / FAIL / UNVERIFIED
    reasons: list = field(default_factory=list)
    evidence: list = field(default_factory=list)  # row ids
    semantic_key: tuple = ()
    flags: list = field(default_factory=list)  # e.g. "novelty_weak": a sentence might say it (fallback tier)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "checks": self.checks, "reasons": self.reasons, "flags": self.flags,
                "evidence": self.evidence, "semantic_key": list(map(str, self.semantic_key))}


class ClaimValidator:
    def __init__(self, inv: FactInventory, input_texts: set[str] | None = None, as_of: str = "2026-10-29"):
        self.inv, self.lex, self.as_of = inv, inv.lex, as_of
        self.input_texts = input_texts or set()
        self.errors = 0  # validator errors (malformed forms); reported, never silently "unverified"

    # ---- semantic identity ---------------------------------------------------------------------
    def canon(self, name: str) -> str:
        return self.lex.canonical(name) or name

    def semantic_key(self, ctype: str, f: dict) -> tuple:
        if ctype == "compare":
            a = (self.canon(f["left"]["subject"]), f["left"]["attribute"])
            b = (self.canon(f["right"]["subject"]), f["right"]["attribute"])
            return ("compare", a, f["op"], b) if a <= b else ("compare", b, FLIP[f["op"]], a)
        if ctype == "change":
            return ("change", self.canon(f["subject"]), f["attribute"], nz.key(f["from"]), nz.key(f["to"]))
        if ctype == "at_event":
            return ("at_event", self.canon(f["subject"]), f["attribute"], self.canon(f["event"]), nz.key(f["value"]))
        if ctype == "relation":
            return ("relation", self.canon(f["subject"]), f["link"], self.canon(f["target"]), f["attribute"],
                    nz.key(f["value"]))
        return ("fact", self.canon(f["subject"]), f["attribute"], nz.key(f["value"]))

    # ---- evidence ------------------------------------------------------------------------------
    def _trusted(self, s, a):
        return self.inv.trusted(self.canon(s), a)

    def _in_force(self, s, a, day):
        """(value, rows) of the latest trusted mention on or before `day`; (None, []) if unknown/ambiguous."""
        ev = [e for e in self._trusted(s, a) if e.date <= day]
        if not ev:
            return None, []
        last = ev[-1]
        if len({e.key for e in ev if e.date == last.date}) > 1:
            return None, []
        later_wrong = [e for e in self.inv.later(self.canon(s), a, last.date) if e.source == "wrong" and e.key == last.key]
        if later_wrong and day >= later_wrong[0].date:
            return None, []
        return last.value, [r for e in ev if e.key == last.key for r in e.row_ids]

    def _held_on(self, s, a, day):
        """(PASS|FAIL|UNVERIFIED, value, rows): what the fact held on one day."""
        ev = self._trusted(s, a)
        exact = {e.key: e for e in ev if e.date == day}
        if len(exact) == 1:
            e = next(iter(exact.values()))
            return PASS, e.value, list(e.row_ids)
        if len(exact) > 1:
            return UNVERIFIED, None, []
        before = [e for e in ev if e.date < day]
        after = [e for e in ev if e.date > day]
        if before and after and before[-1].key == after[0].key:
            return PASS, before[-1].value, list(before[-1].row_ids + after[0].row_ids)
        if before and after:
            return UNVERIFIED, None, []  # it changed around that day: no way to say which held
        return UNVERIFIED, None, []

    @staticmethod
    def _matches(claimed: dict, known: dict) -> bool:
        return nz.same(claimed, known) or (claimed.get("kind") == "date" and nz.date_prefix_related(claimed, known)
                                           and len(str(claimed["value"])) >= len(str(known["value"])))

    # ---- truth per type ------------------------------------------------------------------------
    def truth(self, ctype: str, f: dict) -> tuple[str, list, list]:
        fn = {"fact": self._t_fact, "compare": self._t_compare, "change": self._t_change,
              "at_event": self._t_at_event, "relation": self._t_relation}[ctype]
        try:
            return fn(f)
        except (KeyError, TypeError, ValueError) as e:  # a malformed form: invalid, and said so
            self.errors += 1
            return FAIL, [f"truth: validator error {e!r}"[:120]], []

    def _t_fact(self, f):
        v, rows = self._in_force(f["subject"], f["attribute"], self.as_of)
        if v is None:
            return UNVERIFIED, ["no value in force on the as-of date"], []
        if self._matches(f["value"], v):
            # a more precise date must itself be stated after the last unrelated date
            if f["value"].get("kind") == "date" and not nz.same(f["value"], v):
                ev = self._trusted(f["subject"], f["attribute"])
                cv = str(v["value"])
                unrelated = [e.date for e in ev
                             if not (str(e.value["value"]).startswith(cv) or cv.startswith(str(e.value["value"])))]
                since = max(unrelated, default="")
                ok = [e for e in ev if nz.same(e.value, f["value"]) and e.date > since]
                if not ok:
                    return UNVERIFIED, ["the precise date is not stated since the value in force began"], []
                rows = rows + [r for e in ok for r in e.row_ids]
            return PASS, [], rows
        return FAIL, [f"value in force is {v}, not {f['value']}"], rows

    def _t_compare(self, f):
        L, R = f["left"], f["right"]
        if (self.canon(L["subject"]), L["attribute"]) == (self.canon(R["subject"]), R["attribute"]):
            return FAIL, ["a fact compared with itself"], []
        lv, lr = self._in_force(L["subject"], L["attribute"], self.as_of)
        rv, rr = self._in_force(R["subject"], R["attribute"], self.as_of)
        if lv is None or rv is None:
            return UNVERIFIED, ["a side has no value in force on the as-of date"], []
        if nz.dimension(lv) is None or nz.dimension(lv) != nz.dimension(rv):
            return FAIL, ["the two facts do not compare (kind, unit, currency or period)"], []
        op = nz.compare(lv, rv)
        if op is None:
            return UNVERIFIED, ["the values cannot be ordered (date precision)"], []
        if op != f["op"]:
            return FAIL, [f"values compare {op}, the claim says {f['op']}"], lr + rr
        return PASS, [], lr + rr

    def _t_change(self, f):
        ev = self._trusted(f["subject"], f["attribute"])
        a = [e for e in ev if nz.same(e.value, f["from"])]
        b = [e for e in ev if nz.same(e.value, f["to"])]
        if nz.same(f["from"], f["to"]):
            return FAIL, ["from and to are the same value"], []
        if not a or not b:
            return UNVERIFIED, ["a value of the change is not in the trusted timeline"], []
        if min(e.date for e in a) >= max(e.date for e in b):
            return FAIL, ["'to' is never seen after 'from'"], []
        return PASS, [], [r for e in a + b for r in e.row_ids]

    def _t_at_event(self, f):
        days = sorted({str(e.value["value"]) for e in self._trusted(f["event"], "date") if len(str(e.value["value"])) == 10})
        if not days:
            return UNVERIFIED, ["the event has no known day"], []
        rows = [r for e in self._trusted(f["event"], "date") for r in e.row_ids]
        for d in days:
            st, v, rr = self._held_on(f["subject"], f["attribute"], d)
            if st != PASS:
                return UNVERIFIED, [f"no evidence of the value on {d}"], []
            if not self._matches(f["value"], v):
                return FAIL, [f"on {d} the value was {v}"], rr
            rows += rr
        return PASS, [], rows

    def _t_relation(self, f):
        target = self.canon(f["target"])
        links = [e for e in self._trusted(f["subject"], f["link"])
                 if e.value.get("kind") == "name" and self.canon(e.value["value"]) == target]
        if not links:
            return UNVERIFIED, ["the link never names the target"], []
        lo, hi = min(e.date for e in links), max(e.date for e in links)
        tev = self._trusted(target, f["attribute"])
        if not tev:
            return UNVERIFIED, ["the target's fact is unknown"], []
        inside = [e for e in tev if lo <= e.date <= hi]
        rows = [r for e in links for r in e.row_ids]
        if inside:
            if all(self._matches(f["value"], e.value) for e in inside):
                return PASS, [], rows + [r for e in inside for r in e.row_ids]
            return FAIL, ["the target's value differs while the link named it"], rows
        before = [e for e in tev if e.date < lo]
        after = [e for e in tev if e.date > hi]
        if before and after and len({e.key for e in tev}) == 1 and self._matches(f["value"], tev[0].value):
            # the target's fact never changed and its mentions bracket the whole link period
            return PASS, [], rows + [r for e in (before[-1], after[0]) for r in e.row_ids]
        return UNVERIFIED, ["no evidence of the value while the link named the target"], []

    # ---- novelty, provenance -------------------------------------------------------------------
    def claim_items(self, ctype: str, f: dict) -> list[tuple[tuple[str, str], dict]]:
        if ctype == "fact":
            return [((self.canon(f["subject"]), f["attribute"]), f["value"])]
        if ctype == "change":
            p = (self.canon(f["subject"]), f["attribute"])
            return [(p, f["from"]), (p, f["to"])]
        if ctype == "relation":
            return [((self.canon(f["subject"]), f["link"]), {"kind": "name", "value": f["target"]}),
                    ((self.canon(f["target"]), f["attribute"]), f["value"])]
        if ctype == "at_event":
            return [((self.canon(f["subject"]), f["attribute"]), f["value"])]  # + the event's date, below
        return []

    def _index(self):
        if not hasattr(self, "_rows_by_fact"):
            from collections import defaultdict

            self._rows_by_fact, self._rows_by_pair = defaultdict(set), defaultdict(set)
            for rid, items in self.inv._by_row.items():
                for p, v in items:
                    self._rows_by_fact[(p, nz.key(v))].add(rid)
                    self._rows_by_pair[p].add(rid)
        return self._rows_by_fact, self._rows_by_pair

    def reformulates(self, ctype: str, f: dict) -> str | None:
        """Row id of an input sentence this claim merely restates, else None."""
        by_fact, by_pair = self._index()
        if ctype == "compare":
            sides = [(self.canon(f[k]["subject"]), f[k]["attribute"]) for k in ("left", "right")]
            for rid in sorted(by_pair[sides[0]] & by_pair[sides[1]]):
                about = {p for p, _ in self.inv._by_row[rid]}
                if about == set(sides) and COMPARISON_WORDS.search(self.inv.rows.get(rid, {}).get("text") or ""):
                    return rid
            return None
        items = self.claim_items(ctype, f)
        rows = None
        for p, v in items:
            r = by_fact[(p, nz.key(v))]
            rows = set(r) if rows is None else rows & r
        if ctype == "at_event" and rows:
            ev = (self.canon(f["event"]), "date")
            days = {str(e.value["value"]) for e in self._trusted(f["event"], "date")}
            rows = {r for r in rows if any(p == ev and str(v.get("value")) in days for p, v in self.inv._by_row[r])}
        # A sentence restates the claim only if it also NAMES the subjects: "He begins with a crew of 4
        # engineers" or a table row "Call: 38 min, 44 min, …" gives the value without its subject, and a
        # claim that ties that value to its named subject connects what the universe says.
        subjects = {p[0] for p, _ in items}
        if ctype == "at_event":
            subjects.add(self.canon(f["event"]))
        rows = {r for r in rows or () if subjects <= self._named_in(r)}
        return sorted(rows)[0] if rows else None

    # ---- every input sentence, as text (labelled or not) ----------------------------------------
    NEAR = 80  # characters between a subject name and a value for "the sentence says it"

    def text_index(self):
        if not hasattr(self, "_text_index"):
            from app.validate.textindex import RowTextIndex

            self._text_index = RowTextIndex(self.inv.rows, self.lex)
        return self._text_index

    def text_restates(self, ctype: str, f: dict) -> tuple[str | None, str | None]:
        """("strong"|"weak"|None, row id): does some input sentence already say this claim, as text?"""
        from app.validate.rules import attr_spans
        from app.validate.textindex import COMPARE_WORDS as CMP, claim_value_keys

        ix = self.text_index()

        def near(rid, canon, positions):  # same sentence and close together
            return any(abs(a - b) <= self.NEAR and ix.sentence(rid, a) == ix.sentence(rid, b)
                       for a in ix.name_pos(rid, canon) for b in positions)

        def rows_stating(value):
            if value.get("kind") == "name":
                c = self.canon(value["value"])
                return ix.by_name.get(c, set()), (lambda rid: ix.name_pos(rid, c))
            keys = claim_value_keys(value)
            rows = set().union(*(ix.rows_with_value(k) for k in keys)) if keys else set()
            return rows, (lambda rid: [p for k in keys for p in ix.value_pos(rid, k)])

        if ctype == "compare":
            a, b = self.canon(f["left"]["subject"]), self.canon(f["right"]["subject"])
            best = None
            for rid in sorted(ix.by_name.get(a, set()) & ix.by_name.get(b, set())):
                text = ix.text[rid]
                pa, pb = ix.name_pos(rid, a), ix.name_pos(rid, b)
                words = [m.start() for m in CMP.finditer(text)]
                if not words:
                    continue
                for x in pa:
                    for y in pb:
                        if ix.sentence(rid, x) != ix.sentence(rid, y):
                            continue
                        lo, hi = min(x, y), max(x, y)
                        if not any(lo <= w <= hi for w in words):
                            continue
                        best = best or rid
                        # strong only when that sentence compares this attribute ("costs more" for prices)
                        seg = text[max(0, lo - 40):hi + 80]
                        if attr_spans(f["left"]["attribute"], seg) or (
                                f["left"]["attribute"] == f["right"]["attribute"] == "price"
                                and re.search(r"\b(costs?|cheaper|pricier|expensive)\b", seg, re.I)):
                            return "strong", rid
            return ("weak", best) if best else (None, None)

        subj = self.canon(f["subject"])
        cand = set(ix.by_name.get(subj, set()))
        if ctype == "at_event":
            cand &= ix.by_name.get(self.canon(f["event"]), set())
        values = [f["from"], f["to"]] if ctype == "change" else [f["value"]]
        if ctype == "relation":
            target = self.canon(f["target"])
            cand = {r for r in cand if target in {c for c, _ in ix.names[r]}
                    or attr_spans(f["link"], ix.text[r])}
        if values[0].get("kind") == "boolean":
            from app.validate.rules import _has_negation

            hits = []
            for rid in sorted(cand):
                text = ix.text[rid]
                for s, e in attr_spans(f["attribute"], text):
                    sid = ix.sentence(rid, s)
                    bounds = [0] + ix.breaks.get(rid, []) + [len(text)]
                    says_no = _has_negation(text[bounds[sid]:bounds[sid + 1]])
                    if says_no != (values[0]["value"] is False):
                        continue  # the sentence states the opposite: not a restatement
                    hits.append(rid)
                    if near(rid, subj, [s]):
                        return "strong", rid
            return ("weak", hits[0]) if hits else (None, None)
        weak = None
        for value in values:
            rows, pos = rows_stating(value)
            cand &= rows
        for rid in sorted(cand):
            if all(near(rid, subj, rows_stating(v)[1](rid)) for v in values):
                return "strong", rid
            weak = weak or rid
        return ("weak", weak) if weak else (None, None)

    def _named_in(self, rid: str) -> set[str]:
        """Canonical lexicon names a row's text names (whole names, longest first, any case)."""
        cache = self.__dict__.setdefault("_named_cache", {})
        if rid not in cache:
            from app.validate.rules import named_entities

            text = self.inv.rows.get(rid, {}).get("text") or ""
            names = {c for _, _, c in named_entities(text, self.lex)}
            for ev in self.lex.events:  # events that are not also subjects of the lexicon
                if ev and ev not in names and re.search(r"(?<![\w/])" + r"\s*".join(map(re.escape, ev.split()))
                                                        + r"(?![\w/])", text, re.I):
                    names.add(self.canon(ev))
            cache[rid] = names
        return cache[rid]

    def provenance(self, rows: list[str]) -> tuple[str, list]:
        known = [r for r in dict.fromkeys(rows) if r in self.inv.rows]
        return (PASS, known) if known else (UNVERIFIED, [])

    # ---- one claim, then a set -----------------------------------------------------------------
    def check(self, claim, meta: dict | None = None) -> Verdict:
        f, ctype = claim.form or {}, claim.type
        v = Verdict(PASS)
        if norm_text(claim.text) in self.input_texts:  # a copy is never new, whatever else holds
            v.checks["novelty"], v.verdict = FAIL, FAIL
            v.reasons.append("novelty: the text is an input row")
            return v
        cp = validate_claim(claim if meta is None else _with_meta(claim, meta), self.lex, None)
        v.checks["contract"] = FAIL if cp else PASS
        v.reasons += [f"contract: {p}" for p in cp[:3]]
        if cp:
            v.verdict = FAIL
            return v
        v.semantic_key = self.semantic_key(ctype, f)
        st, why, rows = self.truth(ctype, f)
        v.checks["truth"] = st
        v.reasons += [f"truth: {w}" for w in why]
        pst, prow = self.provenance(rows)
        v.checks["provenance"] = pst if st == PASS else UNVERIFIED
        v.evidence = prow
        if norm_text(claim.text) in self.input_texts:
            v.checks["novelty"] = FAIL
            v.reasons.append("novelty: the text is an input row")
        else:
            rid = self.reformulates(ctype, f)
            v.checks["novelty"] = FAIL if rid else PASS
            if rid:
                v.reasons.append("novelty: reformulates an input row")
                v.reasons.append(f"novelty: row {rid} already states it")
            else:  # every input sentence, labelled or not, read as text
                strength, trid = self.text_restates(ctype, f)
                if strength == "strong":
                    v.checks["novelty"] = FAIL
                    v.reasons.append("novelty: an input sentence says it")
                    v.reasons.append(f"novelty: row {trid} names the subject next to the value")
                elif strength == "weak":
                    v.flags.append("novelty_weak")
                    v.reasons.append(f"novelty: row {trid} names the subject and the value far apart (kept as fallback)")
        order = [v.checks["contract"], v.checks["truth"], v.checks["novelty"], v.checks["provenance"]]
        v.verdict = FAIL if FAIL in order else UNVERIFIED if UNVERIFIED in order else PASS
        return v

    def check_set(self, claims, verdicts: list[Verdict] | None = None) -> list[Verdict]:
        """Every claim, plus uniqueness across the set (first one with a key keeps it)."""
        verdicts = verdicts or [self.check(c, getattr(c, "meta", None)) for c in claims]
        seen: dict[tuple, str] = {}
        for c, v in zip(claims, verdicts):
            if not v.semantic_key:
                continue
            if v.semantic_key in seen:
                v.checks["uniqueness"] = FAIL
                v.reasons.append(f"uniqueness: same information as {seen[v.semantic_key]}")
                v.verdict = FAIL
            else:
                v.checks["uniqueness"] = PASS
                seen[v.semantic_key] = c.id
        return verdicts


def _with_meta(claim, meta):
    try:
        claim.meta = meta
    except Exception:  # noqa: BLE001
        pass
    return claim


def report(claims, verdicts: list[Verdict], candidate_stats: dict | None = None) -> dict:
    from collections import Counter

    per_claim = [{"id": c.id, "type": c.type, "text": c.text, "confidence": c.confidence, **v.as_dict()}
                 for c, v in zip(claims, verdicts)]
    return {
        "selected": {
            "n": len(claims),
            "verdicts": dict(Counter(v.verdict for v in verdicts)),
            "by_type": dict(Counter(c.type for c in claims)),
            "checks_failed": dict(Counter(k for v in verdicts for k, s in v.checks.items() if s != PASS)),
        },
        "candidates": candidate_stats or {},
        "claims": per_claim,
    }

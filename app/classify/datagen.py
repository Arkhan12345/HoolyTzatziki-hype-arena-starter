"""Labelled training items in the 5 claim types, generated from the truth KB (practice gold).

  true     the claim generator's validated candidates (app/generate), one per meaning; a `fact` that only
           rewords one prose input sentence is already dropped there (app/generate/restate.py)
  false    a true item with its logic flipped: another value of the pair (an older one when there is one),
           the reverse change, the opposite comparison
  unsure   a (subject, attribute) pair the lexicon allows but no gold row states, with a value that pair
           could plausibly have (taken from the same attribute of other subjects): unverifiable by design

Each item carries its claim form, the date it speaks about, the facts to query against the classifier's KB
and its source rows (the gold rows the truth came from).
"""
from __future__ import annotations

import random
from collections import defaultdict

from app.generate import candidates as C
from app.generate import templates as T
from app.generate.select import screen
from app.kb import normalize as nz
from app.kb.inventory import FactInventory
from app.kb.store import AS_OF

CAP = {"compare": 400, "at_event": 300, "change": 300, "fact": 300, "relation": 60}
SEED = 0
SIBLING_JACCARD = 0.2  # subjects whose attribute sets overlap this much are siblings (phones, offices, ...)


def _perturb(value: dict, pool: list[dict], rng: random.Random):
    """A different value of the same kind: from the pool when possible, else a changed magnitude."""
    k = nz.key_inclusive(value)
    others = [v for v in pool if nz.key_inclusive(v) != k and v.get("kind") == value.get("kind")]
    if others:
        return dict(rng.choice(others))
    w, kind = dict(value), value.get("kind")
    w.pop("op", None)
    x = value.get("value")
    if kind in ("quantity", "money", "duration") and isinstance(x, (int, float)):
        w["value"] = x * 2 if x else 3
    elif kind == "percent" and isinstance(x, (int, float)):
        w["value"] = round(x / 2, 2) if x > 50 else round(x * 1.5 + 1, 2)
    elif kind == "boolean":
        w["value"] = not x
    elif kind == "date" and isinstance(x, str) and len(x) >= 4:
        w["value"] = str(int(x[:4]) - 1) + x[4:]
    elif kind == "version" and isinstance(x, str) and x.split(".")[0].isdigit():
        parts = x.split(".")
        w["value"] = ".".join([str(int(parts[0]) + 1)] + parts[1:])
    else:
        return None
    return w


class Generator:
    def __init__(self, kb, lexicon: dict, gold_rows: list[dict], input_texts: set[str], seed: int = SEED):
        self.inv = FactInventory.from_kb(kb)
        self.lex = lexicon
        self.texts = input_texts
        self.rng = random.Random(seed)
        self.values_by_attr = defaultdict(list)
        for (s, a) in self.inv.pairs():
            for e in self.inv.trusted(s, a):
                self.values_by_attr[a].append(e.value)
        self.report: dict = {}

    # ------------------------------------------------------------------ queries per item
    def _change_dates(self, s, a, frm, to):
        rs = C.runs(self.inv.trusted(s, a))
        kf, kt = nz.key(frm), nz.key(to)
        for i in range(len(rs) - 1):
            if rs[i][0] == kf and rs[i + 1][0] == kt:
                return rs[i][3], rs[i + 1][2]
        return AS_OF, AS_OF

    def _item(self, ctype, label, text, form, evidence=(), origin="candidate", date=None):
        q = []
        if ctype == "fact":
            date = date or AS_OF
            q = [(form["subject"], form["attribute"], form["value"], date)]
        elif ctype == "relation":
            date = date or AS_OF
            q = [(form["target"], form["attribute"], form["value"], date),
                 (form["subject"], form["link"], {"kind": "name", "value": form["target"]}, date)]
        elif ctype == "at_event":
            days = C.event_days(self.inv, form["event"]) or [AS_OF]
            date = date or days[0]
            q = [(form["subject"], form["attribute"], form["value"], d) for d in days]
        elif ctype == "change":
            last_from, first_to = self._change_dates(form["subject"], form["attribute"], form["from"], form["to"])
            date = date or first_to
            q = [(form["subject"], form["attribute"], form["to"], first_to),
                 (form["subject"], form["attribute"], form["from"], last_from)]
        elif ctype == "compare":
            date = date or AS_OF
        return {"type": ctype, "label": label, "text": text, "form": form, "date": str(date)[:10],
                "queries": q, "sources": list(dict.fromkeys(evidence))[:6], "origin": origin}

    # ------------------------------------------------------------------ true
    def trues(self):
        rep: dict = {}
        valid = screen(self.inv, self.texts, rep)
        by_key = {}
        # facts that only reword a prose input sentence never get here: screen() drops them (app/generate/restate.py)
        skipped = rep.get("rejected", {}).get("fact", {}).get("novelty: rewords a prose input sentence (team rule)", 0)
        for c in valid:
            k = c.meta.get("vkey") or c.semantic_key
            if k in by_key:
                continue
            by_key[k] = c
        per_type = defaultdict(list)
        for c in by_key.values():
            per_type[c.type].append(c)
        out = []
        for t, cs in per_type.items():
            self.rng.shuffle(cs)
            for c in cs[:CAP.get(t, 200)]:
                out.append(self._item(c.type, "true", c.text, c.form, c.evidence))
        self.report["true_candidates"] = {t: len(v) for t, v in per_type.items()}
        self.report["fact_restatements_skipped"] = skipped
        return out

    # ------------------------------------------------------------------ false (flip a true one)
    def flip(self, item):
        t, f = item["type"], dict(item["form"])
        if t == "compare":
            op = {"gt": "lt", "lt": "gt"}.get(f["op"])
            if not op:
                return None
            f["op"] = op
            L, R = f["left"], f["right"]
            between = "date" if self._kind(L) == "date" else "amount"
            text = T.compare_text((L["subject"], L["attribute"]), (R["subject"], R["attribute"]), op, between,
                                  self._kind(L))
            return self._item("compare", "false", text, f, item["sources"], "flip")
        if t == "change":
            rs = C.runs(self.inv.trusted(f["subject"], f["attribute"]))
            kf, kt = nz.key(f["to"]), nz.key(f["from"])
            if any(rs[i][0] == kf and rs[i + 1][0] == kt for i in range(len(rs) - 1)):
                return None  # the reverse change happened too: not false
            f["from"], f["to"] = item["form"]["to"], item["form"]["from"]
            text = T.change_text(f["subject"], f["attribute"], f["from"], f["to"])
            out = self._item("change", "false", text, f, item["sources"], "flip")
            # the reversed claim speaks about the same moment as the true one
            out["queries"] = [(f["subject"], f["attribute"], f["to"], item["queries"][0][3]),
                              (f["subject"], f["attribute"], f["from"], item["queries"][1][3])]
            out["date"] = item["date"]
            return out
        s, a = (f["target"], f["attribute"]) if t == "relation" else (f["subject"], f["attribute"])
        pool = [e.value for e in reversed(self.inv.trusted(s, a))]  # newest first: a stale value when there is one
        pool += [v for v in self.values_by_attr.get(a, []) if v not in pool]
        held = {nz.key_inclusive(q[2]) for q in item["queries"][:1]}
        w = None
        for v in pool:
            if nz.key_inclusive(v) not in held and v.get("kind") == f["value"].get("kind"):
                w = dict(v)
                break
        w = w or _perturb(f["value"], [], self.rng)
        if w is None:
            return None
        f["value"] = w
        if t == "fact":
            text = T.fact_text(s, a, w)
        elif t == "relation":
            text = T.relation_text(f["subject"], f["link"], a, w)
        else:
            text = T.at_event_text(s, a, f["event"], w)
        if not text:
            return None
        return self._item(t, "false", text, f, item["sources"], "flip", date=item["date"] if t != "at_event" else None)

    def _kind(self, side):
        v, _, _ = C.current(self.inv, side["subject"], side["attribute"])
        return (v or {}).get("kind")

    # ------------------------------------------------------------------ unsure (unverifiable pairs)
    def unseen_pairs(self):
        """Pairs no gold row states: the subject's own lexicon attributes, and attributes of its siblings
        (subjects with a similar attribute set: "Hooly Pixel 9" + "memory" of the other phones), which is how
        the unsure statements of the 300-statement set are built."""
        known = set(self.inv.pairs())
        attrs = {s["name"]: set(s.get("attributes", [])) for s in self.lex.get("subjects", [])}
        out = set()
        for s, own in attrs.items():
            for a in own:
                if (s, a) not in known and self.values_by_attr.get(a):
                    out.add((s, a))
            for o, theirs in attrs.items():
                if o == s or not own or not theirs:
                    continue
                if len(own & theirs) / len(own | theirs) < SIBLING_JACCARD:
                    continue
                for a in theirs - own:
                    if (s, a) not in known and self.values_by_attr.get(a):
                        out.add((s, a))
        out = sorted(out)
        self.rng.shuffle(out)
        self.report["unseen_pairs"] = len(out)
        return out

    def unsures(self, n_per_type: dict):
        pairs = self.unseen_pairs()
        known = [p for p in self.inv.pairs() if C.current(self.inv, *p)[0] is not None]
        events = [e for e in sorted(self.inv.lex.event_set) if C.event_days(self.inv, e)]
        rels = [i for i in self._true_relations]
        out, used = [], defaultdict(int)
        for k, (s, a) in enumerate(pairs * 3):
            vals = self.values_by_attr[a]
            v = dict(self.rng.choice(vals))
            t = ["fact", "at_event", "change", "compare", "relation"][k % 5]
            if used[t] >= n_per_type.get(t, 0):
                continue
            item = None
            if t == "fact":
                text = T.fact_text(s, a, v)
                item = text and self._item("fact", "unsure", text, {"subject": s, "attribute": a, "value": v},
                                           origin="unseen")
            elif t == "at_event" and events:
                ev = self.rng.choice(events)
                text = T.at_event_text(s, a, ev, v)
                item = text and self._item("at_event", "unsure", text,
                                           {"subject": s, "attribute": a, "event": ev, "value": v}, origin="unseen")
            elif t == "change":
                v2 = _perturb(v, vals, self.rng)
                text = v2 and T.change_text(s, a, v, v2)
                item = text and self._item("change", "unsure", text,
                                           {"subject": s, "attribute": a, "from": v, "to": v2}, origin="unseen",
                                           date=AS_OF)
                if item:
                    item["queries"] = [(s, a, v2, AS_OF), (s, a, v, AS_OF)]
            elif t == "compare" and nz.dimension(v):
                same = [p for p in known if p[1] == a and p[0] != s] or \
                       [p for p in known if nz.dimension(C.current(self.inv, *p)[0] or {}) == nz.dimension(v)]
                if same:
                    rs, ra = self.rng.choice(same)
                    op = self.rng.choice(["gt", "lt"])
                    between = "date" if v.get("kind") == "date" else "amount"
                    text = T.compare_text((s, a), (rs, ra), op, between, v.get("kind"))
                    item = self._item("compare", "unsure", text,
                                      {"left": {"subject": s, "attribute": a},
                                       "right": {"subject": rs, "attribute": ra}, "op": op}, origin="unseen")
            elif t == "relation" and rels:
                r = self.rng.choice(rels)["form"]
                tgt_unseen = [b for (x, b) in pairs if x == r["target"]]
                if tgt_unseen:
                    b = self.rng.choice(tgt_unseen)
                    vb = dict(self.rng.choice(self.values_by_attr[b]))
                    text = T.relation_text(r["subject"], r["link"], b, vb)
                    item = text and self._item("relation", "unsure", text,
                                               {"subject": r["subject"], "link": r["link"], "target": r["target"],
                                                "attribute": b, "value": vb}, origin="unseen")
            if item:
                out.append(item)
                used[t] += 1
        return out

    # ------------------------------------------------------------------ all
    def generate(self):
        trues = self.trues()
        self._true_relations = [i for i in trues if i["type"] == "relation"]
        falses = [f for f in (self.flip(i) for i in trues) if f]
        n_true = defaultdict(int)
        for i in trues:
            n_true[i["type"]] += 1
        unsure = self.unsures({t: max(20, n_true[t]) for t in CAP})
        items = trues + falses + unsure
        for k, it in enumerate(items):
            it["id"] = f"gen-{it['type']}-{it['label']}-{k}"
        counts = defaultdict(lambda: defaultdict(int))
        for it in items:
            counts[it["type"]][it["label"]] += 1
        self.report["items"] = {t: dict(v) for t, v in counts.items()}
        self.report["total"] = len(items)
        return items


def generate_items(kb, lexicon, gold_rows, input_texts, report=None):
    g = Generator(kb, lexicon, gold_rows, input_texts)
    items = g.generate()
    if report is not None:
        report.update(g.report)
    return items

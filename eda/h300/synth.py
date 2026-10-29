"""Synthetic statements for validation and augmentation, built PER FOLD from the training fold's facts only.

No LLM: sentences are filled templates + the contract's value renderer (app/generate/render.py, read-only import).
Every case comes from a (subject, attribute) timeline of the TRAINING rows' gold forms:

  true_current   the latest true value, dated after it                         practice: true   closed-world: true
  update         a new value on the date it first appears (later it holds)     practice: true   closed-world: true
  stale          an older true value, dated after the value changed            practice: false  closed-world: false
  perturbed      the latest value x2 / x0.5 / x2.5 / x0.4 / x5 / x10           practice: false  closed-world: false
  premature      a later value, dated well before it first appears             practice: false  closed-world: false
  swapped        another subject's value of the same attribute                 practice: false  closed-world: false
  unseen_pair    a lexicon attribute the subject never states                  practice: true   closed-world: unsure
  joke           a true statement with a joke / hype tail                      practice: unsure closed-world: unsure

Two label mixes: practice rule (~83 / 15 / 2 true / false / unsure) and closed-world rule (1/3 each).
Guards: no sentence is a copy (rapidfuzz ratio >= 90) of any held-out text; the generator's labels must agree
>= 95% with the timeline rules recomputed from the training KB (sanity check), otherwise the run stops.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.generate.render import attr_words, value_text  # noqa: E402

from rapidfuzz import fuzz, process  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib.values import canonical  # noqa: E402

TEMPLATES = ["{S}'s {A} is {V}.", "{S} has a {A} of {V}.", "The {A} of {S} is {V}.", "For {S}, the {A} is {V}.",
             "{S} now lists its {A} as {V}.", "{S}: {A} {V}", "Current {A} for {S}: {V}.",
             "We can confirm the {A} of {S} is {V}.", "{S} reports a {A} of {V}.", "According to the latest page, "
             "the {A} of {S} stands at {V}.", "{S} {A}: {V}.", "Update on {S}: its {A} is {V}.",
             "As of {D}, the {A} of {S} is {V}.", "On {D}, {S} put its {A} at {V}.", "{S} ({D}): {A} {V}.",
             "Since {D}, {S}'s {A} has been {V}.", "{D}: {S} lists a {A} of {V}."]
DATE_MIN, DATE_MAX = "2025-10-07", "2026-10-28"
JOKES = [", and the curve is vertical.", ", which is more than anyone expected, I promise.",
         " and I am not even tired.", ", and nobody believed it.", ", while I wait for my coffee."]
PRACTICE_MIX = {"true_current": 0.50, "update": 0.20, "unseen_pair": 0.13, "stale": 0.05, "perturbed": 0.04,
                "premature": 0.02, "swapped": 0.04, "joke": 0.02}
CLOSED_MIX = {"true_current": 0.20, "update": 0.133, "stale": 0.10, "perturbed": 0.10, "premature": 0.067,
              "swapped": 0.067, "unseen_pair": 0.233, "joke": 0.10}
LABEL = {"practice": {"true_current": "true", "update": "true", "stale": "false", "perturbed": "false",
                      "premature": "false", "swapped": "false", "unseen_pair": "true", "joke": "unsure"},
         "closed_world": {"true_current": "true", "update": "true", "stale": "false", "perturbed": "false",
                          "premature": "false", "swapped": "false", "unseen_pair": "unsure", "joke": "unsure"}}
FACTORS = [2, 0.5, 2.5, 0.4, 5, 10, 0.2, 0.1, 3, 1.5, 4, 0.25]


def _d(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _fmt(d):
    return d.strftime("%Y-%m-%d")


class FoldKB:
    """(subject, attribute) -> [(date, value dict, label)] from the training rows' gold forms."""

    def __init__(self, rows, lexicon):
        self.tl = defaultdict(list)
        for r in rows:
            for f in json.loads(r.get("form") or "[]"):
                self.tl[(f["subject"], f["attribute"])].append((str(r["date"])[:10], f["value"], str(r["label"])))
        for k in self.tl:
            self.tl[k].sort(key=lambda t: t[0])
        self.lex_attrs = {s["name"]: list(s.get("attributes", [])) for s in lexicon["subjects"]}
        self.by_attr = defaultdict(list)
        for (s, a), t in self.tl.items():
            for d, v, lab in t:
                if lab == "true":
                    self.by_attr[a].append((s, v))

    def true_steps(self, key):
        """[(first date, value)] of the true values in order of first appearance."""
        steps, seen = [], set()
        for d, v, lab in self.tl[key]:
            c = canonical(v)
            if lab == "true" and c not in seen:
                seen.add(c)
                steps.append((d, v))
        return steps

    def latest_at(self, key, date):
        best = None
        for d, v, lab in self.tl[key]:
            if lab == "true" and d <= date:
                best = v
        return best


def _clamp(date):
    return min(max(date, DATE_MIN), DATE_MAX)


def _render(subject, attribute, value, rng, date):
    vt = value_text(value)
    if not vt:
        return None
    dt = value_text({"kind": "date", "value": date}) or date
    return rng.choice(TEMPLATES).format(S=subject, A=attr_words(attribute), V=vt, D=dt)


def _scale(value, k):
    v = dict(value)
    x = v.get("value")
    if not isinstance(x, (int, float)) or isinstance(x, bool) or v.get("kind") not in ("quantity", "money", "percent",
                                                                                      "duration"):
        return None
    y = x * k
    y = round(y) if float(x).is_integer() and abs(y) >= 10 else round(y, 2)
    if v.get("kind") == "percent" and y > 100:
        return None
    v["value"] = y
    return v


def make_case(kb, case, rng):
    keys = [k for k, t in kb.tl.items() if any(lab == "true" for _, _, lab in t)]
    for _ in range(60):
        s, a = rng.choice(keys)
        steps = kb.true_steps((s, a))
        if not steps:
            continue
        last_d, last_v = steps[-1]
        if case in ("true_current", "joke"):
            date, value = _fmt(_d(last_d) + timedelta(days=rng.randint(1, 40))), last_v
        elif case == "update":
            if len(steps) < 2:
                continue
            i = rng.randrange(1, len(steps))
            date, value = steps[i][0], steps[i][1]
        elif case == "stale":
            if len(steps) < 2:
                continue
            i = rng.randrange(0, len(steps) - 1)
            date, value = _fmt(_d(steps[i + 1][0]) + timedelta(days=rng.randint(10, 60))), steps[i][1]
            if canonical(value) == canonical(kb.latest_at((s, a), date) or {}):
                continue
        elif case == "perturbed":
            value = _scale(last_v, rng.choice(FACTORS))
            if value is None:
                continue
            date = _fmt(_d(last_d) + timedelta(days=rng.randint(1, 40)))
        elif case == "premature":
            if len(steps) < 2:
                continue
            i = rng.randrange(1, len(steps))
            date = _fmt(_d(steps[i][0]) - timedelta(days=rng.randint(45, 120)))
            value = steps[i][1]
            prev = kb.latest_at((s, a), date)
            if prev is None or canonical(prev) == canonical(value):
                continue
        elif case == "swapped":
            others = [(s2, v) for s2, v in kb.by_attr[a] if s2 != s and canonical(v) != canonical(last_v)]
            if not others:
                continue
            value = rng.choice(others)[1]
            date = _fmt(_d(last_d) + timedelta(days=rng.randint(1, 40)))
        elif case == "unseen_pair":
            free = [x for x in kb.lex_attrs.get(s, []) if (s, x) not in kb.tl and kb.by_attr.get(x)]
            if not free:
                continue
            a = rng.choice(free)
            value = rng.choice(kb.by_attr[a])[1]
            date = _fmt(_d(last_d) + timedelta(days=rng.randint(1, 40)))
        else:
            raise ValueError(case)
        date = _clamp(date)
        if case == "stale" and canonical(value) == canonical(kb.latest_at((s, a), date) or {}):
            continue
        if case == "premature" and (kb.latest_at((s, a), date) is None
                                    or canonical(kb.latest_at((s, a), date)) == canonical(value)):
            continue
        text = _render(s, a, value, rng, date)
        if not text:
            continue
        if case == "joke":
            text = text.rstrip(".") + rng.choice(JOKES)
        return {"text": text, "date": date, "form": json.dumps([{"subject": s, "attribute": a, "value": value}])}
    return None


def rule_label(kb, row, mode):
    """Independent recomputation from the training KB (the sanity check)."""
    f = json.loads(row["form"])[0]
    key = (f["subject"], f["attribute"])
    if any(j in row["text"] for j in JOKES):
        return "unsure"
    if key not in kb.tl:
        return "true" if mode == "practice" else "unsure"
    latest = kb.latest_at(key, row["date"])
    if latest is not None and canonical(latest) == canonical(f["value"]):
        return "true"
    return "false"


COLUMNS = ["id", "text", "date", "label", "form", "case_type", "chain_length", "broken_link", "fold", "mode"]
CHAIN_SHARE = 0.17


def _chain_rows(train_rows, lexicon, kb, n, rng, mode):
    """Fact-chain statements from eda/h300/chains.py (1-3 hops, ~20% broken links), training rows only."""
    try:
        import importlib
        import numpy as np
        import pandas as pd
        chains = importlib.import_module("chains")
        facts = pd.DataFrame([{"id": r["id"], "date": str(r["date"])[:10], "label": r["label"],
                               "subject": f["subject"], "attribute": f["attribute"], "kind": f["value"]["kind"],
                               "value_raw": f["value"].get("value"), "unit": f["value"].get("unit"),
                               "op": f["value"].get("op"), "currency": f["value"].get("currency"),
                               "per": f["value"].get("per")}
                              for r in train_rows for f in json.loads(r.get("form") or "[]")])
        graph = chains.build_link_graph(facts, lexicon, {r["id"] for r in train_rows})
        df = chains.chain_cases(graph, n, np.random.default_rng(rng.randrange(1 << 30)),
                                "practice" if mode == "practice" else "closed", lexicon)
    except Exception as e:
        return [], f"chains failed ({type(e).__name__}: {e})"
    out = []
    for r in df.to_dict("records"):
        out.append({"text": str(r["text"]), "date": _clamp(str(r["date"])[:10]),
                    "form": r["form"] if isinstance(r["form"], str) else json.dumps(r["form"], default=str),
                    "label": str(r["label"]), "case_type": "chain_" + str(r.get("case_type", "valid")),
                    "chain_length": int(r.get("chain_length", 1) or 1),
                    "broken_link": int(r.get("broken_link", 0) if r.get("broken_link") == r.get("broken_link") else 0)})
    return out, f"chains ok ({len(out)})"


def generate(train_rows, lexicon, heldout_texts, mode, n, seed, prefix, fold="all"):
    """-> (rows with COLUMNS, stats). Never a copy (ratio >= 90) of a held-out text; label sanity checked."""
    rng = random.Random(seed)
    kb = FoldKB(train_rows, lexicon)
    mix = PRACTICE_MIX if mode == "practice" else CLOSED_MIX
    held = [t.strip() for t in heldout_texts if t]
    chain_rows, chain_note = _chain_rows(train_rows, lexicon, kb, int(CHAIN_SHARE * n), rng, mode)
    n_plain = n - len(chain_rows)
    out, tries, dropped = [], 0, 0
    seen, short = set(), {}

    def accept(r):
        nonlocal dropped
        t = r["text"].strip()
        if t in seen:
            return False
        if held:
            best = process.extractOne(t, held, scorer=fuzz.ratio)
            if best and best[1] >= 90:
                dropped += 1
                return False
        seen.add(t)
        r.update(id=f"{prefix}-{mode[:2]}-{len(out) + 1}", fold=fold, mode=mode)
        out.append({c: r.get(c) for c in COLUMNS})
        return True

    for case, w in mix.items():  # fill each case to its quota (bounded tries)
        quota, got, budget = max(1, round(w * n_plain)), 0, max(1, round(w * n_plain)) * 12
        while got < quota and budget > 0:
            budget -= 1
            tries += 1
            r = make_case(kb, case, rng)
            if r is None:
                continue
            r.update(label=LABEL[mode][case], case_type=case, chain_length=0, broken_link=0)
            got += accept(r)
        short[case] = quota - got
    for r in chain_rows:
        accept(r)
    rng.shuffle(out)
    non_chain = [r for r in out if not r["chain_length"]]
    agree = sum(rule_label(kb, r, mode) == r["label"] for r in non_chain) / max(len(non_chain), 1)
    labs = [r["label"] for r in out]
    return out, {"n": len(out), "tried": tries, "dropped_near_copies": dropped, "chain_rows": len(chain_rows),
                 "chain_note": chain_note,
                 "short_of_quota": {k: v for k, v in short.items() if v > 0}, "rule_agreement": round(agree, 4),
                 "mix": {k: round(labs.count(k) / max(len(labs), 1), 3) for k in ("true", "false", "unsure")}}

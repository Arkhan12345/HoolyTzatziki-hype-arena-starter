# Inference-only port of eda/h300/lib (frozen FINAL). Keep in sync with eda/h300/lib; the frozen model in
# models/final/ was pickled from these classes. Fit methods remain for completeness; the runtime never calls them.
"""Row features for the h300 experiments: one statement read on its own against a practice KB.

`Reader` = a fitted extractor + attribute model + KB, built from TRAINING rows only (gold forms of training rows build
the KB anchors; every scored row - training or validation - is read from its text, so features match inference).
`read(rows)` returns one feature dict per row. Feature groups (used by the ladder in eda/h300/run.py):

  core      reading basics + closed-world facts about the read value
  hist      staircase history of the pair: stale / ever-false / support / ratio to the held value      (A4)
  win       agree / disagree mention counts per date window (same day, 30 d, 90 d, older)              (A14)
  dissent   disagreeing mentions counted only when credible (labelled true)                            (A15)
  cw        the closed-world decision as a feature (latest -> true, other -> false, unseen -> unsure)  (A7)
  misread   second number / range / negation / hedge in the sentence                                    (A3)
  smooth    attribute probabilities shrunk toward the attribute prior                                   (A6)
  second    second-chance reading with a wide extractor when the first finds nothing                   (A12)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from .closed_world import (BOOL, AttributeModel, PracticeKB, _mention_features, _restrict, _subject_spans,
                           attribute_examples, fallback_names, subject_tokens, tune_margin, unsure_margin,
                           value_dim)
from .extractor import FactExtractor, Lexicon
from .values import canonical

HEDGE = re.compile(r"\b(about|around|nearly|almost|roughly|over|more than|up to|under|less than|at least|some)\b", re.I)
JOKE = re.compile(r"(,| and | while | which | but ).*\b(I|me|my|nobody|everyone|curve|tired|lunch|coffee|personal best|"
                  r"poetry|flawless|learning|impossible|higher|vertical|bargain)\b", re.I)
RANGE = re.compile(r"\b(between|from \S+ to|\d\s*(-|–|to)\s*\d)\b", re.I)
NEG = re.compile(r"\b(not|no|never|none|without|n't)\b", re.I)
FIRST = re.compile(r"\b(I|I'm|me|my)\b")
NUM = re.compile(r"\d[\d,.]*")
PERTURB = [2, 0.5, 2.5, 0.4, 5, 0.2, 10, 0.1]
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y", "%m/%d/%Y")


def norm_date(s) -> str:
    """Any of a few common date spellings -> YYYY-MM-DD; unknown -> the first 10 characters (never crash)."""
    s = str(s or "").strip()
    for cand in (s, s[:10], s.split("T")[0]):
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return s[:10]


def _days(a: str, b: str) -> float:
    try:
        return float((datetime.strptime(a, "%Y-%m-%d") - datetime.strptime(b, "%Y-%m-%d")).days)
    except ValueError:
        return float("nan")


def _num(key):
    if isinstance(key, tuple):
        for x in key[1:]:
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return float(x)
    return float("nan")


@dataclass
class Reader:
    extractor: FactExtractor
    wide: FactExtractor
    attributes: AttributeModel
    kb: PracticeKB
    margin: float
    attr_prior: dict
    subject_mode: str = "last"

    @classmethod
    def fit(cls, rows: Sequence[dict], lexicon: Lexicon, margin: float | None = None, C: float = 1.0,
            subject_mode: str = "last") -> "Reader":
        """Fit on TRAINING rows only. `margin` is reused from an enclosing fit to save time (inner refits)."""
        rows = [dict(r, date=norm_date(r.get("date"))) for r in rows]
        extractor = FactExtractor(lexicon).fit(rows)
        wide = FactExtractor(lexicon, wide=True).fit(rows)
        kb = PracticeKB.from_rows(rows)
        if margin is None:
            margin, _, _ = tune_margin(rows, extractor, kb, C=C, subject_mode=subject_mode, synonyms=False)
        ex = attribute_examples(rows, extractor, subject_mode)
        attributes = AttributeModel(C=C).fit([e[0] for e in ex], [e[1] for e in ex])
        cnt = Counter(e[1] for e in ex)
        tot = sum(cnt.values()) or 1
        prior = {a: n / tot for a, n in cnt.items()}
        return cls(extractor, wide, attributes, kb, float(margin), prior, subject_mode)

    # ---------------------------------------------------------------- one row
    def _read_with(self, row, extractor):
        ctx = extractor._contexts([row])[0]
        subjects = list(dict.fromkeys(s for _, _, s in ctx.subjects))
        if not subjects:
            return None, "no_subject", ctx
        spans = _subject_spans(ctx)
        constrained = extractor.extract([row])[0]
        mentions = [m for m in ctx.mentions
                    if not (m.kind == "name" and m.value.get("value") in subjects)
                    and not any(st <= m.start and m.end <= en for st, en in spans)]
        mentions += fallback_names(ctx.text, spans + [(m.start, m.end) for m in ctx.mentions])
        options = mentions or [None]
        preferred = {f.span for f in constrained if f.value.get("kind") != BOOL}
        if not preferred and any(f.value.get("kind") == BOOL for f in constrained):
            options = [None] + mentions
        pairs = [(m, s) for m in options for s in subjects]
        feats = [_mention_features(ctx.text, m, spans, subject_tokens(s, self.subject_mode)) for m, s in pairs]
        best = None
        for (m, subj), p in zip(pairs, self.attributes.predict(feats)):
            kind = BOOL if m is None else m.kind
            dim = None if m is None else value_dim(m.value)
            p = _restrict(p, self.kb, kind, dim)
            if not p:
                continue
            row_date = row["date"]
            inside = {a for a in self.kb.by_subject.get(subj, ())
                      if self.kb.fits(a, kind, dim) and self.kb.known(subj, a, row_date)}
            bonus = 1.0 if m is not None and ctx.text[m.start:m.end] in preferred else 0.0
            score = max(p.values()) + bonus
            if best is None or score > best[0]:
                best = (score, m, subj, p, inside, kind, constrained)
        if best is None:
            return None, "no_attribute_of_kind", ctx
        return best, "ok", ctx

    def _features(self, row) -> dict:
        row = dict(row, date=norm_date(row.get("date")))
        text = str(row.get("text") or "")
        f = {"misread_n_numbers": len(NUM.findall(text)), "misread_range": int(bool(RANGE.search(text))),
             "misread_negation": int(bool(NEG.search(text))), "misread_hedge": int(bool(HEDGE.search(text))),
             "text_joke": int(bool(JOKE.search(text))), "text_first_person": int(bool(FIRST.search(text)))}
        f["misread_risk"] = int(f["misread_n_numbers"] >= 2 or f["misread_range"] or f["misread_negation"]
                                or f["misread_hedge"])
        best, status, ctx = self._read_with(row, self.extractor)
        second = 0
        if best is None or not best[4]:
            best2, status2, _ = self._read_with(row, self.wide)
            if best2 is not None and best2[4]:
                best, status, second = best2, "ok", 1
        f["second_chance"] = second
        f["status_no_subject"] = int(status == "no_subject")
        f["status_no_attr"] = int(status == "no_attribute_of_kind")
        out = dict(f)
        out.update(self._decide(row, best, ctx))
        return out

    def _decide(self, row, best, ctx) -> dict:
        d = {"has_reading": 0, "known_pair": 0, "cw_true": 0, "cw_false": 0, "cw_unsure": 1, "p_attr": 0.0,
             "attr_maxp": 0.0, "margin": 0.0, "margin_gap": 0.0, "p_attr_smooth": 0.0, "margin_gap_smooth": 0.0,
             "parse_conf": 0.0, "kind": "none", "matches_latest": 0, "matches_other_attr": 0, "stale": 0,
             "ever_false": 0, "seen_value": 0, "n_true_values": 0, "n_mentions": 0, "days_since_change": np.nan,
             "log_ratio_latest": np.nan, "perturb_hit": 0, "pages_for": 0, "pages_against": 0,
             "dissent_all": 0, "dissent_credible": 0}
        for w in ("w0", "w30", "w90", "wold"):
            d[f"agree_{w}"] = 0
            d[f"disagree_{w}"] = 0
        if best is None:
            return d
        _, m, subj, p, inside, kind, constrained = best
        d["has_reading"] = 1
        d["kind"] = kind
        d["attr_maxp"] = float(max(p.values()))
        d["parse_conf"] = float(max((fc.conf for fc in constrained), default=0.0))
        if not inside:
            return d
        d["known_pair"] = 1
        margin = unsure_margin(p, inside)
        d["margin"] = float(margin)
        d["margin_gap"] = float(margin - self.margin)
        # rare-attribute smoothing: shrink toward the attribute prior (A6)
        alpha = 0.3
        ps = {a: (q + alpha * self.attr_prior.get(a, 0.0)) / (1 + alpha) for a, q in p.items()}
        d["p_attr_smooth"] = float(max(ps.get(a, 0.0) for a in inside))
        d["margin_gap_smooth"] = float(unsure_margin(ps, inside) - self.margin)
        value = {"kind": BOOL, "value": not ctx.negated} if m is None else dict(m.value)
        key = canonical(value)
        attr = max(inside, key=lambda a: p.get(a, 0.0))
        d["p_attr"] = float(p.get(attr, 0.0) / (sum(p.get(a, 0.0) for a in inside) or 1.0))
        rd = row["date"]
        if margin > self.margin:
            d.update(cw_unsure=1, cw_true=0, cw_false=0)
        elif self.kb.matches_latest(subj, attr, rd, key):
            d.update(cw_true=1, cw_unsure=0)
        elif key[0] != BOOL and any(self.kb.matches_latest(subj, a, rd, key) for a in inside):
            d.update(cw_true=1, cw_unsure=0, matches_other_attr=1)
        else:
            d.update(cw_false=1, cw_unsure=0)
        d["matches_latest"] = int(self.kb.matches_latest(subj, attr, rd, key))
        tl = [t for t in self.kb.timeline.get((subj, attr), ()) if t[0] <= rd]
        latest = self.kb.latest_true(subj, attr, rd)
        true_vals = [v for _, v, lab in tl if lab == "true"]
        d["n_true_values"] = len(set(true_vals))
        d["n_mentions"] = len(tl)
        d["seen_value"] = int(any(v == key for _, v, _ in tl))
        d["ever_false"] = int(any(v == key and lab == "false" for _, v, lab in tl))
        d["stale"] = int(key in true_vals and key != latest)
        changes = [dt for (dt, v, lab), prev in zip(tl[1:], tl[:-1]) if lab == "true" and v != prev[1]]
        d["days_since_change"] = _days(rd, changes[-1]) if changes else (_days(rd, tl[0][0]) if tl else np.nan)
        a, b = _num(key), _num(latest) if latest is not None else float("nan")
        if a == a and b == b and a > 0 and b > 0:
            lr = math.log(a / b)
            d["log_ratio_latest"] = lr
            d["perturb_hit"] = int(any(abs(lr - math.log(k)) < 0.02 for k in PERTURB))
        for dt, v, lab in tl:
            gap = abs(_days(rd, dt))
            w = "w0" if gap == 0 else "w30" if gap <= 30 else "w90" if gap <= 90 else "wold"
            if v == key:
                d[f"agree_{w}"] += 1
                d["pages_for"] += 1
            else:
                d[f"disagree_{w}"] += 1
                d["pages_against"] += 1
                d["dissent_all"] += 1
                if lab == "true":
                    d["dissent_credible"] += 1
        return d

    def read(self, rows: Sequence[dict]) -> list[dict]:
        out = []
        for r in rows:
            try:
                out.append(self._features(r))
            except Exception as e:  # never crash on an odd row; it reads as "nothing found"
                out.append({"read_error": 1, "error": type(e).__name__})
        return out


GROUPS = {
    "core": ["has_reading", "known_pair", "p_attr", "attr_maxp", "margin", "parse_conf", "matches_latest",
             "status_no_subject", "status_no_attr"],
    "cw": ["cw_true", "cw_false", "cw_unsure", "margin_gap", "matches_other_attr"],
    "hist": ["stale", "ever_false", "seen_value", "n_true_values", "n_mentions", "days_since_change",
             "log_ratio_latest", "perturb_hit", "pages_for", "pages_against"],
    "win": [f"{s}_{w}" for s in ("agree", "disagree") for w in ("w0", "w30", "w90", "wold")],
    "dissent": ["dissent_all", "dissent_credible"],
    "misread": ["misread_n_numbers", "misread_range", "misread_negation", "misread_hedge", "misread_risk"],
    "text": ["text_joke", "text_first_person"],
    "smooth": ["p_attr_smooth", "margin_gap_smooth"],
    "second": ["second_chance"],
}

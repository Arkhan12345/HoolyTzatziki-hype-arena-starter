# adapted from classify/closed_world.py on arckan/classifier (copied for eda/h300; imports made package-relative)
"""Closed-world labelling: practice.csv is the whole truth.

For statements about "now" (such as data/hooly_statements_300.csv), a row is
- true    when its value equals the latest true value of its (subject,
          attribute) in the practice KB on or before the row's date,
- false   when the KB knows that (subject, attribute) and the value differs,
- unsure  when the KB knows nothing about that (subject, attribute).

This is a different rule from the shipped TruthStack. In practice.csv a value
that contradicts the latest one is true about 2 times in 3 (prices and heads
change), so the stack learned to ignore contradictions. Use this labeller only
where the KB is meant to be complete; keep the stack for arena-style rows.

The hard part is unsure. The classic extractor only scores the attributes the
lexicon lists for a subject, so "Hooly Pixel 9 has 8 GB of memory" (Pixel 9
has no `memory`) gets forced onto some other attribute. Here an attribute
model that never sees the subject reads which attribute the sentence talks
about. If the best attribute outside the subject's known pairs beats the best
inside by more than `margin` (log-odds), the row is unsure. `margin` is tuned
on practice only, by hiding each held-out fact's own attribute from its
subject (a simulated unknown pair) against the same facts left as they are.

Everything is fitted on practice rows (text + gold `form`) and the lexicon.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .extractor import (
    CONTEXT_WINDOW,
    NEGATION_RE,
    FactExtractor,
    Lexicon,
    _words,
    page_ids,
)
from .values import ValueMention, canonical

BOOL = "boolean"
EPS = 1e-6
# A capitalised name the value parser doesn't know ("led by Ron LaFlame").
NAME_RE = re.compile(r"\b[A-Z][a-z][\w'’-]*(?:\s+(?:[A-Z][a-z][\w'’-]*|[A-Z]{2,}))*")
NOT_NAMES = {"The", "A", "An", "Each", "Every", "Free", "From", "Since", "Before", "After", "In",
             "On", "At", "By", "Its", "It", "This", "That", "These", "Our", "We", "He", "She",
             "They", "Hooly", "January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"}
GENERIC_SUBJECT_TOKENS = {"hooly", "the", "by", "of", "and"}


def subject_tokens(subject: str, mode: str) -> list[str]:
    """Words of a subject's name used as features: "last" = its last word
    ("results", "board", "pro"), "all" = every word, "none" = nothing.
    Years and numbers are dropped."""
    toks = [t for t in re.findall(r"[a-z][a-z/]*", subject.lower()) if t not in GENERIC_SUBJECT_TOKENS]
    if mode == "none" or not toks:
        return []
    return toks[-1:] if mode == "last" else toks


def fallback_names(text: str, taken: list[tuple[int, int]]) -> list[ValueMention]:
    """Capitalised word runs outside `taken` spans, leading function words dropped."""
    out = []
    for m in NAME_RE.finditer(text):
        tokens = [t for t in re.finditer(r"\S+", m.group(0))]
        while tokens and tokens[0].group(0) in NOT_NAMES:
            tokens.pop(0)
        if not tokens:
            continue
        start, end = m.start() + tokens[0].start(), m.start() + tokens[-1].end()
        if any(not (end <= st or start >= en) for st, en in taken):
            continue
        out.append(ValueMention("name", {"kind": "name", "value": text[start:end]}, start, end))
    return out


def value_dim(value: dict) -> str | None:
    """Physical dimension of a quantity or duration ("data", "count", ...)."""
    if value.get("kind") in ("quantity", "duration"):
        return canonical(value)[1]
    return None


# ------------------------------------------------------------ attribute model

def _mention_features(text: str, m: ValueMention | None, skip: list[tuple[int, int]],
                      subject_toks: Sequence[str] = ()) -> dict:
    """Features of one value mention (or of the whole sentence for a boolean),
    with the words of the named subjects left out so the model can't guess the
    attribute from who the sentence is about."""
    def words(a: int, b: int) -> list[str]:
        out, pos = [], a
        for st, en in sorted(skip):
            if en <= pos or st >= b:
                continue
            out += _words(text[pos:max(st, pos)])
            pos = max(pos, en)
        return out + _words(text[pos:b])

    f: dict[str, float] = {"bias": 1}
    for t in subject_toks:
        f[f"subj={t}"] = 1
    if m is None:
        f["kind=boolean"] = 1
        for w in set(words(0, len(text))):
            if not NEGATION_RE.fullmatch(w):
                f[f"s={w}"] = 1
        return f
    left = words(0, m.start)[-CONTEXT_WINDOW:]
    right = words(m.end, len(text))[:CONTEXT_WINDOW]
    f[f"kind={m.kind}"] = 1
    for w in left:
        f[f"L={w}"] = 1
        f[f"w={w}"] = 1
    for w in right:
        f[f"R={w}"] = 1
        f[f"w={w}"] = 1
    for w in set(words(0, len(text))):
        f[f"s={w}"] = 0.5
    if left:
        f[f"L1={left[-1]}"] = 1
    if right:
        f[f"R1={right[0]}"] = 1
    for k in ("unit", "per", "currency", "op"):
        if m.value.get(k):
            f[f"{k}={m.value[k]}"] = 1
    if m.extra.get("noun"):
        f[f"noun={m.extra['noun']}"] = 1
    dim = value_dim(m.value)
    if dim:
        f[f"dim={dim}"] = 1
    return f


@dataclass
class AttributeModel:
    """P(attribute | sentence, value mention), subject-blind, multinomial LogReg."""

    C: float = 1.0
    vec: DictVectorizer = field(default_factory=DictVectorizer)
    clf: LogisticRegression | None = None

    def fit(self, X: list[dict], y: list[str]) -> "AttributeModel":
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(self.vec.fit_transform(X), y)
        return self

    def predict(self, X: list[dict]) -> list[dict[str, float]]:
        P = self.clf.predict_proba(self.vec.transform(X))
        return [dict(zip(self.clf.classes_, row)) for row in P]


# ------------------------------------------------------------ practice KB

@dataclass
class PracticeKB:
    """Timelines of every (subject, attribute) in the practice gold forms."""

    timeline: dict[tuple[str, str], list[tuple[str, tuple, str]]]
    attr_kind: dict[str, str]
    attr_dims: dict[str, set[str]]
    by_subject: dict[str, set[str]]

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> "PracticeKB":
        timeline: dict = defaultdict(list)
        attr_kind: dict[str, str] = {}
        attr_dims: dict[str, set[str]] = defaultdict(set)
        by_subject: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            for f in json.loads(r.get("form") or "[]"):
                s, a, v = f["subject"], f["attribute"], f["value"]
                timeline[(s, a)].append((str(r["date"])[:10], canonical(v), str(r["label"])))
                attr_kind.setdefault(a, v["kind"])
                if value_dim(v):
                    attr_dims[a].add(value_dim(v))
                by_subject[s].add(a)
        for k in timeline:
            timeline[k].sort(key=lambda t: t[0])
        return cls(dict(timeline), attr_kind, dict(attr_dims), dict(by_subject))

    def known(self, subject: str, attr: str, row_date: str) -> bool:
        return self.latest_true(subject, attr, row_date) is not None

    def latest_true(self, subject: str, attr: str, row_date: str) -> tuple | None:
        best = None
        for d, v, lab in self.timeline.get((subject, attr), ()):
            if lab == "true" and d <= row_date:
                best = v
        return best

    def matches_latest(self, subject: str, attr: str, row_date: str, key: tuple) -> bool:
        """The value equals the latest true one. A date's op is ignored: "net zero
        by 31 December 2030" states that date."""
        latest = self.latest_true(subject, attr, row_date)
        if latest is None:
            return False
        if key[0] == "date" == latest[0]:
            return key[1] == latest[1]
        return key == latest

    def fits(self, attr: str, kind: str, dim: str | None) -> bool:
        if self.attr_kind.get(attr) != kind:
            return False
        dims = self.attr_dims.get(attr)
        return not (dim and dims and dim not in dims)


# ------------------------------------------------------------ examples

def _subject_spans(ctx) -> list[tuple[int, int]]:
    return [(st, en) for st, en, _ in ctx.subjects]


def _match_mention(ctx, value: dict) -> ValueMention | None:
    key = canonical(value)
    return next((m for m in ctx.mentions if m.key == key), None)


def attribute_examples(rows: Sequence[dict], extractor: FactExtractor, subject_mode: str = "none"):
    """One (features, attribute, subject, kind, dim, row index) per gold fact
    whose value the parser finds in the sentence (booleans: the sentence)."""
    out = []
    for i, (row, ctx) in enumerate(zip(rows, extractor._contexts(rows))):
        for f in json.loads(row.get("form") or "[]"):
            v = f["value"]
            if v["kind"] == BOOL:
                m = None
            else:
                m = _match_mention(ctx, v)
                if m is None:
                    continue
            feats = _mention_features(ctx.text, m, _subject_spans(ctx),
                                      subject_tokens(f["subject"], subject_mode))
            out.append((feats, f["attribute"], f["subject"], v["kind"], value_dim(v), i))
    return out


def _restrict(p: dict[str, float], kb: PracticeKB, kind: str, dim: str | None) -> dict[str, float]:
    p = {a: q for a, q in p.items() if kb.fits(a, kind, dim)}
    z = sum(p.values()) or 1.0
    return {a: q / z for a, q in p.items()}


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def unsure_margin(p: dict[str, float], inside: set[str],
                  confusable: dict[str, set[str]] | None = None) -> float:
    """log-odds of the best attribute outside `inside` minus the best inside.
    An outside attribute the model often mistakes for an inside one
    (`confusable[outside]`, e.g. employees for headcount) is no evidence."""
    confusable = confusable or {}
    p_in = max((q for a, q in p.items() if a in inside), default=0.0)
    p_out = max((q for a, q in p.items()
                 if a not in inside and not (confusable.get(a, set()) & inside)), default=0.0)
    return _logit(p_out) - _logit(p_in)


def confusions(oof: list[tuple[str, dict[str, float]]], min_rate: float = 0.15,
               min_count: int = 2) -> dict[str, set[str]]:
    """predicted attribute -> gold attributes it is often predicted for, out of fold."""
    total: dict[str, int] = defaultdict(int)
    hits: dict[tuple[str, str], int] = defaultdict(int)
    for gold, p in oof:
        total[gold] += 1
        top = max(p, key=p.get)
        if top != gold:
            hits[(gold, top)] += 1
    out: dict[str, set[str]] = defaultdict(set)
    for (gold, top), n in hits.items():
        if n >= min_count and n / total[gold] >= min_rate:
            out[top].add(gold)
    return dict(out)


def tune_margin(rows: Sequence[dict], extractor: FactExtractor, kb: PracticeKB,
                C: float = 1.0, n_splits: int = 5, subject_mode: str = "none",
                synonyms: bool = True) -> tuple[float, dict[str, set[str]], dict]:
    """Out-of-fold simulation on practice: each held-out fact is scored once
    as is (its pair known: should not be unsure) and once with its attribute
    hidden from its subject (should be unsure). Returns the margin with the
    best F1 on the hidden half, the confusable attributes, and the scores."""
    ex = attribute_examples(rows, extractor, subject_mode)
    groups = np.asarray(page_ids(rows))[[e[5] for e in ex]]
    oof = []
    for tr, te in GroupKFold(n_splits=n_splits).split(ex, groups=groups):
        model = AttributeModel(C=C).fit([ex[i][0] for i in tr], [ex[i][1] for i in tr])
        for i, p in zip(te, model.predict([ex[i][0] for i in te])):
            _, attr, subj, kind, dim, _ = ex[i]
            p = _restrict(p, kb, kind, dim)
            if p:
                oof.append((attr, subj, kind, dim, p))
    confusable = confusions([(a, p) for a, _, _, _, p in oof]) if synonyms else {}
    known_m, hidden_m = [], []
    for attr, subj, kind, dim, p in oof:
        inside = {a for a in kb.by_subject.get(subj, ()) if kb.fits(a, kind, dim)}
        known_m.append(unsure_margin(p, inside, confusable))
        hidden_m.append(unsure_margin(p, inside - {attr}, confusable))
    known_m, hidden_m = np.asarray(known_m), np.asarray(hidden_m)
    best = (-1.0, 0.0)
    for t in np.unique(np.concatenate([known_m, hidden_m])):
        tp = (hidden_m > t).sum()
        fp = (known_m > t).sum()
        fn = (hidden_m <= t).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (f1, float(t)))
    t = best[1]
    stats = {"f1": round(float(best[0]), 4), "margin": round(t, 3), "n": len(known_m),
             "known_flagged": round(float((known_m > t).mean()), 4),
             "hidden_caught": round(float((hidden_m > t).mean()), 4),
             "confusable_pairs": sum(len(v) for v in confusable.values())}
    return t, confusable, stats


# ------------------------------------------------------------ labeller

@dataclass
class ClosedWorldLabeller:
    extractor: FactExtractor
    attributes: AttributeModel
    kb: PracticeKB
    margin: float
    subject_mode: str = "last"  # subject-name words as attribute features (see subject_tokens)
    rescue: bool = True  # non-boolean value equals another known attribute of the subject -> true
    prefer_extractor: bool = False  # pair from the classic extractor, not the attribute model
    confusable: dict[str, set[str]] = field(default_factory=dict)  # see confusions()
    tuning: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, practice_rows: Sequence[dict], lexicon: Lexicon, C: float = 1.0,
            subject_mode: str = "last", synonyms: bool = False, **options) -> "ClosedWorldLabeller":
        extractor = FactExtractor(lexicon).fit(practice_rows)
        kb = PracticeKB.from_rows(practice_rows)
        margin, confusable, stats = tune_margin(practice_rows, extractor, kb, C=C,
                                                subject_mode=subject_mode, synonyms=synonyms)
        ex = attribute_examples(practice_rows, extractor, subject_mode)
        attributes = AttributeModel(C=C).fit([e[0] for e in ex], [e[1] for e in ex])
        return cls(extractor, attributes, kb, margin, subject_mode, confusable=confusable,
                   tuning=stats, **options)

    def _label_one(self, row: dict) -> dict:
        """Each statement is its own page: no context from neighbouring rows."""
        out = {"id": row["id"], "label": "unsure", "confidence": None, "reason": "", "fact": None}
        row_date = str(row["date"])[:10]
        ctx = self.extractor._contexts([row])[0]
        subjects = list(dict.fromkeys(s for _, _, s in ctx.subjects))
        if not subjects:
            out["reason"] = "no subject"
            return out
        spans = _subject_spans(ctx)
        constrained = self.extractor.extract([row])[0]

        mentions = [m for m in ctx.mentions
                    if not (m.kind == "name" and m.value.get("value") in subjects)
                    and not any(st <= m.start and m.end <= en for st, en in spans)]
        mentions += fallback_names(ctx.text, spans + [(m.start, m.end) for m in ctx.mentions])
        options: list[ValueMention | None] = mentions or [None]
        preferred = {f.span for f in constrained if f.value.get("kind") != BOOL}
        if not preferred and any(f.value.get("kind") == BOOL for f in constrained):
            options = [None] + mentions

        # Score every (mention, subject); keep the one the attribute model is
        # surest about, favouring the mention the classic extractor used.
        pairs = [(m, s) for m in options for s in subjects]
        feats = [_mention_features(ctx.text, m, spans, subject_tokens(s, self.subject_mode))
                 for m, s in pairs]
        best = None
        for (m, subj), p in zip(pairs, self.attributes.predict(feats)):
            kind = BOOL if m is None else m.kind
            dim = None if m is None else value_dim(m.value)
            p = _restrict(p, self.kb, kind, dim)
            if not p:
                continue
            inside = {a for a in self.kb.by_subject.get(subj, ())
                      if self.kb.fits(a, kind, dim) and self.kb.known(subj, a, row_date)}
            bonus = 1.0 if m is not None and ctx.text[m.start:m.end] in preferred else 0.0
            score = max(p.values()) + bonus
            if best is None or score > best[0]:
                best = (score, m, subj, p, inside)
        if best is None:
            out["reason"] = "no attribute of this kind"
            return out
        _, m, subj, p, inside = best
        if not inside:
            out["reason"] = "no known pair for subject"
            return out
        margin = unsure_margin(p, inside, self.confusable)
        if margin > self.margin:
            out["reason"] = f"attribute {max(p, key=p.get)} unknown for subject (margin {margin:.2f})"
            return out

        value = {"kind": BOOL, "value": not ctx.negated} if m is None else dict(m.value)
        key = canonical(value)
        attr = None
        if self.prefer_extractor:
            attr = next((f.attribute for f in sorted(constrained, key=lambda f: -f.conf)
                         if f.subject == subj and f.attribute in inside and canonical(f.value) == key), None)
        if attr is None:
            attr = max(inside, key=lambda a: p.get(a, 0.0))
        p_attr = p.get(attr, 0.0) / (sum(p.get(a, 0.0) for a in inside) or 1.0)
        out["fact"] = {"subject": subj, "attribute": attr, "value": value}
        if self.kb.matches_latest(subj, attr, row_date, key):
            label, conf, reason = "true", 0.5 + 0.5 * p_attr, "matches latest value"
        elif (self.rescue and key[0] != BOOL
              and any(self.kb.matches_latest(subj, a, row_date, key) for a in inside)):
            label, conf, reason = "true", 0.5 + 0.25 * p_attr, "matches another known attribute"
        else:
            label, conf, reason = "false", 0.5 + 0.5 * p_attr, "contradicts latest value"
        out.update(label=label, confidence=float(min(max(conf, 0.5), 1.0)), reason=reason)
        return out

    def label(self, rows: Sequence[dict]) -> list[dict]:
        """{id, label, confidence (None for unsure), reason, fact} per row."""
        return [self._label_one(r) for r in rows]

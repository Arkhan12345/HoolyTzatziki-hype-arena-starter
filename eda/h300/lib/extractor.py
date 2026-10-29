# adapted from extract/extractor.py on arckan/classifier (copied for eda/h300; imports made package-relative)
"""Text -> facts (subject, attribute, value), trained on the gold `form` column.

Pipeline per row:
1. Subjects: lexicon gazetteer (names, aliases, and safe short forms such as
   "Hooly Cloud Next" for "Hooly Cloud Next 2026"), longest match first.
2. Values: `values.parse_values` (regexes from contracts/value-forms.md).
3. Coreference: subjects named in earlier rows of the same page stay
   candidates, so "He" / "Its" / "The platform" can resolve to them.
4. Ranker: for every value, each candidate (subject, attribute) whose
   attribute has the value's kind is scored by a logistic regression over
   context-word x attribute features. The best pair above a threshold wins.
5. Booleans: each candidate (subject, boolean attribute) is scored on the
   whole sentence; the value is "no" when the sentence has a negation word.

A page is a contiguous block of rows with the same date (see data/RULES.md).
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .values import ValueMention, canonical, parse_values

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|nobody|nothing|neither|nor|without|cannot)\b|n't\b", re.I
)
PRONOUN_START_RE = re.compile(
    r"^(?:He|She|It|Its|It's|They|Their|His|Her|This|That|These|The \w+)\b"
)
WORD_RE = re.compile(r"[a-z][a-z'-]*")
SPEAKER_RE = re.compile(r"^\s*([A-Z][\w'-]+(?: [A-Z][\w'-]+)?):\s")
GENERIC_FIRST_TOKENS = {"Hooly", "The", "Made", "Gemini", "Brightline", "Meridian", "Quayside"}
# Short forms too common to stand for one subject on their own.
GENERIC_SHORT_FORMS = {
    "Next", "Day", "Night", "Report", "Office", "Board", "Results", "Survey", "Labs",
    "Health", "Hall", "Launch", "Programme", "Team", "One", "Pro", "Free", "Plan", "App",
    "API",
}
# Attribute-name tokens that say nothing about which attribute is meant.
GENERIC_ATTR_TOKENS = {"hooly", "total", "number", "count", "date", "time", "last", "first",
                       "current", "with", "from", "share", "rate", "size"}
CONTEXT_WINDOW = 6
MAX_CONTEXT_SUBJECTS = 4


@dataclass
class Fact:
    row_id: str
    subject: str
    attribute: str
    value: dict
    span: str
    conf: float
    anchored: bool = True  # subject named in this sentence (not guessed from the page)

    @property
    def key(self) -> tuple:
        return (self.subject, self.attribute, canonical(self.value))


@dataclass
class Lexicon:
    attributes: dict[str, set[str]]
    surfaces: list[tuple[str, str, bool]]  # (surface, subject, case_sensitive), longest first
    wide_surfaces: set[str] = field(default_factory=set)  # suffix / first-name forms

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Lexicon":
        attributes: dict[str, set[str]] = {}
        exact: dict[str, str] = {}
        for s in raw["subjects"]:
            attributes[s["name"]] = set(s.get("attributes", []))
            for surface in [s["name"], *s.get("aliases", [])]:
                exact.setdefault(surface.lower(), s["name"])
        for e in raw.get("events", []):
            name = e["name"] if isinstance(e, dict) else e
            attributes.setdefault(name, set())
            exact.setdefault(name.lower(), name)

        # Short forms, matched case-sensitively and kept only when they point to
        # one subject: drop "Hooly " and/or a trailing year; token suffixes
        # ("Photos Plus", "Plus" for "Hooly Photos Plus"); a person's first
        # name ("Bertram" for "Bertram Gilfoil").
        primary: dict[str, set[str]] = defaultdict(set)
        secondary: dict[str, set[str]] = defaultdict(set)
        for name in attributes:
            forms = {name}
            for f in list(forms):
                forms.add(re.sub(r"\s+(?:19|20)\d\d$", "", f))
            for f in list(forms):
                if f.startswith("Hooly ") and len(f) > 6:
                    forms.add(f[6:])
            extra = set()
            for f in forms:
                tokens = f.split()
                for k in range(1, len(tokens)):
                    if not re.fullmatch(r"[\d.]+", tokens[k]):
                        extra.add(" ".join(tokens[k:]))
            tokens = name.split()
            if len(tokens) == 2 and tokens[0] not in GENERIC_FIRST_TOKENS:
                extra.add(tokens[0])
            for f in forms:
                if f.lower() not in exact and len(f) >= 4 and f[0].isupper():
                    primary[f].add(name)
            for f in extra - forms:
                if (f.lower() not in exact and len(f) >= 3 and f[0].isupper()
                        and f not in GENERIC_SHORT_FORMS):
                    secondary[f].add(name)

        surfaces = [(k, v, False) for k, v in exact.items()]
        surfaces += [(k, next(iter(v)), True) for k, v in primary.items() if len(v) == 1]
        wide = {k for k, v in secondary.items() if len(v) == 1 and k not in primary}
        surfaces += [(k, next(iter(secondary[k])), True) for k in wide]
        surfaces.sort(key=lambda t: -len(t[0]))
        return cls(attributes=attributes, surfaces=surfaces, wide_surfaces=wide)

    def find_subjects(self, text: str, wide: bool = False) -> list[tuple[int, int, str]]:
        """Subjects named in `text`. `wide` adds suffix and first-name forms."""
        lowered = text.lower()
        taken: list[tuple[int, int]] = []
        found: list[tuple[int, int, str]] = []
        for surface, subject, case_sensitive in self.surfaces:
            if not wide and surface in self.wide_surfaces:
                continue
            haystack = text if case_sensitive else lowered
            for m in re.finditer(rf"(?<![\w-]){re.escape(surface)}(?![\w-])", haystack):
                if any(not (m.end() <= s or m.start() >= e) for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append((m.start(), m.end(), subject))
        return sorted(found)

    def name_surfaces(self) -> list[tuple[str, str]]:
        return [(s, subj) for s, subj, cs in self.surfaces if not cs]


def page_ids(rows: Sequence[dict]) -> list[int]:
    out, page, prev = [], -1, None
    for row in rows:
        d = str(row["date"]).strip()
        if d != prev:
            page, prev = page + 1, d
        out.append(page)
    return out


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _stem(word: str) -> str:
    return word[:5]


def attr_stems(attr: str) -> set[str]:
    return {_stem(t) for t in attr.lower().split("_")
            if len(t) >= 3 and t not in GENERIC_ATTR_TOKENS}


def subject_period_gap(subject: str, row_date: str) -> float:
    """Days between a dated subject ("Hooly Q1 2026 Results") and the row's date.

    Quarterly results are published about a month after the quarter ends.
    Subjects without a quarter in their name get infinity.
    """
    m = re.search(r"\bQ([1-4]) ((?:19|20)\d\d)\b", subject)
    if not m:
        return math.inf
    try:
        y, mo, d = (int(x) for x in str(row_date).strip()[:10].split("-"))
        row = date(y, mo, d)
    except ValueError:
        return math.inf
    quarter_end_month = 3 * int(m.group(1))
    published = date(int(m.group(2)) + (quarter_end_month == 12), (quarter_end_month % 12) + 1, 28)
    return abs((row - published).days)


def _bucket(x: float, edges: Iterable[float]) -> int:
    return int(sum(x > e for e in edges))


@dataclass
class _RowContext:
    row: dict
    text: str
    subjects: list[tuple[int, int, str]]
    context_subjects: list[str]  # from earlier rows of the page, most recent first
    mentions: list[ValueMention]
    negated: bool
    pronoun_start: bool
    words: list[str] = field(default_factory=list)
    speaker: str | None = None  # subject named in a "Bertram: ..." prefix
    topic_subjects: list[str] = field(default_factory=list)  # best match for the page's words
    first_person: bool = False


class FactExtractor:
    def __init__(self, lexicon: Lexicon, *, wide: bool = False, C: float = 2.0,
                 threshold: float = 0.35, bool_threshold: float | None = None,
                 context_subjects: int = MAX_CONTEXT_SUBJECTS, use_following: bool = True,
                 use_company: bool | None = None, use_topics: bool | None = None,
                 use_lexical: bool | None = None):
        """`wide=False` (classic) is what the truth classifier uses: it scores
        best downstream. `wide=True` adds more subject candidates (company,
        page topics, attribute-word matches, short forms, speakers) and generic
        attribute-word features: higher fact recall (F1 0.868 vs 0.836), but its
        extra subject guesses add noise to the KB timeline, so the classifier
        does about 1.3 points worse with it. Use it where fact recall matters
        (claim generation, KB building)."""
        self.lexicon = lexicon
        self.wide = bool(wide)
        self.use_company = self.wide if use_company is None else use_company
        self.use_topics = self.wide if use_topics is None else use_topics
        self.use_lexical = self.wide if use_lexical is None else use_lexical
        if bool_threshold is None:
            bool_threshold = 0.5 if self.wide else 0.4
        self.context_subjects = int(context_subjects)
        self.use_following = bool(use_following)
        self.C = C
        self.threshold = threshold
        self.bool_threshold = bool_threshold
        self.attr_kind: dict[str, str] = {}
        self.extra_names: list[tuple[str, str]] = []
        self.value_vec = DictVectorizer()
        self.value_clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                            solver="liblinear")
        self.bool_vec = DictVectorizer()
        self.bool_clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                           solver="liblinear")
        self.has_bool = False
        self.lex_index: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------ context
    def _names(self) -> list[tuple[str, str]]:
        names = {s.lower(): (s, c) for s, c in self.lexicon.name_surfaces()}
        for s, c in self.extra_names:
            names.setdefault(s.lower(), (s, c))
        return sorted(names.values(), key=lambda t: -len(t[0]))

    def _contexts(self, rows: Sequence[dict]) -> list[_RowContext]:
        names = self._names()
        pages = page_ids(rows)
        found = [self.lexicon.find_subjects(str(row["text"]), wide=self.wide) for row in rows]
        page_first: dict[int, list[str]] = defaultdict(list)
        page_all: dict[int, list[str]] = defaultdict(list)
        for page, subs in zip(pages, found):
            for _, _, s in subs:
                page_all[page].append(s)
            if not page_first[page]:
                page_first[page] = [s for _, _, s in subs]
        page_stems: dict[int, set[str]] = defaultdict(set)
        for page, row in zip(pages, rows):
            page_stems[page].update(_stem(w) for w in _words(str(row["text"])) if len(w) >= 3)
        first_row: dict[int, int] = {}
        for i, page in enumerate(pages):
            first_row.setdefault(page, i)
        page_topics: dict[int, list[str]] = defaultdict(list)
        if self.use_topics:
            for page, stems in page_stems.items():
                page_topics[page] = self._page_topics(stems, rows[first_row[page]]["date"])
        out: list[_RowContext] = []
        history: list[str] = []
        prev_page = None
        for row, page, subjects in zip(rows, pages, found):
            if page != prev_page:
                history, prev_page = [], page
            text = str(row["text"])
            mentions = parse_values(text, names)
            # Most recent earlier subjects, then the page headline's subjects,
            # then (optionally) any other subject named on the page.
            pool = list(reversed(history)) + page_first[page]
            if self.use_following:
                pool += page_all[page]
            ctx_subjects: list[str] = []
            for s in pool:
                if s not in ctx_subjects:
                    ctx_subjects.append(s)
                if len(ctx_subjects) >= self.context_subjects:
                    break
            speaker = None
            sm = SPEAKER_RE.match(text) if self.wide else None
            if sm:
                speaker = next((subj for st, en, subj in subjects
                                if st >= sm.start(1) and en <= sm.end(1)), None)
            out.append(_RowContext(
                row=row, text=text, subjects=subjects, context_subjects=ctx_subjects,
                mentions=mentions, negated=bool(NEGATION_RE.search(text)),
                pronoun_start=bool(PRONOUN_START_RE.match(text.strip())),
                words=_words(text), speaker=speaker, topic_subjects=page_topics[page],
                first_person=bool(re.search(r"\b(?:I|I'm|I've|me|my|we|our)\b", text)),
            ))
            history.extend(s for _, _, s in subjects)
        return out

    def _candidate_subjects(self, ctx: _RowContext) -> list[tuple[str, dict]]:
        """(subject, info) for subjects in the sentence, then from page context."""
        seen: dict[str, dict] = {}
        for i, (start, end, s) in enumerate(ctx.subjects):
            if s not in seen:
                seen[s] = {"in_sent": True, "start": start, "end": end, "order": i}
        for rank, s in enumerate(ctx.context_subjects):
            if s not in seen:
                seen[s] = {"in_sent": False, "rank": rank}
        for rank, s in enumerate(ctx.topic_subjects if self.use_topics else ()):
            if s not in seen:
                seen[s] = {"in_sent": False, "rank": f"topic{rank}"}
        if (self.use_company or not seen) and "Hooly" not in seen and "Hooly" in self.lexicon.attributes:
            seen["Hooly"] = {"in_sent": False, "rank": "company"}
        if ctx.speaker:
            seen[ctx.speaker]["speaker"] = True
        return list(seen.items())

    def _page_topics(self, stems: set[str], row_date: str, top: int = 2) -> list[str]:
        """Subjects whose attribute names match the most words on the page.

        Ties (e.g. every quarter's results) go to the subject dated closest to
        the page. Needs at least two matched attributes."""
        scored = []
        for subject, attrs in self.lexicon.attributes.items():
            hits = sum(1 for a in attrs if a in self.attr_kind and attr_stems(a) & stems)
            if hits >= 2:
                scored.append((-hits, subject_period_gap(subject, row_date), subject))
        return [s for _, _, s in sorted(scored)[:top]]

    def _lexical_pairs(self, ctx: _RowContext, covered: set[str]) -> list[tuple[str, str]]:
        """(subject, attribute) pairs whose attribute name shares a word with the
        sentence, for subjects not already candidates ("4 recruiters" ->
        Hooly Hiring Programme.recruiters). Only attributes owned by at most
        three subjects: a shared word like "price" says nothing about whose."""
        out = []
        if not self.use_lexical:
            return out
        for st in {_stem(w) for w in ctx.words if len(w) >= 3}:
            for subject, attr in self.lex_index.get(st, ()):
                if subject not in covered and (subject, attr) not in out:
                    out.append((subject, attr))
        return out

    def _overlap_features(self, f: dict, attr: str, window: list[str], ctx: _RowContext):
        if not self.wide:
            return
        stems = attr_stems(attr)
        if not stems:
            return
        win = {_stem(w) for w in window}
        sent = {_stem(w) for w in ctx.words}
        f["ov_win"] = len(stems & win) / len(stems)
        f["ov_sent"] = len(stems & sent) / len(stems)
        f["ov_win_any"] = float(bool(stems & win))

    # ------------------------------------------------------------ features
    def _value_features(self, ctx: _RowContext, m: ValueMention, subject: str,
                        info: dict, attr: str, n_same_kind: int) -> dict[str, float]:
        left = _words(ctx.text[: m.start])[-CONTEXT_WINDOW:]
        right = _words(ctx.text[m.end:])[:CONTEXT_WINDOW]
        f: dict[str, float] = {f"a={attr}": 1, f"sa={subject}|{attr}": 1, "bias": 1}
        for w in left:
            f[f"a|L={attr}|{w}"] = 1
            f[f"a|w={attr}|{w}"] = 1
        for w in right:
            f[f"a|R={attr}|{w}"] = 1
            f[f"a|w={attr}|{w}"] = 1
        for w in set(ctx.words):
            f[f"a|s={attr}|{w}"] = 0.5
        if left:
            f[f"a|L1={attr}|{left[-1]}"] = 1
        if right:
            f[f"a|R1={attr}|{right[0]}"] = 1
        v = m.value
        for k in ("unit", "per", "currency", "op"):
            if v.get(k):
                f[f"a|{k}={attr}|{v[k]}"] = 1
        if m.extra.get("noun"):
            f[f"a|noun={attr}|{m.extra['noun']}"] = 1
        if m.approx:
            f["approx"] = 1
        self._overlap_features(f, attr, left + right, ctx)
        f[f"nkind={_bucket(n_same_kind, (1, 2, 4))}"] = 1
        if info.get("speaker"):
            f["subj_speaker"] = 1
            f[f"subj_speaker_fp={int(ctx.first_person)}"] = 1
        if info["in_sent"]:
            dist = min(abs(m.start - info["end"]), abs(info["start"] - m.end))
            f["subj_in_sent"] = 1
            f[f"subj_dist={_bucket(dist, (5, 15, 40, 80))}"] = 1
            f["subj_before"] = float(info["start"] < m.start)
            nearest = min((abs(m.start - e) if s < m.start else abs(s - m.end), subj)
                          for s, e, subj in ctx.subjects)[1]
            f["subj_nearest"] = float(nearest == subject)
        else:
            f[f"subj_ctx_rank={info['rank']}"] = 1
            f[f"subj_ctx_pron={int(ctx.pronoun_start)}"] = 1
            f[f"subj_ctx_nsent={min(len(ctx.subjects), 2)}"] = 1
        if m.kind == "name" and m.value["value"] == subject:
            f["self_name"] = 1
        return f

    def _bool_features(self, ctx: _RowContext, subject: str, info: dict,
                       attr: str) -> dict[str, float]:
        f: dict[str, float] = {f"a={attr}": 1, f"sa={subject}|{attr}": 1, "bias": 1}
        for w in set(ctx.words):
            f[f"a|s={attr}|{w}"] = 1
        attr_words = set(attr.lower().split("_"))
        overlap = len(attr_words & set(ctx.words))
        f["attr_word_overlap"] = overlap / max(len(attr_words), 1)
        self._overlap_features(f, attr, ctx.words, ctx)
        if info.get("speaker"):
            f["subj_speaker"] = 1
        f[f"nvalues={min(len(ctx.mentions), 3)}"] = 1
        f[f"neg={int(ctx.negated)}"] = 1
        if info["in_sent"]:
            f["subj_in_sent"] = 1
        else:
            f[f"subj_ctx_rank={info['rank']}"] = 1
            f[f"subj_ctx_pron={int(ctx.pronoun_start)}"] = 1
        return f

    def _value_candidates(self, ctx: _RowContext):
        subjects = self._candidate_subjects(ctx)
        kinds = defaultdict(int)
        for m in ctx.mentions:
            kinds[m.kind] += 1
        lexical = self._lexical_pairs(ctx, {s for s, _ in subjects})
        lex_info = {"in_sent": False, "rank": "lexical"}
        for mi, m in enumerate(ctx.mentions):
            pairs = [(s, info, a) for s, info in subjects for a in self.lexicon.attributes.get(s, ())]
            pairs += [(s, lex_info, a) for s, a in lexical]
            for subject, info, attr in pairs:
                if m.kind == "name" and m.value["value"] == subject:
                    continue
                if self.attr_kind.get(attr) != m.kind:
                    continue
                yield mi, m, subject, attr, self._value_features(
                    ctx, m, subject, info, attr, kinds[m.kind])

    def _bool_candidates(self, ctx: _RowContext):
        subjects = self._candidate_subjects(ctx)
        pairs = [(s, info, a) for s, info in subjects for a in self.lexicon.attributes.get(s, ())]
        lex_info = {"in_sent": False, "rank": "lexical"}
        pairs += [(s, lex_info, a) for s, a in self._lexical_pairs(ctx, {s for s, _ in subjects})]
        for subject, info, attr in pairs:
            if self.attr_kind.get(attr) == "boolean":
                yield subject, attr, self._bool_features(ctx, subject, info, attr)

    # ------------------------------------------------------------ fit / predict
    def fit(self, rows: Sequence[dict]) -> "FactExtractor":
        """`rows` need id, text, date and form (JSON list of gold facts)."""
        gold_per_row: list[set[tuple]] = []
        names: dict[str, str] = {}
        for row in rows:
            facts = json.loads(row["form"]) if row.get("form") else []
            keys = set()
            for f in facts:
                self.attr_kind.setdefault(f["attribute"], f["value"]["kind"])
                keys.add((f["subject"], f["attribute"], canonical(f["value"])))
                if f["value"]["kind"] == "name":
                    names[f["value"]["value"]] = f["value"]["value"]
            gold_per_row.append(keys)
        self.extra_names = sorted(names.items(), key=lambda t: -len(t[0]))

        owners: dict[str, set[str]] = defaultdict(set)
        for subject, attrs in self.lexicon.attributes.items():
            for attr in attrs:
                owners[attr].add(subject)
        index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for attr, subs in owners.items():
            if len(subs) > 3 or attr not in self.attr_kind:
                continue
            for st in attr_stems(attr):
                index[st].extend((subject, attr) for subject in sorted(subs))
        self.lex_index = dict(index)

        vx, vy, bx, by = [], [], [], []
        for ctx, gold in zip(self._contexts(rows), gold_per_row):
            for _, m, subject, attr, feats in self._value_candidates(ctx):
                vx.append(feats)
                vy.append(int((subject, attr, m.key) in gold))
            for subject, attr, feats in self._bool_candidates(ctx):
                bx.append(feats)
                by.append(int(any(g[0] == subject and g[1] == attr for g in gold)))

        self.value_clf.fit(self.value_vec.fit_transform(vx), vy)
        self.has_bool = len(set(by)) == 2
        if self.has_bool:
            self.bool_clf.fit(self.bool_vec.fit_transform(bx), by)
        return self

    def extract(self, rows: Sequence[dict]) -> list[list[Fact]]:
        """Facts per row, in input order."""
        contexts = self._contexts(rows)
        out: list[list[Fact]] = []
        for ctx in contexts:
            facts: list[Fact] = []
            named = {s for _, _, s in ctx.subjects}
            cands = list(self._value_candidates(ctx))
            if cands:
                probs = self.value_clf.predict_proba(
                    self.value_vec.transform([c[4] for c in cands]))[:, 1]
                best: dict[int, tuple[float, str, str]] = {}
                for (mi, _, subject, attr, _), p in zip(cands, probs):
                    if p > best.get(mi, (-1.0,))[0]:
                        best[mi] = (float(p), subject, attr)
                for mi, (p, subject, attr) in sorted(best.items()):
                    if p < self.threshold:
                        continue
                    m = ctx.mentions[mi]
                    facts.append(Fact(str(ctx.row["id"]), subject, attr, dict(m.value),
                                      ctx.text[m.start:m.end], p, subject in named))
            if self.has_bool:
                bc = list(self._bool_candidates(ctx))
                if bc:
                    probs = self.bool_clf.predict_proba(
                        self.bool_vec.transform([c[2] for c in bc]))[:, 1]
                    for (subject, attr, _), p in zip(bc, probs):
                        if p >= self.bool_threshold:
                            facts.append(Fact(str(ctx.row["id"]), subject, attr,
                                              {"kind": "boolean", "value": not ctx.negated},
                                              ctx.text, float(p), subject in named))
            out.append(facts)
        return out


def fact_scores(gold_rows: Sequence[dict], predicted: Sequence[Sequence[Fact]]) -> dict[str, float]:
    """Fact-level precision / recall / F1 against the gold `form` column."""
    tp = fp = fn = 0
    for row, facts in zip(gold_rows, predicted):
        gold = {(f["subject"], f["attribute"], canonical(f["value"]))
                for f in json.loads(row["form"] or "[]")}
        pred = {f.key for f in facts}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "tp": tp, "fp": fp, "fn": fn}

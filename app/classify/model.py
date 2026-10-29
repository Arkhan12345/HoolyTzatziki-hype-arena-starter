"""Features, training and inference of the truth classifier.

Rows of two kinds go through the same features:
  sentence  a universe row: its extracted facts are in the KB (app/classify/timeline.py)
  item      a generated training claim (app/classify/datagen.py): its facts only QUERY the KB, never anchor or vote
The KB's labels come only from the input's labelled rows (TZ_KB_LABELS=0 turns them off: a label-free KB):
  inference   unlabelled rows and items are scored by a KB pinned with every labelled input row
  training    a labelled row is scored by a KB pinned with the labelled rows of the OTHER page folds (out-of-fold),
              its own page's rows voting like unlabelled rows, so its features look like an unlabelled row's
Features per row: text flags, page shape, extraction confidence, the KB status of each fact (holds / outdated /
ahead of time / unbacked / unseen) with its support from other pages and pins, corpus support of the value, page
trust, and the claim type with its type-specific checks (compare: does the KB order the two values that way; change:
is from -> to a step of the KB's stairs). Model: LightGBM, the research's frozen hyperparameters.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from app.classify import text as TX
from app.classify.timeline import (FEATURES as TL_COLS, FORBID, IGNORE, PIN, VOTE, Timeline, page_trust, supporters,
                                   text_prior)
from app.kb import normalize as nz

CLASSES = np.array(["false", "true", "unsure"])
TYPES = ["sentence", "fact", "relation", "at_event", "change", "compare"]
FOLDS = 5
SEED = 0


def lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                          verbose=-1, n_jobs=2)


def _num(v):
    try:
        x = float(v.get("value"))
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError, AttributeError):
        return np.nan


class World:
    """The KB over the universe rows (+ the queries of generated items), and the features of any row or item."""

    def __init__(self, rows: list[dict], facts: list[dict], items: list[dict], use_labels: bool = True,
                 gold_forms: dict | None = None):
        self.rows = rows
        self.items = items
        page, prev, pages = 0, None, []
        for r in rows:
            if r.get("date") != prev:
                page += 1
                prev = r.get("date")
            pages.append(page)
        self.page_of = {r["id"]: p for r, p in zip(rows, pages)}
        self.flags = {r["id"]: TX.flags(r.get("text", ""), r.get("date", "")) for r in rows}
        by_page = defaultdict(list)
        for r in rows:
            by_page[self.page_of[r["id"]]].append(r["id"])
        self.page_rows = by_page
        self.page_hype = {p: float(np.mean([self.flags[i]["hype_score"] > 0 for i in ids])) for p, ids in by_page.items()}
        self.page_fp = {p: float(np.mean([self.flags[i]["first_person"] for i in ids])) for p, ids in by_page.items()}
        date_of = {r["id"]: r.get("date", "") for r in rows}
        self.text_of = {r["id"]: r.get("text", "") for r in rows}
        self.with_quotes = True  # the arena run turns this off: 18,000 rows of quotes would crowd /tmp
        self.label_of = {r["id"]: r.get("label") for r in rows if r.get("label") in ("true", "false", "unsure")}
        self.labelled = set(self.label_of) if use_labels else set()
        self.use_labels = use_labels

        # A labelled row pins with its gold form when the image has it (exact facts; the contract lets the image hold
        # the practice forms) and only while it is pinned; whenever it is scored or votes it uses its extracted facts,
        # exactly like an unlabelled row.
        gold_forms = {rid: fs for rid, fs in (gold_forms or {}).items() if rid in self.labelled and fs}
        recs = []
        for f in facts:
            rid = str(f.get("row_id"))
            if rid not in self.page_of or not isinstance(f.get("value"), dict):
                continue
            recs.append({"row_id": rid, "subject": f["subject"], "attribute": f["attribute"], "value": f["value"],
                         "date": date_of[rid], "page": self.page_of[rid], "kind": "sentence", "src": "extracted",
                         "conf": float(f.get("conf", 0.5) or 0.5), "agree": bool(f.get("agree", False))})
        for rid, fs in gold_forms.items():
            for g in fs:
                if isinstance(g.get("value"), dict):
                    recs.append({"row_id": rid, "subject": g["subject"], "attribute": g["attribute"], "value": g["value"],
                                 "date": date_of[rid], "page": self.page_of[rid], "kind": "sentence", "src": "gold",
                                 "conf": 1.0, "agree": True})
        for k, it in enumerate(items):
            for (s, a, v, d) in it["queries"]:
                recs.append({"row_id": it["id"], "subject": s, "attribute": a, "value": v, "date": d,
                             "page": -(k + 1), "kind": "item", "src": "query", "conf": 1.0, "agree": True})
        cols = ["row_id", "subject", "attribute", "value", "date", "page", "kind", "src", "conf", "agree"]
        self.tl = Timeline(pd.DataFrame(recs, columns=cols))
        f = self.tl.f
        self.gold_pinned = set(gold_forms)
        # facts that pin a labelled row: its gold form if it has one, else its extracted facts
        pin_src = np.where(f["row_id"].isin(self.gold_pinned), "gold", "extracted")
        self.pins_with = f["src"].to_numpy() == pin_src
        nfacts = f[(f.kind == "sentence") & self.pins_with].groupby("row_id").size()
        self.nfacts = nfacts.to_dict()
        self.prior = text_prior(self.page_hype)
        self.fold_of_page = {p: p % FOLDS for p in by_page}
        self._build()
        self._support()

    # ------------------------------------------------------------------ KB runs
    def anchors(self, labelled: set) -> np.ndarray:
        """PIN / FORBID for the facts of labelled rows, VOTE for the other sentences, IGNORE for items."""
        f = self.tl.f
        a = np.full(self.tl.n, IGNORE)
        for m, (rid, kind, src) in enumerate(zip(f["row_id"], f["kind"], f["src"])):
            if kind != "sentence":
                continue
            if rid not in labelled:
                if src == "extracted":
                    a[m] = VOTE  # an unlabelled row (or a labelled one being scored) speaks through what was read
            elif self.pins_with[m]:
                if self.label_of[rid] == "true":
                    a[m] = PIN
                elif self.label_of[rid] == "false" and self.nfacts.get(rid, 0) == 1:
                    a[m] = FORBID
        return a

    def _build(self):
        tl, f = self.tl, self.tl.f
        self.oof_trust: dict = {}
        feats = None
        if tl.n == 0:
            self.trust = {}
            self.F = pd.concat([f, pd.DataFrame(columns=TL_COLS)], axis=1)
            return
        oof_parts = []
        if self.labelled:  # out-of-fold: each fold's labelled rows scored by a KB pinned with the other folds
            lab_rows = f["row_id"].isin(self.labelled).to_numpy() & (f["src"] == "extracted").to_numpy()
            fold = np.array([self.fold_of_page.get(p, -1) for p in f["page"]])
            for k in range(FOLDS):
                scored = lab_rows & (fold == k)
                if not scored.any():
                    continue
                keep = {r for r in self.labelled if self.fold_of_page[self.page_of[r]] != k}
                a = self.anchors(keep)
                trust_k, w = page_trust(tl, a, self.prior)
                fk, _ = tl.run(a, w, scored)
                oof_parts.append(fk[scored])
                for p in set(f["page"][scored]):
                    self.oof_trust[p] = trust_k.get(p)
        a = self.anchors(self.labelled)  # the inference KB: every labelled input row pinned (last run: stairs kept)
        self.trust, w = page_trust(tl, a, self.prior)
        feats, _ = tl.run(a, w)
        for part in oof_parts:
            feats.loc[part.index] = part
        self.F = pd.concat([f.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    def row_trust(self, rid):
        p = self.page_of.get(rid)
        if p is None:
            return np.nan
        if rid in self.labelled and p in self.oof_trust:
            return self.oof_trust[p]
        return self.trust.get(p, np.nan)

    def _support(self):
        """Label-free corpus support of each fact's value (every sentence fact), and each fact's status."""
        # a row is described by what was READ from it (gold forms only pin): training rows look like inference rows
        F = self.F = self.F[self.F.src != "gold"].copy()
        sent = F[F.kind == "sentence"]
        pair_rows = sent.groupby(["subject", "attribute"])["row_id"].nunique()
        pair_vals = sent.groupby(["subject", "attribute"])["vkey"].nunique()
        same = sent.groupby(["subject", "attribute", "vkey"])["row_id"].nunique()
        med = sent.assign(x=[_num(v) for v in sent["value"]]).groupby(["subject", "attribute"])["x"].median()
        keys = list(zip(F.subject, F.attribute))
        own = (F.kind == "sentence").astype(int).to_numpy()
        F["pair_rows"] = [pair_rows.get(k, 0) for k in keys]
        F["pair_n_values"] = [pair_vals.get(k, 0) for k in keys]
        F["same_value_rows"] = np.array([same.get((s, a, v), 0) for s, a, v in zip(F.subject, F.attribute, F.vkey)]) - own
        F["value_share"] = (F["same_value_rows"] + own) / F["pair_rows"].clip(lower=1)
        x = np.array([_num(v) for v in F["value"]])
        m = np.array([med.get(k, np.nan) for k in keys], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            F["log_dev_from_pair_median"] = np.log10((np.abs(x) + 1e-9) / (np.abs(m) + 1e-9))
        F["op_present"] = [int(bool(v.get("op"))) for v in F["value"]]
        others = pd.to_numeric(F["pair_mentions"], errors="coerce").fillna(0) - own
        F["st_unseen"] = (others <= 0).astype(int)
        F["st_holds"] = ((F.tl_agree == 1) & (F.st_unseen == 0)).astype(int)
        F["st_outdated"] = (F.tl_was_held_before == 1).astype(int)
        F["st_ahead"] = (F.tl_held_later == 1).astype(int)
        F["st_unbacked"] = ((F.tl_agree == 0) & (F.st_outdated == 0) & (F.st_ahead == 0) & (F.st_unseen == 0)).astype(int)
        self.by_row = {k: g for k, g in F.groupby("row_id")}

    # ------------------------------------------------------------------ type-specific checks (inference stairs)
    def compare_check(self, form, date):
        L, R = form.get("left", {}), form.get("right", {})
        _, lv = self.tl.value_at(L.get("subject"), L.get("attribute"), date)
        _, rv = self.tl.value_at(R.get("subject"), R.get("attribute"), date)
        if lv is None or rv is None:
            return {"cmp_resolvable": 0, "cmp_holds": np.nan, "cmp_sides_known": int(lv is not None) + int(rv is not None)}
        rel = nz.compare(lv, rv)
        return {"cmp_resolvable": int(rel is not None), "cmp_holds": np.nan if rel is None else int(rel == form.get("op")),
                "cmp_sides_known": 2}

    def change_check(self, form):
        seq = self.tl.sequence(form.get("subject"), form.get("attribute"))
        kf, kt = nz.key_inclusive(form.get("from", {})), nz.key_inclusive(form.get("to", {}))
        steps = list(zip(seq, seq[1:]))
        return {"chg_adjacent": int((kf, kt) in steps), "chg_reverse": int((kt, kf) in steps),
                "chg_from_seen": int(kf in seq), "chg_to_seen": int(kt in seq)}

    # ------------------------------------------------------------------ one feature row
    def row_features(self, rid, ctype, text, date, form=None):
        out = dict(TX.flags(text, date)) if ctype != "sentence" else dict(self.flags[rid])
        for t in TYPES:
            out[f"type_{t}"] = int(ctype == t)
        if ctype == "sentence":
            p = self.page_of[rid]
            ids = self.page_rows[p]
            out.update(page_size=len(ids), pos_rel=ids.index(rid) / max(len(ids) - 1, 1),
                       page_hype_share=self.page_hype[p], page_first_person_share=self.page_fp[p],
                       page_trust=self.row_trust(rid))
        else:
            out.update(page_size=1, pos_rel=0.0, page_hype_share=np.nan, page_first_person_share=np.nan,
                       page_trust=np.nan)
        g = self.by_row.get(rid)
        out["n_facts"] = 0 if g is None else len(g)
        if g is not None and len(g):
            for c in TL_COLS + ["same_value_rows", "value_share", "pair_n_values", "pair_rows", "log_dev_from_pair_median"]:
                vals = pd.to_numeric(g[c], errors="coerce")
                out[f"{c}_min"], out[f"{c}_max"] = vals.min(), vals.max()
            for c in ["st_unseen", "st_holds", "st_outdated", "st_ahead", "st_unbacked", "op_present"]:
                out[f"{c}_any"] = int(g[c].max())
                out[f"{c}_share"] = float(g[c].mean())
            out["fact_conf_min"], out["fact_conf_mean"] = float(g.conf.min()), float(g.conf.mean())
            out["fact_agree_share"] = float(g.agree.mean())
        if ctype == "compare":
            out.update(self.compare_check(form or {}, date))
        if ctype == "change":
            out.update(self.change_check(form or {}))
        return out

    def matrix(self, refs):
        """refs: list of (rid, ctype, text, date, form)."""
        return pd.DataFrame([self.row_features(*r) for r in refs])

    # ------------------------------------------------------------------ explanation of one unlabelled sentence
    def trust_and_sources(self, rid, date):
        g = self.by_row.get(rid)
        page_t = self.row_trust(rid)
        page_t = 0.7 if page_t is None or (isinstance(page_t, float) and np.isnan(page_t)) else page_t
        facts, sup_w, ag_w, confs = [], 0.0, 0.0, []
        if g is not None:
            for r in g.itertuples():
                sup, aga = supporters(self.tl, r.subject, r.attribute, r.value, date)
                sup = [s for s in sup if s != rid]
                aga = [s for s in aga if s != rid]
                sf = 0.0 if pd.isna(r.support_for) else float(r.support_for)
                sa = 0.0 if pd.isna(r.support_against) else float(r.support_against)
                sup_w += sf
                ag_w += sa
                confs.append(float(r.conf))
                status = ("unseen" if r.st_unseen else "holds" if r.st_holds else "outdated" if r.st_outdated
                          else "ahead_of_time" if r.st_ahead else "unbacked")
                facts.append({"subject": r.subject, "attribute": r.attribute, "value": r.value,
                              "status": status, "support": round(sf, 3), "against": round(sa, 3),
                              "supporting_rows": sup, "contradicting_rows": aga,
                              "labelled_sources": {i: self.label_of[i] for i in sup + aga if i in self.labelled},
                              "quotes": {i: self.text_of.get(i, "") for i in sup + aga} if self.with_quotes else {},
                              "extraction_conf": round(float(r.conf), 3), "both_extractors": bool(r.agree)})
        support_ratio = sup_w / (sup_w + ag_w) if sup_w + ag_w > 0 else 0.5
        conf = float(np.mean(confs)) if confs else 0.5
        trust = 0.4 * page_t + 0.4 * support_ratio + 0.2 * conf
        return {"trust_score": round(trust, 3),
                "trust_parts": {"page_trust": round(page_t, 3), "support_ratio": round(support_ratio, 3),
                                "extraction_conf": round(conf, 3)},
                "facts": facts}


def item_refs(items):
    return [(it["id"], it["type"], it["text"], it["date"], it["form"]) for it in items]


def sentence_refs(rows):
    return [(r["id"], "sentence", r.get("text", ""), r.get("date", ""), None) for r in rows]


def fit(X: pd.DataFrame, y: np.ndarray):
    m = lgbm()
    m.fit(X, y)
    return m


def predict(model, X: pd.DataFrame) -> np.ndarray:
    P = np.zeros((len(X), 3))
    if len(X):
        P[:, model.classes_] = model.predict_proba(X)
    return P

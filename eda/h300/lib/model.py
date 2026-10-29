"""The learned layer on top of the Reader features, and the saved-model wrapper.

Everything here is fitted on training rows only; calibrators, class offsets and stacking weights are fitted on
inner out-of-fold predictions of those training rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

CLASSES = np.array(["true", "false", "unsure"])
LAB2I = {c: i for i, c in enumerate(CLASSES)}
KINDS = ["none", "quantity", "money", "percent", "duration", "date", "version", "name", "boolean"]


def design(F: pd.DataFrame, cols: list[str], cfg: dict, parse_rel: dict | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cols:
        X[c] = pd.to_numeric(F[c], errors="coerce") if c in F else np.nan
    if "kind" in F:
        for k in KINDS:
            X[f"kind_{k}"] = (F["kind"] == k).astype(float)
    if cfg.get("parse_rel") and parse_rel is not None:
        rel = F.get("kind", pd.Series("none", index=F.index)).map(parse_rel).fillna(parse_rel.get("_all", 0.5))
        X["parse_rel"] = rel.values
        if cfg.get("parse_shrink"):
            for c in ("cw_true", "cw_false", "stale", "matches_latest", "ever_false"):
                if c in F:
                    X[f"{c}_x_rel"] = pd.to_numeric(F[c], errors="coerce").fillna(0).values * rel.values
    return X


def parse_reliability(F: pd.DataFrame, y: np.ndarray) -> dict:
    """Per value kind: how often the closed reading's decision equals the label (smoothed, training rows only)."""
    dec = np.where(F.get("cw_true", 0) == 1, "true", np.where(F.get("cw_false", 0) == 1, "false", "unsure"))
    ok = (dec == CLASSES[y]).astype(float)
    kinds = F.get("kind", pd.Series("none", index=F.index)).values
    allm = ok.mean() if len(ok) else 0.5
    out = {"_all": float(allm)}
    for k in set(kinds):
        m = kinds == k
        out[k] = float((ok[m].sum() + 20 * allm) / (m.sum() + 20))
    return out


def lgbm(cfg: dict, seed: int = 0) -> LGBMClassifier:
    p = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=1.0,
             colsample_bytree=1.0, random_state=seed, n_jobs=2, verbose=-1, deterministic=True)
    if cfg.get("reg") == "strong":   # our default for anything we ship: shallow, few, well-populated leaves
        p.update(n_estimators=100, num_leaves=7, min_child_samples=50, reg_lambda=20.0)
    elif cfg.get("reg"):
        p.update(reg_lambda=10.0, min_child_weight=20.0 / 100, min_child_samples=20)
    if seed:
        p.update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    return LGBMClassifier(**p)


def _proba(clf, X) -> np.ndarray:
    P = np.zeros((len(X), 3))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


class Learned:
    """Fit/predict the learned layer with the optional pieces switched by cfg."""

    def __init__(self, cfg: dict, cols: list[str], seed: int = 0):
        self.cfg, self.cols, self.seed = dict(cfg), list(cols), seed

    def _base_fit(self, X, y, w):
        m = lgbm(self.cfg, self.seed).fit(X, y, sample_weight=w)
        et = None
        if self.cfg.get("et"):
            et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, random_state=self.seed, n_jobs=2)
            et.fit(X.fillna(X.median()).fillna(0), y, sample_weight=w)
        return m, et

    def _base_pred(self, m, et, X):
        P = _proba(m, X)
        if et is not None:
            Q = np.zeros_like(P)
            Q[:, et.classes_] = et.predict_proba(X.fillna(self.med).fillna(0))
            P = 0.7 * P + 0.3 * Q
        return P

    def fit(self, F: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None):
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        self.parse_rel = parse_reliability(F, y) if self.cfg.get("parse_rel") else None
        X = design(F, self.cols, self.cfg, self.parse_rel)
        self.med = X.median()
        self.feature_names = list(X.columns)
        self.m, self.et = self._base_fit(X, y, w)
        self.iso, self.meta, self.offset = None, None, np.zeros(3)
        need_inner = self.cfg.get("iso") or self.cfg.get("stack") or self.cfg.get("offsets")
        if need_inner and len(np.unique(y)) > 1:
            oof = np.zeros((len(y), 3))
            ns = max(2, min(3, int(np.bincount(y).min()) if np.bincount(y).min() >= 2 else 2))
            for tr, te in StratifiedKFold(ns, shuffle=True, random_state=self.seed).split(X, y):
                m, et = self._base_fit(X.iloc[tr], y[tr], w[tr])
                oof[te] = self._base_pred(m, et, X.iloc[te])
            if self.cfg.get("stack"):
                Z = np.column_stack([np.log(np.clip(oof, 1e-4, 1)), F[["cw_true", "cw_false", "cw_unsure"]]
                                     .apply(pd.to_numeric, errors="coerce").fillna(0).values])
                self.meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y, sample_weight=w)
                oof = self._stack(oof, F)
            if self.cfg.get("iso"):
                self.iso = []
                for c in range(3):
                    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                    ir.fit(oof[:, c], (y == c).astype(float), sample_weight=w)
                    self.iso.append(ir)
                oof = self._iso(oof)
            if self.cfg.get("offsets"):
                self.offset = tune_offsets(oof, y)
        return self

    def _stack(self, P, F):
        Z = np.column_stack([np.log(np.clip(P, 1e-4, 1)), F[["cw_true", "cw_false", "cw_unsure"]]
                             .apply(pd.to_numeric, errors="coerce").fillna(0).values])
        Q = np.zeros_like(P)
        Q[:, self.meta.classes_] = self.meta.predict_proba(Z)
        return Q

    def _iso(self, P):
        Q = np.column_stack([self.iso[c].predict(P[:, c]) for c in range(3)]) + 1e-6
        return Q / Q.sum(1, keepdims=True)

    def predict_proba(self, F: pd.DataFrame) -> np.ndarray:
        X = design(F, self.cols, self.cfg, self.parse_rel).reindex(columns=self.feature_names)
        P = self._base_pred(self.m, self.et, X)
        if self.meta is not None:
            P = self._stack(P, F)
        if self.iso is not None:
            P = self._iso(P)
        if self.cfg.get("misread_gate") and "misread_risk" in F:
            risky = (pd.to_numeric(F["misread_risk"], errors="coerce").fillna(0).values == 1) & \
                    (pd.to_numeric(F.get("stale", 0), errors="coerce").fillna(0).values == 0)
            cap = np.minimum(P[:, 1], 0.5)
            P[risky, 1] = cap[risky]
            P = P / P.sum(1, keepdims=True)
        return P

    def predict(self, F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        P = self.predict_proba(F)
        pred = np.argmax(np.log(np.clip(P, 1e-9, 1)) + self.offset, axis=1)
        return pred, P


def macro3(y, p) -> float:
    f = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def tune_offsets(P, y, grid=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)) -> np.ndarray:
    best, off = -1, np.zeros(3)
    L = np.log(np.clip(P, 1e-9, 1))
    for a in grid:
        for b in grid:
            o = np.array([0.0, a, b])
            s = macro3(y, np.argmax(L + o, 1))
            if s > best + 1e-9:
                best, off = s, o
    return off


NO_READING_CAP = 0.6   # a row we could not read into a known fact never gets a confident label


class SavedModel:
    """Reader + learned layer, frozen. label(rows) -> [{id, label, confidence}] (confidence None for unsure).

    Nothing is fitted at prediction time. Rows we cannot read into a known (subject, attribute) - unseen subjects or
    attributes, no extracted fact, empty text - keep the learned layer's label but their confidence is capped at
    NO_READING_CAP, and become unsure when the learned layer itself is undecided (max probability < 0.45)."""

    def __init__(self, reader, learned: Learned, name: str):
        self.reader, self.learned, self.name = reader, learned, name

    def label(self, rows):
        rows = [{"id": str(r.get("id", i)), "text": str(r.get("text") or ""), "date": str(r.get("date") or ""),
                 "label": ""} for i, r in enumerate(rows)]
        F = pd.DataFrame(self.reader.read(rows))
        for c in ("has_reading", "known_pair"):
            if c not in F:
                F[c] = 0
        pred, P = self.learned.predict(F)
        read_ok = (pd.to_numeric(F["has_reading"], errors="coerce").fillna(0).values == 1) & \
                  (pd.to_numeric(F["known_pair"], errors="coerce").fillna(0).values == 1)
        out = []
        for r, c, p, ok in zip(rows, pred, P, read_ok):
            lab = CLASSES[c]
            conf = float(min(max(p[c], 0.5), 1.0))
            if not ok:
                if p.max() < 0.45:
                    lab = "unsure"
                conf = min(conf, NO_READING_CAP)
            out.append({"id": r["id"], "label": lab, "confidence": None if lab == "unsure" else conf})
        return out

    def predict_rows(self, rows, lexicon=None):
        """Entry point for tests/tools: rows (dicts with id, text, date[, label]) -> label dicts. Fits nothing."""
        return self.label(rows)


def load_and_predict(model_file, rows, lexicon=None):
    """predict_entry used in the manifests: load a frozen model and label rows without fitting anything."""
    import joblib
    return joblib.load(model_file).predict_rows(rows, lexicon)


def predict_rows(model, rows, lexicon=None):
    """predict_entry of the manifests: `model` is a loaded SavedModel (or a path to its .joblib). Fits nothing."""
    if isinstance(model, (str, bytes)) or hasattr(model, "__fspath__"):
        import joblib
        model = joblib.load(model)
    return model.predict_rows(rows, lexicon)

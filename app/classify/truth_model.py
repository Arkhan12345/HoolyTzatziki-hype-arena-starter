"""Option A truth classifier: LightGBM + L2 LogReg -> OOF stacking -> isotonic.

Task is binary: is the row false? (`unsure` rows count as not-false.) The row
is labelled `false` when the calibrated P(false) reaches a threshold tuned on
out-of-fold predictions for true/false macro-F1; otherwise `true`. Confidence
is the calibrated probability of the chosen label, clipped to [0.5, 1] (R-22).

KB features for training rows are computed out-of-fold (a row never sees its
own page's labels), so training rows look like unlabelled rows at run time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.classify.timeline_net import SequenceModel, build_sequences, row_scores, safe_logit
from app.classify.truth_features import fact_table, kb_features, label_free_features, _aggregate
from app.extract.extractor import Fact, FactExtractor, Lexicon, page_ids

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 3)
COMPONENTS = ("lgbm", "logreg", "stack")


def macro_f1_true_false(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for label in ("true", "false"):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def labels_from_p_false(p_false: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_false >= threshold, "false", "true")


def tune_threshold(labels: Sequence[str], p_false: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for t in THRESHOLD_GRID:
        score = macro_f1_true_false(labels, labels_from_p_false(p_false, t))
        if score > best[1] + 1e-9:
            best = (float(t), score)
    return best


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _lgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=2, verbose=-1,
    )


def _logreg():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=3000),
    )


@dataclass
class TruthStack:
    """`seq_kind` = "cfc" or "gru" adds option D: a sequence net over fact
    timelines as a third base model (pass `seq` = build_sequences output)."""

    seed: int = 0
    inner_splits: int = 5
    seq_kind: str | None = None
    columns: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    def _meta_inputs(self, p_lgbm, p_lr, p_seq=None):
        cols = [_logit(p_lgbm), _logit(p_lr)]
        if self.seq_kind:
            cols += [safe_logit(p_seq, self.seq_fill_), np.isnan(p_seq).astype(float)]
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, labels: Sequence[str], groups: Sequence[int],
            seq: tuple[np.ndarray, np.ndarray] | None = None) -> "TruthStack":
        labels = np.asarray(labels)
        y = (labels == "false").astype(int)
        groups = np.asarray(groups)
        self.columns = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        n = len(y)
        if self.seq_kind:
            if seq is None:
                raise ValueError("seq_kind is set but no sequences were given")
            positions = X.index.get_indexer(seq[1])
            keep = positions >= 0
            seqs, positions = seq[0][keep], positions[keep]
            y_fact = y[positions].astype(float)
            self.seq_fill_ = float(y.mean())

        oof = np.zeros((n, 2))
        oof_seq = np.full(n, np.nan)
        folds = list(GroupKFold(self.inner_splits).split(Xv, y, groups))
        for k, (tr, va) in enumerate(folds):
            oof[va, 0] = _lgbm(self.seed).fit(Xv[tr], y[tr]).predict_proba(Xv[va])[:, 1]
            oof[va, 1] = _logreg().fit(Xv[tr], y[tr]).predict_proba(Xv[va])[:, 1]
            if self.seq_kind:
                in_tr = np.isin(positions, tr)
                in_va = np.isin(positions, va)
                sm = SequenceModel(self.seq_kind, seed=self.seed + k).fit(seqs[in_tr], y_fact[in_tr])
                oof_seq[va] = row_scores(sm.predict(seqs[in_va]), positions[in_va], va)
        self.lgbm_ = _lgbm(self.seed).fit(Xv, y)
        self.logreg_ = _logreg().fit(Xv, y)
        if self.seq_kind:
            self.seq_ = SequenceModel(self.seq_kind, seed=self.seed).fit(seqs, y_fact)

        # Meta-learner, and its own out-of-fold output for calibration.
        Z = self._meta_inputs(oof[:, 0], oof[:, 1], oof_seq)
        meta_oof = np.zeros(n)
        for tr, va in folds:
            meta_oof[va] = LogisticRegression(C=1.0).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
        self.meta_ = LogisticRegression(C=1.0).fit(Z, y)
        raw = {"lgbm": oof[:, 0], "logreg": oof[:, 1], "stack": meta_oof}
        if self.seq_kind:
            raw["seq"] = np.where(np.isnan(oof_seq), self.seq_fill_, oof_seq)
        self.iso_ = {k: IsotonicRegression(out_of_bounds="clip").fit(p, y) for k, p in raw.items()}
        calibrated = {k: self.iso_[k].predict(p) for k, p in raw.items()}
        self.thresholds = {k: tune_threshold(labels, p)[0] for k, p in calibrated.items()}
        self.oof_scores_ = {k: tune_threshold(labels, p)[1] for k, p in calibrated.items()}
        return self

    def predict_components(self, X: pd.DataFrame,
                           seq: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Calibrated P(false) from each base model and from the stack."""
        Xv = X.reindex(columns=self.columns).to_numpy(dtype=float)
        p_lgbm = self.lgbm_.predict_proba(Xv)[:, 1]
        p_lr = self.logreg_.predict_proba(Xv)[:, 1]
        out = {"lgbm": self.iso_["lgbm"].predict(p_lgbm),
               "logreg": self.iso_["logreg"].predict(p_lr)}
        p_seq = None
        if self.seq_kind:
            if seq is None:
                raise ValueError("this model needs fact sequences (option D)")
            positions = X.index.get_indexer(seq[1])
            keep = positions >= 0
            p_fact = self.seq_.predict(seq[0][keep])
            p_seq = row_scores(p_fact, positions[keep], np.arange(len(X)))
            out["seq"] = self.iso_["seq"].predict(np.where(np.isnan(p_seq), self.seq_fill_, p_seq))
        p_meta = self.meta_.predict_proba(self._meta_inputs(p_lgbm, p_lr, p_seq))[:, 1]
        out["stack"] = self.iso_["stack"].predict(p_meta)
        return out

    def predict_p_false(self, X: pd.DataFrame, seq=None) -> np.ndarray:
        return self.predict_components(X, seq)["stack"]


# ---------------------------------------------------------------- feature tables
def training_table(rows, facts, F, label_free, train_idx, labels, groups,
                   inner_splits: int = 5) -> pd.DataFrame:
    """Features for `train_idx`, with KB features out-of-fold by page."""
    train_idx = np.asarray(train_idx)
    parts = []
    g = np.asarray(groups)[train_idx]
    for tr, va in GroupKFold(inner_splits).split(train_idx, groups=g):
        known = {int(i): labels[i] for i in train_idx[tr]}
        parts.append(predict_table(rows, facts, F, label_free, known, train_idx[va]))
    return pd.concat(parts).loc[train_idx]


def training_sequences(F, train_idx, labels, groups, inner_splits: int = 5):
    """Option D sequences for `train_idx`; known labels out-of-fold like the KB."""
    train_idx = np.asarray(train_idx)
    g = np.asarray(groups)[train_idx]
    parts_s, parts_r = [], []
    for tr, va in GroupKFold(inner_splits).split(train_idx, groups=g):
        known = {int(i): labels[i] for i in train_idx[tr]}
        seqs, fact_rows = build_sequences(F, known, train_idx[va])
        parts_s.append(seqs)
        parts_r.append(fact_rows)
    return np.concatenate(parts_s), np.concatenate(parts_r)


def predict_table(rows, facts, F, label_free, known_labels: dict[int, str],
                  target_idx) -> pd.DataFrame:
    """Features for `target_idx`, with KB built from `known_labels` only."""
    target_idx = np.asarray(target_idx)
    mask = F["row"].isin(set(target_idx.tolist())).to_numpy()
    sub = F[mask]
    kb = kb_features_subset(F, sub, known_labels)
    kb_rows = _aggregate(sub, kb, len(rows)).loc[target_idx]
    return pd.concat([label_free.loc[target_idx], kb_rows], axis=1)


def kb_features_subset(F: pd.DataFrame, sub: pd.DataFrame, known_labels: dict[int, str]):
    """KB features for the facts in `sub`, using every fact of F for the timeline."""
    known_rows = set(known_labels)
    combined = pd.concat([F[F["row"].isin(known_rows)], sub])
    combined = combined[~combined.index.duplicated(keep="last")]
    kb = kb_features(combined, None, known_labels)
    return kb.loc[sub.index]


# ---------------------------------------------------------------- hand-labelled facts
# Labelled rows build the KB timeline and the corpus support. At run time the
# labelled rows are the practice rows, whose hand-labelled `form` facts ship in
# the image (R-12 allows it), so they need not be extracted: only the rows
# being judged carry extraction errors. +1.6 points grouped CV (4 seeds).

def gold_facts_from_rows(rows: Sequence[dict]) -> list[list[Fact]]:
    """Facts from the `form` column (empty list when a row has none)."""
    out = []
    for r in rows:
        form = json.loads(r["form"]) if r.get("form") else []
        out.append([Fact(str(r["id"]), f["subject"], f["attribute"], f["value"], "", 1.0, True)
                    for f in form])
    return out


def mix_facts(extracted, gold, known_labels: dict[int, str], target_idx):
    """Gold facts for labelled rows (not being judged), extracted for the rest."""
    target = set(int(i) for i in target_idx)
    return [gold[i] if (i in known_labels and i not in target and gold[i] is not None)
            else extracted[i] for i in range(len(extracted))]


def mixed_table(rows, extracted, gold, known_labels, target_idx, *,
                truth_discovery: bool = False, td_kwargs: dict | None = None) -> pd.DataFrame:
    """Features for `target_idx` exactly as at run time: judged rows extracted,
    labelled rows gold, KB from `known_labels` only."""
    facts = mix_facts(extracted, gold, known_labels, target_idx)
    F = fact_table(rows, facts)
    label_free = label_free_features(rows, facts, F, truth_discovery=truth_discovery,
                                     td_kwargs=td_kwargs)
    return predict_table(rows, facts, F, label_free, known_labels, np.asarray(target_idx))


def mixed_training_table(rows, extracted, gold, train_idx, labels, groups,
                         inner_splits: int = 5, **kwargs) -> pd.DataFrame:
    """Training features out-of-fold by page, each fold built like run time."""
    train_idx = np.asarray(train_idx)
    g = np.asarray(groups)[train_idx]
    parts = []
    for tr, va in GroupKFold(inner_splits).split(train_idx, groups=g):
        known = {int(i): labels[i] for i in train_idx[tr]}
        parts.append(mixed_table(rows, extracted, gold, known, train_idx[va], **kwargs))
    return pd.concat(parts).loc[train_idx]


# ---------------------------------------------------------------- runtime pipeline
def oof_facts(rows: Sequence[dict], lexicon: Lexicon, n_splits: int = 5) -> list[list[Fact]]:
    """Facts for every row from an extractor that never saw that row's page."""
    pages = page_ids(rows)
    out: list[list[Fact] | None] = [None] * len(rows)
    for tr, va in GroupKFold(n_splits).split(np.zeros(len(rows)), groups=pages):
        ex = FactExtractor(lexicon).fit([rows[i] for i in tr])
        for i, f in zip(va, ex.extract([rows[i] for i in va])):
            out[i] = f
    return out  # type: ignore[return-value]


@dataclass
class TruthPipeline:
    """Extractor + truth stack, trained on practice rows (with gold `form`).

    `gold_by_id` maps a practice row id to (text, hand-labelled facts). At run
    time a LABELLED row whose id and text match uses those facts instead of
    extracted ones; every row being judged is always extracted.
    """

    extractor: FactExtractor
    model: TruthStack
    truth_discovery: bool = False  # option B features on/off
    td_kwargs: dict = field(default_factory=dict)
    gold_by_id: dict = field(default_factory=dict)

    def label(self, rows: Sequence[dict]) -> list[dict]:
        """Label every row of `rows` that has no label; labelled rows feed the KB.

        Rows need id, text, date and label ("" when unknown), in file order.
        Returns {id, label, confidence} for the unlabelled rows.
        """
        facts = self.extractor.extract(rows)
        known = {i: str(r.get("label") or "").strip().lower() for i, r in enumerate(rows)}
        known = {i: lab for i, lab in known.items() if lab in ("true", "false", "unsure")}
        target = [i for i in range(len(rows)) if i not in known]
        if not target:
            return []
        gold = getattr(self, "gold_by_id", {}) or {}
        if gold and not self.model.seq_kind:
            for i in known:
                entry = gold.get(str(rows[i]["id"]))
                if entry and entry[0].strip() == str(rows[i]["text"]).strip():
                    facts[i] = entry[1]
        F = fact_table(rows, facts)
        label_free = label_free_features(rows, facts, F, truth_discovery=self.truth_discovery,
                                         td_kwargs=self.td_kwargs)
        X = predict_table(rows, facts, F, label_free, known, target)
        seq = build_sequences(F, known, target) if self.model.seq_kind else None
        p_false = self.model.predict_p_false(X, seq)
        threshold = self.model.thresholds["stack"]
        out = []
        for i, p in zip(target, p_false):
            label = "false" if p >= threshold else "true"
            conf = p if label == "false" else 1.0 - p
            out.append({"id": rows[i]["id"], "label": label,
                        "confidence": float(min(max(conf, 0.5), 1.0))})
        return out


def fit_pipeline(practice_rows: Sequence[dict], lexicon: Lexicon, seed: int = 0, *,
                 truth_discovery: bool = False, td_kwargs: dict | None = None,
                 seq_kind: str | None = None, use_gold: bool = True) -> TruthPipeline:
    """Train on practice rows. `use_gold` (default) builds the KB from the
    hand-labelled facts of labelled rows; option D (seq_kind) keeps the
    all-extracted path."""
    labels = np.asarray([str(r["label"]).strip().lower() for r in practice_rows])
    groups = page_ids(practice_rows)
    facts = oof_facts(practice_rows, lexicon)
    idx = np.arange(len(practice_rows))
    use_gold = use_gold and not seq_kind
    if use_gold:
        gold = gold_facts_from_rows(practice_rows)
        X = mixed_training_table(practice_rows, facts, gold, idx, labels, groups,
                                 truth_discovery=truth_discovery, td_kwargs=td_kwargs)
        seq = None
    else:
        F = fact_table(practice_rows, facts)
        label_free = label_free_features(practice_rows, facts, F, truth_discovery=truth_discovery,
                                         td_kwargs=td_kwargs)
        X = training_table(practice_rows, facts, F, label_free, idx, labels, groups)
        seq = training_sequences(F, idx, labels, groups) if seq_kind else None
    model = TruthStack(seed=seed, seq_kind=seq_kind).fit(X, labels, groups, seq)
    extractor = FactExtractor(lexicon).fit(practice_rows)
    gold_by_id = {}
    if use_gold:
        gold_by_id = {str(r["id"]): (str(r["text"]), g) for r, g in zip(practice_rows, gold)}
    return TruthPipeline(extractor=extractor, model=model, truth_discovery=truth_discovery,
                         td_kwargs=dict(td_kwargs or {}), gold_by_id=gold_by_id)

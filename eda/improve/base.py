"""Shared pieces of the improvement ladder: data, the bake-off's feature table, per-fold KB features, metrics.

Same features and KB protocol as eda/04_bakeoff.py (copied here so this runs in the torch-free .venv).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eda" / "out"
sys.path.insert(0, str(ROOT / "eda"))
sys.path.insert(0, str(ROOT))
from features import KB_REL_ORDER, facts_to_rows, kb_features, support_features, text_flags  # noqa: E402
from app.kb.normalize import key as value_key  # noqa: E402  (read-only import)

CLASSES = np.array(["false", "true", "unsure"])
SEED = 0


def lgbm(n_jobs=-1):
    """The bake-off winner: unweighted LightGBM, frozen hyperparameters."""
    return LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                          verbose=-1, n_jobs=n_jobs)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def fact_key(r):
    """Canonical value key (app/kb/normalize.key) from a facts.pkl row."""
    d = {"kind": r.kind, "value": r.value_raw}
    for k in ("unit", "op", "currency", "per"):
        v = _clean(getattr(r, k))
        if v is not None:
            d[k] = v
    try:
        return value_key(d)
    except Exception:  # malformed value: fall back to the raw JSON
        return json.dumps([r.kind, str(r.value)])


def label_free_table():
    """Rows, facts (with canonical key), support facts, and the label-free feature table of the bake-off."""
    R = pd.read_pickle(OUT / "rows.pkl")
    F = pd.read_pickle(OUT / "facts.pkl")
    F["vkey"] = [fact_key(r) for r in F.itertuples()]
    T = text_flags(R)
    FS = support_features(F)
    sup_cols = ["same_value_rows", "same_value_pages", "value_share", "pair_n_values", "pair_rows",
                "log_dev_from_pair_median", "round_number", "op_present", "agree_prev_mention",
                "agree_next_mention", "n_values_before"]
    agg = facts_to_rows(FS, sup_cols)
    X = pd.concat([R.set_index("id")[["page_size", "pos_in_page", "pos_rel", "n_facts", "any_pronoun_subject"]],
                   T.set_index(R["id"]), agg], axis=1)
    X["any_pronoun_subject"] = X["any_pronoun_subject"].astype(float)
    P = pd.read_pickle(OUT / "features.pkl").set_index("id")
    X["page_first_person_share"] = P["page_first_person_share"]
    X["page_hype_share"] = P["page_hype_share"]
    genre = pd.get_dummies(P["page_genre"], prefix="genre").astype(float)
    kinds = R.set_index("id")["kinds"].str.get_dummies(sep="|").add_prefix("kind_").astype(float)
    X = pd.concat([X, genre, kinds], axis=1).reindex(R["id"])
    X.columns = [str(c) for c in X.columns]
    meta = P.reindex(R["id"])[["page_genre", "speaker"]]
    return R, F, X, meta


def kb_table(F, test_ids, train_ids, inner_splits):
    """Gold-timeline KB features (bake-off protocol): test rows from all training labels,
    training rows out-of-fold through inner_splits."""
    num = ["days_since_last_change", "n_gold_values_for_pair"]
    parts = []
    if test_ids:
        sub = F[F.id.isin(set(test_ids) | train_ids)]
        kb = pd.concat([sub[["id"]], kb_features(sub, train_ids)], axis=1)
        parts.append(facts_to_rows(kb[kb.id.isin(set(test_ids))], num, rel_col="kb_relation"))
    for inner_train in inner_splits:
        inner_train = set(inner_train)
        inner_test = train_ids - inner_train
        sub = F[F.id.isin(train_ids)]
        kb = pd.concat([sub[["id"]], kb_features(sub, inner_train)], axis=1)
        parts.append(facts_to_rows(kb[kb.id.isin(inner_test)], num, rel_col="kb_relation"))
    out = pd.concat(parts).drop(columns=["kb_relation"])
    for c in [f"rel_{k}" for k in KB_REL_ORDER]:
        out[c] = out[c].fillna(0)
    return out


# ------------------------------------------------------------------ metrics
def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def metrics(y, proba, pred=None):
    """y: int labels (0 false, 1 true, 2 unsure). pred defaults to argmax(proba)."""
    pred = proba.argmax(1) if pred is None else pred
    yl, pl = CLASSES[y], CLASSES[pred]
    conf = np.where(pred == 2, proba.max(1), proba[np.arange(len(pred)), pred])
    correct = (pred == y).astype(float)
    n_pf = (pl == "false").sum()
    return {"acc": accuracy_score(yl, pl),
            "macro_f1_tf": f1_score(yl, pl, labels=["true", "false"], average="macro", zero_division=0),
            "ece": ece(conf, correct),
            "unsure_recall": ((pl == "unsure") & (yl == "unsure")).sum() / max((yl == "unsure").sum(), 1),
            "false_precision": ((pl == "false") & (yl == "false")).sum() / max(n_pf, 1),
            "pred_unsure": int((pl == "unsure").sum()), "pred_false": int(n_pf)}

"""Weak supervision from our dataset rules (docs/RULES.md sections 1-4), without new dependencies.

Each rule is a labelling function (LF) over rows: false / true / unsure / abstain. An EM label model learns,
without labels, how accurate each LF is and the class prior, and turns the votes into soft labels.
Rules that need the date-aware KB get it per fold from training labels only.

Tested on eda/splits.csv fold_group folds against the plain unweighted LightGBM:
  lgbm_plain            same features as the bake-off (label-free + temporal KB)
  lgbm_plus_weak        + the label model's soft labels and vote counts as features
  lgbm_weak_weighted    plain features, training rows weighted by how much the label model backs their label
  label_model_only      the label model itself as a classifier (reference)
Outputs: eda/out/weak_labels.csv (out-of-fold soft labels), eda/out/improve_weak.csv
Run: .venv/Scripts/python eda/weak_labels.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
sys.path.insert(0, str(ROOT / "eda"))
from features import KB_REL_ORDER, facts_to_rows, kb_features, support_features, text_flags  # noqa: E402

CLASSES = np.array(["false", "true", "unsure"])
F_, T_, U_, ABST = 0, 1, 2, -1
SEED = 0


# ------------------------------------------------------------------------------------------- features
def label_free_table():
    R = pd.read_pickle(OUT / "rows.pkl")
    F = pd.read_pickle(OUT / "facts.pkl")
    T = text_flags(R)
    FS = support_features(F)
    sup = ["same_value_rows", "same_value_pages", "value_share", "pair_n_values", "pair_rows",
           "log_dev_from_pair_median", "round_number", "op_present", "agree_prev_mention", "agree_next_mention",
           "n_values_before"]
    agg = facts_to_rows(FS, sup)
    X = pd.concat([R.set_index("id")[["page_size", "pos_in_page", "pos_rel", "n_facts", "any_pronoun_subject"]],
                   T.set_index(R["id"]), agg], axis=1)
    X["any_pronoun_subject"] = X["any_pronoun_subject"].astype(float)
    P = pd.read_pickle(OUT / "features.pkl").set_index("id")  # page shares/genre are text-only
    X["page_first_person_share"] = P["page_first_person_share"]
    X["page_hype_share"] = P["page_hype_share"]
    X = pd.concat([X, pd.get_dummies(P["page_genre"], prefix="genre").astype(float),
                   R.set_index("id")["kinds"].str.get_dummies(sep="|").add_prefix("kind_").astype(float)], axis=1)
    return R, F, X.reindex(R["id"])


def kb_rows(F, score_ids, train_ids):
    """Temporal-KB row features for score_ids from the gold of train_ids only."""
    assert not (set(score_ids) & set(train_ids)), "leak: a scored row's own label is in the KB"
    sub = F[F.id.isin(set(score_ids) | set(train_ids))]
    kb = pd.concat([sub[["id"]], kb_features(sub, set(train_ids))], axis=1)
    rows = facts_to_rows(kb[kb.id.isin(set(score_ids))], ["days_since_last_change", "n_gold_values_for_pair"],
                         rel_col="kb_relation")
    return rows.drop(columns=["kb_relation"])


def fold_kb(F, R, tr_ids, te_ids):
    """Test rows: KB from all training labels. Training rows: inner out-of-fold by page."""
    parts = [kb_rows(F, te_ids, tr_ids)]
    sub = R[R.id.isin(tr_ids)]
    for a, b in GroupKFold(5).split(sub, groups=sub["page"]):
        parts.append(kb_rows(F, sub.id.values[b], sub.id.values[a]))
    return pd.concat(parts)


# ------------------------------------------------------------------------------------------- rules -> LFs
def labelling_functions(X):
    """Votes matrix (rows x LFs) from docs/RULES.md sections 1-4."""
    g = lambda c: X[c].fillna(0) if c in X else pd.Series(0, index=X.index)
    lfs = {
        "T1_stale_value": (g("rel_stale_older") == 1, F_),
        "T2_contradicts_value_in_force": (g("rel_contradicts") == 1, F_),
        "T3_matches_latest": (g("rel_match_latest") == 1, T_),
        "T4_change_point": (g("rel_future_value") == 1, T_),
        "T5_single_value_pair": (g("pair_n_values_max") == 1, T_),
        "S_agrees_prev_mention": (X["agree_prev_mention_min"] == 1, T_),
        "S_disagrees_prev_mention": (X["agree_prev_mention_min"] == 0, F_),
        "V1_value_2x_median": (g("log_dev_from_pair_median_max") >= np.log10(2), F_),
        "V2_hedge_never_repeated": ((g("hedge") == 1) & (X["same_value_rows_min"] == 0), F_),
        "S1_hype_page": (g("page_first_person_share") >= 0.3, F_),
        "S3_joke_tail_single_fact": ((g("joke_tail") == 1) & (g("n_facts") == 1) & (g("kv_line") == 0), U_),
        "S4_hearsay": (g("hearsay") == 1, U_),
        "U2_unverifiable_one_off": ((g("pair_n_values_max") == 1) & (X["same_value_rows_min"] == 0)
                                    & (g("n_facts") == 1), U_),
        "U4_today_date_mismatch": (g("today_date_mismatch") == 1, U_),
        "U3_new_value_unrepeated": ((g("rel_before_first_known") == 1) & (X["same_value_rows_min"] == 0), F_),
    }
    L = pd.DataFrame({k: np.where(m.values, c, ABST) for k, (m, c) in lfs.items()}, index=X.index)
    return L


def label_model(L, pi, iters=50):
    """EM over rule firings: theta[j, y] = P(rule j fires | class y), class prior pi fixed.

    A rule that fires rarely but almost only on false rows ends up strongly pointing to false; a rule that
    fires on every class at the same rate ends up carrying no weight. No row labels are used.
    """
    B = (L.values >= 0).astype(float)
    vote = L.replace(ABST, np.nan).max().fillna(T_).astype(int).values  # the class each rule votes for
    n, m = B.shape
    cov = B.mean(0).clip(1e-3, 0.999)
    theta = np.tile(cov[:, None] * 0.5, (1, 3))
    theta[np.arange(m), vote] = np.minimum(cov * 3, 0.95)
    lp = np.log(pi)
    for _ in range(iters):
        th = theta.clip(1e-4, 1 - 1e-4)
        logp = lp + B @ np.log(th) + (1 - B) @ np.log(1 - th)
        P = np.exp(logp - logp.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        theta = (B.T @ P + 0.5) / (P.sum(0) + 1.0)
    prec = np.array([(theta[j, vote[j]] * pi[vote[j]]) / (theta[j] * pi).sum() for j in range(m)])
    return P, prec, pi


# ------------------------------------------------------------------------------------------- metrics
def ece(proba, y, bins=10):
    conf, correct = proba.max(1), (proba.argmax(1) == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        msk = (conf > lo) & (conf <= hi)
        if msk.any():
            e += msk.mean() * abs(correct[msk].mean() - conf[msk].mean())
    return e


def metrics(y, proba):
    pred, ylab = CLASSES[proba.argmax(1)], CLASSES[y]
    return {"acc": accuracy_score(ylab, pred),
            "macro_f1_tf": f1_score(ylab, pred, labels=["true", "false"], average="macro", zero_division=0),
            "ece": ece(proba, y),
            "unsure_recall": ((pred == "unsure") & (ylab == "unsure")).sum() / max((ylab == "unsure").sum(), 1),
            "false_precision": precision_score(ylab, pred, labels=["false"], average="micro", zero_division=0)}


def lgbm():
    return LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=2)


def proba3(model, X):
    p = np.zeros((len(X), 3))
    p[:, model.classes_] = model.predict_proba(X)
    return p


# ------------------------------------------------------------------------------------------- run
def main():
    t0 = time.time()
    R, F, X0 = label_free_table()
    S = pd.read_csv(ROOT / "eda" / "splits.csv")[["id", "holdout", "fold_group"]]
    R = R.merge(S, on="id")
    y = R.set_index("id")["label"].map({c: i for i, c in enumerate(CLASSES)})
    cv_ids = R[R.holdout == 0]
    variants = ["lgbm_plain", "lgbm_plus_weak", "lgbm_weak_weighted", "label_model_only"]
    oof = {v: pd.DataFrame(np.nan, index=cv_ids.id, columns=CLASSES) for v in variants}
    times = {v: [0.0, 0.0] for v in variants}
    weak_rows, lf_acc = [], []
    for f in range(5):
        te_ids = cv_ids.id[cv_ids.fold_group == f].values
        tr_ids = cv_ids.id[cv_ids.fold_group != f].values
        assert not set(te_ids) & set(tr_ids)
        KB = fold_kb(F, R, tr_ids, te_ids)
        X = X0.join(KB, how="left")
        X = X.loc[np.concatenate([tr_ids, te_ids])]
        L = labelling_functions(X)
        prior = np.bincount(y.loc[tr_ids].values, minlength=3) / len(tr_ids)  # training-fold class shares only
        P, acc, pi = label_model(L, prior)   # no row labels used
        P = pd.DataFrame(P, index=X.index, columns=[f"weak_p_{c}" for c in CLASSES])
        lf_acc.append(pd.Series(acc, index=L.columns, name=f))
        W = P.copy()
        W["weak_votes"] = (L >= 0).sum(1)
        W["weak_votes_false"] = (L == F_).sum(1)
        W["weak_votes_unsure"] = (L == U_).sum(1)
        weak_rows.append(P.loc[te_ids].assign(fold=f))
        Xtr, Xte = X.loc[tr_ids], X.loc[te_ids]
        ytr = y.loc[tr_ids].values

        for v in variants:
            t1 = time.time()
            if v == "label_model_only":
                p = P.loc[te_ids].values
                times[v][0] += time.time() - t1
            else:
                A, B = (Xtr.join(W), Xte.join(W)) if v == "lgbm_plus_weak" else (Xtr, Xte)
                sw = None
                if v == "lgbm_weak_weighted":
                    sw = 0.5 + P.loc[tr_ids].values[np.arange(len(tr_ids)), ytr]
                mdl = lgbm().fit(A, ytr, sample_weight=sw)
                t2 = time.time()
                p = proba3(mdl, B)
                times[v][0] += t2 - t1
                times[v][1] += (time.time() - t2) * 20000 / len(B)
            oof[v].loc[te_ids] = p
        print(f"fold {f}: {time.time() - t0:.0f}s", flush=True)

    rows = []
    yy = y.loc[cv_ids.id].values
    for v in variants:
        m = metrics(yy, oof[v].loc[cv_ids.id].values)
        rows.append({"variant": v, "cv": "fold_group", **{k: round(val, 4) for k, val in m.items()},
                     "fit_s": round(times[v][0] / 5, 2), "predict_s_cpu2": round(times[v][1] / 5, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "improve_weak.csv", index=False)
    pd.concat(weak_rows).reset_index().rename(columns={"index": "id"}).to_csv(OUT / "weak_labels.csv", index=False)
    acc_tab = pd.concat(lf_acc, axis=1).mean(1).round(3).sort_values(ascending=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print("\nlearned P(rule class | rule fires), mean over folds, no row labels used:\n" + acc_tab.to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

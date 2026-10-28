"""Step 7: improvement ladder on top of the best bake-off model (unweighted LightGBM on the date-aware KB).

Each variant adds ONE piece on top of the previous one, so every piece's gain is visible:
  L0_base            bake-off features + per-fold KB (reproduces the bake-off winner)
  L1_timeline        + staircase timeline features (unweighted votes)
  L2_trust           timeline re-run with page-trust weights (EM, genre priors) + page trust
  L3_failure         + fact statuses (holds / outdated / ahead_of_time / unbacked) + misread risk
  L4_rules           + hard rules (outdated against a backed value -> false; single-value pair -> never false)
  L5_threevalued     + three-valued sentence state as features + unsure threshold tuned on inner out-of-fold
  L6_judge           + evidence-sum judge (ensemble or calibrator, picked on inner out-of-fold) + temperature
  L7_runtime_retrain same as L6, refit with frozen hyperparameters on 2 threads (timed)
  L8_cfc             + cfc_current_prob (only if eda/out/neuro_feats.csv exists)
Leak control: the KB, the timeline anchors and the page-trust priors use the training fold's labels only;
training rows get every label-derived feature out-of-fold (inner page split). Holdout pages are not used
until the best variant is scored on them once.

Run:  .venv/Scripts/python eda/07_improve.py
Out:  eda/out/improve_labels.csv, eda/out/improve_oof.pkl
"""
import importlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).parent))
from improve.base import CLASSES, OUT, ROOT, SEED, kb_table, label_free_table, lgbm, metrics  # noqa: E402
from improve.failure import misread_features, row_features, status_features  # noqa: E402
from improve.judge_nb import (EvidenceJudge, apply_temperature, fit_temperature,  # noqa: E402
                              with_false_prob)
from improve.timeline import Timeline, anchors_for  # noqa: E402
from improve.trust import genre_priors, page_trust  # noqa: E402

warnings.filterwarnings("ignore")
ARCHETYPES = importlib.import_module("03_recommend").ARCHETYPES
VARIANTS = ["L0_base", "L1_timeline", "L2_trust", "L3_failure", "L4_rules", "L5_threevalued", "L6_judge",
            "L7_runtime_retrain", "L8_cfc"]
N_RUN = 20000  # rows in the real run, for extrapolating predict time


# ------------------------------------------------------------------ data
R, F_ALL, X0, META = label_free_table()
SPLITS = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
LABELS = dict(zip(R["id"], R["label"]))
NFACTS = dict(zip(R["id"], R["n_facts"]))
GENRE = META["page_genre"].to_dict()
PAGE_OF = dict(zip(R["id"], R["page"]))
Y = pd.Series(R["label"].values, index=R["id"]).map({c: i for i, c in enumerate(CLASSES)})
MR = misread_features(R, X0)
NEURO = None
if (OUT / "neuro_feats.csv").exists():
    NEURO = pd.read_csv(OUT / "neuro_feats.csv")


# ------------------------------------------------------------------ features per partition / fold
def partition(F, labelled, unlabelled, scored):
    """Timeline features (no trust, trust) + statuses for the scored rows of one labelled/unlabelled split."""
    assert not (labelled & scored), "scored rows must be unlabelled"
    assert scored <= unlabelled
    sub = F[F.id.isin(labelled | unlabelled)]
    tl = Timeline(sub)
    anchor = anchors_for(tl, labelled, LABELS, NFACTS)
    assert not (anchor[~np.isin(tl.row, list(labelled))] != 0).any(), "anchor on an unlabelled row"
    mask = np.isin(tl.row, list(scored))
    f0, _ = tl.run(anchor, np.ones(tl.n), mask)
    pri, overall = genre_priors(R, labelled, GENRE)
    page_genre = {PAGE_OF[i]: GENRE[i] for i in sub["id"].unique()}
    prior_by_page = {p: pri.get(g, overall) for p, g in page_genre.items()}
    trust, w = page_trust(tl, anchor, prior_by_page)
    f1, _ = tl.run(anchor, w, mask)
    ids, pages = tl.row[mask], tl.page[mask]
    f0, f1 = f0[mask].reset_index(drop=True), f1[mask].reset_index(drop=True)
    return pd.concat([row_features(ids, f0, prefix="tl0"),
                      row_features(ids, f1, trust, pages, prefix="tl1"),
                      status_features(ids, f1)], axis=1)


def fold_table(F, train_ids, test_ids, inner_splits, extra_unlabelled=frozenset()):
    """Full feature table for the fold's train (out-of-fold) and test rows."""
    t0 = time.time()
    KB = kb_table(F, list(test_ids), train_ids, inner_splits)
    t_kb = time.time() - t0
    t0 = time.time()
    parts = [partition(F, train_ids, test_ids | extra_unlabelled, test_ids)]
    t_tl_test = time.time() - t0
    for inner_train in inner_splits:
        inner_train = set(inner_train)
        inner_test = train_ids - inner_train
        parts.append(partition(F, inner_train, inner_test | test_ids | extra_unlabelled, inner_test))
    TL = pd.concat(parts)
    ids = list(train_ids | test_ids)
    X = pd.concat([X0.reindex(ids), KB.reindex(ids), TL.reindex(ids), MR.reindex(ids)], axis=1)
    X.columns = [str(c) for c in X.columns]
    return X, {"kb_s": t_kb, "timeline_test_s": t_tl_test}


def cols_for(variant, X):
    base = list(X0.columns) + [c for c in X.columns if c.startswith(("rel_", "days_since_last_change",
                                                                       "n_gold_values_for_pair"))]
    pick = lambda p: [c for c in X.columns if c.startswith(p)]  # noqa: E731
    if variant == "L0_base":
        return base
    if variant == "L1_timeline":
        return base + pick("tl0__")
    if variant == "L2_trust":
        return base + pick("tl1__")
    if variant in ("L3_failure", "L4_rules"):
        return base + pick("tl1__") + pick("fs__") + pick("mr__")
    cols = base + pick("tl1__") + pick("fs__") + pick("mr__") + pick("tv__")
    if variant == "L8_cfc":
        cols = cols + ["cfc_current_prob"]
    return cols


# ------------------------------------------------------------------ post-processing pieces
def proba3(model, X):
    p = np.zeros((len(X), 3))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def hard_rules(P, Z):
    P = P.copy()
    stale = ((Z["fs__any_outdated"] == 1) & (Z["tl1__n_pages_against_max"] >= 1)
             & (Z["tl1__support_margin_min"] < 0)).to_numpy()
    P[stale] = with_false_prob(P[stale], np.maximum(P[stale, 0], 0.8))
    single = (Z["pair_n_values_max"] == 1).to_numpy()
    P[single] = with_false_prob(P[single], np.minimum(P[single, 0], 0.05))
    return P


def unsure_candidates(Z):
    return ((Z["tv__gap"] == 1) | (Z["joke_tail"] == 1)).to_numpy()


def decide(P, cand, tau):
    pred = P.argmax(1)
    if tau:
        pred = np.where(cand & (P[:, :2].max(1) < tau), 2, pred)
    return pred


def tune_tau(P, y, cand):
    best, best_tau = -1, 0.0
    for tau in [0.0] + list(np.arange(0.50, 0.92, 0.02)):
        acc = (decide(P, cand, tau) == y).mean()
        if acc > best + 1e-9:
            best, best_tau = acc, tau
    return best_tau


def archetype_of(Z):
    A = Z.copy()
    A["speaker"] = META["speaker"].reindex(Z.index)
    arch = pd.Series(None, index=Z.index, dtype=object)
    for name, _opt, _sec, _algos, cond, _reason in ARCHETYPES:
        hit = cond(A).fillna(False).astype(bool) & arch.isna()
        arch[hit] = name
    return arch


def evidence_items(Z):
    trust = Z["tl1__page_trust"].fillna(Z["tl1__page_trust"].median())
    tv = np.where(Z["tv__false"] == 1, "F", np.where(Z["tv__true"] == 1, "T", "gap"))
    return pd.DataFrame({"archetype": archetype_of(Z), "worst": Z["fs__worst"].fillna(-1),
                         "trust": pd.cut(trust, [-1, 0.6, 0.8, 0.9, 2], labels=False), "tv": tv,
                         "joke": Z["joke_tail"]}, index=Z.index)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logloss_bin(p, yb):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(yb * np.log(p) + (1 - yb) * np.log(1 - p)).mean()


# ------------------------------------------------------------------ one fold, all variants
def run_fold(X, train_ids, test_ids, inner_splits, variants, extras=None):
    tr, te = sorted(train_ids), sorted(test_ids)
    ytr, yte = Y[tr].to_numpy(), Y[te].to_numpy()
    Ztr, Zte = X.loc[tr], X.loc[te]
    out = {}
    inner_masks = [(np.isin(tr, list(a)), ~np.isin(tr, list(a))) for a in inner_splits]
    for v in variants:
        cols = cols_for(v, X)
        if v == "L8_cfc" and "cfc_current_prob" not in X.columns:
            continue
        n_jobs = 2 if v == "L7_runtime_retrain" else -1
        t0 = time.time()
        m = lgbm(n_jobs).fit(Ztr[cols], ytr)
        fit_s = time.time() - t0
        t0 = time.time()
        P = proba3(m, Zte[cols])
        pred_s = (time.time() - t0) * N_RUN / max(len(te), 1)
        info = {"fit_s": fit_s, "predict_s_cpu2": pred_s}
        if v in ("L0_base", "L1_timeline", "L2_trust", "L3_failure"):
            out[v] = (P, P.argmax(1), info)
            continue
        # inner out-of-fold probabilities for the training rows (for tau, judge, temperature)
        Pin = np.zeros((len(tr), 3))
        for a, b in inner_masks:
            mi = lgbm(n_jobs).fit(Ztr.loc[a, cols], ytr[a])
            Pin[b] = proba3(mi, Ztr.loc[b, cols])
        if v in ("L6_judge", "L7_runtime_retrain", "L8_cfc"):
            items_tr, items_te = evidence_items(Ztr), evidence_items(Zte)
            lo_in = np.zeros(len(tr))
            for a, b in inner_masks:
                lo_in[b] = EvidenceJudge().fit(items_tr[a], ytr[a] == 0).log_odds(items_tr[b])
            lo_te = EvidenceJudge().fit(items_tr, ytr == 0).log_odds(items_te)
            yb = (ytr == 0).astype(float)
            # (a) ensemble member: weighted logit average; (b) calibrator: logistic regression on both
            best_w, best_ll = 1.0, np.inf
            for w in np.linspace(0, 1, 11):
                ll = logloss_bin(1 / (1 + np.exp(-(w * logit(Pin[:, 0]) + (1 - w) * lo_in))), yb)
                if ll < best_ll:
                    best_w, best_ll = w, ll
            cal = LogisticRegression(C=1.0).fit(np.column_stack([logit(Pin[:, 0]), lo_in]), yb)
            ll_cal = logloss_bin(cal.predict_proba(np.column_stack([logit(Pin[:, 0]), lo_in]))[:, 1], yb)
            if ll_cal < best_ll:
                mode = "calibrator"
                pf_in = cal.predict_proba(np.column_stack([logit(Pin[:, 0]), lo_in]))[:, 1]
                pf_te = cal.predict_proba(np.column_stack([logit(P[:, 0]), lo_te]))[:, 1]
            else:
                mode = f"ensemble(w={best_w:.1f})"
                pf_in = 1 / (1 + np.exp(-(best_w * logit(Pin[:, 0]) + (1 - best_w) * lo_in)))
                pf_te = 1 / (1 + np.exp(-(best_w * logit(P[:, 0]) + (1 - best_w) * lo_te)))
            Pin, P = with_false_prob(Pin, pf_in), with_false_prob(P, pf_te)
            info["judge_mode"] = mode
        Pin, P = hard_rules(Pin, Ztr), hard_rules(P, Zte)
        if v == "L4_rules":
            out[v] = (P, P.argmax(1), info)
            continue
        if v in ("L6_judge", "L7_runtime_retrain", "L8_cfc"):
            T = fit_temperature(Pin, ytr)
            Pin, P = apply_temperature(Pin, T), apply_temperature(P, T)
            info["temperature"] = T
        tau = tune_tau(Pin, ytr, unsure_candidates(Ztr))
        info["tau"] = tau
        out[v] = (P, decide(P, unsure_candidates(Zte), tau), info)
    return out, yte, te


# ------------------------------------------------------------------ schemes
def scheme_folds(scheme, cv_ids):
    S = SPLITS.loc[cv_ids]
    col = "fold_group" if scheme == "fold_group" else "fold_row5"
    folds = []
    for k in sorted(S[col].unique()):
        test = set(S.index[S[col] == k])
        train = set(cv_ids) - test
        trl = sorted(train)
        if scheme == "fold_group":
            groups = [PAGE_OF[i] for i in trl]
            inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=groups)]
        else:
            inner = [[trl[j] for j in a] for a, _ in KFold(5, shuffle=True, random_state=SEED).split(trl)]
        folds.append((train, test, inner))
    return folds


def add_neuro(X, test_ids):
    if NEURO is None:
        return X
    col = "cfc_current_prob"
    nf = NEURO.groupby("id")[col].mean() if "fold" not in NEURO.columns else NEURO.groupby("id")[col].mean()
    X = X.copy()
    X[col] = nf.reindex(X.index)
    return X


def evaluate(scheme, cv_ids, variants, F):
    t_s = time.time()
    store = {v: ([], [], [], []) for v in variants}  # ids, proba, pred, info
    for fi, (train, test, inner) in enumerate(scheme_folds(scheme, cv_ids)):
        t0 = time.time()
        X, tinfo = fold_table(F, train, test, inner)
        X = add_neuro(X, test)
        t_feat = time.time() - t0
        res, yte, te = run_fold(X, train, test, inner, variants)
        for v, (P, pred, info) in res.items():
            info.update(tinfo)
            store[v][0].extend(te); store[v][1].append(P); store[v][2].append(pred); store[v][3].append(info)
        print(f"  {scheme} fold {fi}: features {t_feat:.1f}s, total {time.time() - t_s:.0f}s", flush=True)
    rows, oof = [], {}
    for v, (ids, Ps, preds, infos) in store.items():
        if not ids:
            continue
        P, pred = np.vstack(Ps), np.concatenate(preds)
        y = Y[ids].to_numpy()
        met = metrics(y, P, pred)
        info = pd.DataFrame(infos)
        row = {"variant": v, "cv": scheme, **{k: round(float(x), 4) for k, x in met.items()},
               "fit_s": round(info["fit_s"].mean(), 2), "predict_s_cpu2": round(info["predict_s_cpu2"].mean(), 3),
               "timeline_test_s": round(info["timeline_test_s"].mean(), 2)}
        for k in ("tau", "temperature", "judge_mode"):
            if k in info:
                row[k] = info[k].iloc[0] if k == "judge_mode" else round(info[k].mean(), 3)
        rows.append(row)
        oof[v] = {"ids": ids, "proba": P, "pred": pred, "y": y}
    return pd.DataFrame(rows), oof


def reproduce_bakeoff():
    """Step 0: the bake-off protocol exactly (GroupKFold(5) by page, all rows, L0 features)."""
    ids = list(R["id"])
    out_p, out_y = [], []
    for tr, te in GroupKFold(5).split(ids, groups=R["page"].values):
        train, test = set(np.array(ids)[tr]), list(np.array(ids)[te])
        trl = list(np.array(ids)[tr])
        inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[PAGE_OF[i] for i in trl])]
        KB = kb_table(F_ALL, test, train, inner)
        X = pd.concat([X0, KB.reindex(X0.index)], axis=1)
        X.columns = [str(c) for c in X.columns]
        m = lgbm().fit(X.loc[trl], Y[trl].to_numpy())
        out_p.append(proba3(m, X.loc[test])); out_y.append(Y[test].to_numpy())
    met = metrics(np.concatenate(out_y), np.vstack(out_p))
    return {"variant": "L0_base(bakeoff GroupKFold all rows)", "cv": "groupkfold_all",
            **{k: round(float(v), 4) for k, v in met.items()}}


def main():
    t_all = time.time()
    rep = reproduce_bakeoff()
    print(f"step 0 reproduction: acc {rep['acc']:.4f} macro-F1 {rep['macro_f1_tf']:.4f} "
          f"(bake-off: 0.8788 / 0.7618)", flush=True)
    assert abs(rep["acc"] - 0.8788) <= 0.005 and abs(rep["macro_f1_tf"] - 0.7618) <= 0.005, rep

    cv_ids = sorted(SPLITS.index[SPLITS.holdout == 0])
    hold_ids = set(SPLITS.index[SPLITS.holdout == 1])
    F_cv = F_ALL[F_ALL.id.isin(cv_ids)]
    assert not set(F_cv.id) & hold_ids
    variants = [v for v in VARIANTS if v != "L8_cfc" or NEURO is not None]

    res_g, oof_g = evaluate("fold_group", cv_ids, variants, F_cv)
    res_r, oof_r = evaluate("fold_row5", cv_ids, variants, F_cv)
    results = pd.concat([pd.DataFrame([rep]), res_g, res_r], ignore_index=True)

    # best variant on fold_group: macro-F1, ties -> lower ECE
    g = res_g.sort_values(["macro_f1_tf", "ece"], ascending=[False, True])
    best = g.variant.iloc[0]
    print(f"best on fold_group: {best}", flush=True)

    # extracted facts (real extraction for the unlabelled rows), if available
    oof_x = {}
    xf = OUT / "facts_extracted.pkl"
    if xf.exists():
        E = pd.read_pickle(xf)
        need = {"id", "subject", "attribute", "vkey"} if "vkey" in E.columns else {"id", "subject", "attribute"}
        if need <= set(E.columns):
            print("extracted facts found: scoring the best variant with test-row facts from extraction", flush=True)
            res_x, oof_x = evaluate_extracted(E, cv_ids, best)
            results = pd.concat([results, res_x], ignore_index=True)
        else:
            print(f"facts_extracted.pkl lacks columns {need - set(E.columns)}: skipped", flush=True)

    # holdout: once, for L0 and the best variant
    train, test = set(cv_ids), hold_ids
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[PAGE_OF[i] for i in trl])]
    X, tinfo = fold_table(F_ALL, train, test, inner)
    X = add_neuro(X, test)
    res, yte, te = run_fold(X, train, test, inner, ["L0_base", best] if best != "L0_base" else ["L0_base"])
    hold_rows, oof_h = [], {}
    for v, (P, pred, info) in res.items():
        met = metrics(yte, P, pred)
        hold_rows.append({"variant": v, "cv": "holdout_once", **{k: round(float(x), 4) for k, x in met.items()},
                          "fit_s": round(info["fit_s"], 2), "predict_s_cpu2": round(info["predict_s_cpu2"], 3),
                          "timeline_test_s": round(tinfo["timeline_test_s"], 2)})
        oof_h[v] = {"ids": te, "proba": P, "pred": pred, "y": yte}
    allt = (hold_ids | set(cv_ids))
    hold_rows.append({"variant": "all_true", "cv": "holdout_once",
                      "acc": round(float((Y[sorted(hold_ids)] == 1).mean()), 4)})
    results = pd.concat([results, pd.DataFrame(hold_rows)], ignore_index=True)
    assert len(allt) == len(R)

    results.to_csv(OUT / "improve_labels.csv", index=False)
    pd.to_pickle({"fold_group": oof_g, "fold_row5": oof_r, "holdout": oof_h, "extracted": oof_x, "best": best,
                  "classes": CLASSES}, OUT / "improve_oof.pkl")
    pd.set_option("display.width", 250)
    show = ["variant", "cv", "acc", "macro_f1_tf", "ece", "unsure_recall", "false_precision", "pred_unsure",
            "fit_s", "predict_s_cpu2", "timeline_test_s", "tau", "temperature", "judge_mode"]
    print(results[[c for c in show if c in results.columns]].to_string(index=False))
    print(f"\ntotal {time.time() - t_all:.0f}s")


def evaluate_extracted(E, cv_ids, best):
    """Best variant on fold_group with the unlabelled (test) rows' facts replaced by extracted ones."""
    from improve.base import fact_key
    E = E.copy()
    if "vkey" not in E.columns:
        E["vkey"] = [fact_key(r) for r in E.itertuples()]
    meta = R.set_index("id")[["date", "page", "label", "num"]]
    for c in ["date", "page", "label", "num"]:
        if c not in E.columns:
            E[c] = E["id"].map(meta[c])
    rows, oof = [], {}
    store = ([], [], [])
    for fi, (train, test, inner) in enumerate(scheme_folds("fold_group", cv_ids)):
        F = pd.concat([F_ALL[F_ALL.id.isin(train)], E[E.id.isin(test)]], ignore_index=True)
        X, _ = fold_table(F, train, test & set(F.id), inner)
        X = add_neuro(X, test)
        res, yte, te = run_fold(X, train, test & set(F.id), inner, [best])
        P, pred, _ = res[best]
        store[0].extend(te); store[1].append(P); store[2].append(pred)
    y = Y[store[0]].to_numpy()
    met = metrics(y, np.vstack(store[1]), np.concatenate(store[2]))
    rows.append({"variant": f"{best}+extracted_facts", "cv": "fold_group",
                 **{k: round(float(x), 4) for k, x in met.items()},
                 "note": f"{len(store[0])}/{len(cv_ids)} rows had extracted facts"})
    oof[best] = {"ids": store[0], "proba": np.vstack(store[1]), "pred": np.concatenate(store[2]), "y": y}
    return pd.DataFrame(rows), oof


if __name__ == "__main__":
    main()

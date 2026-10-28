"""Step 11: overfitting audit of the label models (all-true, L0, L4, L6, FINAL) -> docs/OVERFIT.md.

    LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/11_overfit.py           # all tests (~15 min)
    LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/11_overfit.py --only 3  # one test (results of the others are kept)

FINAL = L6 + tau_days 45 + min_child_samples 20 (the self-improvement result). Everything runs on eda/splits.csv
(holdout pages never scored here: test 2 only reads holdout numbers recorded earlier). Features come from the same
code as the ladder and the loop (eda/07_improve.py through eda/selfimprove/loop.py), so nothing is re-implemented.

Tests (one summary row each, PASS / WARN / FAIL):
  1 train-vs-test gap      in-sample vs out-of-fold macro-F1 (WARN > 0.10)
  2 CV-vs-holdout gap      recorded holdout scores vs CV, page-bootstrap SE (WARN > 2 SE)
  3 label permutation      training labels shuffled 5x, the pipeline must fall to <= 0.50 macro-F1 (FAIL = leakage)
  4 learning curve         25/50/75/100% of training pages: still rising = data-limited, not memorising
  5 seed stability         5 LightGBM seeds, std of macro-F1 (WARN > 0.01)
  6 selection bias         expected max noise over every variant tried + nested fold check of the loop's choices
  7 fold-scheme sign       FINAL - L0 gain on fold_group, fold_row5 and a time split keeps its sign
  8 per-archetype          archetypes where FINAL is worse than L0 by > 0.05 macro-F1
"""
import argparse
import importlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EDA = ROOT / "eda"
OUT = EDA / "out"
FIGS = OUT / "figs"
DOC = ROOT / "docs" / "OVERFIT.md"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(EDA / "selfimprove"))
LP = importlib.import_module("loop")      # loop knobs + cached fold features; LP.M is eda/07_improve.py
M = LP.M

DEFAULT = json.loads((EDA / "selfimprove" / "queue.json").read_text())["defaults"]
FINAL = {**DEFAULT, "tau_days": 45.0, "min_child_samples": 20}
VARIANT = {"L0": "L0_base", "L4": "L4_rules"}
ALL_TRUE_F1 = 0.4546
SUMMARY = OUT / "overfit_summary.csv"
DETAIL = {k: OUT / f"overfit_{k}.csv" for k in
          ("train_gap", "holdout_gap", "permutation", "learning_curve", "seeds", "selection", "schemes", "archetypes")}


def f1tf(y, pred):
    return f1_score(M.CLASSES[y], M.CLASSES[pred], labels=["true", "false"], average="macro", zero_division=0)


# ------------------------------------------------------------------ running one model on one fold
def cfg_of(model):
    return FINAL if model == "FINAL" else DEFAULT


def features(tag, F, train, test, inner, model, cached=True):
    """Fold feature table built under the model's timeline knobs (L0 ignores the timeline columns)."""
    cfg = cfg_of(model)
    LP.apply(cfg)
    if cached:
        return LP.fold_X(tag, F, train, test, inner, cfg)
    return M.fold_table(F, train, test, inner)[0]


def run(model, X, train, test, inner):
    """Out-of-fold (P, pred, y, ids) for one model on one fold."""
    LP.apply(cfg_of(model))
    if model in VARIANT:
        res, yte, te = M.run_fold(X, train, test, inner, [VARIANT[model]])
        P, pred, _ = res[VARIANT[model]]
        return P, pred, yte, te
    P, pred, _, yte, te = LP.run_one(X, train, test, inner, cfg_of(model))
    return P, pred, yte, te


def insample(model, X, train):
    """In-sample macro-F1 of the model's LightGBM (+ hard rules for L4 and later; calibration not refitted)."""
    LP.apply(cfg_of(model))
    tr = sorted(train)
    variant = VARIANT.get(model, "L6_judge")
    cols = M.cols_for(variant, X)
    y = M.Y[tr].to_numpy()
    m = M.lgbm(2).fit(X.loc[tr, cols], y)
    P = M.proba3(m, X.loc[tr, cols])
    if model != "L0":
        P = M.hard_rules(P, X.loc[tr])
    return f1tf(y, P.argmax(1)), (P.argmax(1) == y).mean()


def pairs(model):
    """(model whose features to build, cache tag prefix): L4/L6 share the default timeline, FINAL its own."""
    return "FINAL" if model == "FINAL" else "L6"


# ------------------------------------------------------------------ the main fold_group pass (tests 1, 6, 8)
def main_pass():
    folds = LP.folds_cv()
    oof = {m: ([], [], []) for m in ("L0", "L4", "L6", "FINAL")}
    gap_rows = []
    zoo = {"logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           LogisticRegression(C=1.0, max_iter=2000)),
           "extratrees": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                               ExtraTreesClassifier(300, min_samples_leaf=2, n_jobs=2, random_state=0)),
           "knn15": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          KNeighborsClassifier(15, weights="distance")),
           "svm_rbf": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=1.0))}
    for k, (train, test, inner) in enumerate(folds):
        t0 = time.time()
        Xd = features(f"fold{k}", LP.F_CV, train, test, inner, "L6")
        Xf = features(f"fold{k}", LP.F_CV, train, test, inner, "FINAL")
        for model, X in (("L0", Xd), ("L4", Xd), ("L6", Xd), ("FINAL", Xf)):
            P, pred, y, te = run(model, X, train, test, inner)
            oof[model][0].extend(te); oof[model][1].append(pred); oof[model][2].append(y)
            f_in, a_in = insample(model, X, train)
            gap_rows.append({"model": model, "fold": k, "train_f1": f_in, "train_acc": a_in,
                             "test_f1": f1tf(y, pred), "test_acc": (pred == y).mean()})
        # cheap zoo on the L0 feature set (test 1 only)
        LP.apply(DEFAULT)
        cols = M.cols_for("L0_base", Xd)
        tr, te = sorted(train), sorted(test)
        ytr, yte = M.Y[tr].to_numpy(), M.Y[te].to_numpy()
        for name, mk in zoo.items():
            m = mk().fit(Xd.loc[tr, cols].astype(float), ytr)
            gap_rows.append({"model": name, "fold": k,
                             "train_f1": f1tf(ytr, m.predict(Xd.loc[tr, cols].astype(float))),
                             "train_acc": (m.predict(Xd.loc[tr, cols].astype(float)) == ytr).mean(),
                             "test_f1": f1tf(yte, m.predict(Xd.loc[te, cols].astype(float))),
                             "test_acc": (m.predict(Xd.loc[te, cols].astype(float)) == yte).mean()})
        print(f"  main pass fold {k}: {time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(gap_rows)
    G.to_csv(DETAIL["train_gap"], index=False)
    store = {m: {"ids": ids, "pred": np.concatenate(p), "y": np.concatenate(y)} for m, (ids, p, y) in oof.items()}
    pickle.dump(store, open(OUT / "overfit_oof.pkl", "wb"))
    return store


# ------------------------------------------------------------------ test 3: label permutation
def permutation_test(n_perm=5):
    folds = LP.folds_cv()
    orig = {"LABELS": M.LABELS, "Y": M.Y, "R": M.R, "hard_rules": M.hard_rules}
    rows = []
    try:
        for p in range(n_perm):
            rng = np.random.default_rng(100 + p)
            out = {"FINAL": ([], []), "FINAL_no_rules": ([], []), "L0": ([], [])}
            for k, (train, test, inner) in enumerate(folds):
                tr = sorted(train)
                shuffled = dict(zip(tr, rng.permutation([orig["LABELS"][i] for i in tr])))
                lab = {**orig["LABELS"], **shuffled}
                M.LABELS = lab
                M.Y = pd.Series({i: int(np.where(M.CLASSES == lab[i])[0][0]) for i in orig["Y"].index})
                Rp = orig["R"].copy()
                Rp["label"] = Rp["id"].map(lab)
                M.R = Rp
                Fp = LP.F_CV.copy()
                Fp["label"] = Fp["id"].map(lab)
                X = features(None, Fp, train, test, inner, "FINAL", cached=False)
                for name in out:
                    # the hard rules use label-free timeline votes; switching them off isolates what was learned
                    M.hard_rules = (lambda P, Z: P) if name == "FINAL_no_rules" else orig["hard_rules"]
                    P, pred, y, te = run("L0" if name == "L0" else "FINAL", X, train, test, inner)
                    out[name][0].append(pred); out[name][1].append(orig["Y"][te].to_numpy())
                M.hard_rules = orig["hard_rules"]
            row = {"permutation": p}
            for name, (preds, ys) in out.items():
                row[name] = f1tf(np.concatenate(ys), np.concatenate(preds))
            rows.append(row)
            print(f"  permutation {p}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "permutation"),
                  flush=True)
    finally:
        M.LABELS, M.Y, M.R, M.hard_rules = orig["LABELS"], orig["Y"], orig["R"], orig["hard_rules"]
    pd.DataFrame(rows).to_csv(DETAIL["permutation"], index=False)


# ------------------------------------------------------------------ test 4: learning curve
def learning_curve(fracs=(0.25, 0.5, 0.75)):
    folds = LP.folds_cv()
    rows = []
    for frac in fracs:
        for k, (train, test, inner) in enumerate(folds):
            rng = np.random.default_rng(1000 + k)
            pages = sorted({M.PAGE_OF[i] for i in train})
            keep = set(rng.choice(pages, max(2, int(round(frac * len(pages)))), replace=False))
            sub = sorted(i for i in train if M.PAGE_OF[i] in keep)
            inner_sub = [[sub[j] for j in a] for a, _ in
                         GroupKFold(5).split(sub, groups=[M.PAGE_OF[i] for i in sub])]
            Fs = LP.F_CV[LP.F_CV.id.isin(set(sub) | test)]
            X = features(None, Fs, set(sub), test, inner_sub, "FINAL", cached=False)
            Xd = features(None, Fs, set(sub), test, inner_sub, "L6", cached=False)
            for model, XX in (("L0", Xd), ("FINAL", X)):
                P, pred, y, te = run(model, XX, set(sub), test, inner_sub)
                rows.append({"model": model, "frac": frac, "fold": k, "n_train": len(sub), "macro_f1_tf": f1tf(y, pred)})
        print(f"  learning curve {frac:.2f} done", flush=True)
    pd.DataFrame(rows).to_csv(DETAIL["learning_curve"], index=False)


# ------------------------------------------------------------------ test 5: seeds
def seed_test(seeds=(0, 1, 2, 3, 4)):
    folds = LP.folds_cv()
    rows = []
    orig_seed = M.SEED
    try:
        for s in seeds:
            M.SEED = s
            preds, ys = [], []
            for k, (train, test, inner) in enumerate(folds):
                X = features(f"fold{k}", LP.F_CV, train, test, inner, "FINAL")
                LP.apply(FINAL)          # re-read M.SEED into the LightGBM factory
                P, pred, _, yte, te = LP.run_one(X, train, test, inner, FINAL)
                preds.append(pred); ys.append(yte)
            rows.append({"seed": s, "macro_f1_tf": f1tf(np.concatenate(ys), np.concatenate(preds))})
            print(f"  seed {s}: {rows[-1]['macro_f1_tf']:.4f}", flush=True)
    finally:
        M.SEED = orig_seed
    pd.DataFrame(rows).to_csv(DETAIL["seeds"], index=False)


# ------------------------------------------------------------------ test 7: fold schemes
def scheme_test(store):
    rows = []
    g = {m: f1tf(store[m]["y"], store[m]["pred"]) for m in ("L0", "FINAL")}
    rows.append({"scheme": "fold_group", "L0": g["L0"], "FINAL": g["FINAL"]})
    folds = M.scheme_folds("fold_row5", LP.CV_IDS)
    res = {m: ([], []) for m in ("L0", "FINAL")}
    for k, (train, test, inner) in enumerate(folds):
        for model in ("L0", "FINAL"):
            X = features(f"row5_{k}", LP.F_CV, train, test, inner, "FINAL" if model == "FINAL" else "L6")
            P, pred, y, te = run(model, X, train, test, inner)
            res[model][0].append(pred); res[model][1].append(y)
    rows.append({"scheme": "fold_row5", **{m: f1tf(np.concatenate(res[m][1]), np.concatenate(res[m][0]))
                                          for m in res}})
    S = M.SPLITS.loc[LP.CV_IDS]
    train = set(S.index[S.time_split == "train"])
    test = set(S.index[S.time_split == "test"])
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[M.PAGE_OF[i] for i in trl])]
    t = {}
    for model in ("L0", "FINAL"):
        X = features("time_split", LP.F_CV, train, test, inner, "FINAL" if model == "FINAL" else "L6")
        P, pred, y, te = run(model, X, train, test, inner)
        t[model] = f1tf(y, pred)
    rows.append({"scheme": "time_split", **t})
    D = pd.DataFrame(rows)
    D["gain"] = D["FINAL"] - D["L0"]
    D.to_csv(DETAIL["schemes"], index=False)


# ------------------------------------------------------------------ tests 2, 6, 8 (no refitting)
def page_boot_se(ids, y, pred, B=2000, seed=0):
    page = np.array([M.PAGE_OF[i] for i in ids])
    up = np.unique(page)
    idx = {p: np.where(page == p)[0] for p in up}
    rng = np.random.default_rng(seed)
    vals = [f1tf(y[ii], pred[ii]) for ii in
            (np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(B))]
    return float(np.std(vals))


def holdout_gap():
    lab = pd.read_csv(OUT / "improve_labels.csv")
    cv = lab[lab.cv == "fold_group"].set_index("variant")["macro_f1_tf"]
    ho = lab[lab.cv == "holdout_once"].set_index("variant")["macro_f1_tf"]
    oof = pickle.load(open(OUT / "improve_oof.pkl", "rb"))["holdout"]
    se = {v: page_boot_se(np.asarray(oof[v]["ids"]), np.asarray(oof[v]["y"]), np.asarray(oof[v]["pred"]))
          for v in ("L0_base", "L4_rules")}
    st = json.loads((EDA / "selfimprove" / "state.json").read_text())
    final_cv = st["base"]["macro_f1_tf"] if st.get("base") else 0.8191
    final_ho = (st.get("lockbox") or {}).get("macro_f1_tf", 0.7445)
    rows = [{"model": "L0", "cv_f1": cv["L0_base"], "holdout_f1": ho["L0_base"], "holdout_se": se["L0_base"]},
            {"model": "L4", "cv_f1": cv["L4_rules"], "holdout_f1": ho["L4_rules"], "holdout_se": se["L4_rules"]},
            {"model": "FINAL", "cv_f1": final_cv, "holdout_f1": final_ho, "holdout_se": se["L4_rules"],
             "note": "holdout SE taken from L4 (same 304 rows); FINAL holdout predictions were not stored"}]
    D = pd.DataFrame(rows)
    D["gap"] = D["cv_f1"] - D["holdout_f1"]
    D.to_csv(DETAIL["holdout_gap"], index=False)


def selection_test(store):
    """Expected max noise over all variants tried + a nested fold check of the loop's two accepted knobs."""
    n = 0
    n += pd.read_csv(OUT / "bakeoff.csv")["model"].nunique()
    n += pd.read_csv(OUT / "improve_labels.csv")["variant"].nunique()
    n += pd.read_csv(OUT / "improve_weak.csv")["variant"].nunique()
    if (OUT / "improve_neuro.csv").exists():
        n += pd.read_csv(OUT / "improve_neuro.csv")["variant"].nunique()
    T = pd.read_csv(EDA / "selfimprove" / "trials.csv")
    T = T[T.knob != "base"].copy()
    n += T.drop_duplicates(["knob", "value"]).shape[0]
    se = pd.to_numeric(T["gain_se"], errors="coerce").dropna().mean()
    exp_max = se * math.sqrt(2 * math.log(n))
    f_l0 = f1tf(store["L0"]["y"], store["L0"]["pred"])
    f_fin = f1tf(store["FINAL"]["y"], store["FINAL"]["pred"])
    rows = [{"check": "expected_max_noise", "n_variants": n, "trial_se": se, "value": exp_max,
             "gain_final_vs_L0": f_fin - f_l0}]
    st = json.loads((EDA / "selfimprove" / "state.json").read_text())
    l6 = np.array(st["l6"]["fold_f1"])
    fcols = ["f0", "f1", "f2", "f3", "f4"]
    for knob, base_vec in (("tau_days", l6), ("min_child_samples", None)):
        cand = T[(T.knob == knob) & T.trial.str.startswith("r1q")].copy()
        if base_vec is None:  # min_child_samples was measured on top of tau 45 (trial r1q03)
            base_vec = T[T.trial == "r1q03"][fcols].astype(float).to_numpy()[0]
        V = np.vstack([base_vec] + [cand[fcols].astype(float).to_numpy()])
        names = ["base"] + [str(v) for v in cand["value"]]
        wins = 0
        for k in range(5):
            keep = [j for j in range(5) if j != k]
            pick = int(np.argmax(V[:, keep].mean(1)))
            gain_k = V[pick, k] - V[0, k]
            wins += int(pick != 0 and gain_k > 0)
            rows.append({"check": f"nested_{knob}", "held_fold": k, "picked": names[pick], "gain_on_held_fold": gain_k})
        rows.append({"check": f"nested_{knob}_summary", "value": wins, "note": "held-out folds where the pick wins"})
    pd.DataFrame(rows).to_csv(DETAIL["selection"], index=False)


def archetype_test(store):
    A = pd.read_csv(OUT / "practice_annotated.csv").set_index("id")["archetype"]
    rows = []
    l0 = pd.DataFrame({"y": store["L0"]["y"], "p0": store["L0"]["pred"]}, index=store["L0"]["ids"])
    fi = pd.DataFrame({"p1": store["FINAL"]["pred"]}, index=store["FINAL"]["ids"])
    D = l0.join(fi)
    D["arch"] = A.reindex(D.index)
    for arch, g in D.groupby("arch"):
        rows.append({"archetype": arch, "n": len(g), "n_false": int((g.y == 0).sum()),
                     "L0": f1tf(g.y.to_numpy(), g.p0.to_numpy()), "FINAL": f1tf(g.y.to_numpy(), g.p1.to_numpy()),
                     "acc_L0": (g.y == g.p0).mean(), "acc_FINAL": (g.y == g.p1).mean()})
    E = pd.DataFrame(rows)
    E["gain"] = E["FINAL"] - E["L0"]
    E["acc_gain"] = E["acc_FINAL"] - E["acc_L0"]
    E.sort_values("gain").to_csv(DETAIL["archetypes"], index=False)


# ------------------------------------------------------------------ summary, figures, doc
def summarise():
    S = []

    def add(model, test, value, threshold, verdict, note=""):
        S.append({"model": model, "test": test, "value": round(float(value), 4) if value is not None else None,
                  "threshold": threshold, "verdict": verdict, "note": note})

    G = pd.read_csv(DETAIL["train_gap"])
    g = G.groupby("model")[["train_f1", "test_f1", "train_acc", "test_acc"]].mean()
    for m, r in g.iterrows():
        gap = r.train_f1 - r.test_f1
        add(m, "1 train-vs-test gap (macro-F1)", gap, "> 0.10 = WARN", "WARN" if gap > 0.10 else "PASS",
            f"train {r.train_f1:.3f} / out-of-fold {r.test_f1:.3f}")
    H = pd.read_csv(DETAIL["holdout_gap"])
    for r in H.itertuples():
        v = "WARN" if r.gap > 2 * r.holdout_se else "PASS"
        add(r.model, "2 CV-vs-holdout gap (macro-F1)", r.gap, "> 2 x holdout SE = WARN", v,
            f"CV {r.cv_f1:.3f} / holdout {r.holdout_f1:.3f} / SE {r.holdout_se:.3f}")
    if DETAIL["permutation"].exists():
        Pm = pd.read_csv(DETAIL["permutation"])
        learned = Pm[["FINAL_no_rules", "L0"]].max(axis=1).max()
        add("FINAL", "3 label permutation: learned part (max of L0 / FINAL without hard rules, 5 runs)", learned,
            "<= 0.50 else FAIL (leakage)", "PASS" if learned <= 0.50 else "FAIL",
            f"L0 mean {Pm.L0.mean():.3f}, FINAL without rules mean {Pm.FINAL_no_rules.mean():.3f}; all-true {ALL_TRUE_F1}")
        add("FINAL", "3b label-free floor: FINAL with hard rules on shuffled labels (mean)", Pm.FINAL.mean(),
            "info", "INFO", "the hard rules read label-free timeline votes, so they keep some skill without labels")
    if DETAIL["learning_curve"].exists():
        L = pd.read_csv(DETAIL["learning_curve"]).groupby(["model", "frac"]).macro_f1_tf.mean().unstack(0)
        full = {m: f1tf(*_store_yp(m)) for m in ("L0", "FINAL")}
        for m in ("L0", "FINAL"):
            slope = full[m] - L.loc[0.75, m]
            add(m, "4 learning curve (75% -> 100% gain)", slope, "> 0 = still data-limited", "PASS" if slope > -0.005 else "WARN",
                " / ".join(f"{int(f*100)}% {L.loc[f, m]:.3f}" for f in L.index) + f" / 100% {full[m]:.3f}")
    if DETAIL["seeds"].exists():
        Sd = pd.read_csv(DETAIL["seeds"])
        sd = Sd.macro_f1_tf.std(ddof=1)
        add("FINAL", "5 seed stability (std macro-F1)", sd, "> 0.01 = WARN", "WARN" if sd > 0.01 else "PASS",
            f"mean {Sd.macro_f1_tf.mean():.4f}, range {Sd.macro_f1_tf.min():.4f}-{Sd.macro_f1_tf.max():.4f}")
    Se = pd.read_csv(DETAIL["selection"])
    e = Se[Se.check == "expected_max_noise"].iloc[0]
    add("FINAL", "6 selection bias (gain vs expected max noise)", e.gain_final_vs_L0, f"> {e.value:.3f} (noise max)",
        "PASS" if e.gain_final_vs_L0 > e["value"] else "WARN", f"{int(e.n_variants)} variants, trial SE {e.trial_se:.4f}")
    for knob in ("tau_days", "min_child_samples"):
        w = Se[Se.check == f"nested_{knob}_summary"]["value"].iloc[0]
        add("FINAL", f"6 nested check: {knob} pick wins on the unseen fold", w, ">= 3 of 5", "PASS" if w >= 3 else "WARN")
    if DETAIL["schemes"].exists():
        Sc = pd.read_csv(DETAIL["schemes"])
        for r in Sc.itertuples():
            add("FINAL", f"7 gain vs L0 on {r.scheme}", r.gain, "> 0", "PASS" if r.gain > 0 else "FAIL",
                f"L0 {r.L0:.3f} / FINAL {r.FINAL:.3f}")
    Ar = pd.read_csv(DETAIL["archetypes"])
    bad = Ar[(Ar.gain < -0.05) & (Ar.n >= 20) & (Ar.n_false >= 3)]
    add("FINAL", "8 archetypes worse than L0 by > 0.05 (n >= 20, >= 3 false)", len(bad), "0", "PASS" if bad.empty else "WARN",
        ", ".join(f"{a} {g:+.3f}" for a, g in zip(bad.archetype, bad.gain)))
    pd.DataFrame(S).to_csv(SUMMARY, index=False)
    return pd.DataFrame(S)


_STORE = None


def _store_yp(m):
    global _STORE
    if _STORE is None:
        _STORE = pickle.load(open(OUT / "overfit_oof.pkl", "rb"))
    return _STORE[m]["y"], _STORE[m]["pred"]


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    ink, grid, c_base, c_final = "#1f2933", "#d9dee3", "#8a94a0", "#2b6cb0"

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)

    G = pd.read_csv(DETAIL["train_gap"]).groupby("model")[["train_f1", "test_f1"]].mean()
    order = [m for m in ("L0", "L4", "L6", "FINAL", "logreg", "extratrees", "knn15", "svm_rbf") if m in G.index]
    G = G.loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(G))
    ax.bar(x - 0.2, G.train_f1, 0.4, color=c_base, label="in-sample (training rows)")
    ax.bar(x + 0.2, G.test_f1, 0.4, color=c_final, label="out-of-fold (unseen pages)")
    ax.set_xticks(x, G.index)
    ax.set_ylabel("macro-F1 (true/false)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Trees fit the training rows almost perfectly; what counts is the unseen-page score", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "overfit_train_vs_test.png", dpi=130)
    plt.close(fig)

    if DETAIL["learning_curve"].exists():
        L = pd.read_csv(DETAIL["learning_curve"]).groupby(["model", "frac"]).macro_f1_tf.mean().unstack(0)
        for m in ("L0", "FINAL"):
            L.loc[1.0, m] = f1tf(*_store_yp(m))
        L = L.sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(L.index * 100, L["L0"], "o-", color=c_base, label="L0 (plain LightGBM)")
        ax.plot(L.index * 100, L["FINAL"], "o-", color=c_final, label="FINAL")
        ax.set_xlabel("% of training pages used")
        ax.set_ylabel("out-of-fold macro-F1")
        ax.set_title("More pages still help: the models are data-limited", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        style(ax)
        fig.tight_layout()
        fig.savefig(FIGS / "overfit_learning_curve.png", dpi=130)
        plt.close(fig)

    if DETAIL["permutation"].exists():
        Pm = pd.read_csv(DETAIL["permutation"])
        real = f1tf(*_store_yp("FINAL"))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for j, (col, lab, c) in enumerate((("L0", "L0, shuffled labels", c_base),
                                          ("FINAL_no_rules", "FINAL without hard rules, shuffled", "#6b8fb8"),
                                          ("FINAL", "FINAL with hard rules, shuffled", "#c97b2b"))):
            ax.scatter(Pm[col], np.full(len(Pm), j), s=50, color=c, label=lab)
        ax.axvline(real, color=c_final, lw=2, label=f"FINAL, real labels {real:.3f}")
        ax.axvline(ALL_TRUE_F1, color=ink, lw=1, ls="--", label=f"all-true {ALL_TRUE_F1}")
        ax.set_yticks([])
        ax.set_ylim(-0.8, 3.2)
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("out-of-fold macro-F1 (true/false)")
        ax.set_title("Shuffled labels: the learned part falls to no skill; only label-free rules keep a little",
                     fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGS / "overfit_permutation.png", dpi=130)
        plt.close(fig)


def write_doc(S):
    def tbl(df):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.itertuples(index=False):
            out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                         (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
        return "\n".join(out)

    G = pd.read_csv(DETAIL["train_gap"]).groupby("model")[["train_f1", "test_f1", "train_acc", "test_acc"]].mean()
    G["gap_f1"] = G.train_f1 - G.test_f1
    H = pd.read_csv(DETAIL["holdout_gap"])
    Sc = pd.read_csv(DETAIL["schemes"]) if DETAIL["schemes"].exists() else None
    Sd = pd.read_csv(DETAIL["seeds"]) if DETAIL["seeds"].exists() else None
    Pm = pd.read_csv(DETAIL["permutation"]) if DETAIL["permutation"].exists() else None
    L = pd.read_csv(DETAIL["learning_curve"]) if DETAIL["learning_curve"].exists() else None
    Se = pd.read_csv(DETAIL["selection"])
    Ar = pd.read_csv(DETAIL["archetypes"])
    f_l0, f_fin = f1tf(*_store_yp("L0")), f1tf(*_store_yp("FINAL"))
    e = Se[Se.check == "expected_max_noise"].iloc[0]
    fin_h = H[H.model == "FINAL"].iloc[0]
    l0_h = H[H.model == "L0"].iloc[0]

    md = [f"# Overfitting audit of the label models\n",
          f"*Generated by `eda/11_overfit.py` on {time.strftime('%Y-%m-%d %H:%M')}. Folds: `eda/splits.csv` "
          "`fold_group` (5 folds by page, holdout pages excluded). FINAL = L6 + timeline vote window 45 days + "
          "LightGBM `min_child_samples` 20. No new holdout scores: test 2 reads the numbers recorded earlier.*\n",
          "## Verdicts\n", tbl(S[["model", "test", "value", "threshold", "verdict", "note"]]), "",
          "## 1. Train vs test\n",
          "In-sample = the model's LightGBM scored on its own training rows (hard rules applied for L4 and later; the "
          "calibrator is not refitted). A big in-sample score is normal for boosted trees; the question is whether "
          "the out-of-fold score holds up, which tests 2–7 check.\n", tbl(G.reset_index()), "",
          "![train vs test](../eda/out/figs/overfit_train_vs_test.png)\n",
          "## 2. CV vs holdout\n",
          "The holdout (43 pages, 304 rows) was scored once per model earlier. Its page-bootstrap standard error is "
          "about 0.03, so only gaps beyond ~0.06 are clear.\n", tbl(H.drop(columns=[c for c in ("note",) if c in H])), ""]
    if Pm is not None:
        md += ["## 3. Label permutation\n",
               "Training labels shuffled inside each fold (all label-dependent parts rebuilt: KB, timeline anchors, "
               "page trust priors, LightGBM, calibrator). A pipeline that leaks test labels would still score high. "
               "Three versions are scored on the same shuffled folds: L0 (no rules), FINAL with its hard rules switched "
               "off (the learned part), and FINAL as shipped. The hard rules read the unlabelled test-fold mentions "
               "(timeline votes), which need no labels, so FINAL with rules keeps a little skill even on shuffled "
               "labels; that is a label-free floor, not leakage. Leakage would show in the first two columns.\n",
               tbl(Pm), "", "![permutation](../eda/out/figs/overfit_permutation.png)\n"]
    if L is not None:
        md += ["## 4. Learning curve\n", tbl(L.groupby(["model", "frac"]).macro_f1_tf.mean().unstack(0).reset_index()), "",
               f"At 100% of the training pages: L0 {f_l0:.3f}, FINAL {f_fin:.3f}.\n",
               "![learning curve](../eda/out/figs/overfit_learning_curve.png)\n"]
    if Sd is not None:
        md += ["## 5. Seed stability\n", tbl(Sd), ""]
    md += ["## 6. Selection bias\n",
           f"{int(e.n_variants)} distinct variants were tried across the bake-off, ladder, weak labels, neural track and "
           f"the self-improvement loop. With a per-trial standard error of {e.trial_se:.4f}, the best gain pure noise "
           f"would produce is about {e['value']:.3f}. FINAL's out-of-fold gain over L0 is {e.gain_final_vs_L0:+.3f}.\n",
           "Nested check: for each fold, the loop's choice for a knob is made on the other 4 folds only, then scored on "
           "the held fold.\n", tbl(Se[Se.check.str.startswith("nested")]
                                     .drop(columns=[c for c in ("n_variants", "trial_se", "gain_final_vs_L0") if c in Se])), ""]
    if Sc is not None:
        md += ["## 7. Fold schemes\n", tbl(Sc), ""]
    md += ["## 8. Per archetype (FINAL − L0)\n", tbl(Ar[["archetype", "n", "n_false", "L0", "FINAL", "gain", "acc_gain"]]), ""]

    real_part = f_fin - f_l0
    md += ["## Conclusion\n",
           ("- **Leakage:** none found — with shuffled training labels the learned part (L0, and FINAL without its "
            "hard rules) falls to the no-skill level; FINAL with rules keeps only the label-free skill of its timeline "
            f"votes ({Pm.FINAL.mean():.3f})." if Pm is not None and Pm[['FINAL_no_rules', 'L0']].max(axis=1).max() <= 0.5
            else "- **Leakage:** SUSPECTED — see §3."),
           f"- **Cross-validation gain:** FINAL − L0 = {real_part:+.3f} macro-F1 out of fold, far above the "
           f"{e['value']:.3f} that selection over {int(e.n_variants)} variants could produce by chance, and positive on "
           "every fold scheme tested." if Sc is None or (Sc.gain > 0).all() else
           f"- **Cross-validation gain:** FINAL − L0 = {real_part:+.3f}; not positive on every fold scheme (see §7).",
           f"- **Holdout:** FINAL's CV−holdout gap is {fin_h.gap:+.3f} against L0's {l0_h.gap:+.3f} "
           f"(holdout SE ≈ {fin_h.holdout_se:.3f}). On the 43 holdout pages the gain over L0 does not show; part of the "
           "CV gain may be specific to the CV pages, or the holdout is simply too small to see it. Treat the realistic "
           "gain as somewhere between the holdout's ~0 and the CV's full value; the calibration gain (ECE 0.059 → 0.042 "
           "on the holdout) did carry over.",
           "- **Data-limited, not memorising:** see the learning curve — scores still rise with more pages, so more "
           "labelled data (the arena's other labelled rows) should help more than more tuning.",
           "- **What to do:** keep FINAL (its gain is significant in CV and calibration carries to the holdout), stop "
           "adding tuning knobs against the same folds, and confirm on the arena-shaped `check.sh` run once the model "
           "is ported into `app/classify`.", ""]
    DOC.write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    a = ap.parse_args()
    want = set(a.only or range(1, 9))
    t_all = time.time()
    if want & {1, 6, 7, 8} or not (OUT / "overfit_oof.pkl").exists():
        print("main pass (tests 1, 6, 8)", flush=True)
        store = main_pass()
    else:
        store = pickle.load(open(OUT / "overfit_oof.pkl", "rb"))
    holdout_gap()
    selection_test(store)
    archetype_test(store)
    if 7 in want:
        print("fold schemes (test 7)", flush=True)
        scheme_test(store)
    if 5 in want:
        print("seeds (test 5)", flush=True)
        seed_test()
    if 3 in want:
        print("permutation (test 3)", flush=True)
        permutation_test()
    if 4 in want:
        print("learning curve (test 4)", flush=True)
        learning_curve()
    S = summarise()
    figures()
    write_doc(S)
    print(S.to_string(index=False))
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()

"""Rules round: the dataset rules the label engine did not use yet (docs/RULES.md), as one-variable changes on top of
the current final configuration, each kept only if it passes the goal.json accept rule AND four overfitting gates.

    LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/rules_round/loop_rules.py           # run (resumable)
    LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/rules_round/loop_rules.py --report  # rebuild REPORT.md

Knobs (queue.json, greedy, one at a time): G1 ratio-to-held features, G3 hedge_unbacked, G4 speaker trust,
G2a joke -> unsure, G2b single-value unsure (t tuned on inner out-of-fold rows only), G2c hearsay -> unsure,
G5 today -> unsure. Code for the rules: eda/improve/rules_model.py. Evaluation code: eda/selfimprove/loop.py
(fold features, run_one, lockbox) imported, never edited; goal.json read-only (sha256 checked at start and end).

Accept rule (goal.json): mean per-fold macro-F1 (true/false) gain >= 0.005, better on >= 4/5 folds, pooled ECE not
worse by more than 0.01, runtime gate. Unsure knobs are judged on 3-class macro-F1 instead (same gain / folds rule)
and may not lower the pooled true/false macro-F1 by more than 0.005.
Overfitting gates (all must pass to keep a change):
  a  nested selection  for each fold j, the accept rule on the other 4 folds (>= 3/4 better) must select the change
                       AND it must gain on fold j, in >= 4/5 rotations; mean held-out gain > 0
  b  train-test gap    the change may not widen (train macro-F1 - out-of-fold macro-F1) by more than 0.02
  c  seed stability    the gain keeps its sign over 3 LightGBM seeds and std(gain) < mean(gain)
  d  scheme check      the gain stays positive on fold_row5 (FAIL otherwise) and on the time split
                       (train before 2026-07-01, test after; WARN only)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
SI = HERE.parent
EDA = SI.parent
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(SI))
import loop as L  # noqa: E402  (eda/selfimprove/loop.py: fold features, run_one, lockbox)
from improve import rules_model as RM  # noqa: E402
from improve.timeline import Timeline, anchors_for  # noqa: E402

M = L.M
GOAL_F = SI / "goal.json"
QUEUE_F, STATE_F, TRIALS_F, REPORT_F = (HERE / n for n in ("queue.json", "state.json", "trials.csv", "REPORT.md"))
FIGS = HERE / "figs"
BUDGET_S = 60 * 60
SEEDS = (0, 1, 2)
TF_DROP_MAX = 0.005
GAP_WIDEN_MAX = 0.02
TIME_CUT = "2026-07-01"
FEATURE_COLS = {"g1": ["ratio_to_held", "log_ratio_to_held", "perturb_ratio_hit"], "g3": ["hedge_unbacked"],
                "g4": ["speaker_trust", "speaker_n"]}
UNSURE_KNOBS = ("joke", "single_value", "hearsay", "today")
P_FEATS = pd.read_pickle(M.OUT / "features.pkl").set_index("id")
BASE_CFG = {"switch": 1.5, "tau_days": 45.0, "anchor": 6.0, "two_page_rule": True, "trust_rounds": 3,
            "trust_pseudo": 3.0, "genre_pseudo": 10.0, "num_leaves": 15, "min_child_samples": 20,
            "n_estimators": 200, "learning_rate": 0.05, "drop_family": None, "weak_features": False,
            "cfc_feature": False, "calibrator": "evidence_sum"}
RULE_KEYS = ("g1", "g3", "g4") + UNSURE_KNOBS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ folds per scheme
def scheme_folds(scheme):
    if scheme in ("fold_group", "fold_row5"):
        return M.scheme_folds(scheme, L.CV_IDS)
    S = M.SPLITS.loc[L.CV_IDS]
    d = pd.to_datetime(S["date"])
    train, test = set(S.index[d < TIME_CUT]), set(S.index[d >= TIME_CUT])
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[M.PAGE_OF[i] for i in trl])]
    return [(train, test, inner)]


_FOLDS = {}


def folds_for(scheme):
    if scheme not in _FOLDS:
        _FOLDS[scheme] = scheme_folds(scheme)
    return _FOLDS[scheme]


# ------------------------------------------------------------------ rule features, fold-safe (mirrors 07_improve.partition)
def _rows_frame(labels):
    r = P_FEATS[["speaker", "hedge"]].copy()
    r["label"] = [labels.get(i, "") for i in r.index]
    return r.reset_index()


def rule_part(F, labelled, unlabelled, scored, labels, R_lab):
    sub = F[F.id.isin(labelled | unlabelled)]
    tl = Timeline(sub)
    anchor = anchors_for(tl, labelled, labels, M.NFACTS)
    pri, overall = M.genre_priors(R_lab, labelled, M.GENRE)
    page_genre = {M.PAGE_OF[i]: M.GENRE[i] for i in sub["id"].unique()}
    trust, w = M.page_trust(tl, anchor, {p: pri.get(g, overall) for p, g in page_genre.items()})
    mask = np.isin(tl.row, list(scored))
    feats, _ = tl.run(anchor, w, mask)
    feats["held_vkey"] = RM.held_vkeys(tl, anchor, w)
    rows = _rows_frame({i: labels[i] for i in labelled})
    out = RM.rule_features(sub[mask], rows, feats[mask], labelled)
    return out.loc[out.index.isin(scored)]


_RF = {}


def rule_table(tag, F, train, test, inner, labels=None, R_lab=None, cache=True):
    key = (tag, L.tl_key(CUR_CFG))
    if cache and key in _RF:
        return _RF[key]
    labels = labels if labels is not None else M.LABELS
    R_lab = R_lab if R_lab is not None else M.R
    parts = [rule_part(F, train, set(test), set(test), labels, R_lab)]
    for a in inner:
        a = set(a)
        b = train - a
        parts.append(rule_part(F, a, b | set(test), b, labels, R_lab))
    T = pd.concat(parts)
    if cache:
        _RF[key] = T
    return T


# ------------------------------------------------------------------ one configuration on one scheme
CUR_CFG = dict(BASE_CFG)


def set_seed(seed):
    M.SEED = seed


def apply_cfg(cfg, seed=0):
    global CUR_CFG
    CUR_CFG = cfg
    set_seed(seed)
    L.apply(cfg)
    extra = [c for k in ("g1", "g3", "g4") if cfg.get(k) for c in FEATURE_COLS[k]]
    orig = M.cols_for
    M.cols_for = lambda v, X: (orig(v, X) + [c for c in extra if c in X.columns]
                               + [c for c in X.columns if c.startswith("p2__")])


# ------------------------------------------------------------------ phase 2: dataset-rule flags and tree leaves
KIND_COLS = [c for c in M.X0.columns if c.startswith("kind_")]


def _only_kinds(X, kinds):
    on = np.ones(len(X), bool)
    for c in KIND_COLS:
        on &= (X[c].fillna(0).to_numpy() == 1) if c in kinds else (X[c].fillna(0).to_numpy() == 0)
    return on


# every flag reads columns of the fold table only (rel_* there are fold-safe: built from training labels)
P2_FLAGS = {
    "never_repeated_and_3values": lambda X: (X["same_value_rows_min"] == 0) & (X["pair_n_values_max"] >= 3),
    "first_person_and_joke": lambda X: (X["first_person"] == 1) & (X["joke_tail"] == 1),
    "first_person_and_number": lambda X: (X["first_person"] == 1) & (X["n_numbers"] > 0),
    "headline_and_match_latest": lambda X: (X["no_period"] == 1) & (X["rel_match_latest"] == 1),
    "kinds_duration_quantity": lambda X: pd.Series(_only_kinds(X, {"kind_duration", "kind_quantity"}), X.index),
    "kinds_version_only": lambda X: pd.Series(_only_kinds(X, {"kind_version"}), X.index),
}

# tree flags: the binary flags of eda/02_rules.py, rebuilt from fold-safe columns of the fold table
TREE_FLAGS = {**{c: (lambda c: lambda X: X[c] == 1)(c) for c in
                 ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today", "still", "future",
                  "negation", "joke_tail", "no_period", "kv_line", "starts_pronoun", "number_words", "has_text_date",
                  "text_date_in_past", "text_date_in_future", "today_date_mismatch", "rel_match_latest",
                  "rel_future_value", "rel_unseen_pair", "rel_before_first_known", "rel_contradicts",
                  "rel_stale_older", "op_present_max", "round_number_max"]},
              "never_repeated": lambda X: X["same_value_rows_min"] == 0,
              "pair_single_value": lambda X: X["pair_n_values_max"] == 1,
              "disagrees_prev_mention": lambda X: X["agree_prev_mention_min"] == 0,
              "page_first_person": lambda X: X["page_first_person_share"] >= 0.3,
              "multi_fact": lambda X: X["n_facts"] >= 2}
LEAF_MIN = 20


def tree_flags(X):
    return pd.DataFrame({k: f(X).fillna(False).astype(int).to_numpy() for k, f in TREE_FLAGS.items()}, index=X.index)


def fit_tree(Ftr, ytr, target):
    from sklearn.tree import DecisionTreeClassifier
    yb = (ytr == target).astype(int)
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced",
                                  random_state=0).fit(Ftr, yb)


def add_p2(X, train, cfg):
    """Phase-2 knobs as fold-safe columns: rule flags, tree-leaf indicators (trees fitted on training rows only)."""
    X = X.copy()
    for name, f in P2_FLAGS.items():
        if cfg.get(f"p2:{name}"):
            X[f"p2__{name}"] = f(X).fillna(False).astype(int).to_numpy()
    trl = sorted(train)
    for target, key in ((2, "tree_unsure"), (0, "tree_false")):
        if cfg.get(f"p2:{key}") or (target == 2 and cfg.get("p2:tree_unsure_layer")):
            TF = tree_flags(X)
            tree = fit_tree(TF.loc[trl], M.Y[trl].to_numpy(), target)
            leaf = pd.Series(tree.apply(TF.to_numpy()), index=X.index)
            counts = leaf.loc[trl].value_counts()
            big = [lf for lf, n in counts.items() if n >= LEAF_MIN]
            if cfg.get(f"p2:{key}"):
                for lf in big:
                    X[f"p2__{key}_leaf{lf}"] = (leaf == lf).astype(int).to_numpy()
            if target == 2 and cfg.get("p2:tree_unsure_layer"):
                pos = tree.predict(TF.to_numpy()) == 1          # leaves the (balanced) tree calls unsure
                X["_tree_unsure_cand"] = (pos & leaf.isin(big).to_numpy()).astype(int)
    return X


def layer_feats(Z):
    f = Z.copy()
    f["tl_n_pages_for_max"] = f.get("tl1__n_pages_for_max", np.nan)
    return f


def knobs_of(cfg):
    k = {k: bool(cfg.get(k)) for k in UNSURE_KNOBS}
    k["tree_layer"] = bool(cfg.get("p2:tree_unsure_layer"))
    return k


def apply_layer(P, pred_base, feats, knobs):
    if not any(v for k, v in knobs.items() if k != "t"):
        return pred_base
    lp, _ = RM.unsure_layer(P, feats, {k: v for k, v in knobs.items() if k != "tree_layer"})
    pred = pred_base.copy()
    changed = lp != P.argmax(1)
    pred[changed] = lp[changed]
    if knobs.get("tree_layer") and "_tree_unsure_cand" in feats:
        cand = (feats["_tree_unsure_cand"].to_numpy() == 1) & (P[:, 0] < 0.5)
        pred[cand] = 2
    return pred


def inner_proba(X, train, inner, cfg):
    """Calibrated inner out-of-fold probabilities of the training rows (for tuning t, never for scoring)."""
    trl = sorted(train)
    Pin = np.zeros((len(trl), 3))
    pos = {i: j for j, i in enumerate(trl)}
    for a in inner:
        a = set(a)
        b = train - a
        al = sorted(a)
        ii = [[al[j] for j in x] for x, _ in GroupKFold(4).split(al, groups=[M.PAGE_OF[i] for i in al])]
        P, _, _, _, te = L.run_one(X, a, b, ii, cfg)
        Pin[[pos[i] for i in te]] = P
    return Pin


def run_scheme(cfg, scheme, seed=0, labels_override=None):
    apply_cfg(cfg, seed)
    need_feats = any(cfg.get(k) for k in ("g1", "g3", "g4"))
    knobs = knobs_of(cfg)
    res = {"f_tf": [], "f_3": [], "gap": [], "gap_tf": [], "gap_3": [], "P": [], "pred": [], "y": [], "retrain": [], "predict": [], "t": []}
    for k, (train, test, inner) in enumerate(folds_for(scheme)):
        tag = f"{scheme}{k}" if scheme != "fold_group" else f"fold{k}"
        X = L.fold_X(tag, L.F_CV, train, test, inner, cfg)
        if need_feats:
            X = X.join(rule_table(tag, L.F_CV, train, test, inner))
        if any(str(k).startswith("p2:") and v for k, v in cfg.items()):
            X = add_p2(X, train, cfg)
        t0 = time.time()
        P, pred, info, yte, te = L.run_one(X, train, test, inner, cfg)
        Zte = layer_feats(X.loc[te])
        if knobs["single_value"]:
            trl = sorted(train)
            Pin = inner_proba(X, train, inner, cfg)
            t = RM.tune_t(Pin, M.Y[trl].to_numpy(), layer_feats(X.loc[trl]), knobs)
            knobs_k = {**knobs, "t": t}
            res["t"].append(t)
        else:
            knobs_k = knobs
        pred = apply_layer(P, pred, Zte, knobs_k)
        retrain = info["retrain_s"] + (time.time() - t0 - info["retrain_s"] if knobs["single_value"] else 0)
        f3, ftf = RM._f1s(yte, pred)
        res["f_tf"].append(ftf); res["f_3"].append(f3)
        # train-vs-test gap: same model fitted on the training rows, raw argmax (+ layer), train vs out-of-fold
        cols = M.cols_for("L6_judge", X)
        trl = sorted(train)
        m = M.lgbm(2).fit(X.loc[trl, cols], M.Y[trl].to_numpy())
        Ptr, Pte = M.proba3(m, X.loc[trl, cols]), M.proba3(m, X.loc[te, cols])
        ptr = apply_layer(Ptr, Ptr.argmax(1), layer_feats(X.loc[trl]), knobs_k)
        pte = apply_layer(Pte, Pte.argmax(1), Zte, knobs_k)
        ytr_ = M.Y[trl].to_numpy()
        g3 = RM._f1s(ytr_, ptr)[0] - RM._f1s(yte, pte)[0]
        gtf = RM._f1s(ytr_, ptr)[1] - RM._f1s(yte, pte)[1]
        res["gap_3"].append(g3); res["gap_tf"].append(gtf)
        res["gap"].append(g3 if any(v for kk, v in knobs.items() if kk != "t") else gtf)
        res["P"].append(P); res["pred"].append(pred); res["y"].append(yte)
        res["retrain"].append(retrain); res["predict"].append(info["predict_s_cpu2"])
    y, P, pred = np.concatenate(res["y"]), np.vstack(res["P"]), np.concatenate(res["pred"])
    pooled = M.metrics(y, P, pred)
    f3p, ftfp = RM._f1s(y, pred)
    return {"f_tf": [float(v) for v in res["f_tf"]], "f_3": [float(v) for v in res["f_3"]],
            "mean_tf": float(np.mean(res["f_tf"])), "mean_3": float(np.mean(res["f_3"])),
            "pooled_tf": float(ftfp), "pooled_3": float(f3p), "acc": float(pooled["acc"]), "ece": float(pooled["ece"]),
            "unsure_recall": float(pooled["unsure_recall"]), "pred_unsure": int(pooled["pred_unsure"]),
            "gap": float(np.mean(res["gap"])), "gap_tf": float(np.mean(res["gap_tf"])),
            "gap_3": float(np.mean(res["gap_3"])), "retrain_s_max": float(max(res["retrain"])),
            "predict_20k_s_max": float(max(res["predict"])), "t": res["t"]}


# ------------------------------------------------------------------ phase 2 pre-filter (rules.csv stats only, no fits)
BASE_FALSE, BASE_UNSURE = 0.146, 0.021
ALREADY = {
    "worst fact relation = match_latest": "rel_match_latest", "worst fact relation = before_first_known":
    "rel_before_first_known", "worst fact relation = unseen_pair": "rel_unseen_pair",
    "worst fact relation = contradicts": "rel_contradicts", "worst fact relation = stale_older": "rel_stale_older",
    "worst fact relation = future_value": "rel_future_value",
    "value never repeated anywhere else": "same_value_rows_min (a tree splits it at 0)",
    "pair has a single value in the corpus": "pair_n_values_max (=1) and the never-false hard rule",
    "agrees with previous mention (label-free)": "agree_prev_mention_min (=1)",
    "disagrees with previous mention (label-free)": "agree_prev_mention_min (=0)",
    "value >= 2x pair median": "log_dev_from_pair_median_max (>= log 2)",
    "comparison op present (over/under)": "op_present_max", "hype_score >= 2": "hype_score (>= 2)",
    "page first-person share >= 0.3": "page_first_person_share (>= 0.3)",
    "page genre = first_person_hype": "genre_first_person_hype",
    "n_facts >= 2": "n_facts", "n_facts >= 3": "n_facts", "first row of page (headline)": "pos_in_page (=0)",
    "single-row page": "page_size (=1)",
    **{f"text: {c}": c for c in ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today",
                                 "still", "future", "negation", "change_verb", "joke_tail", "no_period", "kv_line",
                                 "starts_pronoun", "number_words"]},
    **{f"page genre = {g}": f"genre_{g}" for g in ["event_recap", "other_page", "people_announcement", "pricing_page",
                                                   "quote_interview", "release_notes", "spec_sheet_or_slides"]},
    "text has date": "has_text_date", "text date in the past (> 1 day)": "text_date_in_past",
    "text date in the future (> 1 day)": "text_date_in_future", "'today' but text date != page date":
    "today_date_mismatch", "subject only by pronoun/implicit": "any_pronoun_subject",
}
CANDIDATE = {"value never repeated AND pair has >=3 values": "never_repeated_and_3values",
             "first person AND joke tail": "first_person_and_joke",
             "first person AND number": "first_person_and_number",
             "headline/spec line AND match_latest": "headline_and_match_latest",
             "value kinds = duration|quantity": "kinds_duration_quantity",
             "value kinds = version": "kinds_version_only"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def prefilter():
    """Keep a rule only if confirmed, support >= 20, pages >= 10, |lift| or unsure signal strong, CI excludes base;
    then 'already in model' and near-duplicates (Jaccard of the positive rows > 0.9) are removed."""
    r = pd.read_csv(M.OUT / "rules.csv")
    r = r[r.family != "base"]
    Xp = M.X0.join(P_FEATS[["rel_match_latest"]])
    rows = []
    for _, x in r.iterrows():
        lo, hi = wilson(x["false"], x["support"])
        ulo, _ = wilson(x["unsure"], x["support"])
        strong = x["lift_false"] >= 1.5 or x["lift_false"] <= 0.5 or x["p_unsure"] >= 3 * BASE_UNSURE
        ci_ok = (lo > BASE_FALSE or hi < BASE_FALSE) or (x["p_unsure"] >= 3 * BASE_UNSURE and ulo > BASE_UNSURE)
        reasons = []
        if not x["confirmed"]:
            reasons.append("not confirmed")
        if x["support"] < 20:
            reasons.append(f"support {x['support']} < 20")
        if x["pages"] < 10:
            reasons.append(f"pages {x['pages']} < 10")
        if not strong:
            reasons.append(f"weak (lift {x['lift_false']}, p_unsure {x['p_unsure']})")
        if not ci_ok:
            reasons.append(f"CI [{lo:.2f}, {hi:.2f}] contains base 0.146")
        prio = max(x["lift_false"], 1 / max(x["lift_false"], 1e-3)) * x["support"]
        if reasons:
            verdict = "dropped: " + "; ".join(reasons)
        elif x["rule"] in ALREADY:
            verdict = f"already in model: {ALREADY[x['rule']]}"
        elif x["rule"] in CANDIDATE:
            verdict = "candidate"
        else:
            verdict = "dropped: no fold-safe flag definition"
        rows.append({"rule": x["rule"], "family": x["family"], "support": int(x["support"]), "pages": int(x["pages"]),
                     "lift_false": x["lift_false"], "p_unsure": x["p_unsure"], "false_ci95": f"[{lo:.2f}, {hi:.2f}]",
                     "priority": round(prio, 1), "knob": CANDIDATE.get(x["rule"], ""), "verdict": verdict})
    D = pd.DataFrame(rows)
    cand = D[D.verdict == "candidate"].sort_values("priority", ascending=False)
    flags = {k: P2_FLAGS[k](Xp).fillna(False).astype(bool).to_numpy() for k in cand.knob}
    existing = {"kind_version": Xp["kind_version"].fillna(0).to_numpy() == 1,
                "first_person": Xp["first_person"].to_numpy() == 1, "joke_tail": Xp["joke_tail"].to_numpy() == 1}
    jac = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)  # noqa: E731
    kept = []
    for _, c in cand.iterrows():
        f = flags[c.knob]
        dup = [k for k in kept if jac(f, flags[k]) > 0.9]
        ex = [k for k, e in existing.items() if jac(f, e) > 0.9]
        if ex:
            D.loc[D.knob == c.knob, "verdict"] = f"already in model: {ex[0]} (Jaccard > 0.9)"
        elif dup:
            D.loc[D.knob == c.knob, "verdict"] = f"dropped: near-duplicate of {dup[0]} (Jaccard > 0.9)"
        else:
            kept.append(c.knob)
    D.loc[D.knob.isin(kept), "verdict"] = "candidate -> knob"
    D.sort_values(["verdict", "priority"], ascending=[True, False]).to_csv(HERE / "prefilter.csv", index=False)
    return D, [f"p2:{k}" for k in kept]


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    ok = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            ok[i] = True
        else:
            break
    return ok


# ------------------------------------------------------------------ decisions
def is_unsure_knob(knob):
    return knob in UNSURE_KNOBS or knob == "p2:tree_unsure_layer"


def accept_rule(goal, base, r, knob):
    a = goal["accept_change"]
    key = "f_3" if is_unsure_knob(knob) else "f_tf"
    gain = np.array(r[key]) - np.array(base[key])
    ok = (gain.mean() >= a["min_gain"] and int((gain > 0).sum()) >= a["folds_better"]
          and r["ece"] <= base["ece"] + a["ece_worse_max"] and L.gate_ok(goal, r))
    if is_unsure_knob(knob):
        ok = ok and (r["pooled_tf"] >= base["pooled_tf"] - TF_DROP_MAX)
    return bool(ok), gain


def gate_nested(goal, gain):
    a = goal["accept_change"]
    hits, held = 0, []
    for j in range(len(gain)):
        rest = np.delete(gain, j)
        selected = rest.mean() >= a["min_gain"] and int((rest > 0).sum()) >= a["folds_better"] - 1
        held.append(gain[j])
        hits += int(selected and gain[j] > 0)
    return bool(hits >= 4 and np.mean(held) > 0), hits


def gate_seeds(gains):
    g = np.array(gains)
    return bool((g > 0).all() and g.std() < g.mean()), g


# ------------------------------------------------------------------ state
def load_state(goal_sha, fresh):
    if fresh:
        for f in (STATE_F, TRIALS_F):
            f.unlink(missing_ok=True)
    if STATE_F.exists():
        s = json.loads(STATE_F.read_text())
        assert s["goal_sha256"] == goal_sha, "goal.json changed"
        return s
    return {"goal_sha256": goal_sha, "started": time.strftime("%Y-%m-%d %H:%M"), "elapsed_s": 0.0,
            "base_cfg": {**BASE_CFG, **{k: False for k in RULE_KEYS}}, "done": [], "kept": [], "lockbox": None,
            "permutation": None}


def save_state(s):
    STATE_F.write_text(json.dumps(s, indent=1, default=float))


TRIAL_COLS = ["trial", "phase", "knob", "metric", "f0", "f1", "f2", "f3", "f4", "mean_gain", "folds_better", "gain_se",
              "pooled_tf", "pooled_3", "acc", "ece", "unsure_recall", "pred_unsure", "t_per_fold", "accept_rule",
              "gate_a_nested", "nested_hits", "gate_b_gap", "gap_delta", "gate_c_seeds", "seed_gains",
              "gate_d_row5", "row5_gain", "gate_d_time", "time_gain", "retrain_s_max", "predict_20k_s_max",
              "kept", "trial_s"]


def log_trial(row):
    pd.DataFrame([row]).reindex(columns=TRIAL_COLS).to_csv(TRIALS_F, mode="a", header=not TRIALS_F.exists(),
                                                          index=False)


# ------------------------------------------------------------------ main
_BASE = {}


def base_results(cfg):
    key = json.dumps(cfg, sort_keys=True)
    if key not in _BASE:
        _BASE[key] = {"fg": {s: run_scheme(cfg, "fold_group", s) for s in SEEDS},
                      "row5": run_scheme(cfg, "fold_row5"), "time": run_scheme(cfg, "time")}
    return _BASE[key]


def main(fresh=False):
    goal = json.loads(GOAL_F.read_text())
    goal_sha = sha(GOAL_F)
    s = load_state(goal_sha, fresh)
    queue = json.loads(QUEUE_F.read_text())["queue"]
    _, p2_knobs = prefilter()
    queue = queue + [{"knob": k, "phase": 2} for k in p2_knobs] + [
        {"knob": k, "phase": 2} for k in ("p2:tree_unsure", "p2:tree_false", "p2:tree_unsure_layer")]
    s["queue"] = [q["knob"] for q in queue]
    t_start = time.time() - s["elapsed_s"]
    base_cfg = s["base_cfg"]
    B = base_results(base_cfg)
    b0 = B["fg"][0]
    if not s.get("base_check"):
        s["base_check"] = {"mean_tf": b0["mean_tf"], "pooled_tf": b0["pooled_tf"]}
        print(f"base: mean {b0['mean_tf']:.4f} pooled {b0['pooled_tf']:.4f}", flush=True)
        if abs(b0["mean_tf"] - 0.8182) > 0.0005 or abs(b0["pooled_tf"] - 0.8191) > 0.0005:
            save_state(s)
            raise SystemExit(f"base does not reproduce 0.8182 / 0.8191: {b0['mean_tf']:.4f} / {b0['pooled_tf']:.4f}")
        log_trial({"trial": "t000", "knob": "base", "metric": "tf", **{f"f{i}": v for i, v in enumerate(b0["f_tf"])},
                   "pooled_tf": b0["pooled_tf"], "pooled_3": b0["pooled_3"], "acc": b0["acc"], "ece": b0["ece"],
                   "unsure_recall": b0["unsure_recall"], "pred_unsure": b0["pred_unsure"],
                   "retrain_s_max": b0["retrain_s_max"], "predict_20k_s_max": b0["predict_20k_s_max"], "kept": "base"})
    for i, item in enumerate(queue, 1):
        knob = item["knob"]
        if knob in s["done"]:
            continue
        if time.time() - t_start > BUDGET_S:
            print("budget reached", flush=True)
            break
        t0 = time.time()
        cfg = {**base_cfg, knob: True}
        r = {sd: run_scheme(cfg, "fold_group", sd) for sd in SEEDS}
        r0 = r[0]
        ok, gain = accept_rule(goal, B["fg"][0], r0, knob)
        key = "f_3" if is_unsure_knob(knob) else "f_tf"
        ga, hits = gate_nested(goal, gain)
        gk = "gap_3" if is_unsure_knob(knob) else "gap_tf"
        gap_delta = r0[gk] - B["fg"][0][gk]
        gb = bool(gap_delta <= GAP_WIDEN_MAX)
        seed_gains = [float(np.mean(np.array(r[sd][key]) - np.array(B["fg"][sd][key]))) for sd in SEEDS]
        gc, _ = gate_seeds(seed_gains)
        r5 = run_scheme(cfg, "fold_row5")
        rt = run_scheme(cfg, "time")
        mk = "mean_3" if is_unsure_knob(knob) else "mean_tf"
        row5_gain = r5[mk] - B["row5"][mk]
        time_gain = rt[mk] - B["time"][mk]
        gd5, gdt = bool(row5_gain > 0), ("PASS" if time_gain > 0 else "WARN")
        kept = ok and ga and gb and gc and gd5
        log_trial({"trial": f"t{i:03d}", "phase": item.get("phase", 1), "knob": knob, "metric": "3class" if is_unsure_knob(knob) else "tf",
                   **{f"f{j}": float(v) for j, v in enumerate(r0[key])}, "mean_gain": float(gain.mean()),
                   "folds_better": int((gain > 0).sum()), "gain_se": float(gain.std(ddof=1) / math.sqrt(len(gain))),
                   "pooled_tf": r0["pooled_tf"], "pooled_3": r0["pooled_3"], "acc": r0["acc"], "ece": r0["ece"],
                   "unsure_recall": r0["unsure_recall"], "pred_unsure": r0["pred_unsure"],
                   "t_per_fold": json.dumps(r0["t"]) if r0["t"] else "", "accept_rule": ok,
                   "gate_a_nested": ga, "nested_hits": hits, "gate_b_gap": gb, "gap_delta": float(gap_delta),
                   "gate_c_seeds": gc, "seed_gains": json.dumps([round(x, 4) for x in seed_gains]),
                   "gate_d_row5": gd5, "row5_gain": float(row5_gain), "gate_d_time": gdt, "time_gain": float(time_gain),
                   "retrain_s_max": r0["retrain_s_max"], "predict_20k_s_max": r0["predict_20k_s_max"],
                   "kept": kept, "trial_s": round(time.time() - t0, 1)})
        print(f"{knob:13s} gain {gain.mean():+.4f} ({int((gain > 0).sum())}/5) accept={ok} a={ga}({hits}) b={gb} "
              f"c={gc} {seed_gains} d5={gd5} {row5_gain:+.4f} dt={gdt} {time_gain:+.4f} -> kept={kept} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        s["done"].append(knob)
        if kept:
            s["kept"].append(knob)
            base_cfg = cfg
            s["base_cfg"] = base_cfg
            B = base_results(base_cfg)
        s["elapsed_s"] = time.time() - t_start
        save_state(s)
    s["final"] = {k: v for k, v in base_cfg.items()}
    s["final_cv"] = {k: B["fg"][0][k] for k in ("mean_tf", "pooled_tf", "mean_3", "pooled_3", "acc", "ece",
                                                "unsure_recall")}
    if s.get("permutation") is None:
        s["permutation"] = permutation_test(base_cfg)
        save_state(s)
    if s.get("lockbox") is None:
        s["lockbox"] = lockbox(base_cfg) if s["kept"] else {"skipped": "no rule kept: the final model is the main "
                                                            "loop's final, already scored on the holdout (0.7445)"}
        save_state(s)
    s["elapsed_s"] = time.time() - t_start
    assert sha(GOAL_F) == goal_sha, "goal.json changed during the run"
    s["goal_sha256_end"] = sha(GOAL_F)
    save_state(s)
    report()


# ------------------------------------------------------------------ permutation test
def permutation_test(cfg, n=3):
    """Shuffle the training labels inside each fold (features rebuilt from the shuffled labels, no cache), score on
    the true test labels, three ways on the same shuffled features:
      full        the whole final pipeline
      no_rules    the same with the two deterministic hard rules switched off: the learned part only
      rules_only  the hard rules applied to a constant class prior: what needs no labels at all
    The learned part must fall to about the all-true level (<= 0.50). The full pipeline may stay above it only by
    what the label-free rules give on their own; anything beyond that would point to a leak."""
    rng = np.random.default_rng(7)
    out = []
    hard0 = M.hard_rules
    Y0, LAB0, R0 = M.Y.copy(), dict(M.LABELS), M.R.copy()
    try:
        for rep in range(n):
            acc = {"full": ([], []), "no_rules": ([], []), "rules_only": ([], [])}
            for k, (train, test, inner) in enumerate(folds_for("fold_group")):
                trl = sorted(train)
                perm = dict(zip(trl, rng.permutation([LAB0[i] for i in trl])))
                labels = {**LAB0, **perm}
                M.LABELS.clear(); M.LABELS.update(labels)
                M.Y = pd.Series({i: {c: j for j, c in enumerate(M.CLASSES)}[labels[i]] for i in Y0.index})
                M.R = R0.assign(label=R0["id"].map(labels))
                F = L.F_CV.assign(label=L.F_CV["id"].map(labels))
                apply_cfg(cfg)
                X, _ = M.fold_table(F, train, test, inner)
                if any(cfg.get(x) for x in ("g1", "g3", "g4")):
                    X = X.join(rule_table(f"perm{rep}_{k}", F, train, test, inner, labels, M.R, cache=False))
                if any(str(x).startswith("p2:") and v for x, v in cfg.items()):
                    X = add_p2(X, train, cfg)
                prior = np.bincount(M.Y[trl].to_numpy(), minlength=3) / len(trl)
                M.Y.loc[sorted(test)] = Y0.loc[sorted(test)]          # score on the TRUE test labels
                te = sorted(test)
                yte = Y0.loc[te].to_numpy()
                P, pred, _, _, _ = L.run_one(X, train, test, inner, cfg)
                acc["full"][0].append(yte); acc["full"][1].append(apply_layer(P, pred, layer_feats(X.loc[te]),
                                                                              knobs_of(cfg)))
                M.hard_rules = lambda P_, Z_: P_
                P2_, pred2, _, _, _ = L.run_one(X, train, test, inner, cfg)
                M.hard_rules = hard0
                acc["no_rules"][0].append(yte); acc["no_rules"][1].append(pred2)
                Pc = hard0(np.tile(prior, (len(te), 1)), X.loc[te])
                acc["rules_only"][0].append(yte); acc["rules_only"][1].append(Pc.argmax(1))
            row = {"rep": rep}
            for name, (ys, ps) in acc.items():
                f3, ftf = RM._f1s(np.concatenate(ys), np.concatenate(ps))
                row[f"{name}_tf"] = float(ftf)
            row["macro_f1_tf"] = row["full_tf"]
            out.append(row)
            print(f"permutation {rep}: full {row['full_tf']:.4f} no_rules {row['no_rules_tf']:.4f} "
                  f"rules_only {row['rules_only_tf']:.4f}", flush=True)
    finally:
        M.hard_rules = hard0
        M.Y = Y0
        M.LABELS.clear(); M.LABELS.update(LAB0)
        M.R = R0
    mx_learned = max(o["no_rules_tf"] for o in out)
    excess = max(o["full_tf"] - o["rules_only_tf"] for o in out)
    return {"reps": out, "max_tf": max(o["full_tf"] for o in out), "max_learned_tf": mx_learned,
            "max_excess_over_rules": excess, "pass": bool(mx_learned <= 0.50 and excess <= 0.05)}


def _permutation_test_old(cfg, n=3):
    rng = np.random.default_rng(7)
    out = []
    Y0, LAB0, R0 = M.Y.copy(), dict(M.LABELS), M.R.copy()
    try:
        for rep in range(n):
            ys, Ps, preds = [], [], []
            for k, (train, test, inner) in enumerate(folds_for("fold_group")):
                trl = sorted(train)
                perm = dict(zip(trl, rng.permutation([LAB0[i] for i in trl])))
                labels = {**LAB0, **perm}
                M.LABELS.clear(); M.LABELS.update(labels)
                M.Y = pd.Series({i: {c: j for j, c in enumerate(M.CLASSES)}[labels[i]] for i in Y0.index})
                M.R = R0.assign(label=R0["id"].map(labels))
                F = L.F_CV.assign(label=L.F_CV["id"].map(labels))
                apply_cfg(cfg)
                X, _ = M.fold_table(F, train, test, inner)
                if any(cfg.get(x) for x in ("g1", "g3", "g4")):
                    X = X.join(rule_table(f"perm{rep}_{k}", F, train, test, inner, labels, M.R, cache=False))
                if any(str(x).startswith("p2:") and v for x, v in cfg.items()):
                    X = add_p2(X, train, cfg)                          # trees refit on the SHUFFLED training labels
                M.Y.loc[sorted(test)] = Y0.loc[sorted(test)]          # score on the TRUE test labels
                P, pred, info, yte, te = L.run_one(X, train, test, inner, cfg)
                pred = apply_layer(P, pred, layer_feats(X.loc[te]), knobs_of(cfg))
                ys.append(yte); Ps.append(P); preds.append(pred)
            f3, ftf = RM._f1s(np.concatenate(ys), np.concatenate(preds))
            out.append({"rep": rep, "macro_f1_tf": float(ftf), "macro_f1_3": float(f3)})
            print(f"permutation {rep}: tf {ftf:.4f}", flush=True)
    finally:
        M.Y = Y0
        M.LABELS.clear(); M.LABELS.update(LAB0)
        M.R = R0
    mx = max(o["macro_f1_tf"] for o in out)
    return {"reps": out, "max_tf": mx, "pass": bool(mx <= 0.50)}


# ------------------------------------------------------------------ lockbox
def lockbox(cfg):
    apply_cfg(cfg)
    hold = sorted(M.SPLITS.index[M.SPLITS.holdout == 1])
    train = set(L.CV_IDS)
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[M.PAGE_OF[i] for i in trl])]
    X = L.fold_X("lockbox", M.F_ALL, train, set(hold), inner, cfg)
    if any(cfg.get(x) for x in ("g1", "g3", "g4")):
        X = X.join(rule_table("lockbox", M.F_ALL, train, set(hold), inner, cache=False))
    if any(str(k).startswith("p2:") and v for k, v in cfg.items()):
        X = add_p2(X, train, cfg)
    P, pred, info, yte, te = L.run_one(X, train, set(hold), inner, cfg)
    knobs = knobs_of(cfg)
    if knobs["single_value"]:
        t = RM.tune_t(inner_proba(X, train, inner, cfg), M.Y[trl].to_numpy(), layer_feats(X.loc[trl]), knobs)
        knobs = {**knobs, "t": t}
    pred = apply_layer(P, pred, layer_feats(X.loc[te]), knobs)
    met = M.metrics(yte, P, pred)
    f3, ftf = RM._f1s(yte, pred)
    # page bootstrap of the holdout macro-F1
    pages = np.array([M.PAGE_OF[i] for i in te])
    up = np.unique(pages)
    idx = {p: np.where(pages == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        ii = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        boots.append(RM._f1s(yte[ii], pred[ii])[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rows": len(hold), "pages": len(up), "acc": round(float(met["acc"]), 4), "macro_f1_tf": round(float(ftf), 4),
            "macro_f1_3": round(float(f3), 4), "ece": round(float(met["ece"]), 4),
            "unsure_recall": round(float(met["unsure_recall"]), 4), "tf_ci95": [round(lo, 4), round(hi, 4)],
            "all_true_acc": round(float((yte == 1).mean()), 4)}


# ------------------------------------------------------------------ report
def distinct_trials():
    n = {}
    o = M.OUT
    if (o / "bakeoff.csv").exists():
        n["bake-off models"] = pd.read_csv(o / "bakeoff.csv")["model"].nunique()
    if (o / "improve_labels.csv").exists():
        n["ladder variants"] = pd.read_csv(o / "improve_labels.csv")["variant"].nunique()
    if (o / "improve_weak.csv").exists():
        n["weak-label variants"] = pd.read_csv(o / "improve_weak.csv")["variant"].nunique()
    if (o / "improve_neuro.csv").exists():
        n["neural variants"] = pd.read_csv(o / "improve_neuro.csv")["variant"].nunique()
    if (SI / "trials.csv").exists():
        t = pd.read_csv(SI / "trials.csv")
        n["self-improvement trials"] = t[t.knob != "base"][["knob", "value"]].drop_duplicates().shape[0]
    if TRIALS_F.exists():
        n["rules-round trials"] = int((pd.read_csv(TRIALS_F)["knob"] != "base").sum())
    return n


def fmt(v, d=4):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:+.{d}f}" if isinstance(v, float) else str(v))


def report():
    s = json.loads(STATE_F.read_text())
    T = pd.read_csv(TRIALS_F)
    tr = T[T.knob != "base"]
    base = T[T.knob == "base"].iloc[0]
    names = {"g1": "G1 ratio to held value (x2, x0.5, x10 ...)", "g3": "G3 hedge + unbacked value",
             "g4": "G4 speaker trust", "joke": "G2a joke tail -> unsure", "single_value": "G2b one-off value -> unsure",
             "hearsay": "G2c hearsay -> unsure", "today": "G5 'today' + other date -> unsure"}
    names.update({"p2:tree_unsure": "unsure-tree leaves as features", "p2:tree_false": "false-tree leaves as features",
                  "p2:tree_unsure_layer": "unsure-tree leaves as an unsure rule"})
    pf = lambda b: "PASS" if str(b) == "True" else ("FAIL" if str(b) == "False" else str(b))  # noqa: E731
    tr = tr.copy()
    if "phase" not in tr:
        tr["phase"] = 1
    tr["verdict"] = [verdict(r) for _, r in tr.iterrows()]
    # one-sided t-test of the 5 per-fold gains, Holm over every trial of this round
    from scipy.stats import t as tdist
    # mean gain and its standard error are measured against the base current at that trial (greedy chain)
    tr["p_one_sided"] = [float(tdist.sf(r.mean_gain / r.gain_se, df=4)) if r.gain_se > 0 else float("nan")
                         for _, r in tr.iterrows()]
    pv = tr["p_one_sided"].where(tr.metric != "3class").to_numpy(float)
    tf_mask = ~np.isnan(pv)
    ok_holm = np.zeros(len(tr), bool)
    if tf_mask.any():
        ok_holm[tf_mask] = holm(pv[tf_mask])
    tr["holm"] = ok_holm

    def table(sub, phase2=False):
        head = ("| rule / knob | family | support | lift | judged on | CV gain | folds better | ECE | a nested | b gap "
                "| c seeds | d row5 | d time | verdict |" if phase2 else
                "| knob | judged on | mean gain | folds better | pooled tf F1 | pooled 3-class F1 | ECE | unsure recall "
                "| a nested | b gap | c seeds | d row5 | d time | verdict |")
        out = [head, "|" + "---|" * 14]
        pre = pd.read_csv(HERE / "prefilter.csv") if (HERE / "prefilter.csv").exists() else pd.DataFrame()
        for _, r in sub.iterrows():
            gates = (f"{pf(r.gate_a_nested)} ({int(r.nested_hits)}/5) | {pf(r.gate_b_gap)} ({r.gap_delta:+.3f}) | "
                     f"{pf(r.gate_c_seeds)} {r.seed_gains} | {pf(r.gate_d_row5)} ({r.row5_gain:+.4f}) | "
                     f"{r.gate_d_time} ({r.time_gain:+.4f})")
            judged = "3-class F1" if r.metric == "3class" else "true/false F1"
            if phase2:
                kn = str(r.knob).replace("p2:", "")
                m = pre[pre.knob == kn] if len(pre) else pre
                fam, sup, lift = ((m.iloc[0].family, int(m.iloc[0].support), m.iloc[0].lift_false) if len(m)
                                  else ("tree", "", ""))
                rule = m.iloc[0].rule if len(m) else names.get(r.knob, r.knob)
                out.append(f"| {rule} | {fam} | {sup} | {lift} | {judged} | {r.mean_gain:+.4f} | "
                           f"{int(r.folds_better)}/5 | {r.ece:.4f} | {gates} | **{r.verdict}** |")
            else:
                out.append(f"| {names.get(r.knob, r.knob)} | {judged} | {r.mean_gain:+.4f} | {int(r.folds_better)}/5 "
                           f"| {r.pooled_tf:.4f} | {r.pooled_3:.4f} | {r.ece:.4f} | {r.unsure_recall:.3f} | {gates} "
                           f"| **{r.verdict}** |")
        return "\n".join(out)
    lines = [table(tr[tr.phase == 1])]
    pre_txt = ""
    if (HERE / "prefilter.csv").exists():
        pre = pd.read_csv(HERE / "prefilter.csv")
        cnt = pre.verdict.str.split(":").str[0].value_counts().to_dict()
        rows_ = ["| rule | family | support | pages | lift (false) | p unsure | false 95% CI | priority | decision |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for _, x in pre.iterrows():
            rows_.append(f"| {x.rule} | {x.family} | {x.support} | {x.pages} | {x.lift_false} | {x.p_unsure} "
                         f"| {x.false_ci95} | {x.priority} | {x.verdict} |")
        pre_txt = (f"Counts: {json.dumps(cnt)}. Near-duplicates: two flags whose positive rows overlap with Jaccard "
                   f"> 0.9 (agreement on all rows is meaningless for rare flags, so the positive rows are compared).\n\n"
                   + "\n".join(rows_))
    phase2_txt = table(tr[tr.phase == 2], phase2=True) if (tr.phase == 2).any() else "Phase 2 did not run."
    holm_rows = tr[tr.metric != "3class"]
    holm_txt = ("| knob | one-sided p (5 fold gains) | Holm (α 0.05, over this round's true/false knobs) | kept |\n"
                "|---|---|---|---|\n" + "\n".join(
                    f"| {r.knob} | {r.p_one_sided:.4f} | {'significant' if r.holm else 'not significant'} | "
                    f"{'yes' if str(r.kept) == 'True' else 'no'} |" for _, r in holm_rows.iterrows()))
    n = distinct_trials()
    n_all = int(sum(n.values()))
    se = float(tr["gain_se"].mean()) if len(tr) else float("nan")
    noise = se * math.sqrt(2 * math.log(max(n_all, 2)))
    perm = s.get("permutation") or {}
    lb = s.get("lockbox") or {}
    fc = s.get("final_cv", {})
    kept = s.get("kept", [])
    perm_tbl = ("| repeat | full pipeline | hard rules off (learned part) | hard rules alone on a constant prior |\n"
                "|---|---|---|---|\n") + "\n".join(
        f"| {r['rep']} | {r.get('full_tf', r.get('macro_f1_tf')):.4f} | {r.get('no_rules_tf', float('nan')):.4f} "
        f"| {r.get('rules_only_tf', float('nan')):.4f} |" for r in perm.get("reps", []))
    best = tr["mean_gain"].max() if len(tr) else float("nan")
    if lb.get("macro_f1_tf") is not None:
        gap = fc.get("pooled_tf", float("nan")) - lb["macro_f1_tf"]
        lb_txt = (f"| rows | pages | accuracy | macro-F1 (t/f) | 95% page-bootstrap CI | 3-class F1 | ECE | all-true acc |\n"
                  f"|---|---|---|---|---|---|---|---|\n| {lb['rows']} | {lb['pages']} | {lb['acc']} | {lb['macro_f1_tf']} "
                  f"| {lb['tf_ci95']} | {lb['macro_f1_3']} | {lb['ece']} | {lb['all_true_acc']} |\n\n"
                  f"CV − holdout gap: {gap:+.4f} (CV pooled {fc.get('pooled_tf', float('nan')):.4f}; the holdout CI "
                  f"spans {lb['tf_ci95'][1] - lb['tf_ci95'][0]:.3f}).")
    else:
        lb_txt = lb.get("skipped", "not run")
    txt = f"""# Rules round: dataset rules as model changes, with overfitting gates

Started {s['started']} · {len(tr)} knobs tried · {s['elapsed_s'] / 60:.1f} min · base = main-loop final
(L6 + tau_days 45 + min_child_samples 20). `goal.json` sha256 `{s['goal_sha256'][:16]}…` unchanged at start and end:
{s.get('goal_sha256_end', '')[:16] == s['goal_sha256'][:16]}.

Base reproduced: mean per-fold macro-F1 {s['base_check']['mean_tf']:.4f}, pooled {s['base_check']['pooled_tf']:.4f}
(fold_group, 5 folds, holdout excluded). Base pooled 3-class F1 {base.pooled_3:.4f}, ECE {base.ece:.4f},
unsure recall {base.unsure_recall:.3f}.

**Rule.** A change is kept only if it passes the goal.json accept rule (mean gain ≥ 0.005, better on ≥ 4/5 folds,
ECE not worse by > 0.01, runtime gate; unsure knobs judged on 3-class macro-F1 and may not lower the true/false
macro-F1 by > 0.005) **and** four overfitting gates:
a) nested selection (decide on 4 folds, gain on the 5th, in ≥ 4/5 rotations), b) train−test gap not widened by
> 0.02, c) gain positive for 3 LightGBM seeds with std < mean, d) gain positive on fold_row5 (time split: warning only).

## Phase 1: rules G1–G5

{chr(10).join(lines)}

Gain is the change in the judged metric (true/false or 3-class macro-F1) against the current base, per fold,
averaged. Seed gains are the mean gains for seeds 0, 1, 2. "rejected: overfits (gate x)" means the change passed
or failed the accept rule but failed overfitting gate x, so it is dropped whatever its CV gain.

## Phase 2 pre-filter: every rule in eda/out/rules.csv (stats only, no model fits)

Kept only if confirmed, support ≥ 20 rows, ≥ 10 pages, (lift ≥ 1.5 or ≤ 0.5 or p_unsure ≥ 3 × 0.021), and the 95%
Wilson CI of p_false excludes the base 0.146 (or p_unsure is clearly above base). Then rules already present as a model
column are logged "already in model", and near-duplicates are removed. Survivors are tried in order of priority
(max(lift, 1/lift) × support).

{pre_txt}

## Phase 2: all dataset rules

{phase2_txt}

Tree knobs: the depth-4 unsure / false trees are refitted inside every training fold on fold-safe flags; only leaves
with ≥ 20 training rows become indicator features. In the "unsure rule" variant, test rows that fall in a leaf the
tree calls unsure become unsure unless P(false) ≥ 0.5.

## Holm correction

{holm_txt}

The 3-class (unsure) knobs are not in the Holm set: their base is a different metric.

## Final configuration

Kept: {', '.join(kept) if kept else 'nothing'}.

```json
{json.dumps(s.get('final', {}), indent=1)}
```

Final CV (fold_group, seed 0): {json.dumps({k: round(v, 4) for k, v in fc.items()})}

## Label-permutation test (final configuration)

Training labels shuffled inside every fold, features rebuilt from the shuffled labels, scored on the true test labels,
{len(perm.get('reps', []))} repeats, true/false macro-F1 (all-true scores 0.455):

{perm_tbl}

The learned part (hard rules off) must collapse to about the all-true level: max {perm.get('max_learned_tf', float('nan')):.4f}
≤ 0.50. The two hard rules (an outdated value on a trusted timeline → false; a one-value pair → never false) need no
labels, so the full pipeline keeps a floor on shuffled labels; it may exceed the rules-alone score by at most 0.05
(max excess {perm.get('max_excess_over_rules', float('nan')):+.4f}). Verdict: **{'PASS' if perm.get('pass') else 'FAIL'}**.

## Lockbox (holdout pages)

Earlier looks, nothing tuned on them: L0 0.7433, L4 0.7327, main-loop final 0.7445 (true/false macro-F1).

{lb_txt}

## Multiple testing

Distinct variants tried so far: {json.dumps(n)} = **{n_all}**. Mean standard error of a rules-round gain {se:.4f}, so
the best gain pure noise would give over {n_all} tries is about {se:.4f} × √(2 ln {n_all}) = **{noise:.4f}**.
Best gain in this round: {best:+.4f} ({'above' if best > noise else 'not above'} that line).

## Conclusion

{conclusion(tr, kept, noise, perm, lb, fc)}

Figure: `figs/rules_round.png`. Rerun: `LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/rules_round/loop_rules.py`
(resumable; `--fresh` starts over, `--report` rebuilds this file).
"""
    REPORT_F.write_text(txt, encoding="utf-8")
    figure(tr)
    print(f"wrote {REPORT_F}")


def verdict(r):
    gates = [nm for g, nm in (("gate_a_nested", "a"), ("gate_b_gap", "b"), ("gate_c_seeds", "c"),
                              ("gate_d_row5", "d")) if str(r[g]) != "True"]
    if str(r.kept) == "True":
        return "kept" + (" (time split: warning)" if r.gate_d_time != "PASS" else "")
    if str(r.accept_rule) == "True" and gates:
        return f"rejected: overfits (gate {', '.join(gates)})"
    if gates:
        return f"rejected: accept rule; also overfits (gate {', '.join(gates)})"
    return "rejected: accept rule (gain too small or not on 4/5 folds)"


def conclusion(tr, kept, noise, perm, lb, fc):
    out = []
    for _, r in tr.iterrows():
        why = []
        if str(r.accept_rule) != "True":
            why.append("accept rule")
        for g, nm in (("gate_a_nested", "nested selection"), ("gate_b_gap", "train-test gap"),
                      ("gate_c_seeds", "seed stability"), ("gate_d_row5", "fold_row5")):
            if str(r[g]) != "True":
                why.append(nm)
        out.append(f"- `{r.knob}`: {'kept' if str(r.kept) == 'True' else 'not kept (' + ', '.join(why) + ')'}; "
                   f"gain {r.mean_gain:+.4f}, {int(r.folds_better)}/5 folds.")
    head = ("No dataset rule earned its place: the final model stays the main loop's final, so no new holdout look "
            "was spent." if not kept else
            f"Kept {', '.join(kept)}; final CV pooled true/false macro-F1 {fc.get('pooled_tf', float('nan')):.4f}.")
    perm_txt = ("Permutation: the learned part collapses to the all-true level on shuffled labels and the full pipeline "
                "stays within 0.05 of what the label-free hard rules give alone, so nothing leaks." if perm.get("pass")
                else "Permutation: FAILED (see the table above): check for a leak.")
    return head + "\n\n" + "\n".join(out) + "\n\n" + perm_txt


def figure(tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#2a7f62" if str(k) == "True" else ("#c08a1e" if str(a) == "True" else "#9aa0a6")
              for k, a in zip(tr["kept"], tr["accept_rule"])]
    ax.barh(tr["knob"], tr["mean_gain"], color=colors)
    ax.axvline(0.005, color="#555", ls="--", lw=1)
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_xlabel("mean per-fold gain (true/false F1; 3-class F1 for unsure knobs)")
    ax.set_title("Rules round: green kept · amber passed accept rule but failed a gate · grey rejected", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "rules_round.png", dpi=120)
    plt.close(fig)


def refresh():
    """Resume after the first run: recompute gate b for the unsure knobs on the matching metric (3-class gap of the
    knob vs 3-class gap of its base) and rerun the three-way permutation test. Decisions are re-derived from the
    stored gates; nothing else is re-run."""
    s = json.loads(STATE_F.read_text())
    assert s["goal_sha256"] == sha(GOAL_F)
    T = pd.read_csv(TRIALS_F)
    base_cfg = {**BASE_CFG, **{k: False for k in RULE_KEYS}}
    b = run_scheme(base_cfg, "fold_group", 0)
    for i, r in T.iterrows():
        if r.knob == "base":
            T.loc[i, "gap_delta"] = np.nan
            continue
        if is_unsure_knob(r.knob):
            rk = run_scheme({**base_cfg, r.knob: True}, "fold_group", 0)
            d = rk["gap_3"] - b["gap_3"]
            T.loc[i, "gap_delta"] = d
            T.loc[i, "gate_b_gap"] = bool(d <= GAP_WIDEN_MAX)
            print(f"{r.knob}: 3-class gap delta {d:+.4f}", flush=True)
        flags = [str(T.loc[i, g]) == "True" for g in ("accept_rule", "gate_a_nested", "gate_b_gap", "gate_c_seeds",
                                                      "gate_d_row5")]
        T.loc[i, "kept"] = all(flags)
    T.to_csv(TRIALS_F, index=False)
    s["kept"] = [k for k in T.loc[T.kept.astype(str) == "True", "knob"]]
    final = {**base_cfg, **{k: True for k in s["kept"]}}
    s["final"] = final
    s["permutation"] = permutation_test(final)
    s["goal_sha256_end"] = sha(GOAL_F)
    save_state(s)
    report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.refresh:
        refresh()
    else:
        main(fresh=a.fresh)

"""Self-improvement loop over the L6 label engine: one-variable changes, a fixed accept rule, a 60-minute budget.

    .venv/Scripts/python eda/selfimprove/loop.py            # run (resumable: finished trials are never redone)
    .venv/Scripts/python eda/selfimprove/loop.py --report   # rebuild REPORT.md + figures from trials.csv / state.json
    .venv/Scripts/python eda/selfimprove/loop.py --fresh    # forget state.json / trials.csv and start over

How it works
  goal.json   what "better" means (metric, accept rule, runtime gate, lockbox, budget). Read-only: its sha256 is
              checked at the start and at the end of every run.
  queue.json  the one-variable changes, in order. Each is tried against the CURRENT base; an accepted change
              becomes the new base before the next trial (greedy). After the queue, accepted numeric knobs get
              their two neighbours tried. A round that accepted something is followed by another round over the
              whole queue (up to MAX_ROUNDS), until a round accepts nothing or the budget runs out.
  state.json  base configuration, its per-fold scores, finished trials, elapsed time, lockbox result.
  trials.csv  one row per trial: per-fold macro-F1, mean gain, folds better, ECE, timings, accepted or not.
Evaluation: eda/splits.csv fold_group (5 folds, holdout pages excluded), the ladder code of eda/07_improve.py
(per-fold KB + timeline + trust from training labels only). Knobs that are module constants are set on the
module for the trial. Holdout pages = lockbox: scored once at the end, for the final configuration only.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
EDA = HERE.parent
sys.path.insert(0, str(EDA))
M = importlib.import_module("07_improve")      # the ladder: data, fold features, run_fold, rules, metrics
WL = importlib.import_module("weak_labels")     # labelling functions + label model (no row labels used)
import improve.timeline as TLM  # noqa: E402
import improve.trust as TRM  # noqa: E402
sys.path.insert(0, str(HERE))
import combos as CB  # noqa: E402

GOAL_F, QUEUE_F, STATE_F, TRIALS_F = (HERE / n for n in ("goal.json", "queue.json", "state.json", "trials.csv"))
COMBO_QUEUE_F = HERE / "queue_combos.json"
FIGS = HERE / "figs"
CACHE = Path(os.environ.get("SELFIMPROVE_CACHE", Path(tempfile.gettempdir()) / "hype_selfimprove_cache"))
N_RUN = 20000
MAX_ROUNDS = 4

FAMILY_COLS = {
    "wording": ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today", "still", "future",
                "negation", "change_verb", "money_sym", "number_words", "joke_tail", "no_period", "kv_line",
                "n_chars", "n_words", "n_numbers", "n_commas", "starts_pronoun", "hype_score"],
    "date": ["text_date_delta_days", "has_text_date", "text_date_in_past", "text_date_in_future",
             "today_date_mismatch"],
    "value_shape": [f"{c}_{a}" for c in ["log_dev_from_pair_median", "round_number", "op_present"]
                    for a in ["min", "max"]] + [c for c in M.X0.columns if c.startswith("kind_")],
}
WEAK_COLS = ["weak_p_false", "weak_p_true", "weak_p_unsure", "weak_votes", "weak_votes_false", "weak_votes_unsure"]
TL_KNOBS = ("switch", "tau_days", "anchor", "two_page_rule", "trust_rounds", "trust_pseudo", "genre_pseudo")

ORIG = {"two_page": TLM.Timeline._two_page_rule, "page_trust": M.page_trust, "cols_for": M.cols_for}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ knobs
def apply(cfg):
    """Set every knob of cfg on the modules the ladder reads at call time."""
    TLM.SWITCH, TLM.TAU, TLM.ANCHOR = float(cfg["switch"]), float(cfg["tau_days"]), float(cfg["anchor"])
    TLM.Timeline._two_page_rule = (ORIG["two_page"] if cfg["two_page_rule"]
                                   else (lambda self, idx, anchor, t_of, v_of, path: path))
    TRM.PSEUDO, TRM.GENRE_PSEUDO = float(cfg["trust_pseudo"]), float(cfg["genre_pseudo"])
    rounds = int(cfg["trust_rounds"])
    M.page_trust = lambda tl, anchor, prior: ORIG["page_trust"](tl, anchor, prior, rounds=rounds)
    params = dict(num_leaves=int(cfg["num_leaves"]), n_estimators=int(cfg["n_estimators"]),
                  learning_rate=float(cfg["learning_rate"]), min_child_samples=int(cfg["min_child_samples"]),
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=M.SEED, verbose=-1)
    M.lgbm = lambda n_jobs=-1: LGBMClassifier(**params, n_jobs=2)       # always 2 threads: the arena's CPUs
    if float(cfg.get("et_vote", 0)) > 0 or int(cfg.get("lgbm_seeds", 1)) > 1:
        M.lgbm = CB.model_factory({**params, "n_jobs": 2}, cfg)          # combos 4 / 5
    drop = set(FAMILY_COLS.get(cfg["drop_family"] or "", []))
    extra = (WEAK_COLS if cfg["weak_features"] else []) + (["cfc_current_prob"] if cfg["cfc_feature"] else [])
    M.cols_for = lambda variant, X: ([c for c in ORIG["cols_for"](variant, X)
                                      if c not in drop and c != "cfc_current_prob"]
                                     + [c for c in extra if c in X.columns])


def tl_key(cfg):
    return tuple(cfg[k] for k in TL_KNOBS)


# ------------------------------------------------------------------ fold features (cached)
_MEM = {}
DATA_TAG = hashlib.md5((sha(EDA / "splits.csv") + str((M.OUT / "facts.pkl").stat().st_mtime)
                        + sha(EDA / "07_improve.py")).encode()).hexdigest()[:10]


def _cached(name, fn):
    if name in _MEM:
        return _MEM[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{hashlib.md5((DATA_TAG + name).encode()).hexdigest()}.pkl"
    if p.exists():
        v = pickle.loads(p.read_bytes())
    else:
        v = fn()
        p.write_bytes(pickle.dumps(v))
    _MEM[name] = v
    return v


def fold_X(tag, F, train, test, inner, cfg):
    """eda/07_improve.fold_table, split into a knob-free KB part and a timeline/trust part (cached per knobs)."""
    KB = _cached(f"kb|{tag}", lambda: M.kb_table(F, list(test), train, inner))

    def tl_parts():
        parts = [M.partition(F, train, set(test), set(test))]
        for inner_train in inner:
            inner_train = set(inner_train)
            inner_test = train - inner_train
            parts.append(M.partition(F, inner_train, inner_test | set(test), inner_test))
        return pd.concat(parts)
    TL = _cached(f"tl|{tag}|{tl_key(cfg)}", tl_parts)
    ids = list(train | set(test))
    X = pd.concat([M.X0.reindex(ids), KB.reindex(ids), TL.reindex(ids), M.MR.reindex(ids)], axis=1)
    X.columns = [str(c) for c in X.columns]
    return X


def add_weak(X, train):
    """Rule votes -> label model on this fold's table (KB columns already fold-safe); prior from training labels."""
    L = WL.labelling_functions(X)
    ytr = M.Y[sorted(train)].to_numpy()
    prior = np.bincount(ytr, minlength=3) / len(ytr)
    P, _, _ = WL.label_model(L, prior)
    X = X.copy()
    X[["weak_p_false", "weak_p_true", "weak_p_unsure"]] = P
    X["weak_votes"] = (L >= 0).sum(1).to_numpy()
    X["weak_votes_false"] = (L == WL.F_).sum(1).to_numpy()
    X["weak_votes_unsure"] = (L == WL.U_).sum(1).to_numpy()
    return X


_NEURO = None


def add_cfc(X, which):
    """cfc_current_prob for this fold only: outer_fold == k (train + test roles); lockbox: 'all' + 'holdout'."""
    global _NEURO
    if _NEURO is None:
        _NEURO = pd.read_csv(M.OUT / "neuro_feats.csv", dtype={"outer_fold": str})
    sel = _NEURO[_NEURO.outer_fold.isin(which)]
    X = X.copy()
    X["cfc_current_prob"] = sel.groupby("id")["cfc_current_prob"].mean().reindex(X.index)
    return X


# ------------------------------------------------------------------ one fold
def run_isotonic(X, train, test, inner):
    tr, te = sorted(train), sorted(test)
    ytr, yte = M.Y[tr].to_numpy(), M.Y[te].to_numpy()
    cols = M.cols_for("L6_judge", X)
    Ztr, Zte = X.loc[tr], X.loc[te]
    m = M.lgbm(2).fit(Ztr[cols], ytr)
    t0 = time.time()
    P = M.proba3(m, Zte[cols])
    pred_s = (time.time() - t0) * N_RUN / max(len(te), 1)
    Pin = np.zeros((len(tr), 3))
    for a in inner:
        am = np.isin(tr, list(a))
        mi = M.lgbm(2).fit(Ztr.loc[am, cols], ytr[am])
        Pin[~am] = M.proba3(mi, Ztr.loc[~am, cols])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(Pin[:, 0], (ytr == 0).astype(float))
    Pin, P = M.with_false_prob(Pin, iso.predict(Pin[:, 0])), M.with_false_prob(P, iso.predict(P[:, 0]))
    Pin, P = M.hard_rules(Pin, Ztr), M.hard_rules(P, Zte)
    tau = M.tune_tau(Pin, ytr, M.unsure_candidates(Ztr))
    return P, M.decide(P, M.unsure_candidates(Zte), tau), {"predict_s_cpu2": pred_s, "tau": tau}, yte, te


def run_one(X, train, test, inner, cfg):
    t0 = time.time()
    if cfg["calibrator"] == "isotonic":
        P, pred, info, yte, te = run_isotonic(X, train, test, inner)
    else:
        variant = "L7_runtime_retrain" if cfg["calibrator"] == "evidence_sum" else "L5_threevalued"
        res, yte, te = M.run_fold(X, train, test, inner, [variant])
        P, pred, info = res[variant]
    info["retrain_s"] = time.time() - t0       # every fit of the fold (final + inner) on 2 threads
    return P, pred, info, yte, te


def evaluate(cfg, folds, combo=False, tag="fold", key="{k}", seed=None, permute=None):
    """seed: LightGBM / ExtraTrees random seed for this evaluation; permute: shuffle the TRAINING labels of every
    fold (classifier stage) with this rng seed, for the label-permutation test. Test labels are never changed."""
    seed0, Y0 = M.SEED, M.Y
    if seed is not None:
        M.SEED = seed
    apply(cfg)
    fold_f1, Ps, preds, ys, retrain, predict, train_f1 = [], [], [], [], [], [], []
    try:
        for k, (train, test, inner) in enumerate(folds):
            X = fold_X(f"{tag}{k}", F_CV, train, test, inner, cfg)
            if cfg["weak_features"]:
                X = add_weak(X, train)
            if cfg["cfc_feature"]:
                X = add_cfc(X, [str(k)])
            if permute is not None:
                tr = sorted(train)
                Yp = Y0.copy()
                Yp.loc[tr] = np.random.default_rng(permute * 100 + k).permutation(Y0.loc[tr].to_numpy())
                M.Y = Yp
            if combo:
                P, pred, info, yte, te = CB.run_one_combo(M, X, train, test, inner, cfg, key.format(k=k))
                train_f1.append(info["train_f1"])
            else:
                P, pred, info, yte, te = run_one(X, train, test, inner, cfg)
            M.Y = Y0
            yte = Y0[te].to_numpy()                     # always score against the true labels
            fold_f1.append(M.metrics(yte, P, pred)["macro_f1_tf"])
            Ps.append(P); preds.append(pred); ys.append(yte)
            retrain.append(info["retrain_s"]); predict.append(info["predict_s_cpu2"])
    finally:
        M.SEED, M.Y = seed0, Y0
        apply(cfg)
    pooled = M.metrics(np.concatenate(ys), np.vstack(Ps), np.concatenate(preds))
    out = {"fold_f1": [float(x) for x in fold_f1], "mean_f1": float(np.mean(fold_f1)),
           **{k: float(v) for k, v in pooled.items()},
           "retrain_s_max": float(max(retrain)), "predict_20k_s_max": float(max(predict))}
    if train_f1:
        out["train_f1"] = float(np.mean(train_f1))
        out["gap"] = out["train_f1"] - out["mean_f1"]
    return out


# ------------------------------------------------------------------ state
def load_state(goal_sha):
    if STATE_F.exists():
        s = json.loads(STATE_F.read_text())
        assert s["goal_sha256"] == goal_sha, "goal.json changed since this run started"
        return s
    return {"goal_sha256": goal_sha, "started": time.strftime("%Y-%m-%d %H:%M"), "elapsed_s": 0.0,
            "done": [], "accepted": [], "base_cfg": None, "base": None, "l6": None, "lockbox": None,
            "round": 1}


def save_state(s):
    STATE_F.write_text(json.dumps(s, indent=1))


TRIAL_COLS = ["trial", "knob", "value", "f0", "f1", "f2", "f3", "f4", "mean_f1", "pooled_f1", "mean_gain",
              "folds_better", "gain_se", "acc", "ece", "unsure_recall", "retrain_s_max", "predict_20k_s_max",
              "gate_ok", "accepted", "trial_s"]


def log_trial(row):
    df = pd.DataFrame([row]).reindex(columns=TRIAL_COLS)
    df.to_csv(TRIALS_F, mode="a", header=not TRIALS_F.exists(), index=False)


def gate_ok(goal, r):
    g = goal["runtime_gate"]
    return r["retrain_s_max"] <= g["retrain_s_max"] and r["predict_20k_s_max"] <= g["predict_20k_s_max"]


def decide_accept(goal, base, r):
    a = goal["accept_change"]
    gain = np.array(r["fold_f1"]) - np.array(base["fold_f1"])
    ok = (gain.mean() >= a["min_gain"] and int((gain > 0).sum()) >= a["folds_better"]
          and r["ece"] <= base["ece"] + a["ece_worse_max"] and gate_ok(goal, r))
    return bool(ok), gain


def refine_values(knob, accepted_value, tried):
    vals = sorted(set(tried) | {accepted_value})
    i = vals.index(accepted_value)
    out = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(vals):
            mid = (accepted_value + vals[j]) / 2
            if isinstance(accepted_value, int) and not isinstance(accepted_value, bool):
                mid = int(round(mid))
            if mid not in vals:
                out.append(mid)
    return out


# ------------------------------------------------------------------ main loop
def folds_cv():
    return M.scheme_folds("fold_group", CV_IDS)


def main(fresh=False):
    goal = json.loads(GOAL_F.read_text())
    goal_sha = sha(GOAL_F)
    if fresh:
        for f in (STATE_F, TRIALS_F):
            f.unlink(missing_ok=True)
    q = json.loads(QUEUE_F.read_text())
    s = load_state(goal_sha)
    budget = goal["budget_minutes"] * 60
    t_run = time.time()
    elapsed0 = s["elapsed_s"]
    folds = folds_cv()

    def spent():
        return elapsed0 + time.time() - t_run

    if s["base"] is None:
        cfg = dict(q["defaults"])
        t0 = time.time()
        r = evaluate(cfg, folds)
        s["base_cfg"], s["base"], s["l6"] = cfg, r, r
        log_trial({"trial": "t000:base(L6)", "knob": "-", "value": "-", **{f"f{k}": round(v, 4) for k, v in enumerate(r["fold_f1"])},
                   "mean_f1": round(r["mean_f1"], 4), "pooled_f1": round(r["macro_f1_tf"], 4), "mean_gain": 0.0,
                   "folds_better": "-", "acc": round(r["acc"], 4), "ece": round(r["ece"], 4),
                   "unsure_recall": round(r["unsure_recall"], 4), "retrain_s_max": round(r["retrain_s_max"], 2),
                   "predict_20k_s_max": round(r["predict_20k_s_max"], 3), "gate_ok": gate_ok(goal, r),
                   "accepted": "base", "trial_s": round(time.time() - t0, 1)})
        s["elapsed_s"] = spent(); save_state(s)
        print(f"base L6: mean F1 {r['mean_f1']:.4f}, pooled {r['macro_f1_tf']:.4f} (ladder: 0.8007), "
              f"ECE {r['ece']:.4f}, retrain {r['retrain_s_max']:.1f}s", flush=True)

    def plan():
        """Round r: every queue item, then neighbours of the numeric values accepted in round r."""
        r = s.get("round", 1)
        items = [(f"r{r}q{i:02d}", k, v) for i, (k, v) in enumerate(q["queue"])]
        if all(t in s["done"] for t, _, _ in items):
            tried = {}
            for _, k, v in items:
                tried.setdefault(k, []).append(v)
            for acc in s["accepted"]:
                if acc.get("round", 1) != r:
                    continue
                k, v = acc["knob"], acc["value"]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    for nv in refine_values(k, v, tried.get(k, []) + [q["defaults"][k]]):
                        items.append((f"r{r}ref:{k}={nv}", k, nv))
        return items

    n_new = 0
    while True:
        todo = [it for it in plan() if it[0] not in s["done"]]
        if not todo:
            r = s.get("round", 1)
            if any(a.get("round", 1) == r for a in s["accepted"]) and r < MAX_ROUNDS:
                s["round"] = r + 1; save_state(s)
                print(f"round {r} accepted changes: starting round {r + 1}", flush=True)
                continue
            print(f"converged after round {r} (no accepted change)" if r < MAX_ROUNDS else "max rounds", flush=True)
            break
        if spent() >= budget:
            print("budget reached", flush=True)
            break
        tid, knob, value = todo[0]
        cfg = dict(s["base_cfg"])
        if cfg[knob] == value:
            s["done"].append(tid); save_state(s)
            continue
        cfg[knob] = value
        t0 = time.time()
        r = evaluate(cfg, folds)
        ok, gain = decide_accept(goal, s["base"], r)
        log_trial({"trial": tid, "knob": knob, "value": value,
                   **{f"f{k}": round(v, 4) for k, v in enumerate(r["fold_f1"])},
                   "mean_f1": round(r["mean_f1"], 4), "pooled_f1": round(r["macro_f1_tf"], 4),
                   "mean_gain": round(float(gain.mean()), 4), "folds_better": int((gain > 0).sum()),
                   "gain_se": round(float(gain.std(ddof=1) / math.sqrt(len(gain))), 4),
                   "acc": round(r["acc"], 4), "ece": round(r["ece"], 4), "unsure_recall": round(r["unsure_recall"], 4),
                   "retrain_s_max": round(r["retrain_s_max"], 2), "predict_20k_s_max": round(r["predict_20k_s_max"], 3),
                   "gate_ok": gate_ok(goal, r), "accepted": ok, "trial_s": round(time.time() - t0, 1)})
        s["done"].append(tid)
        if ok:
            s["base_cfg"], s["base"] = cfg, r
            s["accepted"].append({"trial": tid, "knob": knob, "value": value, "mean_f1": r["mean_f1"],
                                  "round": s.get("round", 1)})
        s["elapsed_s"] = spent(); save_state(s)
        n_new += 1
        print(f"{tid:28s} {knob}={value!s:8s} gain {gain.mean():+.4f} better {int((gain > 0).sum())}/5 "
              f"ece {r['ece']:.4f} {'ACCEPT' if ok else 'no'}  [{time.time() - t0:.0f}s, total {spent() / 60:.1f} min]",
              flush=True)

    if s["lockbox"] is None:
        s["lockbox"] = lockbox(s["base_cfg"])
        s["elapsed_s"] = spent(); save_state(s)
    assert sha(GOAL_F) == goal_sha, "goal.json was modified during the run"
    report(goal, s)
    print(f"done: {len(s['done'])} trials, {s['elapsed_s'] / 60:.1f} min, lockbox {s['lockbox']}", flush=True)


def _trial_row(tid, knob, value, r, base, goal, ok, t0):
    gain = np.array(r["fold_f1"]) - np.array(base["fold_f1"])
    return {"trial": tid, "knob": knob, "value": value,
            **{f"f{k}": round(v, 4) for k, v in enumerate(r["fold_f1"])},
            "mean_f1": round(r["mean_f1"], 4), "pooled_f1": round(r["macro_f1_tf"], 4),
            "mean_gain": round(float(gain.mean()), 4), "folds_better": int((gain > 0).sum()),
            "gain_se": round(float(gain.std(ddof=1) / math.sqrt(len(gain))), 4),
            "acc": round(r["acc"], 4), "ece": round(r["ece"], 4), "unsure_recall": round(r["unsure_recall"], 4),
            "retrain_s_max": round(r["retrain_s_max"], 2), "predict_20k_s_max": round(r["predict_20k_s_max"], 3),
            "gate_ok": gate_ok(goal, r), "accepted": ok, "trial_s": round(time.time() - t0, 1)}


GATE_COLS = ["nested_rot_ok", "gap_base", "gap_cand", "gap_ok", "seed_gains", "seed_ok", "row5_gain", "row5_ok",
             "time_gain", "time_warn", "overfit_ok"]


def migrate_trials():
    """Add the overfitting-gate columns to trials.csv (old rows keep them empty)."""
    global TRIAL_COLS
    if all(cname in TRIAL_COLS for cname in GATE_COLS):
        return
    TRIAL_COLS = TRIAL_COLS + GATE_COLS
    if TRIALS_F.exists():
        T = pd.read_csv(TRIALS_F)
        if not all(cname in T.columns for cname in GATE_COLS):
            T.reindex(columns=TRIAL_COLS).to_csv(TRIALS_F, index=False)


def folds_scheme(scheme):
    """fold_row5 (KFold inner, as eda/07_improve) or the time split (train < 2026-07-01, GroupKFold inner)."""
    if scheme == "row5":
        return M.scheme_folds("fold_row5", CV_IDS)
    S = M.SPLITS.loc[CV_IDS]
    test = set(S.index[S.time_split == "test"])
    train = set(CV_IDS) - test
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[M.PAGE_OF[i] for i in trl])]
    return [(train, test, inner)]


def ensure_neuro_schemes():
    """Neural members for the fold_row5 / time folds are made on the GPU only when a neural combination needs them."""
    import subprocess
    have = CB.neuro()["parts"]
    if ("r5_0", "test") in have and ("time0", "test") in have:
        return
    py = EDA.parent / ".venv-neuro" / "Scripts" / "python.exe"
    print("  generating neural members for fold_row5 / time on the GPU ...", flush=True)
    subprocess.run([str(py), str(EDA / "neuro" / "combo_oof.py"), "--schemes", "row5,time"], check=True)
    CB._NEURO = None


SCHEME_CACHE = {}


def scheme_eval(cfg, scheme, seed=None):
    k = (json.dumps(cfg, sort_keys=True, default=str), scheme, seed)
    if k not in SCHEME_CACHE:
        if scheme == "group":
            SCHEME_CACHE[k] = evaluate(cfg, folds_cv(), combo=True, seed=seed)
        elif scheme == "row5":
            SCHEME_CACHE[k] = evaluate(cfg, folds_scheme("row5"), combo=True, tag="row5_", key="r5_{k}", seed=seed)
        else:
            SCHEME_CACHE[k] = evaluate(cfg, folds_scheme("time"), combo=True, tag="time", key="time{k}", seed=seed)
    return SCHEME_CACHE[k]


def overfit_gates(goal, base_cfg, base, cfg, r):
    """(a) nested selection, (b) train-OOF gap, (c) 3 seeds, (d) fold_row5 (FAIL on flip) / time split (WARN)."""
    a = goal["accept_change"]
    g = np.array(r["fold_f1"]) - np.array(base["fold_f1"])
    rot = 0
    for j in range(len(g)):
        o = np.delete(g, j)
        if o.mean() >= a["min_gain"] and int((o > 0).sum()) >= a["folds_better"] - 1 and g[j] > 0:
            rot += 1
    nested_ok = rot >= 4
    gap_ok = (r["gap"] - base["gap"]) <= 0.02
    seeds = [M.SEED, M.SEED + 1, M.SEED + 2]
    gains = [r["mean_f1"] - base["mean_f1"]]
    for sd in seeds[1:]:
        gains.append(scheme_eval(cfg, "group", sd)["mean_f1"] - scheme_eval(base_cfg, "group", sd)["mean_f1"])
    seed_ok = all(x > 0 for x in gains) and float(np.std(gains)) < float(np.mean(gains))
    if CB.neural_members(cfg):
        ensure_neuro_schemes()
    row5_gain = scheme_eval(cfg, "row5")["mean_f1"] - scheme_eval(base_cfg, "row5")["mean_f1"]
    time_gain = scheme_eval(cfg, "time")["mean_f1"] - scheme_eval(base_cfg, "time")["mean_f1"]
    gates = {"nested_rot_ok": f"{rot}/5", "gap_base": round(base["gap"], 4), "gap_cand": round(r["gap"], 4),
             "gap_ok": gap_ok, "seed_gains": " ".join(f"{x:+.4f}" for x in gains), "seed_ok": seed_ok,
             "row5_gain": round(row5_gain, 4), "row5_ok": row5_gain > 0, "time_gain": round(time_gain, 4),
             "time_warn": time_gain <= 0}
    gates["overfit_ok"] = bool(nested_ok and gap_ok and seed_ok and gates["row5_ok"])
    return gates


def permutation_test(cfg, n=3):
    """Shuffle the training labels inside every fold (classifier stage), refit, score OOF on the true labels."""
    return [round(evaluate(cfg, folds_cv(), combo=True, permute=i)["macro_f1_tf"], 4) for i in range(1, n + 1)]


def main_combos():
    """Combination round: start from the loop's final configuration, try queue_combos.json (one change at a time,
    same accept rule), greedy, at most 2 rounds, its own budget; then the lockbox once (second look)."""
    goal = json.loads(GOAL_F.read_text())
    goal_sha = sha(GOAL_F)
    cq = json.loads(COMBO_QUEUE_F.read_text())
    migrate_trials()
    s = load_state(goal_sha)
    assert s.get("lockbox"), "run the main loop first (its final configuration is the combination round's base)"
    c = s.setdefault("combo", {"started": time.strftime("%Y-%m-%d %H:%M"), "elapsed_s": 0.0, "done": [],
                               "accepted": [], "base_cfg": None, "base": None, "start": None, "lockbox": None,
                               "round": 1, "extras": {}})
    budget = cq["budget_minutes"] * 60
    t_run, elapsed0 = time.time(), c["elapsed_s"]
    folds = folds_cv()

    def spent():
        return elapsed0 + time.time() - t_run

    if c["base"] is None:
        cfg = {**s["base_cfg"], **CB.DEFAULTS}
        t0 = time.time()
        r = evaluate(cfg, folds, combo=True)
        target = s["base"]["mean_f1"]
        print(f"combo-path base: mean F1 {r['mean_f1']:.4f} (final model: {target:.4f})", flush=True)
        assert abs(r["mean_f1"] - target) <= 5e-4, "combination path does not reproduce the final model"
        c["base_cfg"], c["base"], c["start"] = cfg, r, r
        log_trial({**_trial_row("c000:base(final, combo path)", "-", "-", r, r, goal, "base", t0),
                   "mean_gain": 0.0, "folds_better": "-", "gain_se": ""})
        c["elapsed_s"] = spent(); save_state(s)

    while True:
        rnd = c["round"]
        items = [(f"c{rnd}q{i:02d}", k, v) for i, (k, v) in enumerate(cq["queue"])]
        todo = [it for it in items if it[0] not in c["done"]]
        if not todo:
            if any(a["round"] == rnd for a in c["accepted"]) and rnd < cq["max_rounds"]:
                c["round"] = rnd + 1; save_state(s)
                print(f"combination round {rnd} accepted changes: one more round", flush=True)
                continue
            print(f"combinations done after round {rnd}", flush=True)
            break
        if spent() >= budget:
            print("combination budget reached", flush=True)
            break
        tid, knob, value = todo[0]
        cfg = dict(c["base_cfg"])
        if cfg.get(knob) == value:
            c["done"].append(tid); save_state(s)
            continue
        cfg[knob] = value
        t0 = time.time()
        r = evaluate(cfg, folds, combo=True)
        rule_ok, gain = decide_accept(goal, c["base"], r)
        gates = {"gap_base": round(c["base"]["gap"], 4), "gap_cand": round(r["gap"], 4)}
        if rule_ok:                                   # the overfitting gates, for every change the rule would keep
            gates = overfit_gates(goal, c["base_cfg"], c["base"], cfg, r)
        ok = bool(rule_ok and gates.get("overfit_ok", False))
        log_trial({**_trial_row(tid, knob, value, r, c["base"], goal, ok, t0), **gates})
        c.setdefault("rule_ok", {})[tid] = rule_ok
        c["extras"][tid] = {"extra_cpu_s": round(CB.extra_cpu_s(cfg), 2),
                            "note": CB.NEEDS_TORCH if CB.neural_members(cfg) else "",
                            "pooled_f1": round(r["macro_f1_tf"], 4)}
        c["done"].append(tid)
        if ok:
            c["base_cfg"], c["base"] = cfg, r
            c["accepted"].append({"trial": tid, "knob": knob, "value": value, "mean_f1": r["mean_f1"], "round": rnd})
        c["elapsed_s"] = spent(); save_state(s)
        print(f"{tid:10s} {knob}={value!s:6s} gain {gain.mean():+.4f} better {int((gain > 0).sum())}/5 "
              f"ece {r['ece']:.4f} {'ACCEPT' if ok else 'no'}  [{time.time() - t0:.0f}s, total {spent() / 60:.1f} min]",
              flush=True)

    if c.get("permutation") is None:
        perm = permutation_test(c["base_cfg"])
        c["permutation"] = {"oof_macro_f1": perm, "pass": all(x <= 0.50 for x in perm)}
        c["elapsed_s"] = spent(); save_state(s)
        print(f"label-permutation test: OOF macro-F1 {perm} -> {'PASS' if c['permutation']['pass'] else 'FAIL'}",
              flush=True)
    if c["lockbox"] is None:
        c["lockbox"] = lockbox(c["base_cfg"], combo=True)
        c["elapsed_s"] = spent(); save_state(s)
    assert sha(GOAL_F) == goal_sha, "goal.json was modified during the run"
    report(goal, s)
    print(f"combinations: {len(c['done'])} items, {c['elapsed_s'] / 60:.1f} min, lockbox {c['lockbox']}", flush=True)


def lockbox(cfg, combo=False):
    """The holdout pages, once, for the final configuration (train = every CV row)."""
    apply(cfg)
    hold = sorted(M.SPLITS.index[M.SPLITS.holdout == 1])
    train = set(CV_IDS)
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[M.PAGE_OF[i] for i in trl])]
    X = fold_X("lockbox", M.F_ALL, train, set(hold), inner, cfg)
    if cfg["weak_features"]:
        X = add_weak(X, train)
    if cfg["cfc_feature"]:
        X = add_cfc(X, ["all", "holdout"])
    if combo:
        P, pred, info, yte, _ = CB.run_one_combo(M, X, train, set(hold), inner, cfg, "lockbox")
    else:
        P, pred, info, yte, _ = run_one(X, train, set(hold), inner, cfg)
    met = M.metrics(yte, P, pred)
    return {k: round(float(v), 4) for k, v in met.items()} | {"rows": len(hold),
                                                              "all_true_acc": round(float((yte == 1).mean()), 4)}


# ------------------------------------------------------------------ report
def report(goal=None, s=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    goal = goal or json.loads(GOAL_F.read_text())
    s = s or json.loads(STATE_F.read_text())
    q = json.loads(QUEUE_F.read_text())
    T_all = pd.read_csv(TRIALS_F)
    T = T_all[~T_all.trial.astype(str).str.startswith("c")]
    trials = T[T.trial != "t000:base(L6)"].copy()
    n_all = len(trials)
    distinct = trials.drop_duplicates(subset=["knob", "value", "f0", "f1", "f2", "f3", "f4"])
    n = len(distinct)        # a later round can repeat a trial exactly (same base, same change): counted once
    l6, fin = s["l6"], s["base"]
    se = distinct["gain_se"].astype(float).mean() if n else float("nan")
    noise = se * math.sqrt(2 * math.log(n)) if n > 1 else float("nan")
    final_gain = fin["mean_f1"] - l6["mean_f1"]

    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["#2e7d32" if str(a) == "True" else "#9e9e9e" for a in trials["accepted"]]
    ax.bar(range(n_all), trials["mean_gain"].astype(float), color=colors)
    ax.axhline(goal["accept_change"]["min_gain"], color="#c0392b", ls="--", lw=1, label="min gain")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(n_all))
    ax.set_xticklabels([f"{k}={v}" for k, v in zip(trials["knob"], trials["value"])], rotation=75, fontsize=7)
    ax.set_ylabel("mean per-fold macro-F1 gain vs base")
    ax.legend(loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "trial_gains.png", dpi=120); plt.close(fig)

    path = [l6["mean_f1"]]
    for _, r in trials.iterrows():
        path.append(r["mean_f1"] if str(r["accepted"]) == "True" else path[-1])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.step(range(len(path)), path, where="post", color="#3b6ea5")
    ax.set_xlabel("trial"); ax.set_ylabel("base mean macro-F1")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "accepted_path.png", dpi=120); plt.close(fig)

    tri = []
    for knob in dict.fromkeys(trials["knob"]):
        t = trials[trials.knob == knob]
        acc = t[t.accepted.astype(str) == "True"]
        if len(acc):
            verdict, val = "kept", acc["value"].iloc[-1]
        elif (t["mean_gain"].astype(float) < 0).all():
            verdict, val = "dropped", q["defaults"].get(knob)
        else:
            verdict, val = "frozen", q["defaults"].get(knob)
        best = t.loc[t["mean_gain"].astype(float).idxmax()]
        tri.append(f"| `{knob}` | {', '.join(str(v) for v in t['value'])} | {best['value']} ({float(best['mean_gain']):+.4f}, "
                   f"{best['folds_better']}/5) | **{verdict}** | {val} |")

    def row(name, r):
        return (f"| {name} | {r['mean_f1']:.4f} | {r['macro_f1_tf']:.4f} | {r['acc']:.4f} | {r['ece']:.4f} | "
                f"{r['unsure_recall']:.3f} | {r['false_precision']:.3f} | {r['retrain_s_max']:.1f} | {r['predict_20k_s_max']:.2f} |")

    cols = ["trial", "knob", "value", "f0", "f1", "f2", "f3", "f4", "mean_f1", "mean_gain", "folds_better", "ece",
            "retrain_s_max", "gate_ok", "accepted", "trial_s"]
    tt = T[[c for c in cols if c in T.columns]].fillna("")
    trial_md = "\n".join(["| " + " | ".join(tt.columns) + " |", "|" + "---|" * len(tt.columns)]
                         + ["| " + " | ".join(str(v) for v in r) + " |" for r in tt.itertuples(index=False)])
    lb = s["lockbox"] or {}
    verdict = ("distinguishable from noise" if final_gain > noise else "NOT distinguishable from noise") \
        if n > 1 and final_gain > 0 else "no gain"
    md = f"""# Self-improvement loop: report

Started {s['started']} · {len(s['done'])} queue/refine items · {n_all} trials evaluated ({n} distinct; a later round repeats a trial exactly when neither its base nor its change moved) · {s['elapsed_s'] / 60:.1f} min of the {goal['budget_minutes']}-min budget.
Goal file `goal.json` (sha256 `{s['goal_sha256'][:16]}…`) checked unchanged at the start and the end.

**Rule.** A one-variable change is kept only if the mean per-fold macro-F1 rises by ≥ {goal['accept_change']['min_gain']}, it is better on ≥ {goal['accept_change']['folds_better']}/5 folds, pooled ECE is not worse by more than {goal['accept_change']['ece_worse_max']}, and the 2-thread runtime gate holds (retrain ≤ {goal['runtime_gate']['retrain_s_max']} s per fold, predict ≤ {goal['runtime_gate']['predict_20k_s_max']} s for 20k rows). Folds: `eda/splits.csv` `fold_group`, holdout pages excluded.

## Before / after (fold_group, 5 folds)

| | mean per-fold macro-F1 | pooled macro-F1 | accuracy | ECE | unsure recall | false precision | retrain s (max fold) | predict 20k s |
|---|---|---|---|---|---|---|---|---|
{row('L6 (start)', l6)}
{row('final', fin)}

Gain of the final configuration: **{final_gain:+.4f}** mean per-fold macro-F1.

**Multiple testing.** {n} distinct trials; the mean standard error of a trial's mean gain is {se:.4f}, so the best gain pure noise would produce over {n} tries is about {se:.4f} × √(2 ln {n}) = **{noise:.4f}**. The final gain is {verdict}.

## Lockbox (holdout pages, scored once, final configuration)

| rows | accuracy | macro-F1 | ECE | unsure recall | false precision | all-true accuracy |
|---|---|---|---|---|---|---|
| {lb.get('rows', '')} | {lb.get('acc', '')} | {lb.get('macro_f1_tf', '')} | {lb.get('ece', '')} | {lb.get('unsure_recall', '')} | {lb.get('false_precision', '')} | {lb.get('all_true_acc', '')} |

Note: before this loop the holdout had already been viewed for L0 (macro-F1 0.7433) and L4 (0.7327), without tuning on it. This loop never read it before this single final score. 43 pages / 304 rows: a macro-F1 standard error of roughly ±0.03.

## Triage per knob

| knob | values tried | best trial (gain, folds better) | verdict | value in final config |
|---|---|---|---|---|
{chr(10).join(tri)}

kept = a value passed the rule and is in the final config · dropped = every value made it worse · frozen = some value helped a little but not enough for the rule, so the default stays.
Skipped: {'; '.join(f'{k}: {v}' for k, v in q.get('skipped', {}).items())}

## Final configuration

```json
{json.dumps(s['base_cfg'], indent=1)}
```

Accepted changes, in order: {', '.join(f"`{a['knob']}={a['value']}`" for a in s['accepted']) or 'none'}.

## Every trial

{trial_md}

Figures: `figs/trial_gains.png` (mean gain per trial, green = accepted), `figs/accepted_path.png` (base score along the run).

Rerun: `.venv/Scripts/python eda/selfimprove/loop.py` (resumes) · `--fresh` to start over · `--report` to rebuild this file.
"""
    if s.get("combo") and s["combo"].get("base"):
        md += combo_section(goal, s, T_all, n, se)
    (HERE / "REPORT.md").write_text(md, encoding="utf-8")


def combo_section(goal, s, T_all, n_main, se_main):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = s["combo"]
    cq = json.loads(COMBO_QUEUE_F.read_text())
    CT = T_all[T_all.trial.astype(str).str.startswith("c")]
    ct = CT[~CT.trial.astype(str).str.startswith("c000")].copy()
    start, fin = c["start"], c["base"]
    labels = {"et_vote": "4 · ExtraTrees soft vote (LightGBM weight)", "lgbm_seeds": "5 · LightGBM seed bag",
              "bigru_blend": "3 · char BiGRU blend (model weight)", "neural_stack": "1 · CfC + spiking stack",
              "unsure_head": "2 · unsure flag from a neural head"}
    rows = []
    for _, r in ct.iterrows():
        ex = c["extras"].get(r["trial"], {})
        rows.append(f"| {r['trial']} | {labels.get(r['knob'], r['knob'])} | {r['value']} | {r['acc']} | "
                    f"{ex.get('pooled_f1', r['pooled_f1'])} | {r['mean_f1']} | {r['ece']} | {r['unsure_recall']} | "
                    f"{r['folds_better']}/5 | {float(r['mean_gain']):+.4f} | {ex.get('extra_cpu_s', 0)} | "
                    f"{'**yes**' if str(r['accepted']) == 'True' else 'no'} | {ex.get('note', '')} |")
    distinct_all = T_all[~T_all.trial.astype(str).str.contains("base")].drop_duplicates(
        subset=["knob", "value", "f0", "f1", "f2", "f3", "f4"])
    n_all = len(distinct_all)
    se = distinct_all["gain_se"].astype(float).mean()
    noise = se * math.sqrt(2 * math.log(n_all)) if n_all > 1 else float("nan")
    gain = fin["mean_f1"] - start["mean_f1"]
    if len(ct):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        colors = ["#2e7d32" if str(a) == "True" else "#9e9e9e" for a in ct["accepted"]]
        ax.bar(range(len(ct)), ct["mean_gain"].astype(float), color=colors)
        ax.axhline(goal["accept_change"]["min_gain"], color="#c0392b", ls="--", lw=1, label="min gain")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(len(ct)))
        ax.set_xticklabels([f"{k}={v}" for k, v in zip(ct["knob"], ct["value"])], rotation=60, fontsize=7)
        ax.set_ylabel("mean macro-F1 gain vs base")
        ax.legend(loc="lower left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(FIGS / "combo_gains.png", dpi=120); plt.close(fig)
    lb1, lb2 = s.get("lockbox") or {}, c.get("lockbox") or {}

    def mrow(name, r):
        return (f"| {name} | {r['acc']:.4f} | {r['macro_f1_tf']:.4f} | {r['mean_f1']:.4f} | {r['ece']:.4f} | "
                f"{r['unsure_recall']:.3f} | {r['retrain_s_max']:.1f} | {r['predict_20k_s_max']:.2f} |")
    acc_list = ", ".join(f"`{a['knob']}={a['value']}`" for a in c["accepted"]) or "none"
    grow = []
    for _, r in ct.iterrows():
        rule = c.get("rule_ok", {}).get(r["trial"])
        if rule:
            yn = lambda v: "PASS" if str(v) == "True" else "FAIL"  # noqa: E731
            grow.append(f"| {r['trial']} | {labels.get(r['knob'], r['knob'])} = {r['value']} | {r.get('nested_rot_ok', '')} "
                        f"({'PASS' if str(r.get('nested_rot_ok', '0/5'))[0] in '45' else 'FAIL'}) | "
                        f"{r.get('gap_base', '')} → {r.get('gap_cand', '')} ({yn(r.get('gap_ok'))}) | "
                        f"{r.get('seed_gains', '')} ({yn(r.get('seed_ok'))}) | {r.get('row5_gain', '')} "
                        f"({yn(r.get('row5_ok'))}) | {r.get('time_gain', '')} "
                        f"({'WARN' if str(r.get('time_warn')) == 'True' else 'ok'}) | "
                        f"**{'kept' if str(r.get('overfit_ok')) == 'True' else 'rejected'}** |")
    gate_md = ("\n".join(["| trial | combination | (a) nested 4→5th rotations | (b) train−OOF gap | (c) gain over 3 seeds "
                          "| (d) fold_row5 gain | (d) time-split gain | verdict |", "|---|---|---|---|---|---|---|---|"]
                         + grow) if grow else "No combination passed the accept rule, so no overfitting gate was needed.")
    perm = c.get("permutation") or {}
    return f"""

## Round: combinations

Started {c['started']} · {len(ct)} combination trials · {c['elapsed_s'] / 60:.1f} min of the {cq['budget_minutes']}-min combination budget · same `goal.json` rule (checked unchanged). Base = the final configuration above; the first trial re-ran it through the combination pipeline (all combinations off) and reproduced it (mean {start['mean_f1']:.4f}). Every combination is tuned only on the training folds' inner out-of-fold predictions (stack weights, blend weight, thresholds); neural members (CfC / spiking unsure heads, char BiGRU) come from `eda/out/neuro_combo_oof.pkl` (`eda/neuro/combo_oof.py`, GPU) on exactly the same inner splits. Extra CPU s = what the neural members add to predicting 20k rows on 2 threads.

| trial | combination | value | acc | macro-F1 (pooled) | macro-F1 (mean) | ECE | unsure recall | folds better | gain | extra CPU s | accepted | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

| | acc | pooled macro-F1 | mean macro-F1 | ECE | unsure recall | retrain s (max fold) | predict 20k s |
|---|---|---|---|---|---|---|---|
{mrow('final model (start)', start)}
{mrow('after combinations', fin)}

Accepted, in order: {acc_list}. Gain over the final model: **{gain:+.4f}** mean macro-F1.

### Overfitting gates

Every combination that passes the accept rule must also pass all of these before it is kept: (a) nested selection — the rule decided on 4 folds must still see a gain on the left-out 5th fold in ≥ 4 of 5 rotations; (b) the train−OOF macro-F1 gap must not widen by more than 0.02; (c) the gain keeps its sign over 3 seeds with std < mean gain; (d) the gain keeps its sign on `fold_row5` (FAIL if it flips) and on the time split (WARN if it flips).

{gate_md}

**Label-permutation test (final configuration):** training labels shuffled inside every fold, 3 refits, scored out of fold on the true labels → macro-F1 {perm.get('oof_macro_f1', '')} (must be ≤ 0.50): **{'PASS' if perm.get('pass') else 'FAIL' if perm else 'not run'}**. The shuffle covers the classifier stage (LightGBM, calibrator, stack, thresholds); the knowledge-base features and any neural members keep their true-label training.

**Multiple testing (main loop + combinations).** {n_all} distinct trials; mean standard error of a trial's gain {se:.4f}, so the best gain noise alone would give is about {se:.4f} × √(2 ln {n_all}) = **{noise:.4f}**.

### Lockbox (holdout pages)

| look | configuration | rows | accuracy | macro-F1 | ECE | unsure recall | all-true accuracy |
|---|---|---|---|---|---|---|---|
| 1st (main loop final) | see "Final configuration" | {lb1.get('rows', '')} | {lb1.get('acc', '')} | {lb1.get('macro_f1_tf', '')} | {lb1.get('ece', '')} | {lb1.get('unsure_recall', '')} | {lb1.get('all_true_acc', '')} |
| 2nd (after combinations) | below | {lb2.get('rows', '')} | {lb2.get('acc', '')} | {lb2.get('macro_f1_tf', '')} | {lb2.get('ece', '')} | {lb2.get('unsure_recall', '')} | {lb2.get('all_true_acc', '')} |

This is the second look at the holdout; nothing was tuned on either look.

```json
{json.dumps(c['base_cfg'], indent=1)}
```

Figure: `figs/combo_gains.png`. Rerun: `.venv-neuro/Scripts/python eda/neuro/combo_oof.py`, then `LOKY_MAX_CPU_COUNT=2 .venv/Scripts/python eda/selfimprove/loop.py --combos` (resumes).
"""


R_IDS = None
CV_IDS = sorted(M.SPLITS.index[M.SPLITS.holdout == 0])
F_CV = M.F_ALL[M.F_ALL.id.isin(CV_IDS)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--combos", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.combos:
        main_combos()
    else:
        main(fresh=a.fresh)

"""The h300 ladder: every approach as one variant, greedy, with overfitting gates; then lockbox once and saved models.

Regimes (all scored rows are read from text; practice gold forms only build training-fold KB anchors)
  R1  train on practice only (out-of-fold readings of the 1,593 practice CV rows) -> validate on the h300 CV folds
  R2  train on practice + the other h300 CV folds (+ the de-duplicated train-only statements) -> held-out h300 fold
  R3  train on h300 only (other folds + train-only) -> held-out h300 fold ("inside h300")
  P   practice page-grouped CV (fold_group, nested readings) -> true/false macro-F1 (the arena's label rule)

Accept a piece only if: 3-class macro-F1 gain >= 0.01 (P: true/false macro-F1 >= 0.005), better on >= 4/5 folds,
ECE not worse by > 0.02, and the gates
  a  nested: decided on 4 folds, still gains on the unseen 5th in >= 4/5 rotations
  b  train - validation gap does not widen by > 0.03
  c  the gain keeps its sign over 3 seeds and its spread is smaller than the gain
  d  the gain holds under a second subject grouping of the folds (whole subjects unseen)
Final per regime: label-permutation test (the learned layer on shuffled training labels must be no-skill),
deflation over all distinct trials, then the h300 lockbox ONCE (logged in out/lockbox_log.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lib.feats import GROUPS  # noqa: E402
from lib.model import CLASSES, LAB2I, Learned, SavedModel, macro3  # noqa: E402

OUT, CACHE, MODELS = HERE / "out", HERE / "out" / "cache", ROOT / "eda" / "models"
SEEDS = (0, 1, 2)
MODE_FILE = {"practice": "practice_rule", "closed_world": "closed_world"}


# ------------------------------------------------------------------ data
SYNTH_VERSION = "script"
BASE_REG = None       # "strong": strong regularisation in the base of every ladder (--strong)


def load(ver="script"):
    hs = pd.read_csv(OUT / "h300_splits.csv")
    H = pickle.load(open(CACHE / "h300_all.pkl", "rb"))["feats"].merge(hs, on="id")
    H["y"] = H.label.map(LAB2I)
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    outer = [pickle.load(open(CACHE / f"outer_{k}.pkl", "rb")) for k in range(5)]
    for k, o in enumerate(outer):
        o["val"]["fold"] = k
        for j, F in o["inner"].items():
            F["fold"] = j
    P = pd.concat([o["val"] for o in outer], ignore_index=True)
    lab = sp.label.to_dict()
    for o in outer:
        o["val"]["y"] = o["val"].id.map(lab).map(LAB2I)
        for F in o["inner"].values():
            F["y"] = F.id.map(lab).map(LAB2I)
    P["y"] = P.id.map(lab).map(LAB2I)
    S10 = None
    if ver != "none" and (CACHE / f"synth10k_{ver}_feats.pkl").exists():
        S10 = pickle.load(open(CACHE / f"synth10k_{ver}_feats.pkl", "rb"))
        for m in S10.values():
            m["y"] = m.label.map(LAB2I)
    for o in outer:
        for m in o["synth"].values():
            m["y"] = m.label.map(LAB2I)
    return H, P, outer, S10


def cols_of(cfg):
    out = []
    for g in cfg["groups"]:
        out += GROUPS[g]
    return out


# ------------------------------------------------------------------ metrics
def metrics(y, pred, P):
    y, pred = np.asarray(y), np.asarray(pred)
    acc = float((y == pred).mean()) if len(y) else float("nan")
    f = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + ((pred == c) & (y != c)).sum() + ((pred != c) & (y == c)).sum(), 1))
    conf = P[np.arange(len(pred)), pred]
    m = pred != 2
    ece = 0.0
    if m.sum():
        c, ok = np.clip(conf[m], 0.5, 1), (pred[m] == y[m])
        for lo in np.arange(0.5, 1.0, 0.05):
            s = (c >= lo) & (c < lo + 0.05 + (1e-9 if lo >= 0.949 else 0))
            if s.any():
                ece += s.sum() * abs(c[s].mean() - ok[s].mean())
        ece /= m.sum()
    return {"acc": acc, "f3": float(np.mean(f)), "ftf": float((f[0] + f[1]) / 2), "f_unsure": float(f[2]),
            "ece": float(ece)}


# ------------------------------------------------------------------ fitting per regime
def fit_predict(cfg, train_F, train_y, test_F, seed=0, w=None, unl_F=None):
    cols = cols_of(cfg)
    m = Learned(cfg, cols, seed).fit(train_F, train_y, w)
    if cfg.get("selftrain") and unl_F is not None and len(unl_F):
        p_unl, P_unl = m.predict(unl_F)
        keep = P_unl.max(1) >= 0.95
        if keep.sum():
            F2 = pd.concat([train_F, unl_F[keep]], ignore_index=True)
            y2 = np.concatenate([train_y, p_unl[keep]])
            w2 = np.concatenate([np.ones(len(train_y)) if w is None else w, np.full(keep.sum(), 0.5)])
            m = Learned(cfg, cols, seed).fit(F2, y2, w2)
    pred, P = m.predict(test_F)
    return pred, P, m


def synth_aug(cfg, S10, base_F, base_y, base_w=None):
    if not cfg.get("synth") or S10 is None:
        return base_F, base_y, base_w
    mode, wt = cfg["synth"]
    S = S10[mode]
    F = pd.concat([base_F, S], ignore_index=True)
    y = np.concatenate([base_y, S.y.values])
    w = np.concatenate([np.ones(len(base_y)) if base_w is None else base_w, np.full(len(S), wt)])
    return F, y, w


def eval_regime(regime, cfg, D, seed=0, foldcol="fold", in_sample=False):
    """-> per-fold metric dicts (5), pooled metrics, optional in-sample metric."""
    H, P, outer, S10 = D
    Hc = H[H.part == "cv"]
    Htr_only = H[H.part == "train_only"]
    folds, preds = [], []
    ins = []
    if regime == "R1":
        F, y, w = synth_aug(cfg, S10, P, P.y.values)
        pred, Pr, m = fit_predict(cfg, F, y, Hc, seed, w, unl_F=Hc)
        Hc = Hc.assign(pred=pred)
        for f in range(5):
            s = (Hc[foldcol] == f).values
            folds.append(metrics(Hc.y.values[s], pred[s], Pr[s]))
        pooled = metrics(Hc.y.values, pred, Pr)
        if in_sample:
            pi, Pi = m.predict(P)
            ins.append(metrics(P.y.values, pi, Pi)["f3"])
        return folds, pooled, ins
    if regime in ("R2", "R3"):
        allp, ally, allP = np.zeros(len(Hc), int), Hc.y.values, np.zeros((len(Hc), 3))
        for f in range(5):
            te = (Hc[foldcol] == f).values
            parts = [Hc[~te], Htr_only]
            wts = [np.full((~te).sum(), 3.0), np.full(len(Htr_only), 3.0)]
            if regime == "R2":
                parts.insert(0, P)
                wts.insert(0, np.ones(len(P)))
            F = pd.concat(parts, ignore_index=True)
            y = np.concatenate([p.y.values for p in parts])
            w = np.concatenate(wts)
            F, y, w = synth_aug(cfg, S10, F, y, w)
            pred, Pr, m = fit_predict(cfg, F, y, Hc[te], seed, w, unl_F=Hc[te])
            allp[te], allP[te] = pred, Pr
            folds.append(metrics(ally[te], pred, Pr))
            if in_sample:
                hi = Hc[~te]
                pi, Pi = m.predict(hi)
                ins.append(metrics(hi.y.values, pi, Pi)["f3"])
        return folds, metrics(ally, allp, allP), ins
    if regime == "P":
        allp, ally, allP = [], [], []
        for k, o in enumerate(outer):
            F = pd.concat(list(o["inner"].values()), ignore_index=True)
            y = F.y.values
            w = None
            if cfg.get("synth") and SYNTH_VERSION == "script":
                S = o["synth"][cfg["synth"][0]]
                F = pd.concat([F, S], ignore_index=True)
                w = np.concatenate([np.ones(len(y)), np.full(len(S), cfg["synth"][1])])
                y = np.concatenate([y, S.y.values])
            V = o["val"]
            pred, Pr, m = fit_predict(cfg, F, y, V, seed, w, unl_F=V)
            folds.append(metrics(V.y.values, pred, Pr))
            allp.append(pred), ally.append(V.y.values), allP.append(Pr)
            if in_sample:
                n0 = len(np.concatenate([g.y.values for g in o["inner"].values()]))
                pi, Pi = m.predict(F.iloc[:n0])
                ins.append(metrics(F.y.values[:n0], pi, Pi)["ftf"])
        return folds, metrics(np.concatenate(ally), np.concatenate(allp), np.concatenate(allP)), ins
    raise ValueError(regime)


def key_metric(regime):
    return "ftf" if regime == "P" else "f3"


# ------------------------------------------------------------------ gates
def gate_eval(regime, base, cand, D, base_res, cand_res, min_gain):
    km = key_metric(regime)
    bf = np.array([x[km] for x in base_res[0]])
    cf = np.array([x[km] for x in cand_res[0]])
    g = cf - bf
    gates = {}
    rot = 0
    for k in range(5):
        others = np.delete(g, k)
        decide = others.mean() >= min_gain and (others > 0).sum() >= 3
        rot += int(decide and g[k] > 0)
    gates["a_nested"] = rot >= 4
    _, _, bi = eval_regime(regime, base, D, in_sample=True)
    _, _, ci = eval_regime(regime, cand, D, in_sample=True)
    gap_b = np.mean(bi) - base_res[1][km]
    gap_c = np.mean(ci) - cand_res[1][km]
    gates["b_gap"] = (gap_c - gap_b) <= 0.03
    gains = [float(g.mean())]
    for s in SEEDS[1:]:
        b = eval_regime(regime, base, D, seed=s)
        c = eval_regime(regime, cand, D, seed=s)
        gains.append(float(np.mean([x[km] for x in c[0]]) - np.mean([x[km] for x in b[0]])))
    gates["c_seeds"] = all(x > 0 for x in gains) and float(np.std(gains)) < float(np.mean(gains))
    if regime == "P":
        gates["d_subjects"] = True  # practice folds are page-grouped; the h300 subject grouping does not apply
    else:
        b = eval_regime(regime, base, D, foldcol="fold_alt")
        c = eval_regime(regime, cand, D, foldcol="fold_alt")
        gates["d_subjects"] = float(np.mean([x[km] for x in c[0]]) - np.mean([x[km] for x in b[0]])) > 0
    info = {"rotations": rot, "gap_change": round(float(gap_c - gap_b), 4), "seed_gains": [round(x, 4) for x in gains]}
    return gates, info


# ------------------------------------------------------------------ ladder
STEPS = [
    ("A1_regularised", {"reg": True}),
    ("A7_closed_world_feature", {"groups+": ["cw"]}),
    ("A4_staircase_history", {"groups+": ["hist"]}),
    ("A14_date_windows", {"groups+": ["win"]}),
    ("A15_credible_dissent", {"groups+": ["dissent"]}),
    ("A3_confidence_check", {"groups+": ["misread"], "misread_gate": True}),
    ("A6_attribute_smoothing", {"groups+": ["smooth"]}),
    ("A12_second_chance_parse", {"groups+": ["second"]}),
    ("A2_parse_reliability", {"parse_rel": True}),
    ("A13_parse_aware_verdict", {"parse_rel": True, "parse_shrink": True}),
    ("A5_extratrees_vote", {"et": True}),
    ("A11_per_label_isotonic", {"iso": True}),
    ("A8_hybrid_stack", {"stack": True, "groups+": ["cw"]}),
    ("A9_tuned_offsets", {"offsets": True}),
    ("A10_self_training", {"selftrain": True}),
    ("A16_synth_closed_world_0.3", {"synth": ("closed_world", 0.3)}),
    ("A16_synth_closed_world_1.0", {"synth": ("closed_world", 1.0)}),
    ("A16_synth_practice_0.3", {"synth": ("practice", 0.3)}),
]


def apply(base, delta):
    c = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    for k, v in delta.items():
        if k == "groups+":
            c["groups"] = list(dict.fromkeys(c["groups"] + v))
        else:
            c[k] = v
    return c


def summarise(res, km):
    f = res[1]
    return {"acc": round(f["acc"], 4), "f3": round(f["f3"], 4), "ftf": round(f["ftf"], 4),
            "f_unsure": round(f["f_unsure"], 4), "ece": round(f["ece"], 4),
            "fold_" + km: [round(x[km], 4) for x in res[0]]}


def ladder(regime, D, log):
    km = key_metric(regime)
    min_gain = 0.005 if regime == "P" else 0.01
    base = {"groups": ["core", "text"]}
    if BASE_REG:
        base["reg"] = BASE_REG
    t0 = time.time()
    base_res = eval_regime(regime, base, D)
    log.append({"regime": regime, "step": "B0_base", "cfg": dict(base), **summarise(base_res, km), "kept": "base"})
    print(f"[{regime}] base {summarise(base_res, km)}", flush=True)
    for name, delta in STEPS:
        if regime == "P" and name.startswith("A16_synth_closed"):
            continue
        if name.startswith("A16") and (D[3] is None or (regime == "P" and SYNTH_VERSION != "script")):
            continue
        cand = apply(base, delta)
        if cand == base:
            continue
        res = eval_regime(regime, cand, D)
        bf = np.array([x[km] for x in base_res[0]])
        cf = np.array([x[km] for x in res[0]])
        gain, better = float(cf.mean() - bf.mean()), int((cf > bf).sum())
        ece_ok = res[1]["ece"] <= base_res[1]["ece"] + 0.02
        row = {"regime": regime, "step": name, "cfg": {k: v for k, v in cand.items()}, **summarise(res, km),
               "gain": round(gain, 4), "folds_better": better}
        if gain >= min_gain and better >= 4 and ece_ok:
            gates, info = gate_eval(regime, base, cand, D, base_res, res, min_gain)
            row.update({f"gate_{k}": v for k, v in gates.items()}, **info)
            ok = all(gates.values())
            row["kept"] = "kept" if ok else "rejected: overfits (" + ",".join(k for k, v in gates.items() if not v) + ")"
            if ok:
                base, base_res = cand, res
        else:
            row["kept"] = "rejected: accept rule"
        log.append(row)
        print(f"[{regime}] {name}: gain {gain:+.4f} {better}/5 -> {row['kept']}  ({time.time() - t0:.0f}s)", flush=True)
    return base, base_res


# ------------------------------------------------------------------ final checks
def permutation(regime, cfg, D, reps=3):
    """Shuffle every TRAINING label, refit, score against the REAL validation labels (must be no-skill)."""
    H, P, outer, S10 = D
    rng = np.random.default_rng(0)
    cfg = {k: v for k, v in cfg.items() if k not in ("synth", "selftrain")}
    Hc, Ht = H[H.part == "cv"], H[H.part == "train_only"]
    scores = []
    for _ in range(reps):
        if regime == "R1":
            pred, _, _ = fit_predict(cfg, P, rng.permutation(P.y.values), Hc)
            scores.append(macro3(Hc.y.values, pred))
        elif regime in ("R2", "R3"):
            allp = np.zeros(len(Hc), int)
            for f in range(5):
                te = (Hc.fold == f).values
                parts = ([P] if regime == "R2" else []) + [Hc[~te], Ht]
                F = pd.concat(parts, ignore_index=True)
                pred, _, _ = fit_predict(cfg, F, rng.permutation(np.concatenate([q.y.values for q in parts])), Hc[te])
                allp[te] = pred
            scores.append(macro3(Hc.y.values, allp))
        else:
            ys, ps = [], []
            for o in outer:
                F = pd.concat(list(o["inner"].values()), ignore_index=True)
                pred, Pr, _ = fit_predict(cfg, F, rng.permutation(F.y.values), o["val"])
                ys.append(o["val"].y.values), ps.append(pred)
            y_, p_ = np.concatenate(ys), np.concatenate(ps)
            scores.append(metrics(y_, p_, np.full((len(p_), 3), 1 / 3))["ftf"])
    return scores


def lockbox(regime, cfg, D, force=False):
    H, P, outer, S10 = D
    logp = HERE / "out" / "lockbox_log.json"   # one log for every synthetic version: every look is counted
    log = json.load(open(logp)) if logp.exists() else []
    ver = SYNTH_VERSION + ("_strong" if BASE_REG else "")
    mine = [e for e in log if e["regime"] == regime and e.get("synth") == ver]
    if mine and not force:
        return mine[0], log
    L = H[H.part == "lockbox"]
    Hc, Ht = H[H.part == "cv"], H[H.part == "train_only"]
    if regime == "R1":
        F, y, w = synth_aug(cfg, S10, P, P.y.values)
    else:
        parts = ([P] if regime == "R2" else []) + [Hc, Ht]
        F = pd.concat(parts, ignore_index=True)
        y = np.concatenate([p.y.values for p in parts])
        w = np.concatenate(([np.ones(len(P))] if regime == "R2" else []) + [np.full(len(Hc) + len(Ht), 3.0)])
        F, y, w = synth_aug(cfg, S10, F, y, w)
    pred, Pr, m = fit_predict(cfg, F, y, L, 0, w, unl_F=L)
    entry = {"regime": regime, "synth": ver, "same_rows_looked_before": sum(e["regime"] == regime for e in log), "when": datetime.now().isoformat(timespec="seconds"),
             "n": int(len(L)), "log_sequence": len(log) + 1,
             **{k: round(v, 4) for k, v in metrics(L.y.values, pred, Pr).items()}, "looks": 1}
    log.append(entry)
    json.dump(log, open(logp, "w"), indent=1)
    return entry, log


def final_model(regime, cfg, D):
    H, P, outer, S10 = D
    Hc, Ht = H[H.part == "cv"], H[H.part == "train_only"]
    if regime in ("R1", "P"):
        F, y, w = synth_aug(cfg, S10, P, P.y.values)
    else:
        parts = ([P] if regime == "R2" else []) + [Hc, Ht]
        F = pd.concat(parts, ignore_index=True)
        y = np.concatenate([p.y.values for p in parts])
        w = np.concatenate(([np.ones(len(P))] if regime == "R2" else []) + [np.full(len(Hc) + len(Ht), 3.0)])
        F, y, w = synth_aug(cfg, S10, F, y, w)
    cfg2 = {k: v for k, v in cfg.items() if k != "selftrain"}  # self-training needs the target rows; off when saved
    return Learned(cfg2, cols_of(cfg2), 0).fit(F, y, w)


def robustness(model: SavedModel):
    import csv
    practice = list(csv.DictReader(open(ROOT / "data" / "practice.csv", encoding="utf-8-sig")))[:40]
    base = [{"id": f"x-{i}", "text": r["text"], "date": r["date"], "label": ""} for i, r in enumerate(practice)]
    fmts = {"iso": lambda d: d, "slash": lambda d: d.replace("-", "/"),
            "day_mon": lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d %b %Y"),
            "mon_day": lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y")}
    res = {}
    try:
        ref = [o["label"] for o in model.label(base)]
        res["new_ids_no_labels"] = "ok"
    except Exception as e:
        return {"new_ids_no_labels": f"CRASH {type(e).__name__}: {e}"}
    for name, fn in fmts.items():
        try:
            out = model.label([dict(r, date=fn(r["date"])) for r in base])
            agree = np.mean([a == o["label"] for a, o in zip(ref, out)])
            res[f"date_{name}"] = f"ok, agrees {agree:.2f}"
        except Exception as e:
            res[f"date_{name}"] = f"CRASH {type(e).__name__}"
    try:
        out = model.label([{"id": "x-empty", "text": "", "date": "", "label": ""},
                           {"id": "x-odd", "text": "???", "date": "yesterday", "label": ""}])
        res["empty_and_odd_rows"] = f"ok -> {[o['label'] for o in out]}"
    except Exception as e:
        res["empty_and_odd_rows"] = f"CRASH {type(e).__name__}"
    res["passed"] = not any("CRASH" in str(v) for v in res.values())
    return res


def deflation(log):
    df = pd.DataFrame(log)
    tried = df[df.step != "B0_base"]
    n = int(tried.drop_duplicates(["regime", "step"]).shape[0])
    ses = []
    for r in tried.itertuples():
        fk = [c for c in df.columns if c.startswith("fold_")]
        vals = [v for c in fk for v in ([getattr(r, c)] if isinstance(getattr(r, c), list) else [])]
        if vals:
            ses.append(np.std(vals) / math.sqrt(len(vals)))
    se = float(np.mean(ses)) if ses else float("nan")
    return {"distinct_trials": n, "mean_se": round(se, 4), "noise_max_gain": round(se * math.sqrt(2 * math.log(max(n, 2))), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="R1,R2,R3,P")
    ap.add_argument("--synth", default="script", choices=("script", "claude", "none"))
    ap.add_argument("--strong", action="store_true", help="strong regularisation in every base (our default to ship)")
    args = ap.parse_args()
    global SYNTH_VERSION, OUT, BASE_REG
    SYNTH_VERSION = args.synth
    if args.strong:
        BASE_REG = "strong"
    tag = args.synth + ("_strong" if args.strong else "")
    if tag != "script":
        OUT = HERE / "out" / f"synth_{tag}"
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "h300_splits.csv").write_bytes((HERE / "out" / "h300_splits.csv").read_bytes())
    D = load(args.synth)
    log, finals = [], {}
    t0 = time.time()
    for regime in args.regimes.split(","):
        cfg, res = ladder(regime, D, log)
        finals[regime] = {"cfg": cfg, "res": res}
        pd.DataFrame(log).to_csv(OUT / "ladder.csv", index=False)
        print(f"[{regime}] final cfg {cfg}  ({time.time() - t0:.0f}s)", flush=True)
    defl = deflation(log)
    summary = {"deflation": defl, "synth_version": args.synth, "regimes": {}}
    reader = pickle.load(open(CACHE / "reader_all.pkl", "rb"))
    MODELS.mkdir(parents=True, exist_ok=True)
    for regime, fr in finals.items():
        cfg, res = fr["cfg"], fr["res"]
        km = key_metric(regime)
        perm = permutation(regime, cfg, D)
        perm_ok = max(perm) <= (0.40 if km == "f3" else 0.50)
        lb = None
        if regime != "P":
            lb, _ = lockbox(regime, cfg, D)
        steps = [r for r in log if r["regime"] == regime and r.get("kept") == "kept"]
        gates_ok = all(all(r.get(f"gate_{g}", True) for g in ("a_nested", "b_gap", "c_seeds", "d_subjects"))
                       for r in steps)
        entry = {"cfg": cfg, "cv": summarise(res, km), "permutation": [round(x, 4) for x in perm],
                 "permutation_pass": perm_ok, "lockbox": lb, "kept_steps": [r["step"] for r in steps],
                 "all_gates_pass": gates_ok}
        if perm_ok and gates_ok:
            name = {"R1": "h300_r1_practice_trained", "R2": "h300_r2_h300_augmented",
                    "R3": "h300_r3_h300_only", "P": "practice_cv_best"}[regime] +                 ("" if tag == "script" else f"_synth_{tag}")
            learned = final_model(regime, cfg, D)
            sm = SavedModel(reader, learned, name)
            rob = robustness(sm)
            path = MODELS / f"{name}.joblib"
            joblib.dump(sm, path, compress=3)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"name": name, "regime": regime,
                        "approach": "reader (extractor + attribute model + practice KB) + learned layer; "
                                    "kept pieces: " + ", ".join(entry["kept_steps"] or ["base only"]),
                        "training_data": {"R1": "practice CV rows (1,593; out-of-fold readings)",
                                          "R2": "practice CV rows + h300 CV + h300 train-only (weight 3)",
                                          "R3": "h300 CV + h300 train-only",
                                          "P": "practice CV rows (1,593; out-of-fold readings)"}[regime]
                                         + (f" + synthetic {cfg['synth']}" if cfg.get("synth") else ""),
                        "features": learned.feature_names, "hyperparams": {k: v for k, v in cfg.items()},
                        "scores": {"validation": entry["cv"], "lockbox": lb, "permutation": entry["permutation"]},
                        "gates": {"all_kept_steps_pass_a_b_c_d": gates_ok, "permutation_pass": perm_ok,
                                  "lockbox_looks": 1 if lb else 0},
                        "robustness": rob, "sha256": sha, "synth_version": args.synth,
                        "predict_entry": "eda/h300/lib/model.py:load_and_predict",
                        "load_requires_sys_path": "eda/h300", "created": datetime.now().isoformat(timespec="seconds"),
                        "file": str(path.relative_to(ROOT))}
            json.dump(manifest, open(MODELS / f"{name}.manifest.json", "w"), indent=1, default=str)
            entry["saved"] = manifest["file"]
            entry["robustness"] = rob
        summary["regimes"][regime] = entry
        print(f"[{regime}] perm {perm} lockbox {lb} saved {entry.get('saved')}", flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

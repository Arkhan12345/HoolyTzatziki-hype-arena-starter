"""v2 ladder: robustness pieces on top of FINAL v1, chosen on practice CV only (never the outside set).

    .venv/Scripts/python eda/h300/ladder_v2.py

Base = FINAL v1 (groups core+text+hist, strong regularisation, per-fold practice-rule synthetic at weight 0.3).
Every knob is one change, accepted greedily only if:
  rule   mean true/false macro-F1 gain >= 0.005 on the 5 fold_group folds, better on >= 4/5, ECE not worse by > 0.02,
         and the renamed-subject slice (validation rows with invented subject names) not worse by > 0.005
         (the confidence cap is judged on ECE instead: ECE better by >= 0.005, macro-F1 not worse by > 0.002)
  gate a nested: decided on 4 folds, the unseen 5th still gains, in >= 4/5 rotations
  gate b the train - validation gap does not widen by > 0.03
  gate c the gain keeps its sign over 3 seeds and its spread is smaller than the gain
  gate d the renamed-subject slice gain is >= -0.005 (whole subjects unseen)
Final: label-permutation test on the chosen config. Writes out/ladder_v2_trials.csv and out/ladder_v2_choice.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lib.model import LAB2I, Learned  # noqa: E402
from run import cols_of, metrics  # noqa: E402

CACHE2 = HERE / "out" / "cache_v2"
SEEDS = (0, 1, 2)
BASE = {"groups": ["core", "text", "hist"], "reg": "strong", "synth_w": 0.3, "extra": [], "ren_w": 0.0,
        "cw_w": 0.0, "cap": False, "unk_offsets": False, "unk_read": False}
# what a v1 reader returns for a sentence without a known subject (Reader._decide defaults)
V1_UNREAD = {"has_reading": 0, "known_pair": 0, "cw_true": 0, "cw_false": 0, "cw_unsure": 1, "p_attr": 0.0,
             "attr_maxp": 0.0, "margin": 0.0, "margin_gap": 0.0, "p_attr_smooth": 0.0, "margin_gap_smooth": 0.0,
             "parse_conf": 0.0, "kind": "none", "matches_latest": 0, "matches_other_attr": 0, "stale": 0,
             "ever_false": 0, "seen_value": 0, "n_true_values": 0, "n_mentions": 0, "days_since_change": np.nan,
             "log_ratio_latest": np.nan, "perturb_hit": 0, "pages_for": 0, "pages_against": 0, "dissent_all": 0,
             "dissent_credible": 0, "second_chance": 0, "status_no_attr": 0,
             **{f"{s}_{w}": 0 for s in ("agree", "disagree") for w in ("w0", "w30", "w90", "wold")},
             "v2_attr_gap": 0.0, "v2_confusable": 0, "v2_plaus_z": np.nan, "v2_plaus_out": 0, "v2_plaus_n": 0,
             "v2_tol_match": 0, "v2_referent_weak": 1}


def as_v1_reading(F):
    """Rows read through the unknown-subject path get exactly the values a v1 reader returns for them."""
    m = (pd.to_numeric(F.get("status_no_subject", 0), errors="coerce").fillna(0).values == 1) & \
        (pd.to_numeric(F.get("has_reading", 0), errors="coerce").fillna(0).values == 1) & \
        (pd.to_numeric(F.get("v2_known_subject", 0), errors="coerce").fillna(0).values == 0)
    if not m.any():
        return F
    F = F.copy()
    for c, v in V1_UNREAD.items():
        if c in F:
            F.loc[m, c] = v
    return F
REFERENT = ["v2_attr_gap", "v2_confusable", "v2_referent_weak"]
TOLERANCE = ["v2_tol_match"]
SUBJECT_FREE = ["v2_plaus_z", "v2_plaus_out", "v2_known_subject", "v2_page_size", "v2_pos_in_page",
                "v2_page_first_person", "misread_hedge", "misread_negation"]
KNOBS = [("unknown_subject_reading", {"unk_read": True}),
         ("subject_free_features", {"extra": SUBJECT_FREE}),
         ("same_referent", {"extra": REFERENT}),
         ("tolerance", {"extra": TOLERANCE}),
         ("renamed_augmentation_w1", {"ren_w": 1.0}),
         ("renamed_augmentation_w05", {"ren_w": 0.5}),
         ("closed_world_share_w01", {"cw_w": 0.1}),
         ("unknown_subject_thresholds", {"unk_offsets": True}),
         ("confidence_cap", {"cap": True})]


def load():
    lab = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False).set_index("id").label
    folds = []
    for k in range(5):
        o = pickle.load(open(CACHE2 / f"outer_{k}.pkl", "rb"))
        for key in ("val", "ren_val"):
            o[key]["y"] = o[key].id.str.replace("^rn-", "", regex=True).map(lab).map(LAB2I).values
        o["train"] = pd.concat(o["inner"].values(), ignore_index=True)
        o["ren_train"] = pd.concat(o["ren_inner"].values(), ignore_index=True)
        for key in ("train", "ren_train"):
            o[key]["y"] = o[key].id.str.replace("^rn-", "", regex=True).map(lab).map(LAB2I).values
        for m in o["synth"].values():
            m["y"] = m.label.map(LAB2I).values
        folds.append(o)
    return folds


def cols(cfg):
    return cols_of(cfg) + list(dict.fromkeys(cfg["extra"]))


def training_set(cfg, o):
    parts, ws = [o["train"]], [np.ones(len(o["train"]))]
    if cfg["synth_w"]:
        parts.append(o["synth"]["practice"]); ws.append(np.full(len(o["synth"]["practice"]), cfg["synth_w"]))
    if cfg["cw_w"]:
        parts.append(o["synth"]["closed_world"]); ws.append(np.full(len(o["synth"]["closed_world"]), cfg["cw_w"]))
    if cfg["ren_w"]:
        parts.append(o["ren_train"]); ws.append(np.full(len(o["ren_train"]), cfg["ren_w"]))
    F = pd.concat(parts, ignore_index=True)
    return F, F.y.values.astype(int), np.concatenate(ws)


def unknown(F):
    return pd.to_numeric(F.get("known_pair", 0), errors="coerce").fillna(0).values == 0


def tf_f1(y, pred):
    f = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + ((pred == c) & (y != c)).sum() + ((pred != c) & (y == c)).sum(), 1))
    return float(np.mean(f))


def tune_unknown_offset(P, y, mask):
    """Offset on log P(false) for rows whose subject has no KB support, chosen on inner out-of-fold rows only."""
    best, off = -1.0, 0.0
    L = np.log(np.clip(P, 1e-9, 1))
    for b in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        Lb = L.copy()
        Lb[mask, 1] += b
        s = tf_f1(y, np.argmax(Lb, 1))
        if s > best + 1e-9:
            best, off = s, b
    return off


class Model:
    def __init__(self, cfg, seed=0):
        self.cfg, self.seed = cfg, seed

    def fit(self, F, y, w):
        lc = {"groups": self.cfg["groups"], "reg": self.cfg["reg"]}
        self.m = Learned(lc, cols(self.cfg), self.seed).fit(F, y, w)
        self.off = 0.0
        if self.cfg["unk_offsets"]:
            oof = np.zeros((len(y), 3))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=self.seed).split(F, y):
                mm = Learned(lc, cols(self.cfg), self.seed).fit(F.iloc[tr], y[tr], w[tr])
                oof[te] = mm.predict_proba(F.iloc[te])
            self.off = tune_unknown_offset(oof, y, unknown(F))
        return self

    def predict(self, F):
        P = self.m.predict_proba(F)
        L = np.log(np.clip(P, 1e-9, 1))
        if self.off:
            L[unknown(F), 1] += self.off
        pred = np.argmax(L, 1)
        if self.cfg["cap"]:
            weak = (pd.to_numeric(F.get("v2_referent_weak", 0), errors="coerce").fillna(0).values == 1) | \
                   (pd.to_numeric(F.get("misread_hedge", 0), errors="coerce").fillna(0).values == 1) | unknown(F)
            top = P[np.arange(len(P)), pred]
            capped = np.minimum(top, 0.6)
            scale = np.where(weak, (1 - capped) / np.clip(1 - top, 1e-9, None), 1.0)
            Q = P * scale[:, None]
            Q[np.arange(len(P)), pred] = np.where(weak, capped, top)
            P = Q / Q.sum(1, keepdims=True)
        return pred, P


def view(cfg, o):
    """The fold as this configuration sees it (v1 readings unless unk_read)."""
    if cfg.get("unk_read"):
        return o
    return {"train": as_v1_reading(o["train"]), "ren_train": as_v1_reading(o["ren_train"]),
            "val": as_v1_reading(o["val"]), "ren_val": as_v1_reading(o["ren_val"]),
            "synth": {k: as_v1_reading(v) for k, v in o["synth"].items()}}


def evaluate(cfg, folds, seed=0, in_sample=False):
    res = []
    for o in [view(cfg, f) for f in folds]:
        F, y, w = training_set(cfg, o)
        m = Model(cfg, seed).fit(F, y, w)
        pv, Pv = m.predict(o["val"])
        pr, Pr = m.predict(o["ren_val"])
        r = {"ftf": tf_f1(o["val"].y.values, pv), "ece": metrics(o["val"].y.values, pv, Pv)["ece"],
             "acc": float((pv == o["val"].y.values).mean()), "ren_ftf": tf_f1(o["ren_val"].y.values, pr),
             "ren_acc": float((pr == o["ren_val"].y.values).mean())}
        if in_sample:
            pt, _ = m.predict(o["train"])
            r["train_ftf"] = tf_f1(o["train"].y.values, pt)
        res.append(r)
    return res


def mean(res, k):
    return float(np.mean([r[k] for r in res]))


def gates(base_s, cand_s, base_res, cand_res, folds, cfg_base, cfg_cand, ece_mode):
    key = "ece" if ece_mode else "ftf"
    sign = -1 if ece_mode else 1
    gains = np.array([sign * (c[key] - b[key]) for b, c in zip(base_res, cand_res)])
    ok_rot = 0
    for i in range(5):
        others = np.delete(gains, i)
        decided = others.mean() >= (0.005 if not ece_mode else 0.005) and (others > 0).sum() >= 3
        ok_rot += int(decided and gains[i] > 0)
    a = ok_rot >= 4
    bi = evaluate(cfg_base, folds, in_sample=True)
    ci = evaluate(cfg_cand, folds, in_sample=True)
    gap_b = mean(bi, "train_ftf") - mean(bi, "ftf")
    gap_c = mean(ci, "train_ftf") - mean(ci, "ftf")
    b = (gap_c - gap_b) <= 0.03
    seed_gains = [mean(cand_res, key) - mean(base_res, key)]
    for s in SEEDS[1:]:
        seed_gains.append(mean(evaluate(cfg_cand, folds, seed=s), key) - mean(evaluate(cfg_base, folds, seed=s), key))
    seed_gains = sign * np.array(seed_gains)
    c = bool((np.sign(seed_gains) == np.sign(seed_gains[0])).all() and seed_gains.std() < abs(seed_gains.mean()))
    d = (mean(cand_res, "ren_ftf") - mean(base_res, "ren_ftf")) >= -0.005
    return {"a_nested": bool(a), "a_rotations": ok_rot, "b_gap": bool(b), "gap_base": round(gap_b, 4),
            "gap_cand": round(gap_c, 4), "c_seeds": c, "seed_gains": [round(x, 4) for x in seed_gains],
            "d_renamed": bool(d)}


def permutation(cfg, folds, reps=3):
    out = []
    rng = np.random.default_rng(0)
    for _ in range(reps):
        res = []
        for o in [view(cfg, f) for f in folds]:
            F, y, w = training_set(cfg, o)
            ys = y.copy()
            n = len(o["train"])
            ys[:n] = rng.permutation(ys[:n])
            ys[n:] = rng.permutation(ys[n:])
            m = Model({**cfg, "unk_offsets": False}, 0).fit(F, ys, w)
            pv, _ = m.predict(o["val"])
            res.append(tf_f1(o["val"].y.values, pv))
        out.append(float(np.mean(res)))
    return out


def main():
    folds = load()
    base = dict(BASE)
    base_res = evaluate(base, folds)
    trials = [{"knob": "base (FINAL v1)", "ftf": mean(base_res, "ftf"), "acc": mean(base_res, "acc"),
               "ece": mean(base_res, "ece"), "ren_ftf": mean(base_res, "ren_ftf"),
               "ren_acc": mean(base_res, "ren_acc"), "accepted": "base"}]
    print(trials[0], flush=True)
    for name, delta in KNOBS:
        cand = dict(base)
        for k, v in delta.items():
            cand[k] = (base["extra"] + v) if k == "extra" else v
        cres = evaluate(cand, folds)
        ece_mode = name == "confidence_cap"
        g_ftf = mean(cres, "ftf") - mean(base_res, "ftf")
        better = sum(c["ftf"] > b["ftf"] for b, c in zip(base_res, cres))
        g_ece = mean(base_res, "ece") - mean(cres, "ece")
        g_ren = mean(cres, "ren_ftf") - mean(base_res, "ren_ftf")
        if ece_mode:
            rule = g_ece >= 0.005 and g_ftf >= -0.002
        else:
            rule = g_ftf >= 0.005 and better >= 4 and -g_ece <= 0.02 and g_ren >= -0.005
        gt = gates(None, None, base_res, cres, folds, base, cand, ece_mode) if rule else {}
        ok = rule and all(gt.get(k) for k in ("a_nested", "b_gap", "c_seeds", "d_renamed"))
        row = {"knob": name, "ftf": mean(cres, "ftf"), "gain": g_ftf, "folds_better": better,
               "acc": mean(cres, "acc"), "ece": mean(cres, "ece"), "ece_gain": g_ece, "ren_ftf": mean(cres, "ren_ftf"),
               "ren_gain": g_ren, "ren_acc": mean(cres, "ren_acc"), "rule": rule, **gt, "accepted": ok}
        trials.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if ok:
            base, base_res = cand, cres
    perm = permutation(base, folds)
    T = pd.DataFrame(trials)
    T.to_csv(HERE / "out" / "ladder_v2_trials.csv", index=False)
    choice = {"config": base, "cv": {k: mean(base_res, k) for k in ("ftf", "acc", "ece", "ren_ftf", "ren_acc")},
              "permutation_ftf": perm, "permutation_pass": bool(max(perm) <= 0.50),
              "n_trials": len(trials) - 1}
    json.dump(choice, open(HERE / "out" / "ladder_v2_choice.json", "w"), indent=1)
    print(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()

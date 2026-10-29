"""Freeze the shipped model (FINAL) and a FALLBACK after the ladder chose a configuration. Fits once, then never again.

    python eda/h300/freeze.py [--synth script|claude|none]

1. Generalisation checks on the configuration the practice ladder kept (P regime, the arena's label rule):
     learning curve still rising or flat at 100% of training pages · held-out subjects >= all-true ·
     time-forward (train < 2026-07-01, test after) >= all-true · train - validation gap <= 0.10 ·
     seed spread <= 0.01 · permutation (shuffled training labels) <= 0.50 · and it is not worse than the fallback on
     the synthetic relation cases (chain, update, stale, perturbed) of the per-fold validation sets.
   If any check fails, FINAL = the fallback (base reading + strong regularisation + confidence check), provided the
   fallback passes; the report says which.
2. The practice holdout pages get their single look (the chosen configuration trained on CV rows only).
3. FINAL and FALLBACK are refitted ONCE on every labelled practice row (CV + holdout, all out-of-fold readings) +
   the chosen synthetic weight, with a reader fitted on all practice rows, and saved to eda/models/FINAL|FALLBACK/
   (model.joblib + manifest.json). Inference: eda/h300/predict_frozen.py, one-time scoring: eda/h300/validate_once.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run  # noqa: E402
from lib.extractor import Lexicon  # noqa: E402
from lib.feats import Reader  # noqa: E402
from lib.model import LAB2I, Learned, SavedModel  # noqa: E402

FALLBACK = {"groups": ["core", "text", "misread"], "reg": "strong", "misread_gate": True}
# Strong regularisation (100 trees, 7 leaves, >= 50 rows per leaf, L2 20) was picked among 6 settings on the practice CV
# by the train - validation gap (<= 0.10); it is applied to the ladder's choice before the checks.
RELATION_CASES = ("chain", "update", "stale", "perturbed")


def ftf(y, pred):
    return run.metrics(y, pred, np.full((len(pred), 3), 1 / 3))["ftf"]


def all_true_ftf(y):
    return ftf(y, np.zeros(len(y), int))


def checks(cfg, D, fallback_case_acc=None):
    H, P, outer, S10 = D
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False).set_index("id")
    res = {}
    # learning curve (pages of the training folds)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        ys, ps = [], []
        for o in outer:
            F = pd.concat(list(o["inner"].values()), ignore_index=True)
            pages = F.id.map(sp.page)
            keep_pages = pd.Series(pages.unique()).sample(frac=frac, random_state=0)
            m = pages.isin(set(keep_pages)).values
            pred, _, _ = run.fit_predict(cfg, F[m], F.y.values[m], o["val"])
            ys.append(o["val"].y.values), ps.append(pred)
        curve.append(round(ftf(np.concatenate(ys), np.concatenate(ps)), 4))
    res["learning_curve"] = {"values": curve, "pass": curve[-1] >= curve[-2] - 0.005}
    # held-out subjects (whole subjects unseen by the learned layer)
    subj = P.id.map(lambda i: (json.loads(prac.loc[i, "form"])[0]["subject"]
                               if prac.loc[i, "form"] else "none"))
    pred = np.zeros(len(P), int)
    for tr, te in GroupKFold(5).split(P, groups=subj):
        p, _, _ = run.fit_predict(cfg, P.iloc[tr], P.y.values[tr], P.iloc[te])
        pred[te] = p
    s, base = ftf(P.y.values, pred), all_true_ftf(P.y.values)
    res["held_out_subjects"] = {"ftf": round(s, 4), "all_true": round(base, 4), "pass": s >= base}
    # time-forward
    d = P.id.map(prac.date)
    tr, te = (d < "2026-07-01").values, (d >= "2026-07-01").values
    p, _, _ = run.fit_predict(cfg, P[tr], P.y.values[tr], P[te])
    s, base = ftf(P.y.values[te], p), all_true_ftf(P.y.values[te])
    res["time_forward"] = {"ftf": round(s, 4), "all_true": round(base, 4), "pass": s >= base}
    # train - validation gap, seeds
    folds, pooled, ins = run.eval_regime("P", cfg, D, in_sample=True)
    gap = float(np.mean(ins) - pooled["ftf"])
    res["train_validation_difference"] = {"gap": round(gap, 4), "pass": gap <= 0.10}
    seeds = [run.eval_regime("P", cfg, D, seed=s)[1]["ftf"] for s in run.SEEDS]
    res["seed_spread"] = {"values": [round(x, 4) for x in seeds], "std": round(float(np.std(seeds)), 4),
                          "pass": float(np.std(seeds)) <= 0.01}
    perm = run.permutation("P", cfg, D)
    res["permutation"] = {"values": [round(x, 4) for x in perm], "pass": max(perm) <= 0.50}
    # synthetic per case_type (practice-rule validation sets of every fold)
    rows = []
    for k, o in enumerate(outer):
        F = pd.concat(list(o["inner"].values()), ignore_index=True)
        S = o["synth"]["practice"]
        pred, _, _ = run.fit_predict(cfg, F, F.y.values, S)
        rows.append(pd.DataFrame({"case_type": S.case_type.values, "ok": pred == S.y.values}))
    acc = pd.concat(rows).groupby("case_type").ok.mean().round(4).to_dict()
    rel = float(np.mean([acc.get(c, np.nan) for c in RELATION_CASES if c in acc]))
    res["synthetic_case_accuracy"] = {"by_case": acc, "relation_mean": round(rel, 4)}
    if fallback_case_acc is not None:
        res["synthetic_case_accuracy"]["pass"] = rel >= fallback_case_acc - 0.01
    res["all_pass"] = all(v.get("pass", True) for v in res.values() if isinstance(v, dict))
    return res


def holdout_rows(D):
    """Practice holdout rows read by the reader fitted on CV rows only (they never trained it)."""
    p = run.CACHE / "holdout_feats.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False)
    hold = prac[prac.id.map(sp.holdout) == 1]
    rd = pickle.load(open(run.CACHE / "reader_all.pkl", "rb"))
    F = pd.DataFrame(rd.read([{"id": r.id, "text": r.text, "date": r.date, "label": ""} for r in hold.itertuples()]))
    F.insert(0, "id", hold.id.values)
    F["y"] = hold.label.map(LAB2I).values
    pickle.dump(F, open(p, "wb"))
    return F


def full_reader():
    p = run.CACHE / "reader_full.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False).to_dict("records")
    rd = Reader.fit(prac, Lexicon.load(ROOT / "data" / "lexicon.json"))
    pickle.dump(rd, open(p, "wb"))
    return rd


def freeze(name, cfg, D, gate_info, check_info, scores, synth_version):
    H, P, outer, S10 = D
    HF = holdout_rows(D)
    F = pd.concat([P, HF], ignore_index=True)
    y = F.y.values.astype(int)
    F, y, w = run.synth_aug(cfg, S10, F, y)
    cfg2 = {k: v for k, v in cfg.items() if k != "selftrain"}
    learned = Learned(cfg2, run.cols_of(cfg2), 0).fit(F, y, w)
    sm = SavedModel(full_reader(), learned, name)
    d = ROOT / "eda" / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(sm, d / "model.joblib", compress=3)
    sha = hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest()
    rob = run.robustness(sm)
    man = {"name": name, "config": cfg, "training_rows": {"practice_labelled": int(len(P) + len(HF)),
                                                          "synthetic": int(len(y) - len(P) - len(HF))},
           "synth_version": synth_version if cfg.get("synth") else None,
           "synth_weight": cfg["synth"][1] if cfg.get("synth") else 0, "seeds": [0],
           "features": learned.feature_names, "gates": gate_info, "generalisation_checks": check_info,
           "scores": scores, "robustness": rob, "sha256": sha, "created": datetime.now().isoformat(timespec="seconds"),
           "predict_entry": "eda/h300/lib/model.py:load_and_predict", "load_requires_sys_path": "eda/h300",
           "frozen_inference": "python eda/h300/predict_frozen.py INPUT.csv --model eda/models/" + name,
           "reader": "extractor + wide extractor + attribute model + KB fitted on all 1,897 practice rows"}
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="script", choices=("script", "claude", "none"))
    a = ap.parse_args()
    run.SYNTH_VERSION = a.synth
    D = run.load(a.synth)
    summ = json.load(open(run.HERE / "out" / ("summary.json" if a.synth == "script" else f"synth_{a.synth}/summary.json")))
    chosen = dict(summ["regimes"]["P"]["cfg"], reg="strong")
    if isinstance(chosen.get("synth"), list):
        chosen["synth"] = tuple(chosen["synth"])
    fb = checks(FALLBACK, D)
    ch = checks(chosen, D, fallback_case_acc=fb["synthetic_case_accuracy"]["relation_mean"])
    use_chosen = ch["all_pass"] and summ["regimes"]["P"].get("all_gates_pass", False)
    final_cfg = chosen if use_chosen else FALLBACK
    # practice holdout: its single look, for the configuration we freeze (trained on CV rows only)
    HF = holdout_rows(D)
    H, P, outer, S10 = D
    out = {}
    for nm, cfg in (("FINAL", final_cfg), ("FALLBACK", FALLBACK)):
        Ft, yt, wt = run.synth_aug(cfg, S10, P, P.y.values)
        pred, Pr, _ = run.fit_predict(cfg, Ft, yt, HF, 0, wt)
        hold = {k: round(v, 4) for k, v in run.metrics(HF.y.values, pred, Pr).items()}
        hold["all_true_acc"] = round(float((HF.y.values == 0).mean()), 4)
        folds, pooled, _ = run.eval_regime("P", cfg, D)
        h300 = run.eval_regime("R1", cfg, D)[1]
        scores = {"practice_cv": {k: round(v, 4) for k, v in pooled.items()},
                  "practice_holdout_single_look": hold,
                  "h300_cv_practice_trained": {k: round(v, 4) for k, v in h300.items()}}
        info = ch if nm == "FINAL" and use_chosen else fb
        gates = summ["regimes"]["P"] if nm == "FINAL" and use_chosen else {"note": "fallback: no ladder pieces"}
        out[nm] = freeze(nm, cfg, D, {k: gates.get(k) for k in ("kept_steps", "all_gates_pass", "permutation",
                                                                  "permutation_pass", "note") if k in gates},
                         info, scores, a.synth)
    report = {"chosen_by_ladder": chosen, "chosen_checks": ch, "fallback_checks": fb, "frozen_final_is":
              "ladder choice" if use_chosen else "fallback (ladder choice failed a check)",
              "FINAL": {k: out["FINAL"][k] for k in ("config", "scores", "sha256", "created")},
              "FALLBACK": {k: out["FALLBACK"][k] for k in ("config", "scores", "sha256", "created")}}
    (run.HERE / "out" / "freeze_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()

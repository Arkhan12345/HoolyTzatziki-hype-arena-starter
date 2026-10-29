"""How well do models do on the Claude-written synthetic set, and does it transfer to real rows?

    .venv/Scripts/python eda/h300/claude_cv.py

Four experiments per labelling mode (practice rule, closed world), all with FINAL's feature groups:
  E1  10-fold CV inside the Claude set, grouped by subject (no subject in train and test)
  E2  70 / 15 / 15 train / validation / test split, grouped by subject: the validation part picks between the
      FINAL and FALLBACK configurations, the test part is scored once
  E3  trained on real practice rows (+ script synthetic, as FINAL) -> scored on every Claude row (never seen)
  E4  trained on the Claude set only -> scored on real practice rows and on h300
Facts are read from the text by the all-practice reader (the same extraction the frozen models use), never from `form`.
Writes eda/h300/out/claude_cv.csv and eda/h300/out/claude_cv_case_types.csv.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import feats_df, load_csv  # noqa: E402
from lib.model import LAB2I  # noqa: E402
from run import fit_predict, load, metrics, synth_aug  # noqa: E402

OUT = HERE / "out"
CACHE = OUT / "cache"
SEED = 0
FINAL_CFG = {"groups": ["core", "text", "hist"], "reg": "strong"}
FALLBACK_CFG = {"groups": ["core", "text", "misread"], "reg": "strong", "misread_gate": True}
MODES = {"practice": "practice_rule", "closed_world": "closed_world"}


def claude_feats():
    p = CACHE / "synth10k_claude_full_feats.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    rd = pickle.load(open(CACHE / "reader_all.pkl", "rb"))
    out = {}
    for mode, fname in MODES.items():
        rows = load_csv(OUT / f"synth10k_claude_{fname}.csv")
        blind = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]
        F = feats_df(rd, blind)
        meta = pd.DataFrame(rows)[["id", "label", "case_type", "form"]]
        meta["subject"] = meta.form.map(lambda s: json.loads(s)[0]["subject"] if s else "")
        F = F.merge(meta, on="id")
        F["y"] = F.label.map(LAB2I)
        out[mode] = F
    pickle.dump(out, open(p, "wb"))
    return out


def score(y, pred, P, name, mode, exp, n_train):
    m = metrics(y, pred, P)
    m.update({"experiment": exp, "mode": mode, "model": name, "n_test": len(y), "n_train": n_train,
              "all_true_acc": float((np.asarray(y) == LAB2I["true"]).mean())})
    return m


def main():
    C = claude_feats()
    H, P, outer, S10 = load("script")
    rows, cases = [], []
    for mode, F in C.items():
        y, g = F.y.values, F.subject.values
        # E1: 10-fold grouped CV inside the Claude set
        pred_all, P_all = np.zeros(len(F), int), np.zeros((len(F), 3))
        for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(F, y, groups=g):
            pr, Pr, _ = fit_predict(FINAL_CFG, F.iloc[tr], y[tr], F.iloc[te], seed=SEED)
            pred_all[te], P_all[te] = pr, Pr
        rows.append(score(y, pred_all, P_all, "FINAL cfg", mode, "E1 10-fold CV (grouped by subject)", len(F) * 0.9))
        ct = F.assign(ok=(pred_all == y)).groupby("case_type").ok.agg(["mean", "size"]).reset_index()
        cases += [{"mode": mode, "case_type": r.case_type, "acc": round(r["mean"], 3), "rows": int(r["size"])}
                  for _, r in ct.iterrows()]
        # E2: 70/15/15 grouped split; validation picks the config, test scored once
        sg = list(StratifiedGroupKFold(20, shuffle=True, random_state=SEED + 1).split(F, y, groups=g))
        te_idx = np.concatenate([s[1] for s in sg[:3]])
        va_idx = np.concatenate([s[1] for s in sg[3:6]])
        tr_idx = np.setdiff1d(np.arange(len(F)), np.concatenate([te_idx, va_idx]))
        val = {}
        for nm, cfg in (("FINAL cfg", FINAL_CFG), ("FALLBACK cfg", FALLBACK_CFG)):
            pr, Pr, _ = fit_predict(cfg, F.iloc[tr_idx], y[tr_idx], F.iloc[va_idx], seed=SEED)
            val[nm] = (metrics(y[va_idx], pr, Pr)["f3"], cfg)
            rows.append(score(y[va_idx], pr, Pr, nm, mode, "E2 validation part", len(tr_idx)))
        best = max(val, key=lambda k: val[k][0])
        pr, Pr, _ = fit_predict(val[best][1], F.iloc[tr_idx], y[tr_idx], F.iloc[te_idx], seed=SEED)
        rows.append(score(y[te_idx], pr, Pr, f"{best} (picked on validation)", mode, "E2 test part, scored once",
                          len(tr_idx)))
        # E3: trained on real practice (+ script synthetic at 0.3, as FINAL) -> every Claude row
        tF, ty, tw = synth_aug({**FINAL_CFG, "synth": ["practice", 0.3]}, S10, P, P.y.values)
        pr, Pr, _ = fit_predict(FINAL_CFG, tF, ty, F, seed=SEED, w=tw)
        rows.append(score(y, pr, Pr, "FINAL cfg trained on practice", mode, "E3 practice -> Claude rows", len(tF)))
        # E4: trained on the Claude set only -> real practice rows and h300
        pr, Pr, _ = fit_predict(FINAL_CFG, F, y, P, seed=SEED)
        rows.append(score(P.y.values, pr, Pr, "FINAL cfg trained on Claude rows", mode,
                          "E4 Claude -> real practice (out-of-fold readings)", len(F)))
        pr, Pr, _ = fit_predict(FINAL_CFG, F, y, H, seed=SEED)
        rows.append(score(H.y.values, pr, Pr, "FINAL cfg trained on Claude rows", mode, "E4 Claude -> h300", len(F)))
    R = pd.DataFrame(rows)[["mode", "experiment", "model", "n_train", "n_test", "acc", "all_true_acc", "f3", "ftf",
                            "f_unsure", "ece"]]
    R.to_csv(OUT / "claude_cv.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "claude_cv_case_types.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    main()

"""Item 12: CfC "value still current" probability per fact -> one extra feature for LightGBM.

Each fact is read as an irregular-time sequence of every other corpus mention of its (subject, attribute):
signed day gap to the fact (before and after), same value or not, same page or not, before/after flag. The CfC
gets the real day gaps as its time constants (`timespans`), so a liquid cell can decay old evidence.
Target: the training timeline confirms the value (relation match_latest / future_value = 1, stale_older /
contradicts = 0; facts with no verdict are not trained on). Labels only ever come from the training rows.

eda/out/neuro_feats.csv (row level = min over the row's facts), one value per (id, outer_fold):
  outer_fold k, role test : model trained on the CV rows outside fold k            (use for fold-k test rows)
  outer_fold k, role train: model trained outside fold k and outside the row's fold (use for fold-k train rows)
  outer_fold all, role oof : model trained outside the row's fold                   (final model on all CV rows)
  outer_fold holdout       : model trained on all CV rows, scored on the holdout pages
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from nets import liquid_last  # noqa: E402

SEQ = 16
TAB = ["pair_rows", "pair_n_values", "same_value_rows", "value_share", "agree_prev_mention", "agree_next_mention",
       "n_values_before"]


def fact_sequences(FS, dt="log"):
    """[n_facts, SEQ, 5] features + [n_facts, SEQ] timespans, label-free."""
    FS = FS.reset_index(drop=True)
    seq = np.zeros((len(FS), SEQ, 5), np.float32)
    ts = np.ones((len(FS), SEQ), np.float32)
    for _, g in FS.groupby(["subject", "attribute"]):
        d = g["date"].values.astype("datetime64[D]").astype(np.int64)
        vals, pages, ids, idx = g["value"].values, g["page"].values, g["id"].values, g.index.values
        for a in range(len(g)):
            m = ids != ids[a]
            if not m.any():
                continue
            delta = (d[m] - d[a]).astype(np.float32)
            order = np.argsort(np.abs(delta))[:SEQ]
            order = order[np.argsort(delta[order])]  # chronological
            dl = delta[order]
            enc = {"raw": dl / 365.0, "log": np.sign(dl) * np.log1p(np.abs(dl)) / np.log1p(365.0),
                   "none": np.zeros_like(dl)}[dt]
            rows = np.stack([enc, (vals[m][order] == vals[a]).astype(np.float32),
                             (pages[m][order] == pages[a]).astype(np.float32), (dl > 0).astype(np.float32),
                             np.ones(len(order), np.float32)], 1)
            seq[idx[a], SEQ - len(order):] = rows
            gaps = np.abs(np.diff(np.concatenate([[dl[0]], dl]))) / 30.0 + 1.0
            ts[idx[a], SEQ - len(order):] = gaps if dt != "none" else 1.0
    return seq, ts


class CurrentNet(torch.nn.Module):
    def __init__(self, n_seq, n_tab, units=32):
        super().__init__()
        from ncps.torch import CfC
        self.rnn = CfC(n_seq, units, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(units + n_tab, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))

    def forward(self, s, ts, x):
        return self.head(torch.cat([liquid_last(self.rnn, s, ts), x], 1))


def targets(F, train_ids):
    rel = C.B.kb_features(F[F["id"].isin(train_ids)], train_ids)["kb_relation"]
    y = rel.map({"match_latest": 1, "future_value": 1, "stale_older": 0, "contradicts": 0})
    return y.reindex(F.index)


def fit_predict(F, SQ, TS, XT, train_ids, score_mask, seed=0):
    y = targets(F, train_ids)
    tr = F["id"].isin(train_ids).values & y.notna().values
    yt = y.values[tr].astype(np.int64)
    model, fit_s = C.train_dev(CurrentNet(SQ.shape[2], XT.shape[1]), [SQ[tr], TS[tr], XT[tr]], yt, epochs=40,
                               seed=seed, budget_s=40)
    p = C.predict_dev(model, [SQ[score_mask], TS[score_mask], XT[score_mask]])[:, 1]
    return p, model, fit_s, (tr.sum(), yt.mean())


def main():
    t0 = time.time()
    D = C.load()
    R, FS, y, folds = D["R"], D["FS"].copy(), D["y"], D["folds"]
    FS = FS.reset_index(drop=True)
    SQ, TS = fact_sequences(FS, "log")
    XT = FS[TAB].fillna(-1).astype(np.float32).values
    XT = (XT - XT.mean(0)) / (XT.std(0) + 1e-6)
    print(f"fact sequences {SQ.shape} ({time.time() - t0:.0f}s), device {C.DEVICE}", flush=True)

    fold_of = dict(zip(R["id"], R["fold"]))
    hold = set(R.loc[R["holdout"] == 1, "id"])
    cv_ids = set(R.loc[R["holdout"] == 0, "id"])
    out, fits = [], []

    def run(train_ids, score_ids, outer, role):
        mask = FS["id"].isin(score_ids).values
        p, model, fs, info = fit_predict(FS, SQ, TS, XT, train_ids, mask)
        fits.append(fs)
        row = pd.Series(p, index=FS.index[mask]).groupby(FS.loc[mask, "id"]).min()
        out.append(pd.DataFrame({"id": row.index, "outer_fold": outer, "role": role, "cfc_current_prob": row.values}))
        return model

    for k in range(5):
        not_k = {i for i in cv_ids if fold_of[i] != k}
        run(not_k, {i for i in cv_ids if fold_of[i] == k}, str(k), "test")
        for j in range(5):
            if j != k:
                run({i for i in not_k if fold_of[i] != j}, {i for i in cv_ids if fold_of[i] == j}, str(k), "train")
        print(f"  outer fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    for j in range(5):
        run({i for i in cv_ids if fold_of[i] != j}, {i for i in cv_ids if fold_of[i] == j}, "all", "oof")
    model = run(cv_ids, hold, "holdout", "holdout")
    NF = pd.concat(out, ignore_index=True)
    NF.round(5).to_csv(C.OUT / "neuro_feats.csv", index=False)
    print(f"wrote neuro_feats.csv {NF.shape} ({time.time() - t0:.0f}s)", flush=True)
    cpu2 = C.cpu2_seconds(model, [SQ[:2000], TS[:2000], XT[:2000]]) * len(FS) / len(R)  # facts per row
    evaluate(D, NF, {"predict_s_cpu2": round(cpu2, 2), "gpu_fit_s": round(float(np.sum(fits)), 1)})


def evaluate(D, NF, extra, seeds=5):
    """LightGBM with vs without the feature on the same folds, 5 LightGBM seeds (mean +- std)."""
    from sklearn.metrics import roc_auc_score
    y, folds, R = D["y"], D["folds"], D["R"]
    oof = NF[NF.outer_fold == "all"].set_index("id")["cfc_current_prob"]
    cv = R[R.holdout == 0].set_index("id")
    auc = roc_auc_score(cv["label"].eq("false"), -oof.reindex(cv.index))
    print(f"feature alone: AUC for 'row is false' = {auc:.3f}", flush=True)
    rows = []
    for name, use in [("n12_lgbm_reference", False), ("n12_lgbm_plus_cfc_current", True)]:
        S = []
        for seed in range(seeds):
            probs, fit_t = {}, 0.0
            for k, (tr, te) in folds.items():
                X = D["Xf"][k].copy()
                if use:
                    f = NF[NF.outer_fold == str(k)].set_index("id")["cfc_current_prob"]
                    X["cfc_current_prob"] = f.reindex(X.index).values
                t1 = time.time()
                m = C.lgbm(seed).fit(X.iloc[tr], y[tr])
                fit_t += time.time() - t1
                probs[k] = m.predict_proba(X.iloc[te])
            r, _, _ = C.oof_eval(name, y, folds, probs, {"fit_s": round(fit_t, 2)})
            S.append(r)
        S = pd.DataFrame(S)
        row = S.iloc[0].to_dict()
        row["cv"] = f"group5_splits_x{seeds}lgbm_seeds"
        for c in ["acc", "macro_f1_tf", "ece", "unsure_recall", "false_precision"]:
            row[c], row[f"{c}_std"] = S[c].mean(), S[c].std()
        if use:
            row.update(extra)
            row["feature_auc_false"] = round(auc, 4)
        rows.append(row)
        print(name, f"macro-F1 {row['macro_f1_tf']:.4f}+-{row['macro_f1_tf_std']:.4f} acc {row['acc']:.4f}"
                    f" ece {row['ece']:.4f}", flush=True)
    C.save_rows(rows, "n12_")


if __name__ == "__main__":
    if "--eval-only" in sys.argv:
        prev = pd.read_csv(C.RESULTS).set_index("variant") if C.RESULTS.exists() else pd.DataFrame()
        keep = {}
        if "n12_lgbm_plus_cfc_current" in prev.index:
            keep = prev.loc["n12_lgbm_plus_cfc_current", ["predict_s_cpu2", "gpu_fit_s"]].to_dict()
        evaluate(C.load(), pd.read_csv(C.OUT / "neuro_feats.csv", dtype={"outer_fold": str}), keep)
    else:
        main()

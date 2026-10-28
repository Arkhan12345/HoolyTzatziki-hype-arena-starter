"""Item 13: liquid (CfC) and spiking (LIF) heads trained only for unsure vs not-unsure, stacked with LightGBM.

Per outer fold: LightGBM (3-class) + both binary heads produce inner out-of-fold scores on the training folds;
a logistic-regression stacker learns from those, then everything is refit on the full training folds and scored
on the test fold. Variant `..._tuned` also moves the unsure cut-off to the value that maximises 3-class accuracy
on the inner out-of-fold scores.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from nets import RowNet, SpikingHead, encode_dt  # noqa: E402


def heads(Xtr, Str, Ttr, ytr_uns, Xte, Ste, Tte, seed=0):
    xa, xb = C.tab_scaled(Xtr, Xte)
    cfc, f1 = C.train_dev(RowNet("cfc", Str.shape[2], xa.shape[1], 32, n_out=2), [Str, Ttr, xa], ytr_uns,
                          epochs=50, seed=seed, budget_s=30)
    snn, f2 = C.train_dev(SpikingHead(xa.shape[1], 64, 25, n_out=2), [xa], ytr_uns, epochs=40, seed=seed, budget_s=30)
    p_cfc = C.predict_dev(cfc, [Ste, Tte, xb])[:, 1]
    p_snn = C.predict_dev(snn, [xb])[:, 1]
    return p_cfc, p_snn, f1 + f2, (cfc, snn, xb)


def main():
    t0 = time.time()
    D = C.load()
    R, y, folds, Xf = D["R"], D["y"], D["folds"], D["Xf"]
    SEQ, TS = encode_dt(D["SEQ"], "log")
    fold = R["fold"].values
    probs = {v: {} for v in ["n13_lgbm_reference", "n13_cfc_unsure_head", "n13_snn_unsure_head",
                             "n13_stack_lgbm_cfc_snn", "n13_stack_lgbm_cfc_snn_tuned"]}
    gpu_fit, timing = 0.0, None
    for k, (tr, te) in folds.items():
        X = Xf[k].values.astype(np.float32)
        y_uns = (y == 2).astype(np.int64)
        # inner OOF on the training folds
        Z = np.zeros((len(tr), 5))
        for j in sorted(set(fold[tr])):
            itr, iva = tr[fold[tr] != j], tr[fold[tr] == j]
            pos = np.searchsorted(tr, iva)
            Z[pos, :3] = C.lgbm().fit(X[itr], y[itr]).predict_proba(X[iva])
            pc, ps, fs, _ = heads(X[itr], SEQ[itr], TS[itr], y_uns[itr], X[iva], SEQ[iva], TS[iva])
            Z[pos, 3], Z[pos, 4] = pc, ps
            gpu_fit += fs
        # refit on all training folds, score the test fold
        P_lgb = C.lgbm().fit(X[tr], y[tr]).predict_proba(X[te])
        pc, ps, fs, (cfc, snn, xb) = heads(X[tr], SEQ[tr], TS[tr], y_uns[tr], X[te], SEQ[te], TS[te])
        gpu_fit += fs
        stack = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y[tr])
        Zte = np.column_stack([P_lgb, pc, ps])
        P_st = stack.predict_proba(Zte)
        # tuned unsure cut-off on inner OOF
        Zi = stack.predict_proba(Z)
        best_t, best_acc = None, -1
        for t in np.linspace(0.05, 0.6, 23):
            pr = np.where(Zi[:, 2] >= t, 2, Zi[:, :2].argmax(1))
            acc = (pr == y[tr]).mean()
            if acc > best_acc:
                best_t, best_acc = t, acc
        P_tu = P_st.copy()
        hit = P_tu[:, 2] >= best_t
        P_tu[hit, 2] = 1.0  # force unsure where the cut-off says so
        P_tu = P_tu / P_tu.sum(1, keepdims=True)
        probs["n13_lgbm_reference"][k] = P_lgb
        probs["n13_stack_lgbm_cfc_snn"][k] = P_st
        probs["n13_stack_lgbm_cfc_snn_tuned"][k] = P_tu
        # a head alone: LightGBM's true/false split, the head decides unsure at 0.5
        for name, p in [("n13_cfc_unsure_head", pc), ("n13_snn_unsure_head", ps)]:
            Q = P_lgb.copy()
            Q[:, 2] = np.where(p >= 0.5, 1.0, 0.0)
            probs[name][k] = Q / Q.sum(1, keepdims=True)
        if timing is None:
            Ste = SEQ[te][:2000]
            timing = {"cfc": C.cpu2_seconds(cfc, [Ste, TS[te][:2000], xb[:2000]]),
                      "snn": C.cpu2_seconds(snn, [xb[:2000]])}
        print(f"  fold {k} done, unsure cut-off {best_t:.2f} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for name, pk in probs.items():
        row, _, _ = C.oof_eval(name, y, folds, pk)
        if "cfc" in name or "snn" in name or "stack" in name:
            row["predict_s_cpu2"] = round(timing["cfc"] * ("cfc" in name or "stack" in name)
                                          + timing["snn"] * ("snn" in name or "stack" in name), 2)
            row["gpu_fit_s"] = round(gpu_fit, 1)
        rows.append(row)
        print(name, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    C.save_rows(rows, "n13_")


if __name__ == "__main__":
    main()

"""Neural members for the combination round of the self-improvement loop (eda/selfimprove/loop.py --combos).

For every outer fold of eda/splits.csv fold_group (holdout pages excluded) and for the lockbox:
  role "train": inner out-of-fold predictions for the fold's training rows, on EXACTLY the loop's inner splits
                (GroupKFold(5) by page over the sorted training ids), so they line up with the final model's own
                inner out-of-fold probabilities;
  role "test":  members refit on all of the fold's training rows, predicting its test rows (lockbox: holdout rows;
                holdout labels are never read).
Members: CfC unsure head + spiking (LIF) unsure head (binary unsure vs not; combos 1 and 2) and the character BiGRU
(3-class; combo 3). Output eda/out/neuro_combo_oof.pkl = {"parts": {(outer_fold, role): DataFrame indexed by id},
"cpu2_s_20k": {...}}.

    .venv-neuro/Scripts/python eda/neuro/combo_oof.py
    .venv-neuro/Scripts/python eda/neuro/combo_oof.py --schemes row5,time   # extra keys for the overfitting gates
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from nets import CharBiGRU, RowNet, SpikingHead, char_encode, encode_dt  # noqa: E402

OUTF = C.OUT / "neuro_combo_oof.pkl"


def inner_splits(train_ids, page_of):
    trl = sorted(train_ids)
    return [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[page_of[i] for i in trl])]


def fit_predict(X, SEQ, TS, T, y, tr, te, seed=0):
    """Train the three members on rows tr (positions), predict rows te. Returns (DataFrame cols, models)."""
    y_uns = (y[tr] == 2).astype(np.int64)
    xa, xb = C.tab_scaled(X[tr], X[te])
    cfc, _ = C.train_dev(RowNet("cfc", SEQ.shape[2], xa.shape[1], 32, n_out=2), [SEQ[tr], TS[tr], xa], y_uns,
                         epochs=50, seed=seed, budget_s=25)
    snn, _ = C.train_dev(SpikingHead(xa.shape[1], 64, 25, n_out=2), [xa], y_uns, epochs=40, seed=seed, budget_s=25)
    gru, _ = C.train_dev(CharBiGRU(VOCAB), [T[tr]], y[tr], epochs=25, lr=2e-3, weight=False, seed=seed, budget_s=40)
    out = np.column_stack([C.predict_dev(cfc, [SEQ[te], TS[te], xb])[:, 1],
                           C.predict_dev(snn, [xb])[:, 1],
                           C.predict_dev(gru, [T[te]])])
    return out, (cfc, snn, gru, xb)


def run_split(X, ids_all, pos_of, SEQ, TS, T, y, train_ids, test_ids, page_of, tag, inner=None):
    parts = {}
    tr_ids = sorted(train_ids)
    oof = pd.DataFrame(index=tr_ids, columns=COLS, dtype=float)
    for a in (inner if inner is not None else inner_splits(train_ids, page_of)):
        b = sorted(set(tr_ids) - set(a))
        out, _ = fit_predict(X, SEQ, TS, T, y, [pos_of[i] for i in a], [pos_of[i] for i in b])
        oof.loc[b] = out
    parts[(tag, "train")] = oof
    te = sorted(test_ids)
    out, models = fit_predict(X, SEQ, TS, T, y, [pos_of[i] for i in tr_ids], [pos_of[i] for i in te])
    parts[(tag, "test")] = pd.DataFrame(out, index=te, columns=COLS)
    return parts, models, [pos_of[i] for i in te]


COLS = ["cfc_uns", "snn_uns", "bigru_p0", "bigru_p1", "bigru_p2"]
VOCAB = None


def features_for(train_ids, ids, page_of):
    """Label-free table + gold-timeline KB from train_ids' labels (training rows out-of-fold), as common.load."""
    R0, F, FS, X0 = C.B.label_free_table()
    trl = sorted(train_ids)
    inner = [np.array([trl[j] for j in a]) for a, _ in GroupKFold(5).split(trl, groups=[page_of[i] for i in trl])]
    kb = C.B.kb_table(F, list(ids), set(train_ids), inner)
    XL = pd.concat([X0, kb.reindex(X0.index)], axis=1)
    return XL.reindex(ids).values.astype(np.float32)


def schemes(names):
    """Extra keys for the overfitting gates: fold_row5 (r5_k, KFold inner as eda/07_improve) and the time split."""
    global VOCAB
    from sklearn.model_selection import KFold
    t0 = time.time()
    D = C.load()
    R, y = D["R"], D["y"]
    SEQ, TS = encode_dt(D["SEQ"], "log")
    T, VOCAB = char_encode(R["text"].fillna("").tolist())
    ids = R["id"].values
    pos_of = {i: p for p, i in enumerate(ids)}
    page_of = dict(zip(R["id"], R["page"]))
    S = pd.read_csv(C.ROOT / "eda" / "splits.csv").set_index("id")
    cv_ids = sorted(S.index[S.holdout == 0])
    Scv = S.loc[cv_ids]
    data = pickle.loads(OUTF.read_bytes())
    jobs = []
    if "row5" in names:
        for k in sorted(Scv.fold_row5.unique()):
            test = set(Scv.index[Scv.fold_row5 == k])
            train = set(cv_ids) - test
            trl = sorted(train)
            inner = [[trl[j] for j in a] for a, _ in KFold(5, shuffle=True, random_state=0).split(trl)]
            jobs.append((f"r5_{k}", train, test, inner))
    if "time" in names:
        test = set(Scv.index[Scv.time_split == "test"])
        jobs.append(("time0", set(cv_ids) - test, test, None))
    for key, train, test, inner in jobs:
        X = features_for(train, ids, page_of)
        p, _, _ = run_split(X, ids, pos_of, SEQ, TS, T, y, train, test, page_of, key, inner)
        data["parts"].update(p)
        print(f"  {key} done ({time.time() - t0:.0f}s)", flush=True)
    OUTF.write_bytes(pickle.dumps(data))


def main():
    global VOCAB
    t0 = time.time()
    D = C.load()
    R, y, Xf = D["R"], D["y"], D["Xf"]
    SEQ, TS = encode_dt(D["SEQ"], "log")
    T, VOCAB = char_encode(R["text"].fillna("").tolist())
    ids = R["id"].values
    pos_of = {i: p for p, i in enumerate(ids)}
    page_of = dict(zip(R["id"], R["page"]))
    S = pd.read_csv(C.ROOT / "eda" / "splits.csv").set_index("id")
    cv_ids = sorted(S.index[S.holdout == 0])
    hold_ids = sorted(S.index[S.holdout == 1])
    parts, timing = {}, None
    for k in range(5):
        test = set(S.index[(S.holdout == 0) & (S.fold_group == k)])
        train = set(cv_ids) - test
        X = Xf[k].reindex(ids).values.astype(np.float32)
        p, models, te_pos = run_split(X, ids, pos_of, SEQ, TS, T, y, train, test, page_of, str(k))
        parts.update(p)
        if timing is None:
            cfc, snn, gru, xb = models
            n = min(1000, len(te_pos))
            sl = te_pos[:n]
            timing = {"cfc": C.cpu2_seconds(cfc, [SEQ[sl], TS[sl], xb[:n]]), "snn": C.cpu2_seconds(snn, [xb[:n]]),
                      "bigru": C.cpu2_seconds(gru, [T[sl]])}
        print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    # lockbox: KB features from every CV row's labels (inner out-of-fold for the CV rows), holdout predicted
    R0, F, FS, X0 = C.B.label_free_table()
    trl = sorted(cv_ids)
    inner = [np.array([trl[j] for j in a]) for a, _ in GroupKFold(5).split(trl, groups=[page_of[i] for i in trl])]
    kb = C.B.kb_table(F, list(ids), set(cv_ids), inner)
    XL = pd.concat([X0, kb.reindex(X0.index)], axis=1)
    XL = XL[Xf[0].columns].reindex(ids).values.astype(np.float32) if set(Xf[0].columns) <= set(XL.columns) \
        else XL.reindex(ids).values.astype(np.float32)
    p, _, _ = run_split(XL, ids, pos_of, SEQ, TS, T, y, set(cv_ids), set(hold_ids), page_of, "lockbox")
    parts.update(p)
    print(f"  lockbox done ({time.time() - t0:.0f}s)", flush=True)
    with open(OUTF, "wb") as f:
        pickle.dump({"parts": parts, "cpu2_s_20k": timing, "device": C.DEVICE.type,
                     "gpu_wall_s": time.time() - t0}, f)
    print(f"wrote {OUTF} | cpu2 s/20k {timing} | {time.time() - t0:.0f}s on {C.DEVICE.type}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="")
    a = ap.parse_args()
    if a.schemes:
        schemes(a.schemes.split(","))
    else:
        main()

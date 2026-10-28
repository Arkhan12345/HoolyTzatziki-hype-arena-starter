"""Shared pieces for the neural track (eda/neuro/*.py). Dev-only, runs in .venv-neuro (CUDA torch).

Folds: eda/splits.csv `fold_group` (0-4) on the non-holdout rows; holdout pages are never touched here.
Per-fold tabular features = label-free table + gold-timeline KB features rebuilt from the fold's training labels
(training rows get theirs out-of-fold), exactly as eda/04_bakeoff.py does. They are cached in
eda/out/neuro_cache.pkl so every script sees identical inputs.
"""
import importlib.util
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eda" / "out"
CACHE = OUT / "neuro_cache.pkl"
RESULTS = OUT / "improve_neuro.csv"
CLASSES = np.array(["false", "true", "unsure"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNTIME_ROWS = 20_000

_spec = importlib.util.spec_from_file_location("bakeoff", ROOT / "eda" / "04_bakeoff.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)  # sets torch CPU threads to 2 (the runtime budget)
torch.set_num_threads(4)     # training-side CPU work may use more; timing switches back to 2


def lgbm(seed=0):
    """The research winner: unweighted LightGBM, same settings as the bake-off."""
    return LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                          n_jobs=2, deterministic=True, force_row_wise=True)


def load():
    """R, F, FS, label-free X0, y, folds {k: (train_idx, test_idx)} over non-holdout rows, per-fold X."""
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    R, F, FS, X0 = B.label_free_table()
    S = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id").reindex(R["id"])
    R = R.assign(fold=S["fold_group"].values, holdout=S["holdout"].values)
    y = pd.Series(R["label"].values).map({c: i for i, c in enumerate(CLASSES)}).values
    cv = np.where(R["holdout"].values == 0)[0]
    folds, Xf = {}, {}
    for k in range(5):
        te = cv[R["fold"].values[cv] == k]
        tr = cv[R["fold"].values[cv] != k]
        folds[k] = (tr, te)
        train_ids = set(R["id"].values[tr])
        sub = R.iloc[tr]
        inner = [sub["id"].values[a] for a, _ in GroupKFold(5).split(sub, groups=sub["page"].values)]
        ids = list(R["id"].values[np.concatenate([tr, te])])
        kb = B.kb_table(F, ids, train_ids, inner)
        Xf[k] = pd.concat([X0, kb.reindex(X0.index)], axis=1)
        print(f"  fold {k}: KB features ({time.time() - t0:.0f}s)", flush=True)
    data = dict(R=R, F=F, FS=FS, X0=X0, y=y, folds=folds, Xf=Xf, SEQ=B.sequences(R, FS))
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def metrics(y, proba):
    m = B.metrics(y, proba)
    pred = proba.argmax(1)
    pf = pred == 0
    m["false_precision"] = float((y[pf] == 0).mean()) if pf.any() else np.nan
    return m


def train_dev(model, inputs, y, epochs=60, lr=3e-3, batch=128, seed=0, weight=True, budget_s=60):
    """Device-aware version of the bake-off trainer (same loss, optimiser and class weights)."""
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    k = int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(float)
    w = torch.tensor(len(y) / (k * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE) if weight else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xs = [torch.as_tensor(i, device=DEVICE) for i in inputs]
    y = np.asarray(y, dtype=np.int64)
    yt = torch.as_tensor(y, device=DEVICE)
    n, t0 = len(y), time.time()
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            loss = loss_fn(model(*[x[idx] for x in xs]), yt[idx])
            loss.backward()
            opt.step()
        if time.time() - t0 > budget_s:
            break
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0


def predict_dev(model, inputs, device=DEVICE, batch=4096):
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(inputs[0]), batch):
            out.append(torch.softmax(model(*[torch.as_tensor(i[b:b + batch], device=device) for i in inputs]), 1).cpu())
    return torch.cat(out).numpy()


def cpu2_seconds(model, inputs):
    """CPU inference time with 2 threads (the arena box), scaled to 20k rows."""
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    t0 = time.time()
    predict_dev(model, inputs, device=torch.device("cpu"))
    s = (time.time() - t0) * N_RUNTIME_ROWS / len(inputs[0])
    torch.set_num_threads(prev)
    model.to(DEVICE)
    return s


def tab_scaled(Xtr, Xte):
    """Impute + min-max to [0, 1] (spiking rate coding needs [0, 1]; also fine for the RNN heads)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = MinMaxScaler(clip=True).fit(imp.transform(Xtr))
    f = lambda X: sc.transform(imp.transform(X)).astype(np.float32)
    return f(Xtr), f(Xte)


def save_rows(rows, prefix):
    """Replace this script's rows (variant starting with prefix) in eda/out/improve_neuro.csv."""
    cols = ["variant", "cv", "acc", "macro_f1_tf", "ece", "unsure_recall", "false_precision", "fit_s",
            "predict_s_cpu2", "device", "gpu_fit_s"]
    new = pd.DataFrame(rows)
    for c in cols:
        if c not in new:
            new[c] = np.nan
    new = new[cols + [c for c in new.columns if c not in cols]]
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        old = old[~old["variant"].astype(str).str.startswith(prefix)]
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(RESULTS, index=False)
    return new


def oof_eval(name, y, folds, fold_proba, extra=None):
    """Concatenate per-fold test probabilities into one OOF evaluation row."""
    te_all = np.concatenate([folds[k][1] for k in sorted(fold_proba)])
    P = np.concatenate([fold_proba[k] for k in sorted(fold_proba)])
    row = {"variant": name, "cv": "group5_splits", **metrics(y[te_all], P), "device": DEVICE.type}
    row.update(extra or {})
    return row, te_all, P

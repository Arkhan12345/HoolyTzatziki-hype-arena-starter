"""Shared data + metrics for the neural stream on h300 (eda/h300/neuro/). Dev-only, runs in .venv-neuro (CUDA torch).

Honest protocol:
  practice   eda/splits.csv fold_group (0-4) on non-holdout rows; holdout pages never touched here.
  h300       eda/h300/out/h300_splits.csv: part cv (5 subject-grouped folds) / lockbox (never touched here) /
             train_only (near-copies of practice: allowed in training, never scored).
  synthetic  per-fold files eda/h300/out/synth_fold{k}_{mode}.csv (built from fold k's training rows only) when
             present, else eda/h300/out/synth10k_{mode}.csv with every (subject, attribute) pair that appears in the
             fold's validation pages dropped. 20% of the synthetic rows (fixed seed) are kept out of training and
             used only as a per-case_type validation set.
Facts for the sequence models come from EXTRACTED facts (eda/out/facts_extracted.pkl for practice, the practice-fitted
extractor for h300), never from the gold `form` of a scored row.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
H = ROOT / "eda" / "h300"
OUT = H / "neuro" / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "eda" / "neuro"))
import common as C  # noqa: E402  (device-aware trainer, cpu2 timing, bake-off module B)

CLASSES = np.array(["false", "true", "unsure"])
LAB = {c: i for i, c in enumerate(CLASSES)}
DEVICE = C.DEVICE
MODES = {"practice": "practice_rule", "closed_world": "closed_world"}


def practice():
    """Practice rows with text, int label, page, fold (-1 = holdout)."""
    R = pd.read_pickle(ROOT / "eda" / "out" / "rows.pkl")
    S = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id").reindex(R["id"])
    R = R.assign(fold=np.where(S["holdout"].values == 1, -1, S["fold_group"].values))
    R["y"] = R["label"].map(LAB).values
    R["date"] = pd.to_datetime(R["date"]).dt.strftime("%Y-%m-%d")
    return R[["id", "text", "date", "label", "y", "page", "fold"]].reset_index(drop=True)


def h300():
    """h300 rows with part / fold from the subject-grouped split; lockbox rows are dropped here (never scored)."""
    D = pd.read_csv(ROOT / "data" / "hooly_statements_300.csv")
    S = pd.read_csv(H / "out" / "h300_splits.csv")[["id", "subject", "part", "fold"]]
    D = D.merge(S, on="id", how="left")
    D = D[D["part"] != "lockbox"].reset_index(drop=True)
    D["y"] = D["label"].map(LAB).values
    return D


def gold_pairs(rows):
    pairs = set()
    for f in rows["form"].dropna():
        for x in json.loads(f):
            pairs.add((x["subject"], x["attribute"]))
    return pairs


def synth(mode, k, val_pairs=None, seed=0):
    """(train_part, val_part) of the synthetic rows usable for practice fold k (k=None: no fold filter)."""
    tag = MODES[mode]
    per_fold = H / "out" / f"synth_fold{k}_{tag}.csv"
    big = H / "out" / f"synth10k_{tag}.csv"
    if k is not None and per_fold.exists() and not big.exists():
        S = pd.read_csv(per_fold)
        source = per_fold.name
    elif big.exists():
        S = pd.read_csv(big)
        source = big.name
        if val_pairs:
            keep = [not any((x["subject"], x["attribute"]) in val_pairs for x in json.loads(f)) for f in S["form"]]
            S = S[np.array(keep)]
    else:
        return None, None, None
    S = S.dropna(subset=["text", "label"]).copy()
    S["y"] = S["label"].map(LAB).values
    rng = np.random.default_rng(seed)
    hold = rng.random(len(S)) < 0.2
    return S[~hold].reset_index(drop=True), S[hold].reset_index(drop=True), source


def macro_f1(y, pred, labels):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def ece(P, y, bins=10):
    conf = P.max(1)
    ok = (P.argmax(1) == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def scores(y, P):
    pred = P.argmax(1)
    return {"acc": float((pred == y).mean()), "f1_tf": macro_f1(y, pred, [0, 1]),
            "f1_3": macro_f1(y, pred, [0, 1, 2]), "unsure_f1": macro_f1(y, pred, [2]), "ece": ece(P, y),
            "n": int(len(y))}


def per_case(S, P):
    pred = P.argmax(1)
    out = {}
    for c, g in S.assign(ok=(pred == S["y"].values)).groupby("case_type"):
        out[c] = round(float(g["ok"].mean()), 3)
    return out

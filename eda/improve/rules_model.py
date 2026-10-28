"""Dataset rules the ladder did not use yet (docs/RULES.md), as model pieces.

G1  ratio_to_held / log_ratio_to_held / perturb_ratio_hit: the sentence value against the value the staircase
    holds on that date; false values are the held value times 2, 0.5, 2.5, 0.4, 5, 0.2, 10 or 0.1 (§5, V1).
G3  hedge_unbacked: a hedge word on a fact the timeline never holds (V2).
G4  speaker_trust / speaker_n: Beta-smoothed false rate of the speaker over TRAINING rows only (S5).
G2  unsure layer after calibration, one switch per rule:
      joke          joke tail + single fact + not a key-value line -> unsure (§4.1, S3)
      single_value  one value in the whole corpus, never repeated, nothing corroborates -> never false; unsure
                    when P(true) < t (§4.2, T5)
      hearsay       hearsay -> unsure (S4)
G5    today         'today' with a text date that differs from the page date -> unsure (§4.4)
    The joke / hearsay / today switches leave a confident false (P(false) >= 0.5) alone.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eda"))
sys.path.insert(0, str(ROOT))

CLASSES = np.array(["false", "true", "unsure"])
PERTURB = (2.0, 0.5, 2.5, 0.4, 5.0, 0.2, 10.0, 0.1)
PERTURB_TOL = 0.02          # |log ratio - log k| below this is a hit
SPEAKER_PRIOR = 0.146       # base false rate
SPEAKER_PSEUDO = 10.0
SPEAKER_MIN_N = 2           # a speaker seen fewer times in training keeps the prior
NUMERIC = {"quantity", "duration", "money", "percent"}


# ------------------------------------------------------------------ G1
def _numeric(vkey):
    """(dimension, magnitude) from a canonical key (app/kb/normalize.key), or None."""
    if vkey is None or (isinstance(vkey, float) and math.isnan(vkey)):
        return None
    try:
        k = json.loads(vkey)
    except (TypeError, ValueError):
        return None
    if not isinstance(k, list) or len(k) < 3 or k[0] not in NUMERIC:
        return None
    mag = k[2]
    if not isinstance(mag, (int, float)):
        return None
    return json.dumps(k[1]), float(mag)


def ratio_to_held(vkey, held_vkey):
    """Stated value / held value for comparable numeric values; NaN when not comparable or not positive."""
    a, b = _numeric(vkey), _numeric(held_vkey)
    if a is None or b is None or a[0] != b[0] or a[1] <= 0 or b[1] <= 0:
        return float("nan")
    return a[1] / b[1]


def is_perturbation(r):
    if r is None or not np.isfinite(r) or r <= 0:
        return 0
    lr = math.log(r)
    return int(any(abs(lr - math.log(k)) < PERTURB_TOL for k in PERTURB))


def held_vkeys(tl, anchor, weight):
    """The value key the staircase holds on each fact's date (aligned with tl.f)."""
    uniques = pd.factorize(tl.f["vkey"])[1]
    held = np.empty(tl.n, dtype=object)
    for idx in tl.groups:
        days_u, vals_u, t_of, v_of, path = tl._path(idx, anchor, weight)
        held[idx] = uniques[vals_u[path[t_of]]]
    return held


# ------------------------------------------------------------------ G1 + G3 + G4
def rule_features(facts, rows, timeline_out, train_ids):
    """Row-level G1/G3/G4 features.

    facts: id, vkey (aligned with timeline_out); rows: id, label, speaker, hedge;
    timeline_out: held_vkey, tl_agree, tl_was_held_before, tl_held_later (aligned with facts);
    train_ids: rows whose labels may be read (speaker trust only).
    """
    f = facts.reset_index(drop=True)
    t = timeline_out.reset_index(drop=True)
    ratio = np.array([ratio_to_held(v, h) for v, h in zip(f["vkey"], t["held_vkey"])], dtype=float)
    d = pd.DataFrame({"id": f["id"].to_numpy(), "ratio": ratio,
                      "abslog": np.abs(np.log(np.where(ratio > 0, ratio, np.nan))),
                      "hit": [is_perturbation(r) for r in ratio],
                      "unbacked": ((t["tl_agree"] != 1) & (t["tl_was_held_before"] != 1)
                                   & (t["tl_held_later"] != 1)).to_numpy()})
    r = rows.set_index("id")
    out = pd.DataFrame(index=r.index)
    g = d.groupby("id")
    worst = d.loc[d["abslog"].fillna(-1).groupby(d["id"]).idxmax()].set_index("id")
    worst.loc[worst["abslog"].isna(), "ratio"] = np.nan
    out["ratio_to_held"] = worst["ratio"].reindex(out.index)
    out["log_ratio_to_held"] = np.log(out["ratio_to_held"])
    out["perturb_ratio_hit"] = g["hit"].max().reindex(out.index).fillna(0).astype(int)
    unb = g["unbacked"].any().astype(bool).reindex(out.index, fill_value=False)
    out["hedge_unbacked"] = ((r["hedge"].fillna(0).astype(int) == 1) & unb).astype(int)

    tr = r.loc[r.index.isin(train_ids) & r["speaker"].notna() & (r["label"].astype(str) != "")]
    n = tr.groupby("speaker").size()
    k = (tr["label"] == "false").groupby(tr["speaker"]).sum()
    trust = (k + SPEAKER_PRIOR * SPEAKER_PSEUDO) / (n + SPEAKER_PSEUDO)
    trust[n < SPEAKER_MIN_N] = SPEAKER_PRIOR
    sp = r["speaker"]
    out["speaker_trust"] = sp.map(trust).where(sp.notna())
    out.loc[sp.notna() & out["speaker_trust"].isna(), "speaker_trust"] = SPEAKER_PRIOR
    out["speaker_n"] = sp.map(n).fillna(0).where(sp.notna())
    return out


# ------------------------------------------------------------------ G2 + G5
def _col(feats, name, default=0):
    return feats[name].fillna(default).to_numpy() if name in feats else np.full(len(feats), default)


def unsure_layer(proba, feats, knobs):
    """Apply the unsure rules after calibration. Returns (labels as int codes 0/1/2, confidence; NaN for unsure)."""
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(1)
    guard = proba[:, 0] < 0.5
    if knobs.get("single_value"):
        m = ((_col(feats, "pair_n_values_max", 2) == 1) & (_col(feats, "same_value_rows_min", 1) == 0)
             & (_col(feats, "tl_n_pages_for_max", 0) == 0))
        pred = np.where(m & (pred == 0), 1, pred)
        t = float(knobs.get("t", 0.0))
        pred = np.where(m & (proba[:, 1] < t), 2, pred)
    if knobs.get("joke"):
        m = (_col(feats, "joke_tail") == 1) & (_col(feats, "n_facts", 1) == 1) & (_col(feats, "kv_line") == 0)
        pred = np.where(m & guard, 2, pred)
    if knobs.get("hearsay"):
        pred = np.where((_col(feats, "hearsay") == 1) & guard, 2, pred)
    if knobs.get("today"):
        pred = np.where((_col(feats, "today_date_mismatch") == 1) & guard, 2, pred)
    conf = np.clip(proba[np.arange(len(pred)), pred], 0.5, 1.0)
    conf = np.where(pred == 2, np.nan, conf)
    return pred, conf


def _f1s(y, pred):
    yl, pl = CLASSES[y], CLASSES[pred]
    f3 = f1_score(yl, pl, labels=list(CLASSES), average="macro", zero_division=0)
    ftf = f1_score(yl, pl, labels=["true", "false"], average="macro", zero_division=0)
    return f3, ftf


def tune_t(inner_proba, inner_y, feats, knobs, grid=(0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9),
           max_tf_drop=0.005):
    """Pick t on INNER out-of-fold rows: best 3-class macro-F1 with true/false macro-F1 not lower than at t=0
    by more than max_tf_drop. Ties go to the smaller t."""
    inner_y = np.asarray(inner_y)
    _, tf0 = _f1s(inner_y, unsure_layer(inner_proba, feats, {**knobs, "t": 0.0})[0])
    best_t, best = 0.0, -1.0
    for t in grid:
        f3, ftf = _f1s(inner_y, unsure_layer(inner_proba, feats, {**knobs, "t": t})[0])
        if ftf >= tf0 - max_tf_drop and f3 > best + 1e-12:
            best_t, best = t, f3
    return best_t


# ------------------------------------------------------------------ practice, out of fold
def practice_oof_rule_features(flip_fold=None):
    """G1/G3/G4 features for every non-holdout practice row, built fold by fold (eda/splits.csv fold_group):
    the timeline is anchored on training labels only and test rows vote unlabelled. flip_fold flips the
    true/false labels of that fold's test rows (leak check: the result must not change)."""
    from improve.base import label_free_table
    from improve.timeline import Timeline, anchors_for

    R, F, _, _ = label_free_table()
    P = pd.read_pickle(ROOT / "eda" / "out" / "features.pkl").set_index("id")
    S = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    cv = S.index[S["holdout"] == 0]
    labels = R.set_index("id")["label"].astype(str).to_dict()
    nfacts = R.set_index("id")["n_facts"].to_dict()
    if flip_fold is not None:
        for rid in S.index[(S["holdout"] == 0) & (S["fold_group"] == flip_fold)]:
            labels[rid] = {"true": "false", "false": "true"}.get(labels[rid], labels[rid])
    cols = ["speaker", "hedge", "joke_tail", "n_facts", "kv_line", "hearsay", "today_date_mismatch",
            "pair_n_values_max", "same_value_rows_min"]
    base_rows = P.loc[cv, cols].copy()
    base_rows["label"] = [labels[i] for i in base_rows.index]
    base_rows = base_rows.reset_index()
    sub = F[F["id"].isin(set(cv))].reset_index(drop=True)
    tl = Timeline(sub)
    parts = []
    for k in sorted(S.loc[cv, "fold_group"].unique()):
        test = set(S.index[(S["holdout"] == 0) & (S["fold_group"] == k)])
        train = set(cv) - test
        anchor = anchors_for(tl, train, labels, nfacts)
        w = np.ones(tl.n)
        feats, _ = tl.run(anchor, w, np.isin(tl.row, list(test)))
        feats["held_vkey"] = held_vkeys(tl, anchor, w)
        m = np.isin(tl.row, list(test))
        rows_k = base_rows[base_rows["id"].isin(test) | base_rows["id"].isin(train)]
        out = rule_features(sub[m], rows_k, feats[m], train)
        out = out.loc[out.index.isin(test)]
        out["tl_n_pages_for_max"] = feats[m].assign(id=tl.row[m]).groupby("id")["n_pages_for"].max()
        out["fold"] = k
        parts.append(out)
    res = pd.concat(parts)
    extra = base_rows.set_index("id").drop(columns=["speaker"])
    extra["label"] = R.set_index("id")["label"].reindex(extra.index)  # gold labels for reporting, never features
    return res.join(extra, how="left").loc[[i for i in cv if i in res.index]]

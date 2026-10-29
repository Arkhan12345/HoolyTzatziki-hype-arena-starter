"""Build every feature table the h300 ladder needs, with the honest protocol (cached in eda/h300/out/cache/).

Protocol
  practice   eda/splits.csv: CV rows (holdout==0) in 5 fold_group folds; the practice holdout pages are never used.
  outer k    a Reader (extractor + wide extractor + attribute model + KB) is fitted from scratch on the practice
             rows of the other 4 folds only; fold k's rows lose label AND form and are read from text  -> val feats.
  inner k,j  for each training fold j != k a Reader is refitted on the 3 remaining folds (margin reused from outer
             k, which never saw fold k) and reads fold j  -> training feats for outer k, out of fold.
  all        a Reader fitted on all practice CV rows reads the 300 statements (labels/forms never read)
             and supplies the R1 training feats via the outer-fold readings.
  h300       dedupe vs practice (exact + rapidfuzz ratio >= 90 -> train-only pool); the rest is split grouped by
             subject and stratified by label: 20% lockbox (scored once) + 5 CV folds; a second grouping seed for
             the subject-held-out gate.
  synth      per outer k: synthetic statements from the training folds' KB only (practice rule and closed-world
             rule), never a copy of a held-out sentence, read by the outer-k Reader.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lib.extractor import Lexicon  # noqa: E402
from lib.feats import Reader  # noqa: E402
import synth  # noqa: E402

OUT = HERE / "out"
CACHE = OUT / "cache"
SEED = 300


def load_csv(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False).to_dict("records")


def h300_splits(h300, practice):
    ptexts = [r["text"].strip() for r in practice]
    rows = []
    for r in h300:
        best = process.extractOne(r["text"].strip(), ptexts, scorer=fuzz.ratio)
        subj = json.loads(r["form"])[0]["subject"]
        rows.append({"id": r["id"], "label": r["label"], "subject": subj, "near_score": round(best[1], 1),
                     "near_text": best[0]})
    df = pd.DataFrame(rows)
    df["dedupe"] = (df.near_score >= 90).astype(int)
    df["part"] = "cv"
    df.loc[df.dedupe == 1, "part"] = "train_only"
    clean = df[df.dedupe == 0]
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    lock_idx = next(iter(sg.split(clean, clean.label, groups=clean.subject)))[1]
    df.loc[clean.index[lock_idx], "part"] = "lockbox"
    cv = df[df.part == "cv"]
    df["fold"] = -1
    for f, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1)
                                .split(cv, cv.label, groups=cv.subject)):
        df.loc[cv.index[te], "fold"] = f
    df["fold_alt"] = -1
    for f, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 7)
                                .split(cv, cv.label, groups=cv.subject)):
        df.loc[cv.index[te], "fold_alt"] = f
    return df


def feats_df(reader, rows):
    F = pd.DataFrame(reader.read(rows))
    F.insert(0, "id", [r["id"] for r in rows])
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-all", type=int, default=10000)
    ap.add_argument("--synth-fold", type=int, default=2000)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    lex_raw = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    practice = load_csv(ROOT / "data" / "practice.csv")
    h300 = load_csv(ROOT / "data" / "hooly_statements_300.csv")
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    for r in practice:
        r["fold"] = int(sp.loc[r["id"], "fold_group"])
        r["holdout"] = int(sp.loc[r["id"], "holdout"])
    cv_rows = [r for r in practice if r["holdout"] == 0]
    t0 = time.time()

    hs = h300_splits(h300, practice)
    hs.to_csv(OUT / "h300_splits.csv", index=False)
    hs[hs.dedupe == 1][["id", "near_score", "near_text"]].to_csv(OUT / "h300_dedupe.csv", index=False)
    print(f"h300 parts: {hs.part.value_counts().to_dict()}  dedupe {int(hs.dedupe.sum())}", flush=True)
    blind_h300 = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in h300]

    def blind(rows):
        return [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]

    MODE_FILE = {"practice": "practice_rule", "closed_world": "closed_world"}
    held_h300 = [r["text"] for r in h300]
    rowset = {i for i in sp.index[(sp.holdout == 1) | (sp.fold_row5 == 0)]}
    held_holdout = [r["text"] for r in practice if r["id"] in rowset]   # holdout pages + check.sh hidden rows
    stats_all = {}

    # ------------------------------------------------ synth10k from all practice CV rows, written EARLY
    for mode in ("practice", "closed_world"):
        path = OUT / f"synth10k_script_{MODE_FILE[mode]}.csv"
        if path.exists():
            continue
        srows, st = synth.generate(cv_rows, lex_raw, held_h300 + held_holdout, mode, args.synth_all,
                                   seed=SEED + (0 if mode == "practice" else 1), prefix="s10k", fold="all")
        assert st["rule_agreement"] >= 0.95, f"synthetic labels disagree with the timeline rules: {st}"
        pd.DataFrame(srows, columns=synth.COLUMNS).to_csv(path, index=False)
        pd.DataFrame(srows, columns=synth.COLUMNS).to_csv(OUT / f"synth10k_{MODE_FILE[mode]}.csv", index=False)
        stats_all[mode] = st
        print(f"WROTE {path}  {st}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(stats_all, open(OUT / "synth10k_stats.json", "w"), indent=1)

    # ------------------------------------------------ all-practice reader -> h300 + synth10k features
    p = CACHE / "h300_all.pkl"
    if not p.exists():
        rd = Reader.fit(cv_rows, lexicon)
        pickle.dump({"feats": feats_df(rd, blind_h300), "margin": rd.margin}, open(p, "wb"))
        pickle.dump(rd, open(CACHE / "reader_all.pkl", "wb"))
        print(f"all-practice reader + h300 read  ({time.time() - t0:.0f}s)", flush=True)
    for ver in ("script", "claude"):
        p = CACHE / f"synth10k_{ver}_feats.pkl"
        files = {m: OUT / f"synth10k_{ver}_{MODE_FILE[m]}.csv" for m in MODE_FILE}
        if p.exists() or not all(f.exists() for f in files.values()):
            continue
        rd = pickle.load(open(CACHE / "reader_all.pkl", "rb"))
        out = {}
        for mode, f in files.items():
            meta = pd.read_csv(f, dtype=str, keep_default_na=False)
            out[mode] = meta.merge(feats_df(rd, blind(meta.to_dict("records"))), on="id")
        pickle.dump(out, open(p, "wb"))
        print(f"synth10k[{ver}] read  ({time.time() - t0:.0f}s)", flush=True)

    for k in range(5):
        p = CACHE / f"outer_{k}.pkl"
        if p.exists():
            continue
        train = [r for r in cv_rows if r["fold"] != k]
        val = [r for r in cv_rows if r["fold"] == k]
        rd = Reader.fit(train, lexicon)
        pickle.dump(rd, open(CACHE / f"reader_outer_{k}.pkl", "wb"))
        out = {"val": feats_df(rd, blind(val)), "margin": rd.margin, "inner": {}}
        for j in sorted({r["fold"] for r in train}):
            tr_j = [r for r in train if r["fold"] != j]
            rj = Reader.fit(tr_j, lexicon, margin=rd.margin)
            out["inner"][j] = feats_df(rj, blind([r for r in train if r["fold"] == j]))
        # synthetic statements from this fold's training KB only; never a copy of a held-out text
        held = [r["text"] for r in val] + held_h300 + held_holdout
        out["synth"], out["synth_stats"] = {}, {}
        for mode in ("practice", "closed_world"):
            srows, stats = synth.generate(train, lex_raw, held, mode, args.synth_fold, seed=SEED + 31 * k + 3,
                                          prefix=f"s{k}", fold=k)
            assert stats["rule_agreement"] >= 0.95, f"synthetic labels disagree with the timeline rules: {stats}"
            meta = pd.DataFrame(srows, columns=synth.COLUMNS)
            meta.to_csv(OUT / f"synth_fold{k}_{MODE_FILE[mode]}.csv", index=False)
            out["synth"][mode] = meta.merge(feats_df(rd, blind(srows)), on="id")
            out["synth_stats"][mode] = stats
        pickle.dump(out, open(p, "wb"))
        print(f"outer {k}: val {len(val)}, inner {len(out['inner'])}, synth {out['synth_stats']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

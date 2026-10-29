"""Regenerate the per-fold synthetic sets with the current generator and re-read them with each fold's frozen
reader (eda/h300/out/cache/reader_outer_<k>.pkl), updating outer_<k>.pkl in place. Also drops the cached synth10k
readings so build.py re-reads the current 10k files.

    python eda/h300/resynth.py [--n 2000]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import synth  # noqa: E402
from build import CACHE, OUT, SEED, feats_df, load_csv  # noqa: E402

MODE_FILE = {"practice": "practice_rule", "closed_world": "closed_world"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    lex_raw = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    practice = load_csv(ROOT / "data" / "practice.csv")
    h300 = load_csv(ROOT / "data" / "hooly_statements_300.csv")
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    for r in practice:
        r["fold"], r["holdout"] = int(sp.loc[r["id"], "fold_group"]), int(sp.loc[r["id"], "holdout"])
    cv = [r for r in practice if r["holdout"] == 0]
    rowset = set(sp.index[(sp.holdout == 1) | (sp.fold_row5 == 0)])
    held_base = [r["text"] for r in practice if r["id"] in rowset] + [r["text"] for r in h300]
    blind = lambda rows: [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]  # noqa: E731
    stats = {}
    for k in range(5):
        p = CACHE / f"outer_{k}.pkl"
        o = pickle.load(open(p, "rb"))
        rd = pickle.load(open(CACHE / f"reader_outer_{k}.pkl", "rb"))
        train = [r for r in cv if r["fold"] != k]
        held = [r["text"] for r in cv if r["fold"] == k] + held_base
        o["synth"], o["synth_stats"] = {}, {}
        for mode in MODE_FILE:
            rows, st = synth.generate(train, lex_raw, held, mode, a.n, seed=SEED + 31 * k + 3, prefix=f"s{k}", fold=k)
            assert st["rule_agreement"] >= 0.95, st
            meta = pd.DataFrame(rows, columns=synth.COLUMNS)
            meta.to_csv(OUT / f"synth_fold{k}_{MODE_FILE[mode]}.csv", index=False)
            o["synth"][mode] = meta.merge(feats_df(rd, blind(rows)), on="id")
            o["synth_stats"][mode] = st
            stats[f"fold{k}_{mode}"] = st
        pickle.dump(o, open(p, "wb"))
        print(f"fold {k}: {o['synth_stats']}", flush=True)
    json.dump(stats, open(OUT / "synth_fold_stats.json", "w"), indent=1)
    (CACHE / "synth10k_script_feats.pkl").unlink(missing_ok=True)


if __name__ == "__main__":
    main()

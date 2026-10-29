"""Feature cache for the v2 ladder (eda/h300/out/cache_v2/), the v1 caches stay untouched.

    .venv/Scripts/python eda/h300/build_v2.py [--folds 0,1,2,3,4]

Per outer fold k of eda/splits.csv fold_group (holdout pages never read here):
  val        validation rows read by the reader fitted without fold k              (reader_outer_k from the v1 build)
  inner[j]   training rows of fold j read by a reader fitted without folds j and k (out of fold, as in v1)
  ren_val    the validation rows with every subject renamed to an invented name, same reader as val
  ren_inner  the same for the inner rows (renamed-subject augmentation for training)
  synth      the fold's script synthetic statements (practice rule + closed world), read by reader k
Page features (page size, position, first-person share) are added from file order: they need no labels.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lib.extractor import Lexicon  # noqa: E402
from lib.feats import Reader  # noqa: E402

CACHE, CACHE2 = HERE / "out" / "cache", HERE / "out" / "cache_v2"
FIRST = ["Brenn", "Cald", "Dorn", "Elm", "Fask", "Grim", "Hask", "Isel", "Jorv", "Kest", "Lunn", "Mord", "Nask",
         "Oster", "Prill", "Quen", "Rask", "Sorr", "Tamb", "Ulm", "Vask", "Wend", "Yarr", "Zell"]
SECOND = ["bury", "dale", "holt", "mere", "ridge", "shaw", "thorpe", "wold", "combe", "garth", "haven", "ley"]
KIND = ["", "", " Group", " Digital", " Tech", " Networks", " Studio", " Partners"]
FP = re.compile(r"\b(I|I'm|me|my)\b")


def rename_map(lex_raw, seed=7):
    """Every lexicon subject and event -> a deterministic invented name that is not in the lexicon."""
    rng = random.Random(seed)
    names = [s["name"] for s in lex_raw["subjects"]] + [e if isinstance(e, str) else e["name"]
                                                       for e in lex_raw.get("events", [])]
    taken = {n.lower() for n in names} | {a.lower() for s in lex_raw["subjects"] for a in s.get("aliases", [])}
    out, used = {}, set()
    for n in names:
        while True:
            nm = rng.choice(FIRST) + rng.choice(SECOND) + rng.choice(KIND)
            if nm.lower() not in taken and nm not in used:
                used.add(nm)
                out[n] = nm
                break
    aliases = {s["name"]: s.get("aliases", []) for s in lex_raw["subjects"]}
    return out, aliases


def rename_row(row, mapping, aliases):
    subjects = [f["subject"] for f in json.loads(row.get("form") or "[]")]
    text = row["text"]
    pairs = [(nm, mapping[s]) for s in subjects if s in mapping for nm in [s] + aliases.get(s, [])]
    for nm, new in sorted(pairs, key=lambda p: -len(p[0])):
        text = re.sub(r"(?<![\w-])" + re.escape(nm) + r"(?![\w-])", new, text, flags=re.I)
    return {"id": "rn-" + row["id"], "text": text, "date": row["date"], "label": ""}


def page_features(rows):
    """Contiguous same-date runs in the given order; label-free."""
    pages, cur, prev = [], -1, None
    for r in rows:
        if r["date"] != prev:
            cur += 1
            prev = r["date"]
        pages.append(cur)
    df = pd.DataFrame({"id": [r["id"] for r in rows], "page": pages,
                       "fp": [int(bool(FP.search(r["text"]))) for r in rows]})
    df["v2_page_size"] = df.groupby("page").id.transform("size")
    df["v2_pos_in_page"] = df.groupby("page").cumcount()
    df["v2_page_first_person"] = df.groupby("page").fp.transform("mean")
    return df[["id", "v2_page_size", "v2_pos_in_page", "v2_page_first_person"]]


def read(reader, rows):
    reader.unknown_subject_reading = True   # v2: still read value + attribute when the subject is unknown
    F = pd.DataFrame(reader.read([{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]))
    F.insert(0, "id", [r["id"] for r in rows])
    return F.merge(page_features(rows), on="id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    CACHE2.mkdir(parents=True, exist_ok=True)
    lex_raw = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    mapping, aliases = rename_map(lex_raw)
    json.dump(mapping, open(CACHE2 / "rename_map.json", "w"), indent=1)
    practice = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False).to_dict("records")
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    for r in practice:
        r["fold"] = int(sp.loc[r["id"], "fold_group"])
    cv = [r for r in practice if r["fold"] >= 0]
    t0 = time.time()
    for k in [int(x) for x in a.folds.split(",")]:
        p = CACHE2 / f"outer_{k}.pkl"
        if p.exists():
            continue
        rd = pickle.load(open(CACHE / f"reader_outer_{k}.pkl", "rb"))
        train = [r for r in cv if r["fold"] != k]
        val = [r for r in cv if r["fold"] == k]
        out = {"val": read(rd, val), "ren_val": read(rd, [rename_row(r, mapping, aliases) for r in val]),
               "inner": {}, "ren_inner": {}, "synth": {}}
        for j in sorted({r["fold"] for r in train}):
            rj = Reader.fit([r for r in train if r["fold"] != j], lexicon, margin=rd.margin)
            rows_j = [r for r in train if r["fold"] == j]
            out["inner"][j] = read(rj, rows_j)
            out["ren_inner"][j] = read(rj, [rename_row(r, mapping, aliases) for r in rows_j])
            print(f"  outer {k} inner {j} ({time.time() - t0:.0f}s)", flush=True)
        for mode, fname in (("practice", "practice_rule"), ("closed_world", "closed_world")):
            meta = pd.read_csv(HERE / "out" / f"synth_fold{k}_{fname}.csv", dtype=str, keep_default_na=False)
            out["synth"][mode] = meta[["id", "label", "case_type"]].merge(read(rd, meta.to_dict("records")), on="id")
        pickle.dump(out, open(p, "wb"))
        print(f"outer {k} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""Build the outside validation set ONCE (fixed seed), before any model sees it.

    .venv/Scripts/python eda/sim/make_outside.py

From data/practice.csv, three kinds of rows leave the training universe:
  subject   every row about ~12% of the subjects with >= 5 rows (chosen at random): whole subjects never seen
  time      every row dated in the last 6 weeks of the year (time-forward)
  page      ~10% of the remaining pages (a page = contiguous rows with one date, as in eda/06_splits.py)
A row can carry several flags; its `slice` is the first of subject > time > page.
A second, harder copy of every outside row is added with its subjects (names and aliases) replaced by invented names
that are not in the lexicon, values and dates untouched (`renamed` = 1, id prefixed with `r-`).

Writes eda/sim/out/outside.csv and eda/sim/out/outside.sha256. Refuses to overwrite an existing set.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
SEED = 20261029
SUBJECT_SHARE = 0.12
LAST_DAYS = 42
PAGE_SHARE = 0.10
FIRST = ["Ember", "Fern", "Pask", "Fallow", "Nesh", "Umber", "Holl", "Quill", "Brack", "Corr", "Dun", "Esk", "Gant",
         "Harth", "Ivel", "Jasp", "Kell", "Lorn", "Marr", "Norw", "Orrin", "Pell", "Rook", "Sand", "Tarn", "Vell"]
SECOND = ["keep", "hollow", "wright", "mark", "spar", "path", "fern", "ford", "stone", "brook", "gate", "moor",
          "wick", "vale", "crest", "field"]
KIND = ["", "", " Freight", " Mobile", " Labs", " Systems", " Works", " Cloud"]


def subjects_of(form):
    return [f["subject"] for f in json.loads(form)] if form else []


def invented_names(n, taken, rng):
    names, seen = [], {t.lower() for t in taken}
    while len(names) < n:
        nm = rng.choice(FIRST) + rng.choice(SECOND) + rng.choice(KIND)
        if nm.lower() not in seen and not any(nm.lower() in t or t in nm.lower() for t in seen if len(t) > 3):
            seen.add(nm.lower())
            names.append(nm)
    return names


def rename(text, subjects, aliases, mapping):
    """Replace every name/alias of the row's subjects (longest first) by the subject's invented name."""
    pairs = []
    for s in subjects:
        for nm in [s] + aliases.get(s, []):
            pairs.append((nm, mapping[s]))
    for nm, new in sorted(pairs, key=lambda p: -len(p[0])):
        text = re.sub(r"(?<![\w-])" + re.escape(nm) + r"(?![\w-])", new, text, flags=re.I)
    return text


def main():
    p_out, p_sha = OUT / "outside.csv", OUT / "outside.sha256"
    if p_out.exists():
        sys.exit(f"{p_out} already exists: the outside set is built once and never rebuilt")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    aliases = {s["name"]: s.get("aliases", []) for s in lex["subjects"]}
    for e in lex.get("events", []):
        if isinstance(e, dict):
            aliases.setdefault(e["name"], e.get("aliases", []))
        else:
            aliases.setdefault(e, [])
    df = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False)
    d = pd.to_datetime(df.date)
    df["page"] = (d != d.shift()).cumsum()
    df["subjects"] = df.form.map(subjects_of)

    counts = df.explode("subjects").subjects.value_counts()
    eligible = sorted(counts[counts >= 5].index)
    k = max(1, round(SUBJECT_SHARE * len(eligible)))
    held_subjects = set(rng.sample(eligible, k))
    df["slice_subject"] = df.subjects.map(lambda ss: int(any(s in held_subjects for s in ss)))
    cutoff = d.max() - pd.Timedelta(days=LAST_DAYS)
    df["slice_time"] = (d > cutoff).astype(int)
    rest_pages = sorted(df.loc[(df.slice_subject == 0) & (df.slice_time == 0), "page"].unique())
    held_pages = set(rng.sample(rest_pages, round(PAGE_SHARE * len(rest_pages))))
    df["slice_page"] = df.page.isin(held_pages).astype(int)
    out = df[(df.slice_subject + df.slice_time + df.slice_page) > 0].copy()
    out["slice"] = out.apply(lambda r: "subject" if r.slice_subject else ("time" if r.slice_time else "page"), axis=1)
    out["renamed"] = 0
    out["orig_id"] = out["id"]

    # harder copy: the outside rows with their subjects renamed to invented names unknown to the lexicon
    taken = list(aliases) + [a for v in aliases.values() for a in v]
    subj_all = sorted({s for ss in out.subjects for s in ss})
    mapping = dict(zip(subj_all, invented_names(len(subj_all), taken, rng)))
    ren = out.copy()
    ren["text"] = [rename(t, ss, aliases, mapping) for t, ss in zip(ren.text, ren.subjects)]
    ren["id"] = "r-" + ren["orig_id"]
    ren["renamed"] = 1
    cols = ["id", "orig_id", "text", "date", "label", "slice", "slice_subject", "slice_time", "slice_page", "renamed",
            "page"]
    full = pd.concat([out[cols], ren[cols]], ignore_index=True)
    full.to_csv(p_out, index=False)
    p_sha.write_text(hashlib.sha256(p_out.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    json.dump({"seed": SEED, "held_subjects": sorted(held_subjects), "cutoff_date": str(cutoff.date()),
               "held_pages": len(held_pages), "rename_map": mapping},
              open(OUT / "outside_meta.json", "w", encoding="utf-8"), indent=1)

    base = out
    print(f"practice rows {len(df)}; outside rows {len(base)} ({len(base) / len(df):.1%}); renamed copies {len(ren)}")
    print(f"held subjects ({len(held_subjects)} of {len(eligible)} eligible): {sorted(held_subjects)}")
    print(f"time cutoff > {cutoff.date()}; held pages {len(held_pages)} of {len(rest_pages)}")
    print("slice x label:\n", pd.crosstab(base.slice, base.label, margins=True).to_string())
    print("changed renamed texts:", int((ren.text.values != out.text.values).sum()), "of", len(ren))


if __name__ == "__main__":
    main()

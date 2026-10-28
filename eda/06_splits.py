"""Step 6: one shared split file for the whole team -> data/splits.csv

No copied folders: every script loads practice.csv, joins this file on `id` and filters in memory.
Columns:
  holdout     1 = frozen test pages (~15% of pages). Never used for tuning or model choice; score on it once at the end.
  fold_group  0-4: StratifiedGroupKFold by page on the non-holdout rows (-1 for holdout rows). Main CV.
  fold_row5   0-4: row index % 5 in file order. fold 0 is exactly what scripts/check.sh hides (input.py: i % 5 == 0).
  time_split  train = date < 2026-07-01, test = later. Pessimistic "future rows" check.
Seed 42; rerun gives the same file.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def main():
    df = pd.read_csv(ROOT / "data" / "practice.csv", encoding="utf-8-sig")  # file order, as input.py reads it
    df["fold_row5"] = np.arange(len(df)) % 5
    d = pd.to_datetime(df["date"])
    df["page"] = (d != d.shift()).cumsum()  # contiguous ids with one date (same rule as eda/01_parse.py)

    # page label profile for stratifying pages: worst label on the page
    worst = df.groupby("page")["label"].agg(lambda s: "unsure" if (s == "unsure").any()
                                            else "false" if (s == "false").any() else "true")
    rng = np.random.default_rng(SEED)
    hold_pages = []
    for lab, pages in worst.groupby(worst):
        p = pages.index.to_numpy()
        k = max(1, round(0.15 * len(p)))
        hold_pages += list(rng.choice(p, size=k, replace=False))
    df["holdout"] = df["page"].isin(hold_pages).astype(int)

    df["fold_group"] = -1
    cv = df["holdout"] == 0
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f, (_, te) in enumerate(sgkf.split(df[cv], df.loc[cv, "label"], groups=df.loc[cv, "page"])):
        df.loc[df[cv].index[te], "fold_group"] = f

    df["time_split"] = np.where(d < "2026-07-01", "train", "test")

    out = df[["id", "page", "date", "label", "holdout", "fold_group", "fold_row5", "time_split"]]
    out.to_csv(ROOT / "data" / "splits.csv", index=False)

    pd.set_option("display.width", 200)
    print("holdout:", out.groupby("holdout").label.value_counts().unstack().to_string(), sep="\n")
    print("\nfold_group (non-holdout):",
          out[out.holdout == 0].groupby("fold_group").agg(rows=("id", "size"), pages=("page", "nunique"),
                                                         false=("label", lambda s: (s == "false").sum()),
                                                         unsure=("label", lambda s: (s == "unsure").sum())).to_string(),
          sep="\n")
    print("\nfold_row5 = 0 rows (check.sh hidden):", (out.fold_row5 == 0).sum())
    print("time_split:", out.time_split.value_counts().to_dict())


if __name__ == "__main__":
    main()

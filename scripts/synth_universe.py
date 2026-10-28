"""Make a full-size universe for load tests: practice x N with new ids, ~90% of labels blank.

    python scripts/synth_universe.py OUT_DIR [--times 10]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--times", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(ROOT, "data", "practice.csv"), encoding="utf-8-sig", newline="") as f:
        practice = list(csv.DictReader(f))
    rnd = random.Random(7)
    with open(os.path.join(args.out, "universe.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "date", "label"])
        for r in practice:  # the practice rows keep their labels, as in the real run
            w.writerow([r["id"], r["text"], r["date"], r["label"]])
        for k in range(1, args.times):
            for i, r in enumerate(practice):
                w.writerow([f"s{k}-{i}", r["text"] + ("" if rnd.random() < 0.5 else f" ({k})"), r["date"], ""])
    shutil.copy(os.path.join(ROOT, "data", "lexicon.json"), args.out)
    print(f"{len(practice) * args.times} rows in {args.out}")


if __name__ == "__main__":
    main()

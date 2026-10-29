"""Baseline predictions on the 300 statements, computed once (eda/h300/out/baseline_*.csv).

  shipped   the option A stack saved on the classifier branch (models/truth_stack.joblib), run as shipped:
            practice rows labelled, the 300 statements unlabelled.
  closed    the closed-world labeller's published predictions from the same branch (fitted on practice only).

Run from anywhere with the branch checked out in a separate worktree:
    python eda/h300/baselines.py <path-to-classifier-branch-worktree>
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(wt: Path):
    sys.path.insert(0, str(wt))
    import joblib  # noqa: E402

    practice = list(csv.DictReader(open(ROOT / "data" / "practice.csv", encoding="utf-8-sig")))
    h300 = list(csv.DictReader(open(ROOT / "data" / "hooly_statements_300.csv", encoding="utf-8-sig")))
    pipe = joblib.load(wt / "models" / "truth_stack.joblib")
    universe = [{k: r[k] for k in ("id", "text", "date", "label")} for r in practice]
    universe += [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in h300]
    out = pipe.label(universe)
    with open(HERE / "out" / "baseline_shipped.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "confidence"])
        w.writeheader()
        for o in out:
            w.writerow({"id": o["id"], "label": o["label"], "confidence": o["confidence"]})
    src = wt / "train" / "statements300_predictions.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    with open(HERE / "out" / "baseline_closed.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "label", "confidence"])
        w.writeheader()
        for r in rows:
            w.writerow({"id": r["id"], "label": r["pred"], "confidence": r["confidence"]})
    print("wrote baseline_shipped.csv", len(out), "and baseline_closed.csv", len(rows))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

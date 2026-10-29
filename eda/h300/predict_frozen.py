"""Frozen inference: label a CSV with a saved model. Fits NOTHING.

    python eda/h300/predict_frozen.py INPUT.csv [--model eda/models/FINAL] [--out OUTPUT.csv]

INPUT.csv needs id, text, date (any of a few common date spellings); a label column is ignored. The model's reader
(extractor, attribute model, practice KB) and learned layer were fitted once and frozen in model.joblib. Rows it
cannot read into a known fact (unseen subject or attribute, no value found, empty text) keep a capped confidence
and turn unsure when the learned layer is undecided. Output: id, label, confidence (empty for unsure).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "eda"))   # frozen models are pickled under h300.lib.*


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        r = {str(k).strip().lower(): v for k, v in r.items() if k is not None}
        out.append({"id": (r.get("id") or f"row-{i + 1}").strip(), "text": r.get("text") or "",
                     "date": r.get("date") or "", "label": ""})
    return out


def predict(model_dir, rows):
    import joblib
    model = joblib.load(Path(model_dir) / "model.joblib")
    return model.predict_rows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=str(ROOT / "eda" / "models" / "FINAL"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = read_rows(a.input)
    out = predict(a.model, rows)
    dst = a.out or str(Path(a.input).with_suffix("")) + f".{Path(a.model).name}.labels.csv"
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "confidence"])
        for o in out:
            w.writerow([o["id"], o["label"], "" if o["confidence"] is None else f"{o['confidence']:.4f}"])
    manifest = json.loads((Path(a.model) / "manifest.json").read_text(encoding="utf-8"))
    print(f"{len(out)} rows labelled by {manifest['name']} (sha256 {manifest['sha256'][:12]}) -> {dst}")


if __name__ == "__main__":
    main()

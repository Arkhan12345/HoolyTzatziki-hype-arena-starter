"""Validate the frozen models ONCE on a new dataset (e.g. the 09:00 sample). No tuning is possible after it.

    python eda/h300/validate_once.py NEW.csv

- Runs predict_frozen with eda/models/FINAL and eda/models/FALLBACK (nothing is fitted).
- Scores both if the file has labels (accuracy, 3-class macro-F1, true/false macro-F1, per-label counts).
- Writes eda/h300/out/validate_once_<hash>.md and the two label files next to it.
- Refuses to run a second time on the same file content (sha256 lock in eda/h300/out/validate_once_locks.json),
  so the result can't be used to tune and then re-scored.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from predict_frozen import predict, read_rows  # noqa: E402

LOCKS = HERE / "out" / "validate_once_locks.json"
MODELS = {"FINAL": ROOT / "eda" / "models" / "FINAL", "FALLBACK": ROOT / "eda" / "models" / "FALLBACK"}


def score(gold, pred):
    labs = ["true", "false", "unsure"]
    f1 = {}
    for c in labs:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        f1[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)
    return {"n": len(gold), "accuracy": round(acc, 4), "macro_f1_3": round(sum(f1.values()) / 3, 4),
            "macro_f1_true_false": round((f1["true"] + f1["false"]) / 2, 4),
            "all_true_accuracy": round(sum(g == "true" for g in gold) / max(len(gold), 1), 4)}


def main():
    src = Path(sys.argv[1])
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    locks = json.loads(LOCKS.read_text()) if LOCKS.exists() else {}
    if digest in locks:
        sys.exit(f"REFUSED: {src.name} (sha256 {digest[:12]}) was already validated on {locks[digest]['when']}. "
                 f"A second run would let the result steer the model. Report: {locks[digest]['report']}")
    rows = read_rows(src)
    with open(src, encoding="utf-8-sig", newline="") as fh:
        gold = {r.get("id", ""): (r.get("label") or "").strip().lower() for r in csv.DictReader(fh)}
    labelled = [r["id"] for r in rows if gold.get(r["id"]) in ("true", "false", "unsure")]
    lines = [f"# One-time validation: {src.name}", "", f"- sha256 `{digest}`",
             f"- run {datetime.now().isoformat(timespec='seconds')}", f"- rows {len(rows)}, labelled {len(labelled)}", ""]
    out_dir = HERE / "out"
    for name, mdir in MODELS.items():
        if not (mdir / "model.joblib").exists():
            lines.append(f"## {name}: model missing ({mdir})")
            continue
        man = json.loads((mdir / "manifest.json").read_text(encoding="utf-8"))
        preds = predict(mdir, rows)
        dst = out_dir / f"validate_once_{digest[:8]}.{name}.labels.csv"
        with open(dst, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "label", "confidence"])
            for o in preds:
                w.writerow([o["id"], o["label"], "" if o["confidence"] is None else f"{o['confidence']:.4f}"])
        lines += [f"## {name} (sha256 {man['sha256'][:12]}, created {man['created']})", "",
                  f"- labels: {dict((l, sum(o['label'] == l for o in preds)) for l in ('true', 'false', 'unsure'))}",
                  f"- file: `{dst.relative_to(ROOT)}`"]
        if labelled:
            p = {o["id"]: o["label"] for o in preds}
            lines.append(f"- score: {score([gold[i] for i in labelled], [p[i] for i in labelled])}")
        lines.append("")
    report = out_dir / f"validate_once_{digest[:8]}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    locks[digest] = {"file": str(src), "when": datetime.now().isoformat(timespec="seconds"),
                     "report": str(report.relative_to(ROOT))}
    LOCKS.write_text(json.dumps(locks, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

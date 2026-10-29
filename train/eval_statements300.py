"""Score labellers on data/hooly_statements_300.csv, trained on practice.csv only.

The 300 statements are all dated 2026-10-28, 100 true / 100 false / 100 unsure.
Their labels follow a closed-world rule over practice.csv: true = the latest
true value of the (subject, attribute), false = another value for a known
pair, unsure = a pair practice never states.

Honesty rules:
- Every model is fitted on practice.csv (+ lexicon.json) only.
- At prediction time the labellers see id, text and date; the `form` column
  is read only to score (and for the gold-fact ceiling, marked as such).
- The 300 rows are split once (seeded, stratified) into 100 dev rows, used to
  look at errors, and 200 locked rows, the number to report.

    python train/eval_statements300.py           # closed-world labeller
    python train/eval_statements300.py --stack   # also the shipped TruthStack
    python train/eval_statements300.py --locked  # final score: also the locked 200

Writes train/results_statements300.csv and train/statements300_predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classify.closed_world import ClosedWorldLabeller, PracticeKB  # noqa: E402
from app.extract.extractor import Lexicon  # noqa: E402
from app.extract.values import canonical  # noqa: E402

PRACTICE_PATH = ROOT / "data" / "practice.csv"
STATEMENTS_PATH = ROOT / "data" / "hooly_statements_300.csv"
LEXICON_PATH = ROOT / "data" / "lexicon.json"
MODEL_PATH = ROOT / "models" / "truth_stack.joblib"
RESULTS_PATH = ROOT / "train" / "results_statements300.csv"
PREDICTIONS_PATH = ROOT / "train" / "statements300_predictions.csv"
LABELS = ("true", "false", "unsure")
SPLIT_SEED = 300
N_DEV = 100
RESULT_FIELDS = ["model", "split", "n", "acc", "f1_true", "f1_false", "f1_unsure", "macro_f1_3",
                 "macro_f1_tf", "ece_tf", "excluded_overlap"]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split(rows: list[dict]) -> dict[str, str]:
    """id -> "dev" | "locked", stratified by label, fixed seed."""
    rng = random.Random(SPLIT_SEED)
    out = {}
    for lab in LABELS:
        ids = sorted(r["id"] for r in rows if r["label"] == lab)
        rng.shuffle(ids)
        k = round(N_DEV * len(ids) / len(rows))
        out.update({i: "dev" for i in ids[:k]})
        out.update({i: "locked" for i in ids[k:]})
    return out


def f1(cm: Counter, c: str) -> float:
    tp = cm[(c, c)]
    fp = sum(v for (g, p), v in cm.items() if p == c and g != c)
    fn = sum(v for (g, p), v in cm.items() if g == c and p != c)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = 0.5 + b * 0.05, 0.5 + (b + 1) * 0.05
        sel = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == hi)]
        if sel:
            total += len(sel) * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
    return total / len(pairs)


def score(name: str, gold: list[dict], pred: dict[str, dict], which: str) -> dict:
    cm = Counter((g["label"], pred[g["id"]]["label"]) for g in gold)
    tf = [(pred[g["id"]]["confidence"], pred[g["id"]]["label"] == g["label"]) for g in gold
          if pred[g["id"]]["label"] in ("true", "false") and pred[g["id"]]["confidence"] is not None]
    e = ece(tf)
    res = {"model": name, "split": which, "n": len(gold),
           "acc": sum(cm[(c, c)] for c in LABELS) / max(len(gold), 1),
           **{f"f1_{c}": f1(cm, c) for c in LABELS},
           "macro_f1_3": sum(f1(cm, c) for c in LABELS) / 3,
           "macro_f1_tf": (f1(cm, "true") + f1(cm, "false")) / 2,
           "ece_tf": e if e is not None else ""}
    print(f"\n{name} [{which}] n={len(gold)}  acc {res['acc']:.3f}  macro-F1(3) {res['macro_f1_3']:.3f}"
          f"  F1 t/f/u {res['f1_true']:.3f}/{res['f1_false']:.3f}/{res['f1_unsure']:.3f}"
          f"  ECE(t/f) {res['ece_tf'] if res['ece_tf'] == '' else round(res['ece_tf'], 3)}")
    print("  gold \\ pred  " + "  ".join(f"{p:>6}" for p in LABELS))
    for g in LABELS:
        print(f"  {g:>11}  " + "  ".join(f"{cm[(g, p)]:>6}" for p in LABELS))
    return res


def gold_ceiling(practice: list[dict], rows: list[dict]) -> dict[str, dict]:
    """The closed-world rule on each statement's gold `form`: tests the rule, not the reading."""
    kb = PracticeKB.from_rows(practice)
    out = {}
    for r in rows:
        f = json.loads(r["form"])[0]
        latest = kb.latest_true(f["subject"], f["attribute"], r["date"][:10])
        lab = "unsure" if latest is None else ("true" if canonical(f["value"]) == latest else "false")
        out[r["id"]] = {"label": lab, "confidence": None if lab == "unsure" else 1.0}
    return out


def shipped_stack(practice: list[dict], rows: list[dict]) -> dict[str, dict]:
    import joblib
    pipe = joblib.load(MODEL_PATH)
    universe = [{k: r[k] for k in ("id", "text", "date", "label")} for r in practice]
    universe += [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]
    return {o["id"]: o for o in pipe.label(universe)}


def fact_reading(rows: list[dict], pred: dict[str, dict]) -> Counter:
    """Was the fact the labeller decided on the gold fact? (true/false gold rows)"""
    c = Counter()
    for r in rows:
        if r["label"] == "unsure":
            continue
        g = json.loads(r["form"])[0]
        f = pred[r["id"]].get("fact")
        if f is None:
            c["no fact"] += 1
        elif (f["subject"], f["attribute"]) != (g["subject"], g["attribute"]):
            c["wrong pair"] += 1
        elif canonical(f["value"]) != canonical(g["value"]):
            c["wrong value"] += 1
        else:
            c["gold fact"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", action="store_true", help="also score the shipped TruthStack")
    ap.add_argument("--no-rescue", action="store_true", help="a value equal to another known attribute is not true")
    ap.add_argument("--subject-mode", default="last", choices=("none", "last", "all"))
    ap.add_argument("--synonyms", action="store_true", help="confusable outside attributes are no evidence of unsure")
    ap.add_argument("--extractor-first", action="store_true", help="pair from the classic extractor")
    ap.add_argument("--locked", action="store_true", help="also score the locked 200 (final runs only)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    practice = load(PRACTICE_PATH)
    statements = load(STATEMENTS_PATH)
    practice_texts = {r["text"].strip() for r in practice}
    overlap = {r["id"] for r in statements if r["text"].strip() in practice_texts}
    parts = split(statements)
    blind = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in statements]

    t0 = time.time()
    labeller = ClosedWorldLabeller.fit(practice, Lexicon.load(LEXICON_PATH), subject_mode=args.subject_mode,
                                       synonyms=args.synonyms, rescue=not args.no_rescue,
                                       prefer_extractor=args.extractor_first)
    print(f"fitted on practice in {time.time() - t0:.0f} s; unsure margin tuned on practice: {labeller.tuning}")
    cw = {o["id"]: o for o in labeller.label(blind)}

    models = {"closed-world (practice only)": cw, "gold facts + closed-world rule (ceiling)":
              gold_ceiling(practice, statements)}
    if args.stack:
        models["shipped TruthStack"] = shipped_stack(practice, statements)

    splits = ("dev", "locked") if args.locked else ("dev",)
    results = []
    for name, pred in models.items():
        for which in splits:
            gold = [r for r in statements if parts[r["id"]] == which and r["id"] not in overlap]
            res = score(name, gold, pred, which)
            res["excluded_overlap"] = len(overlap)
            results.append(res)
    for which in splits:
        sub = [r for r in statements if parts[r["id"]] == which and r["id"] not in overlap]
        print(f"\nclosed-world fact reading on true/false rows [{which}]: {dict(fact_reading(sub, cw))}")

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for res in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in res.items()})
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["id", "split", "gold", "pred", "confidence", "reason", "fact", "text"])
        for r in statements:
            o = cw[r["id"]]
            w.writerow([r["id"], "overlap" if r["id"] in overlap else parts[r["id"]], r["label"], o["label"],
                        "" if o["confidence"] is None else f"{o['confidence']:.3f}", o["reason"],
                        json.dumps(o["fact"]) if o["fact"] else "", r["text"]])
    print(f"\nwrote {RESULTS_PATH.relative_to(ROOT)} and {PREDICTIONS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

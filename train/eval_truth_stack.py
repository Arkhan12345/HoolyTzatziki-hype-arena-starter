"""Evaluate options A, B and D on the same folds and save the one that ships.

A     = extractor + temporal KB + LightGBM/LogReg stack.
B     = A + unsupervised truth-discovery features (app/classify/truth_discovery.py).
D     = A + a CfC liquid net over each fact's timeline as a third stack member
        (app/classify/timeline_net.py).
D-GRU = the same with a plain GRU: the control for "does liquid help?".

Another option replaces A only if its grouped-CV stack beats A by SHIP_MARGIN
and its time-forward score is within TIME_TOLERANCE of A's (time-forward is a
single noisy split). By default A and B run; name options to add D:
`python train/eval_truth_stack.py D D-GRU` (needs torch + ncps, ~4x slower).

Honesty rules:
- Facts for every practice row come from an extractor that never saw that
  row's page (5-fold GroupKFold by page). The classifier folds are the same
  5 folds, so a test row's facts never come from a model trained on its page.
- KB features of test rows use training labels only; training rows get theirs
  out-of-fold (inner GroupKFold), never from their own page.
- As at run time, labelled rows contribute their hand-labelled `form` facts to
  the KB and support features; rows being judged always use extracted facts.
- The P(false) threshold and calibration are tuned on training rows only.

Writes train/results.csv (one row per model x CV scheme) and
models/truth_stack.joblib (the pipeline used at run time).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classify.truth_features import fact_table, label_free_features  # noqa: E402
from app.classify.timeline_net import build_sequences  # noqa: E402
from app.classify.truth_model import (  # noqa: E402
    TruthStack,
    fit_pipeline,
    labels_from_p_false,
    macro_f1_true_false,
    predict_table,
    gold_facts_from_rows,
    mixed_table,
    mixed_training_table,
    training_sequences,
    training_table,
)
from app.extract.extractor import FactExtractor, Lexicon, fact_scores, page_ids  # noqa: E402

DATA_PATH = ROOT / "data" / "practice.csv"
LEXICON_PATH = ROOT / "data" / "lexicon.json"
MODEL_PATH = ROOT / "models" / "truth_stack.joblib"
RESULTS_PATH = ROOT / "train" / "results.csv"
TD_KWARGS = {"tau_days": 60.0}
# Seed-to-seed noise: about +-0.5 points grouped, +-3 points time-forward.
SHIP_MARGIN = 0.01
TIME_TOLERANCE = 0.03
OPTIONS = {
    # name: (truth discovery on?, sequence net, results.csv features text)
    "A": (False, None, "judged rows extracted, labelled rows hand-labelled + KB + support + text"),
    "B": (True, None, "A + truth discovery (EM, tau 60 d)"),
    "D": (False, "cfc", "A + CfC liquid net over fact timelines"),
    "D-GRU": (False, "gru", "A + GRU over fact timelines (control)"),
}
RESULT_FIELDS = ["model", "track", "features", "cv", "macro_f1_tf", "acc", "ece", "brier",
                 "false_precision", "false_recall", "n"]


def load_rows() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def ece_binary(labels: np.ndarray, p_false: np.ndarray, predicted: np.ndarray, bins: int = 10) -> float:
    """ECE of the shipped confidence: P(chosen label) vs how often it is right."""
    conf = np.where(predicted == "false", p_false, 1 - p_false)
    correct = (predicted == "false") == (labels == "false")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def scores(labels: np.ndarray, p_false: np.ndarray, predicted: np.ndarray) -> dict:
    is_false = labels == "false"
    pred_false = predicted == "false"
    tp = int(np.sum(is_false & pred_false))
    return {
        "macro_f1_tf": macro_f1_true_false(labels, predicted),
        "acc": float(np.mean(np.where(labels == "unsure", "true", labels) == predicted)),
        "ece": ece_binary(labels, p_false, predicted),
        "brier": float(np.mean((p_false - is_false) ** 2)),
        "false_precision": tp / max(int(pred_false.sum()), 1),
        "false_recall": tp / max(int(is_false.sum()), 1),
        "n": len(labels),
    }


def print_block(name: str, labels: np.ndarray, comp_p: dict, comp_pred: dict) -> list[dict]:
    print(f"\n{name}")
    print(f"{'model':<10}{'macro-F1':>10}{'acc':>8}{'ECE':>8}{'Brier':>8}{'P(false)':>10}{'R(false)':>10}")
    out = []
    for comp in comp_p:
        s = scores(labels, comp_p[comp], comp_pred[comp])
        out.append({"component": comp, **s})
        print(f"{comp:<10}{s['macro_f1_tf']:>10.4f}{s['acc']:>8.4f}{s['ece']:>8.4f}{s['brier']:>8.4f}"
              f"{s['false_precision']:>10.3f}{s['false_recall']:>10.3f}")
    pred = comp_pred["stack"]
    print("Stack confusion (rows actual true/false/unsure, cols predicted true/false):")
    for lab in ("true", "false", "unsure"):
        m = labels == lab
        print(f"  {lab:<7}{int(np.sum(pred[m] == 'true')):>6}{int(np.sum(pred[m] == 'false')):>6}")
    return out


def evaluate(rows, facts, F, label_free, labels, groups, train_idx, test_idx, seed,
             seq_kind=None, gold=None, td=False):
    """One train/test split. With `gold`, labelled rows use hand-labelled facts
    (as at run time); option D keeps the all-extracted path."""
    known = {int(i): labels[i] for i in train_idx}
    if gold is not None and not seq_kind:
        kw = {"truth_discovery": td, "td_kwargs": TD_KWARGS if td else None}
        X_train = mixed_training_table(rows, facts, gold, train_idx, labels, groups, **kw)
        X_test = mixed_table(rows, facts, gold, known, test_idx, **kw)
        seq_train = seq_test = None
    else:
        X_train = training_table(rows, facts, F, label_free, train_idx, labels, groups)
        X_test = predict_table(rows, facts, F, label_free, known, test_idx)
        seq_train = training_sequences(F, train_idx, labels, groups) if seq_kind else None
        seq_test = build_sequences(F, known, test_idx) if seq_kind else None
    model = TruthStack(seed=seed, seq_kind=seq_kind).fit(
        X_train, labels[train_idx], groups[train_idx], seq_train)
    comp = model.predict_components(X_test, seq_test)
    preds = {k: labels_from_p_false(p, model.thresholds[k]) for k, p in comp.items()}
    return model, comp, preds


def main() -> None:
    t0 = time.time()
    # D options take ~4x longer; run them by name: `python train/eval_truth_stack.py D D-GRU`.
    chosen = [a for a in sys.argv[1:] if a in OPTIONS] or ["A", "B"]
    if "A" not in chosen:
        chosen = ["A", *chosen]
    rows = load_rows()
    lexicon = Lexicon.load(LEXICON_PATH)
    labels = np.asarray([r["label"].strip().lower() for r in rows])
    groups = np.asarray(page_ids(rows))
    folds = list(GroupKFold(5).split(np.zeros(len(rows)), groups=groups))

    # 1. Out-of-fold extraction on the classifier's own folds.
    facts = [None] * len(rows)
    for tr, va in folds:
        ex = FactExtractor(lexicon).fit([rows[i] for i in tr])
        for i, f in zip(va, ex.extract([rows[i] for i in va])):
            facts[i] = f
    ext = fact_scores(rows, facts)
    print(f"Extraction (out-of-fold by page): P {ext['precision']:.3f}  R {ext['recall']:.3f}  "
          f"F1 {ext['f1']:.3f}  ({time.time() - t0:.0f}s)")

    F = fact_table(rows, facts)
    gold = gold_facts_from_rows(rows)
    label_free = {
        False: label_free_features(rows, facts, F),
        True: label_free_features(rows, facts, F, truth_discovery=True, td_kwargs=TD_KWARGS),
    }
    results = []
    stack_scores = {}
    for option in chosen:
        td, seq_kind, _ = OPTIONS[option]
        print(f"\n===== Option {option} ({time.time() - t0:.0f}s) =====")
        option_results = run_option(rows, facts, F, label_free[td], labels, groups, folds, seq_kind,
                                    gold=gold, td=td)
        results += [{**r, "option": option} for r in option_results]
        stack_scores[option] = {r["cv"]: r["macro_f1_tf"] for r in option_results
                                if r["component"] == "stack"}

    # 4. Leaderboard rows.
    write_results(results)

    # 5. Ship decision: A unless another option clearly beats it.
    base = stack_scores["A"]
    ship = "A"
    print()
    for option in chosen[1:]:
        gain_group = stack_scores[option]["group"] - base["group"]
        gain_time = stack_scores[option]["time"] - base["time"]
        ok = gain_group >= SHIP_MARGIN and gain_time >= -TIME_TOLERANCE
        print(f"{option} vs A: grouped {gain_group:+.4f}, time-forward {gain_time:+.4f} "
              f"-> {'qualifies' if ok else 'does not qualify'}")
        if ok and stack_scores[option]["group"] > stack_scores[ship]["group"]:
            ship = option
    print(f"Ships: {ship} (needs >= +{SHIP_MARGIN:.2f} grouped, time-forward within "
          f"{TIME_TOLERANCE:.2f} of A)")

    # 6. Final pipeline on all practice rows.
    t1 = time.time()
    td, seq_kind, _ = OPTIONS[ship]
    pipeline = fit_pipeline(rows, lexicon, truth_discovery=td,
                            td_kwargs=TD_KWARGS if td else None, seq_kind=seq_kind)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH} (option {ship}, threshold "
          f"{pipeline.model.thresholds['stack']:.2f}, fit {time.time() - t1:.0f}s, "
          f"total {time.time() - t0:.0f}s)")


def run_option(rows, facts, F, label_free, labels, groups, folds, seq_kind=None,
               gold=None, td=False) -> list[dict]:
    results = []
    # 2. Grouped page CV.
    comp_p: dict[str, np.ndarray] = {}
    comp_pred: dict[str, np.ndarray] = {}
    for fold, (tr, te) in enumerate(folds, start=1):
        model, comp, preds = evaluate(rows, facts, F, label_free, labels, groups, tr, te,
                                      seed=fold, seq_kind=seq_kind, gold=gold, td=td)
        for k in comp:
            comp_p.setdefault(k, np.zeros(len(rows)))[te] = comp[k]
            comp_pred.setdefault(k, np.empty(len(rows), dtype=object))[te] = preds[k]
        print(f"Fold {fold}: threshold {model.thresholds['stack']:.2f}, "
              f"stack macro-F1 {macro_f1_true_false(labels[te], preds['stack']):.3f}")
    for r in print_block("Grouped page CV (5 folds)", labels, comp_p,
                         {k: v.astype(str) for k, v in comp_pred.items()}):
        results.append({**r, "cv": "group"})

    # 3. Time-forward: train on the earliest 80% of dates.
    dates = np.asarray([r["date"].strip() for r in rows])
    unique_dates = sorted(set(dates))
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    tr, te = np.flatnonzero(dates < cutoff), np.flatnonzero(dates >= cutoff)
    model, comp, preds = evaluate(rows, facts, F, label_free, labels, groups, tr, te,
                                  seed=2026, seq_kind=seq_kind, gold=gold, td=td)
    for r in print_block(f"Time-forward (test from {cutoff}, {len(te)} rows)", labels[te], comp, preds):
        results.append({**r, "cv": "time"})
    return results


def write_results(results: list[dict]) -> None:
    names = {"lgbm": "LightGBM", "logreg": "LogReg L2", "stack": "stack",
             "seq": "sequence net alone"}
    evaluated = {r["option"] for r in results}
    existing = []
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8", newline="") as handle:
            # Keep rows of options not re-evaluated in this run.
            existing = [r for r in csv.DictReader(handle)
                        if not r["model"].startswith(tuple(f"{o}: " for o in evaluated))]
    new = [{
        "model": f"{r['option']}: {names[r['component']]}",
        "track": "neuro" if OPTIONS[r["option"]][1] else "classic",
        "features": OPTIONS[r["option"]][2], "cv": r["cv"],
        **{k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
           for k in ("macro_f1_tf", "acc", "ece", "brier", "false_precision", "false_recall", "n")},
    } for r in results]
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + new)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()

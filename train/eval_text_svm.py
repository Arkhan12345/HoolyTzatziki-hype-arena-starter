"""Evaluate the text-only SVM with a tuned P(false) threshold and save it.

The threshold is always tuned on training rows only (inner out-of-fold
probabilities), so the scores below are not optimistic.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.classify.text_svm import (  # noqa: E402
    LABELS,
    apply_false_threshold,
    fit_thresholded_svm,
    macro_f1_true_false,
    make_grouped_splits,
    page_groups_from_rows,
    predict_probabilities,
)

DATA_PATH = ROOT / "data" / "practice.csv"
MODEL_PATH = ROOT / "models" / "text_svm.joblib"


def load_practice_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"text", "date", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    rows = [
        row
        for row in rows
        if row["label"].strip().lower() in LABELS and row["text"].strip()
    ]
    if not rows:
        raise ValueError(f"No labeled rows found in {path}")

    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Top-label ECE: weighted confidence/accuracy gap across equal-width bins."""
    predicted_indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predicted_indices == np.asarray(
        [LABELS.index(label) for label in y_true]
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        if index == bins - 1:
            mask = (confidences >= edges[index]) & (
                confidences <= edges[index + 1]
            )
        else:
            mask = (confidences >= edges[index]) & (
                confidences < edges[index + 1]
            )

        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean())
                - float(confidences[mask].mean())
            )

    return ece


def report(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
) -> None:
    argmax_predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]

    print(f"\n{name}")
    print(f"Accuracy:                     {accuracy_score(y_true, predicted):.4f}")
    print(
        "Macro-F1 true/false (README): "
        f"{macro_f1_true_false(y_true, predicted):.4f}"
        f"   (argmax would give {macro_f1_true_false(y_true, argmax_predicted):.4f})"
    )
    print(
        "Macro-F1 all 3 classes:       "
        f"{f1_score(y_true, predicted, labels=list(LABELS), average='macro', zero_division=0):.4f}"
    )
    print(
        "Top-label ECE (probabilities): "
        f"{expected_calibration_error(y_true, probabilities):.4f}"
    )
    print(
        classification_report(
            y_true,
            predicted,
            labels=list(LABELS),
            target_names=list(LABELS),
            zero_division=0,
            digits=3,
        )
    )
    print(
        "Confusion matrix (actual rows, predicted columns; "
        "order true/false/unsure):"
    )
    print(confusion_matrix(y_true, predicted, labels=list(LABELS)))


def evaluate_grouped(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Out-of-fold evaluation with each contiguous page held out as a unit."""
    splits = make_grouped_splits(
        labels,
        groups,
        requested_splits=5,
        random_state=42,
    )
    oof_probabilities = np.zeros((len(labels), len(LABELS)), dtype=float)
    oof_predicted = np.empty(len(labels), dtype=object)

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        bundle = fit_thresholded_svm(
            texts[train_idx],
            labels[train_idx],
            groups[train_idx],
            calibration_splits=3,
            random_state=100 + fold,
        )
        probabilities = predict_probabilities(bundle.model, texts[valid_idx])
        predicted = apply_false_threshold(probabilities, bundle.false_threshold)
        oof_probabilities[valid_idx] = probabilities
        oof_predicted[valid_idx] = predicted

        print(
            f"Fold {fold}/{len(splits)}: threshold P(false) >= "
            f"{bundle.false_threshold:.3f} (inner macro-F1 {bundle.tuning_macro_f1:.3f}), "
            f"validation macro-F1 {macro_f1_true_false(labels[valid_idx], predicted):.3f}, "
            f"predictions {dict(Counter(predicted.tolist()))}"
        )

    report(
        "Grouped page cross-validation",
        labels,
        oof_probabilities,
        oof_predicted.astype(str),
    )


def evaluate_time_forward(
    rows: list[dict[str, str]],
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Hold out the latest 20% of distinct dates; train only on earlier dates."""
    dates = np.asarray([row["date"].strip() for row in rows])
    unique_dates = sorted(set(dates))

    if len(unique_dates) < 5:
        print("Skipping time-forward evaluation: too few distinct dates")
        return

    cut = max(1, int(len(unique_dates) * 0.8))
    cutoff_date = unique_dates[cut]
    train_idx = np.flatnonzero(dates < cutoff_date)
    valid_idx = np.flatnonzero(dates >= cutoff_date)

    if not len(train_idx) or not len(valid_idx):
        print("Skipping time-forward evaluation: empty train or validation partition")
        return

    bundle = fit_thresholded_svm(
        texts[train_idx],
        labels[train_idx],
        groups[train_idx],
        calibration_splits=3,
        random_state=2026,
    )
    probabilities = predict_probabilities(bundle.model, texts[valid_idx])
    predicted = apply_false_threshold(probabilities, bundle.false_threshold)

    print(
        f"\nTime-forward cutoff: {cutoff_date} "
        f"(validation has {len(valid_idx)} rows, "
        f"threshold P(false) >= {bundle.false_threshold:.3f})"
    )
    report("Time-forward evaluation", labels[valid_idx], probabilities, predicted)


def main() -> None:
    rows = load_practice_rows(DATA_PATH)
    texts = np.asarray([row["text"] for row in rows], dtype=object)
    labels = np.asarray(
        [row["label"].strip().lower() for row in rows]
    )
    groups = page_groups_from_rows(rows)

    evaluate_grouped(texts, labels, groups)
    evaluate_time_forward(rows, texts, labels, groups)

    final_bundle = fit_thresholded_svm(
        texts,
        labels,
        groups,
        calibration_splits=3,
        random_state=42,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_bundle, MODEL_PATH)
    print(
        f"\nSaved final model: {MODEL_PATH} "
        f"(threshold P(false) >= {final_bundle.false_threshold:.3f})"
    )


if __name__ == "__main__":
    main()

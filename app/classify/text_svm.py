"""Text-only SVM baseline with page-aware probability calibration.

Calibrated probabilities stay close to the class prior (83% true), so argmax
always picks `true`. Instead, a row is labelled `false` when P(false) reaches a
threshold tuned on out-of-fold probabilities to maximise true/false macro-F1.
The baseline never outputs `unsure`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

LABELS = ("true", "false", "unsure")
# The README leaderboard scores macro-F1 over these two classes only.
SCORED_LABELS = ("true", "false")
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def page_groups_from_rows(rows: Sequence[dict[str, str]]) -> np.ndarray:
    """Return a page ID per row, treating each contiguous date block as a page."""
    groups: list[int] = []
    page_id = -1
    previous_date: str | None = None

    for row in rows:
        current_date = row["date"].strip()
        if current_date != previous_date:
            page_id += 1
            previous_date = current_date
        groups.append(page_id)

    return np.asarray(groups)


def make_grouped_splits(
    labels: Iterable[str],
    groups: Iterable[int],
    *,
    requested_splits: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified folds that never split a page between train and validation."""
    y = np.asarray(list(labels))
    group_ids = np.asarray(list(groups))

    if len(y) != len(group_ids):
        raise ValueError("labels and groups must have the same length")

    # Every class must appear in at least one training and one validation group.
    groups_per_class = [len(np.unique(group_ids[y == label])) for label in np.unique(y)]
    n_splits = min(requested_splits, min(groups_per_class, default=0))
    if n_splits < 2:
        raise ValueError("Need at least two distinct page groups for every label")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    placeholder_x = np.zeros((len(y), 1), dtype=np.uint8)
    return list(splitter.split(placeholder_x, y, groups=group_ids))


def build_svm_pipeline() -> Pipeline:
    """The baseline learns from text only; no dates, forms, or lexicon features."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def fit_calibrated_svm(
    texts: Iterable[str],
    labels: Iterable[str],
    page_groups: Iterable[int],
    *,
    calibration_splits: int = 3,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Fit a linear SVM and calibrate it using page-grouped folds."""
    x = np.asarray(list(texts), dtype=object)
    y = np.asarray(list(labels), dtype=str)
    groups = np.asarray(list(page_groups))

    if not set(np.unique(y)).issubset(LABELS):
        raise ValueError(f"labels must be drawn from {LABELS}")

    cv = make_grouped_splits(
        y,
        groups,
        requested_splits=calibration_splits,
        random_state=random_state,
    )
    model = CalibratedClassifierCV(
        estimator=build_svm_pipeline(),
        method="sigmoid",
        cv=cv,
        ensemble=True,
    )
    return model.fit(x, y)


def predict_probabilities(model, texts: Iterable[str]) -> np.ndarray:
    """Return probabilities in the fixed LABELS order."""
    raw = model.predict_proba(list(texts))
    aligned = np.zeros((len(raw), len(LABELS)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        aligned[:, LABELS.index(str(label))] = raw[:, source_index]
    return aligned


def macro_f1_true_false(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """Macro-F1 over true/false, the README leaderboard metric."""
    return float(
        f1_score(
            list(y_true),
            list(y_pred),
            labels=list(SCORED_LABELS),
            average="macro",
            zero_division=0,
        )
    )


def apply_false_threshold(
    probabilities: np.ndarray,
    false_threshold: float,
) -> np.ndarray:
    """Label a row `false` when P(false) >= threshold, else `true`."""
    p_false = probabilities[:, LABELS.index("false")]
    return np.where(p_false >= false_threshold, "false", "true")


def tune_false_threshold(
    y_true: Iterable[str],
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Return (threshold, macro-F1) that maximises true/false macro-F1.

    Ties go to the higher threshold, which flags fewer rows as false.
    """
    y = list(y_true)
    best_threshold, best_score = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        score = macro_f1_true_false(y, apply_false_threshold(probabilities, threshold))
        if score >= best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def out_of_fold_probabilities(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    splits: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Calibrated probabilities for every row from a model that never saw its page."""
    probabilities = np.zeros((len(labels), len(LABELS)), dtype=float)
    folds = make_grouped_splits(
        labels, groups, requested_splits=splits, random_state=random_state
    )
    for fold, (train_idx, valid_idx) in enumerate(folds):
        model = fit_calibrated_svm(
            texts[train_idx],
            labels[train_idx],
            groups[train_idx],
            calibration_splits=splits,
            random_state=random_state + fold + 1,
        )
        probabilities[valid_idx] = predict_probabilities(model, texts[valid_idx])
    return probabilities


@dataclass
class ThresholdedSvm:
    """A calibrated SVM plus the P(false) threshold used to pick labels."""

    model: CalibratedClassifierCV
    false_threshold: float
    tuning_macro_f1: float


def fit_thresholded_svm(
    texts: Iterable[str],
    labels: Iterable[str],
    page_groups: Iterable[int],
    *,
    calibration_splits: int = 3,
    random_state: int = 42,
) -> ThresholdedSvm:
    """Tune the threshold on out-of-fold probabilities, then fit on all rows."""
    x = np.asarray(list(texts), dtype=object)
    y = np.asarray(list(labels), dtype=str)
    groups = np.asarray(list(page_groups))

    oof = out_of_fold_probabilities(
        x, y, groups, splits=calibration_splits, random_state=random_state
    )
    threshold, score = tune_false_threshold(y, oof)
    model = fit_calibrated_svm(
        x,
        y,
        groups,
        calibration_splits=calibration_splits,
        random_state=random_state,
    )
    return ThresholdedSvm(model=model, false_threshold=threshold, tuning_macro_f1=score)


def predict_with_confidence(
    bundle: ThresholdedSvm,
    texts: Iterable[str],
) -> list[dict[str, object]]:
    """Return output-ready labels with confidence in [0.5, 1.0] (R-22).

    Confidence is P(label) renormalised over true/false, then clipped to 0.5,
    because a thresholded `false` can have P(false) well below 0.5.
    """
    probabilities = predict_probabilities(bundle.model, texts)
    predicted = apply_false_threshold(probabilities, bundle.false_threshold)
    p_true = probabilities[:, LABELS.index("true")]
    p_false = probabilities[:, LABELS.index("false")]
    total = np.maximum(p_true + p_false, 1e-12)

    results: list[dict[str, object]] = []
    for label, pt, pf, norm in zip(predicted, p_true, p_false, total):
        share = (pf if label == "false" else pt) / norm
        results.append(
            {"label": str(label), "confidence": float(np.clip(share, 0.5, 1.0))}
        )
    return results



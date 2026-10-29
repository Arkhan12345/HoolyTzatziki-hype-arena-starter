"""The frozen FINAL model behind app.classify.classify: loads, labels awkward inputs, never breaks the run."""
import csv
import os
from pathlib import Path

import pytest

from app.classify import classify
from app.io.types import Label

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "final" / "model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="models/final/model.joblib not present")


def _practice(n=20):
    with open(ROOT / "data" / "practice.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:n]


def _check(labels, rows):
    assert len(labels) == len(rows)
    assert {lb.id for lb in labels} == {r["id"] for r in rows}
    for lb in labels:
        assert isinstance(lb, Label)
        assert lb.label in ("true", "false", "unsure")
        if lb.label == "unsure":
            assert lb.confidence is None
        else:
            assert 0.5 <= lb.confidence <= 1.0


def test_new_ids_zero_labelled_rows():
    rows = [{"id": f"new-{i}", "text": r["text"], "date": r["date"], "label": ""} for i, r in enumerate(_practice())]
    _check(classify(rows, [], None), rows)


def test_other_date_format():
    rows = [{"id": f"d-{i}", "text": r["text"], "date": r["date"].replace("-", "/"), "label": ""}
            for i, r in enumerate(_practice())]
    _check(classify(rows, [], None), rows)


def test_unseen_subjects_get_capped_confidence():
    rows = [{"id": "u1", "text": "Emberkeep sells its team plan with 50 seats.", "date": "2026-05-23", "label": ""},
            {"id": "u2", "text": "Neshspar has a satisfaction score of 90%.", "date": "2025-11-29", "label": ""},
            {"id": "u3", "text": "", "date": "", "label": ""}]
    labels = classify(rows, [], None)
    _check(labels, rows)
    assert all(lb.confidence is None or lb.confidence <= 0.6 for lb in labels)


def test_labelled_rows_are_not_relabelled():
    rows = [dict(r, label=r["label"]) for r in _practice(5)]
    assert classify(rows, [], None) == []


def test_alltrue_switch_and_failure_fallback(monkeypatch):
    rows = [{"id": "a1", "text": "Hooly Docs 2.0", "date": "2026-08-25", "label": ""}]
    monkeypatch.setenv("TZ_CLASSIFIER", "alltrue")
    assert classify(rows, [], None)[0].label == "true"
    monkeypatch.setenv("TZ_CLASSIFIER", "final")
    from app.classify import final
    monkeypatch.setattr(final, "predict_rows", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    lb = classify(rows, [], None)
    assert lb[0].label == "true" and lb[0].confidence == pytest.approx(0.83)

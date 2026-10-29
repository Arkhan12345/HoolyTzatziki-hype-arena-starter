"""P3 classify entry point used by app/run.py: classify(rows, facts, kb) -> list[Label].

TZ_CLASSIFIER selects the model:
  final    (default) the frozen FINAL model in models/final/ (reads facts from the text itself; fits nothing)
  alltrue  every unlabelled row true at the baseline confidence
If the selected model fails for any reason, it falls back to all-true, so app.run always writes output.csv.
Option A's research stack (truth_model.TruthStack) has no runtime entry point yet; it is selectable once one exists.
"""
from __future__ import annotations

import os
import sys

from app.io.types import Label

BASELINE_CONF = 0.83


def _all_true(blank):
    return [Label(id=r["id"], label="true", confidence=BASELINE_CONF) for r in blank]


def classify(rows, facts=None, kb=None):
    blank = [r for r in rows if not r.get("label")]
    if not blank:
        return []
    choice = os.environ.get("TZ_CLASSIFIER", "final").strip().lower()
    if choice == "final":
        try:
            from app.classify import final
            out = final.predict_rows([{"id": r["id"], "text": r.get("text", ""), "date": r.get("date", "")}
                                      for r in blank])
            return [Label(id=o["id"], label=o["label"], confidence=o["confidence"]) for o in out]
        except Exception as e:  # never break the run: fall back to the baseline
            print(f"[classify] FINAL failed ({e!r}); falling back to all-true", file=sys.stderr, flush=True)
    return _all_true(blank)

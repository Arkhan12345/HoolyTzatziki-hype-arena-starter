"""Frozen FINAL label model, inference only (see models/final/manifest.json).

The model was pickled under the research package name; `load()` registers aliases so it unpickles from here.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "final"
_MODULES = ("values", "extractor", "closed_world", "feats", "model")
_cached = None


def _register_aliases() -> None:
    pkg = sys.modules[__name__]
    for alias in ("h300.lib", "lib"):
        sys.modules.setdefault(alias, pkg)
    sys.modules.setdefault("h300", type(sys)("h300"))
    for name in _MODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        for alias in (f"h300.lib.{name}", f"lib.{name}"):
            sys.modules.setdefault(alias, mod)


def load(model_dir: Path | str | None = None):
    """Load the frozen SavedModel once (fits nothing)."""
    global _cached
    if _cached is None or model_dir is not None:
        import joblib
        _register_aliases()
        _cached = joblib.load(Path(model_dir or MODEL_DIR) / "model.joblib")
    return _cached


def predict_rows(rows, model_dir: Path | str | None = None) -> list[dict]:
    """rows: dicts with id, text, date -> [{id, label, confidence}] (confidence None for unsure)."""
    out = load(model_dir).predict_rows(rows)
    return [{"id": str(o["id"]), "label": str(o["label"]),
             "confidence": None if o["confidence"] is None else float(o["confidence"])} for o in out]

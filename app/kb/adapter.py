"""Adapter: whatever P3's extractor returns -> uniform fact records for the KB.

Accepts the agreed `Fact(row_id, subject, attribute, value, span, conf)` dataclass, any object with
those attributes, or plain dicts with the same keys. Values must already be claimed_value dicts
(`{"kind": ...}`); anything else is dropped and counted, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = {"quantity", "money", "percent", "date", "version", "duration", "name", "boolean"}


@dataclass(frozen=True)
class FactRecord:
    row_id: str
    subject: str
    attribute: str
    value: dict
    conf: float


def _get(f, name, default=None):
    if isinstance(f, dict):
        return f.get(name, default)
    return getattr(f, name, default)


def to_records(facts, report: dict | None = None) -> list[FactRecord]:
    out, dropped = [], 0
    for f in facts or []:
        value = _get(f, "value")
        row_id, subject, attribute = _get(f, "row_id"), _get(f, "subject"), _get(f, "attribute")
        if not (isinstance(value, dict) and value.get("kind") in KINDS and row_id and subject and attribute):
            dropped += 1
            continue
        try:
            conf = float(_get(f, "conf", 0.5) or 0.5)
        except (TypeError, ValueError):
            conf = 0.5
        out.append(FactRecord(str(row_id), str(subject), str(attribute), value, min(1.0, max(0.0, conf))))
    if report is not None:
        report["extracted_in"] = len(out) + dropped
        report["extracted_dropped"] = dropped
    return out

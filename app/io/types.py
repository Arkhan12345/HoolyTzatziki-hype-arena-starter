"""Shared record types.

P1 owns app/contracts.py. Until it lands (or if it fails to import) these
fallbacks keep the runtime working with the same field names.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:  # pragma: no cover - depends on P1's push
    from app.contracts import Claim, Fact, Label  # type: ignore
except Exception:  # noqa: BLE001

    @dataclass
    class Fact:  # type: ignore[no-redef]
        row_id: str
        subject: str
        attribute: str
        value: dict
        span: str = ""
        conf: float = 1.0

    @dataclass
    class Label:  # type: ignore[no-redef]
        id: str
        label: str
        confidence: float | None

    @dataclass
    class Claim:  # type: ignore[no-redef]
        id: str
        type: str
        text: str
        form: dict
        confidence: float
        meta: dict = field(default_factory=dict)


__all__ = ["Claim", "Fact", "Label"]

"""Team rule on top of the mentor's: a `fact` claim that only rewords one prose input sentence is not new.

"A team of 4 recruiters reads your file." -> "Hooly Hiring Programme has 4 recruiters." is a reformulation even
though the sentence never names the subject (the mentor's rule would let it through: it is stricter, never looser).
A fact read off a table / spec line ("Recruiters: 4", a fragment without a sentence end) stays: turning a table row
into a sentence is new. Used by the claim screening (app/generate/select.py) and the training-item generator
(app/classify/datagen.py).
"""
from __future__ import annotations

import json
import re

from app.kb import normalize as nz

_KV = re.compile(r"^[^.]{2,40}: ")
_END = (".", "!", "?", '"', "”")


def is_table_line(text: str) -> bool:
    """A spec-sheet / table row ("Storage: 2 TB", a fragment without a sentence end)."""
    s = (text or "").strip()
    return bool(_KV.search(s)) or not s.endswith(_END)


def _facts(form) -> list[dict]:
    if isinstance(form, str):
        try:
            return json.loads(form or "[]")
        except ValueError:
            return []
    return form or []


def restates_prose(form: dict, evidence, rows: dict) -> bool:
    """True when one of the fact's source rows is a prose sentence whose gold form already states this fact.

    rows: row id -> {"text", "form"}; a row without a form cannot be checked and does not count."""
    try:
        k = (form["subject"], form["attribute"], nz.key_inclusive(form["value"]))
    except (KeyError, TypeError, AttributeError):
        return False
    for rid in evidence or ():
        r = rows.get(rid)
        if not r or is_table_line(r.get("text", "")):
            continue
        for f in _facts(r.get("form")):
            if isinstance(f.get("value"), dict) and (f["subject"], f["attribute"], nz.key_inclusive(f["value"])) == k:
                return True
    return False

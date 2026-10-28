"""Two ways to feed the same claim generator, for comparing them while debugging.

A  "gold"        (production): evidence = practice rows labelled true (data/practice.csv forms).
B  "classified"  (alternative): evidence = the structured output of X/Arckan's extraction + sentence
                 classification over the universe rows, no practice gold at all: facts of rows the
                 classifier labelled true with confidence >= min_conf.

B's input uses the shared interfaces only: Label(id, label, confidence) and
Fact(row_id, subject, attribute, value, span, conf), as objects or dicts. It comes from
app.extract / app.classify when they exist, or from a JSON file
    {"labels": [{"id", "label", "confidence"}, ...], "facts": [{"row_id", "subject", "attribute", "value", "conf"}, ...]}
Same generator, same validator, same selection: only the evidence differs.
"""
from __future__ import annotations

import json

from app.kb.adapter import _get  # noqa: F401  (re-exported for callers reading labels)
from app.kb.inventory import FactInventory
from app.kb.store import KB

MIN_CONF = 0.9


def inventory_gold(lexicon: dict, gold_rows: list[dict], rows: list[dict] | None = None, facts=None) -> FactInventory:
    return FactInventory.from_kb(KB(lexicon).build(facts or [], gold_rows, rows))


def inventory_classified(lexicon: dict, rows: list[dict], labels, facts, min_conf: float = MIN_CONF) -> FactInventory:
    kb = KB(lexicon)
    kb.add_extracted(facts, rows)  # every extracted fact, for the staleness veto
    kb.add_classified(labels, facts, rows, min_conf=min_conf)
    return FactInventory.from_kb(kb, trust_classified=True)


def load_structured(path: str) -> tuple[list, list]:
    """(labels, facts) from a JSON file in the shared-interface shape above."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return list(data.get("labels") or []), list(data.get("facts") or [])


def from_p3(rows: list[dict], lexicon: dict) -> tuple[list, list] | None:
    """(labels, facts) from the real app.extract + app.classify, or None while they don't exist."""
    try:
        from app.classify import classify  # type: ignore
        from app.extract import extract  # type: ignore
    except ModuleNotFoundError as e:  # only a module that is not there yet means "unavailable"
        if e.name in ("app.extract", "app.classify"):
            return None
        raise  # a broken P3 module (or one of its imports) must show, not fall back to a mock
    facts = extract(rows, lexicon)
    kb = KB(lexicon).build(facts, [], rows)
    return classify(rows, facts, kb), facts

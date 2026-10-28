import json
import os
from collections import Counter

import pytest

from app.generate import generate
from app.generate.render import value_text
from app.io.reader import read_lexicon, read_practice
from app.kb import normalize as nz
from app.kb.store import KB
from app.validate.rules import norm_text, validate_claim

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def world():
    rows = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    kb = KB(lex).build([], rows)
    texts = {norm_text(r["text"]) for r in rows}
    report = {}
    claims = generate(kb, lex, texts, 1000, report)
    return rows, lex, kb, texts, claims, report


def test_exactly_1000_unique_valid(world):
    rows, lex, kb, texts, claims, _ = world
    assert len(claims) == 1000
    assert len({c.id for c in claims}) == 1000
    assert len({norm_text(c.text) for c in claims}) == 1000
    for c in claims:
        c.meta = {"between": "date" if c.type == "compare" and _is_date_compare(kb, c) else "amount"}
        assert validate_claim(c, lex, texts) == [], c.text
        assert 0.5 <= c.confidence <= 1


def _is_date_compare(kb, c):
    v, _, _ = kb.current_value(c.form["left"]["subject"], c.form["left"]["attribute"])
    return v is not None and v.get("kind") == "date"


def test_all_types_present(world):
    by = Counter(c.type for c in world[4])
    assert set(by) == {"fact", "compare", "change", "at_event", "relation"}


def test_change_claims_follow_gold_order(world):
    rows, lex, kb, *_ = world
    for c in world[4]:
        if c.type != "change":
            continue
        ms = kb.timeline(c.form["subject"], c.form["attribute"]).gold()
        first_from = min(m.date for m in ms if nz.same(m.value, c.form["from"]))
        last_to = max(m.date for m in ms if nz.same(m.value, c.form["to"]))
        assert first_from < last_to, c.text


def test_compare_claims_hold_on_current_values(world):
    kb = world[2]
    for c in world[4]:
        if c.type != "compare":
            continue
        lv, _, _ = kb.current_value(c.form["left"]["subject"], c.form["left"]["attribute"])
        rv, _, _ = kb.current_value(c.form["right"]["subject"], c.form["right"]["attribute"])
        assert nz.compare(lv, rv) == c.form["op"], c.text


@pytest.mark.parametrize("value,text", [
    ({"kind": "money", "value": 59, "currency": "EUR", "per": "month"}, "€59 per month"),
    ({"kind": "money", "value": 1999, "currency": "RON"}, "1,999 RON"),
    ({"kind": "quantity", "value": 2026, "unit": "customer"}, "2,026 customers"),
    ({"kind": "quantity", "value": 12, "unit": "person"}, "12 people"),
    ({"kind": "quantity", "value": 128, "unit": "GB"}, "128 GB"),
    ({"kind": "quantity", "value": 1.5, "unit": "kg"}, "1.5 kg"),
    ({"kind": "quantity", "value": 1.234, "unit": "kg"}, None),
    ({"kind": "percent", "value": 99.999}, "99.999%"),
    ({"kind": "date", "value": "2026-03-12"}, "12 March 2026"),
    ({"kind": "date", "value": "2026-03"}, "March 2026"),
    ({"kind": "version", "value": "2.3"}, "version 2.3"),
    ({"kind": "duration", "value": 1, "unit": "week"}, "1 week"),
    ({"kind": "duration", "value": 14, "unit": "day"}, "14 days"),
    ({"kind": "quantity", "value": 10, "unit": "customer", "op": "gt"}, None),
])
def test_value_text(value, text):
    assert value_text(value) == text


def test_units_convert():
    assert nz.same({"kind": "duration", "value": 14, "unit": "day"}, {"kind": "duration", "value": 2, "unit": "week"})
    assert nz.same({"kind": "quantity", "value": 1, "unit": "GB"}, {"kind": "quantity", "value": 1000, "unit": "MB"})
    assert not nz.same({"kind": "duration", "value": 30, "unit": "day"}, {"kind": "duration", "value": 1, "unit": "month"})
    assert nz.compare({"kind": "date", "value": "2026-06"}, {"kind": "date", "value": "2026-06-02"}) is None


def test_report_is_json(world):
    json.dumps(world[5])

import json
import os
from collections import Counter

import pytest

from app.generate import generate, generate_from_inventory
from app.generate.render import value_text
from app.generate.truth import truth_problems
from app.io.reader import read_lexicon, read_practice
from app.kb import normalize as nz
from app.kb.inventory import FactInventory
from app.kb.store import KB
from app.validate.rules import norm_text, validate_claim
from tests.runtime.mocks import gold_extractor

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
    inv = FactInventory.from_kb(kb)
    for c in claims:
        assert validate_claim(c, lex, texts) == [], c.text
        assert truth_problems(c.type, c.form, inv) == [], c.text
        assert 0.5 <= c.confidence <= 1


def test_no_semantic_duplicates_while_distinct_claims_remain(world):
    report = world[5]
    assert report["distinct_semantic"] >= 1000
    assert report["selected_semantic_duplicates"] == 0
    keys = [c.meta["semantic_key"] for c in world[4]]
    assert len(set(keys)) == len(keys)


def test_selection_is_confidence_first(world):
    claims, report = world[4], world[5]
    assert report["min_confidence_selected"] == min(c.confidence for c in claims)
    # nothing left out is more confident than the least confident claim taken
    assert all(c.confidence >= report["min_confidence_selected"] for c in claims)


def test_change_claims_follow_gold_order(world):
    kb = world[2]
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


# ---- synthetic inventory: the generator without any extractor or practice data -------------------
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett", "Kilo",
         "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor",
         "Whiskey", "Xray", "Yankee", "Zulu", "Amber", "Basalt", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Helix"]


def _synthetic(n_products=34):
    products = [f"Zeta {w} Plan" for w in NAMES[:n_products]]
    lex = {"subjects": [{"name": p, "aliases": [], "attributes": ["price"]} for p in products], "events": []}
    recs = [{"subject": p, "attribute": "price", "date": "2026-10-01",
             "value": {"kind": "money", "value": 10 + i, "currency": "EUR", "per": "month"}}
            for i, p in enumerate(products)]
    return FactInventory.from_records(recs, lex), lex


def test_combinatorial_compares_without_quotas():
    inv, lex = _synthetic(34)
    report = {}
    claims = generate_from_inventory(inv, set(), 1000, report)
    assert report["valid_by_type"]["compare"] == 34 * 33  # 1,122 ordered comparisons
    assert report["distinct_by_type"]["compare"] == 34 * 33 // 2  # 561 distinct relations
    assert len(claims) == 1000
    by_type = Counter(c.type for c in claims)
    assert by_type["compare"] + by_type["fact"] == 1000  # whatever the evidence supports, no quota
    assert report["selected_semantic_duplicates"] == 1000 - report["distinct_semantic"]
    for c in claims:
        assert validate_claim(c, lex, set()) == [], c.text


def test_currency_and_period_never_mix():
    lex = {"subjects": [{"name": "Zeta Alpha Plan", "aliases": [], "attributes": ["price"]},
                        {"name": "Zeta Bravo Plan", "aliases": [], "attributes": ["price"]}], "events": []}
    recs = [{"subject": "Zeta Alpha Plan", "attribute": "price", "date": "2026-10-01",
             "value": {"kind": "money", "value": 10, "currency": "EUR", "per": "month"}},
            {"subject": "Zeta Bravo Plan", "attribute": "price", "date": "2026-10-01",
             "value": {"kind": "money", "value": 20, "currency": "USD", "per": "month"}}]
    report = {}
    generate_from_inventory(FactInventory.from_records(recs, lex), set(), 10, report)
    assert "compare" not in report["valid_by_type"]


def test_extractor_output_plugs_in_unchanged(world):
    """Facts from a P3-shaped extractor (dataclasses or dicts) reach the generator through the KB only."""
    rows, lex, _, texts, _, _ = world
    universe = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]
    for as_dicts in (False, True):
        facts = gold_extractor(rows, as_dicts)(universe, lex)
        kb = KB(lex).build(facts, rows, universe)
        assert kb.stats()["extracted"] == len(facts) and kb.stats()["extracted_dropped"] == 0
        report = {}
        assert len(generate(kb, lex, texts, 1000, report)) == 1000


def test_adapter_drops_malformed_facts():
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    kb = KB(lex).build([{"row_id": "x", "subject": "Hooly", "attribute": "ceo", "value": "Gavin"},
                        {"row_id": "y", "subject": "Hooly", "attribute": "ceo",
                         "value": {"kind": "name", "value": "Gavin Bellson"}}], [], [{"id": "y", "date": "2026-01-01"}])
    assert kb.stats()["extracted"] == 1 and kb.stats()["extracted_dropped"] == 1


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

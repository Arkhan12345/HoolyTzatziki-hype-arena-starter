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
    assert report["valid_by_type"]["compare"] >= 34 * 33  # 1,122 ordered comparisons (x wordings)
    pairs = {}  # every one of the 561 pairs must be stated from both sides
    from app.generate.select import screen

    for c in screen(inv, set(), {}):
        if c.type == "compare":
            pairs.setdefault(frozenset((c.form["left"]["subject"], c.form["right"]["subject"])), set()).add(
                c.form["left"]["subject"])
    assert len(pairs) == 561 and all(len(sides) == 2 for sides in pairs.values())
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


def test_classifier_output_feeds_claims_only_when_trusted():
    """Facts of rows the sentence classifier labelled true (X/Arckan's structured output) reach the
    generator through KB.add_classified; they count as evidence only with trust_classified."""
    from app.io.types import Label

    inv_lex = {"subjects": [{"name": "Zeta Alpha Plan", "aliases": [], "attributes": ["price"]},
                            {"name": "Zeta Bravo Plan", "aliases": [], "attributes": ["price"]}], "events": []}
    rows = [{"id": "u-1", "text": "Zeta Alpha Plan costs €10 per month.", "date": "2026-10-01", "label": ""},
            {"id": "u-2", "text": "Zeta Bravo Plan costs €20 per month.", "date": "2026-10-01", "label": ""}]
    facts = [{"row_id": r["id"], "subject": s, "attribute": "price", "conf": 0.9,
              "value": {"kind": "money", "value": v, "currency": "EUR", "per": "month"}}
             for r, s, v in zip(rows, ["Zeta Alpha Plan", "Zeta Bravo Plan"], [10, 20])]
    labels = [Label(id="u-1", label="true", confidence=0.95), Label(id="u-2", label="true", confidence=0.97)]
    kb = KB(inv_lex).build(facts, [], rows, labels=labels)
    assert kb.stats()["classified_facts"] == 2
    assert generate(kb, inv_lex, set(), 10, {}, trust_classified=False) == []
    report = {}
    claims = generate(kb, inv_lex, set(), 10, report, trust_classified=True)
    assert claims and report["valid_by_type"].get("compare")
    srcs = {s["row_id"] for c in claims for s in c.meta["sources"]}
    assert srcs <= {"u-1", "u-2"} and all(s["classified"]["confidence"] >= 0.9
                                           for c in claims for s in c.meta["sources"])


def test_low_confidence_classifications_are_not_evidence():
    from app.io.types import Label

    lex = {"subjects": [{"name": "Zeta Alpha Plan", "aliases": [], "attributes": ["price"]}], "events": []}
    rows = [{"id": "u-1", "text": "x", "date": "2026-10-01", "label": ""}]
    facts = [{"row_id": "u-1", "subject": "Zeta Alpha Plan", "attribute": "price",
              "value": {"kind": "money", "value": 10, "currency": "EUR", "per": "month"}}]
    kb = KB(lex).build(facts, [], rows, labels=[Label(id="u-1", label="true", confidence=0.6)])
    assert kb.stats().get("classified_facts", 0) == 0


def test_every_selected_claim_has_its_sentences(world):
    for c in world[4]:
        assert c.meta["sources"], c.text
        assert all(s["text"] and s["origin"] == "practice" and s["label"] == "true" for s in c.meta["sources"])


def test_compare_generators_script_runs_both_strategies(tmp_path):
    import subprocess
    import sys

    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_generators.py"), "--mock", "--noise", "0.1",
                        "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONPATH": ROOT, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr[-2000:]
    cmp = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["A_gold"]["selected"] == cmp["B_classified"]["selected"] == 1000
    assert "MOCK" in cmp["setting"]["b_source"]
    recs = [json.loads(x) for x in (tmp_path / "claims_B.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["verdict"] in ("holds", "unconfirmed", "contradicted") and r["sources"] for r in recs)


def test_broken_p3_module_is_not_hidden_behind_the_mock(tmp_path, monkeypatch):
    import sys

    import app
    from app.generate import strategies

    for sub, body in (("extract", "import not_a_real_package\n"), ("classify", "def classify(*a):\n    return []\n")):
        (tmp_path / "app" / sub).mkdir(parents=True)
        (tmp_path / "app" / sub / "__init__.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(app, "__path__", list(app.__path__) + [str(tmp_path / "app")])
    for m in ("app.extract", "app.classify"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    with pytest.raises(ModuleNotFoundError) as e:
        strategies.from_p3([], {})
    assert e.value.name == "not_a_real_package"
    for m in ("app.extract", "app.classify"):
        sys.modules.pop(m, None)


def test_absent_p3_modules_mean_unavailable():
    from app.generate import strategies

    assert strategies.from_p3([], {}) is None


def test_precise_date_only_from_the_current_run():
    from app.generate.candidates import current

    lex = {"subjects": [{"name": "Zeta Alpha Plan", "aliases": [], "attributes": ["launched"]}], "events": []}
    d = lambda v: {"kind": "date", "value": v}  # noqa: E731
    recs = [{"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-01-20", "value": d("2026-01-12")},
            {"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-03-01", "value": d("2026-02-03")},
            {"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-09-01", "value": d("2026-01")}]
    v, _, _ = current(FactInventory.from_records(recs, lex), "Zeta Alpha Plan", "launched")
    assert v["value"] == "2026-01"  # the old day sits before an unrelated date: not evidence for "now"
    v2, _, _ = current(FactInventory.from_records([recs[0], recs[2]], lex), "Zeta Alpha Plan", "launched")
    assert v2["value"] == "2026-01-12"


def test_equal_counts_read_as_counts():
    from app.generate.templates import compare_text

    t = compare_text(("Hooly", "employees"), ("Hooly Cloud", "employees"), "eq", "amount", "quantity")
    assert t == "Hooly has as many employees as Hooly Cloud."


def test_change_claims_are_consecutive_steps(world):
    kb = world[2]
    for c in world[4]:
        if c.type != "change":
            continue
        keys = []
        for m in kb.timeline(c.form["subject"], c.form["attribute"]).gold():
            if "op" not in m.value and (not keys or keys[-1] != m.key):
                keys.append(m.key)
        steps = set(zip(keys, keys[1:]))
        assert (nz.key(c.form["from"]), nz.key(c.form["to"])) in steps, c.text


def test_sources_say_which_fact_each_sentence_gives(world):
    from app.generate.candidates import current
    from app.generate.select import claim_pairs

    inv = FactInventory.from_kb(world[2])
    for c in world[4]:
        pairs = set(claim_pairs(c.type, c.form))
        if c.type == "fact":
            used = {(c.form["subject"], c.form["attribute"]): [c.form["value"]]}
        elif c.type == "change":
            used = {(c.form["subject"], c.form["attribute"]): [c.form["from"], c.form["to"]]}
        elif c.type == "compare":
            used = {p: [current(inv, *p)[0]] for p in pairs}
        else:
            used = None
        for s in c.meta["sources"]:
            assert s["states"], (c.text, s["row_id"])
            keys = [(st["subject"], st["attribute"], nz.key(st["value"])) for st in s["states"]]
            assert len(keys) == len(set(keys)), ("repeated state", c.text, s["row_id"])
            for st in s["states"]:
                assert (st["subject"], st["attribute"]) in pairs, (c.text, st)
                if used is not None:
                    # a less precise date of the same day ("January 2026" for 12 January 2026) also backs it
                    assert any(nz.same(st["value"], v) or nz.date_prefix_related(st["value"], v)
                               for v in used[(st["subject"], st["attribute"])]), (c.text, st)


def test_at_event_sources_include_the_event_date():
    from app.generate.candidates import at_event_candidates

    rows = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    inv = FactInventory.from_kb(KB(lex).build([], rows))
    c = next(iter(at_event_candidates(inv)))
    from app.generate.select import claim_pairs

    srcs = inv.sources(c.evidence, limit=50, pairs=claim_pairs(c.type, c.form))
    assert any(st["attribute"] == "date" and st["subject"] == c.form["event"] for s in srcs for st in s["states"])

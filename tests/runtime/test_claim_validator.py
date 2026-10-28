"""Adversarial tests for the independent claim validator (app/validate/claims.py)."""
import os

import pytest

from app.io.reader import read_lexicon, read_practice
from app.io.types import Claim
from app.kb.inventory import FactInventory
from app.kb.store import KB
from app.validate.claims import FAIL, PASS, UNVERIFIED, ClaimValidator
from app.validate.rules import norm_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def V():
    rows = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    return ClaimValidator(FactInventory.from_kb(KB(lex).build([], rows)), {norm_text(r["text"]) for r in rows})


def claim(cid, ctype, text, form, meta=None):
    c = Claim(cid, ctype, text, form, 0.9)
    c.meta = meta or {}
    return c


def cmp_form(l_s, l_a, op, r_s, r_a):
    return {"left": {"subject": l_s, "attribute": l_a}, "right": {"subject": r_s, "attribute": r_a}, "op": op}


AMOUNT = {"between": "amount"}
DATE = {"between": "date"}


def test_true_compare_passes_with_evidence(V):
    c = claim("c1", "compare", "Hooly Search has more monthly active users than Hooly Chrome.",
              cmp_form("Hooly Search", "monthly_active_users", "gt", "Hooly Chrome", "monthly_active_users"), AMOUNT)
    v = V.check(c, c.meta)
    assert v.verdict == PASS, v.reasons
    assert v.evidence and all(r in V.inv.rows for r in v.evidence)


def test_wrong_direction_fails_truth(V):
    c = claim("c1", "compare", "Hooly Search has fewer monthly active users than Hooly Chrome.",
              cmp_form("Hooly Search", "monthly_active_users", "lt", "Hooly Chrome", "monthly_active_users"), AMOUNT)
    v = V.check(c, c.meta)
    assert v.verdict == FAIL and v.checks["truth"] == FAIL


def test_inverse_compare_is_the_same_information(V):
    a = claim("c1", "compare", "Hooly Search has more monthly active users than Hooly Chrome.",
              cmp_form("Hooly Search", "monthly_active_users", "gt", "Hooly Chrome", "monthly_active_users"), AMOUNT)
    b = claim("c2", "compare", "Hooly Chrome has fewer monthly active users than Hooly Search.",
              cmp_form("Hooly Chrome", "monthly_active_users", "lt", "Hooly Search", "monthly_active_users"), AMOUNT)
    va, vb = V.check_set([a, b])
    assert va.verdict == PASS and vb.verdict == FAIL and vb.checks["uniqueness"] == FAIL


def test_paraphrase_and_alias_share_a_key(V):
    f1 = {"subject": "Hooly", "attribute": "employees", "value": {"kind": "quantity", "value": 74, "unit": "employee"}}
    f2 = dict(f1, subject="Hooly Inc.")  # an alias of Hooly
    assert V.semantic_key("fact", f1) == V.semantic_key("fact", f2)
    c1 = cmp_form("Hooly Pixel 9", "price", "gt", "Hooly Pixel Watch", "price")
    c2 = cmp_form("Hooly Pixel Watch", "price", "lt", "Hooly Pixel 9", "price")
    assert V.semantic_key("compare", c1) == V.semantic_key("compare", c2)


def test_fact_restating_an_input_row_is_not_new(V):
    form = {"subject": "Hooly AI Pro", "attribute": "price",
            "value": {"kind": "money", "value": 59, "currency": "EUR", "per": "month"}}
    v = V.check(claim("f1", "fact", "The price of Hooly AI Pro is €59 per month.", form))
    assert v.checks["novelty"] == FAIL and "reformulates" in " ".join(v.reasons)


def test_copy_of_an_input_sentence_fails(V):
    form = {"subject": "Hooly AI Pro", "attribute": "price",
            "value": {"kind": "money", "value": 59, "currency": "EUR", "per": "month"}}
    text = next(t for t in V.input_texts if "hooly ai pro costs €59 per month" in t)
    v = V.check(claim("f1", "fact", text[0].upper() + text[1:] + ".", form))
    assert v.verdict == FAIL and v.checks["novelty"] == FAIL


def test_stale_value_fails_truth(V):
    form = {"subject": "Hooly Chrome", "attribute": "stable_version", "value": {"kind": "version", "value": "71.0"}}
    v = V.check(claim("f1", "fact", "Hooly Chrome is on stable version 71.0.", form))
    assert v.checks["truth"] == FAIL  # 72.0 is in force on 29 Oct 2026


def test_change_against_the_timeline_fails(V):
    form = {"subject": "Hooly Chrome", "attribute": "stable_version",
            "from": {"kind": "version", "value": "72.0"}, "to": {"kind": "version", "value": "64.0"}}
    v = V.check(claim("x1", "change", "The stable version of Hooly Chrome went from version 72.0 to version 64.0.", form))
    assert v.checks["truth"] == FAIL


def test_unknown_value_is_unverified_not_pass(V):
    form = {"subject": "Hooly Chrome", "attribute": "stable_version",
            "from": {"kind": "version", "value": "72.0"}, "to": {"kind": "version", "value": "99.0"}}
    v = V.check(claim("x1", "change", "The stable version of Hooly Chrome went from version 72.0 to version 99.0.", form))
    assert v.verdict == UNVERIFIED and v.checks["truth"] == UNVERIFIED


def test_claim_without_evidence_is_unverified(V):
    # Hooly Calendar has an active_users attribute in the lexicon but no evidence for it
    form = cmp_form("Hooly Calendar", "active_users", "gt", "Hooly Drive", "active_users")
    c = claim("c1", "compare", "Hooly Calendar has more active users than Hooly Drive.", form, AMOUNT)
    v = V.check(c, AMOUNT)
    assert v.checks["contract"] == PASS and v.checks["truth"] == UNVERIFIED and v.verdict == UNVERIFIED


def test_precise_date_needs_current_support():
    lex = {"subjects": [{"name": "Zeta Alpha Plan", "aliases": [], "attributes": ["launched"]}], "events": []}
    d = lambda v: {"kind": "date", "value": v}  # noqa: E731
    recs = [{"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-01-20", "value": d("2026-01-12")},
            {"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-03-01", "value": d("2026-02-03")},
            {"subject": "Zeta Alpha Plan", "attribute": "launched", "date": "2026-09-01", "value": d("2026-01")}]
    v = ClaimValidator(FactInventory.from_records(recs, lex))
    assert v.truth("fact", {"subject": "Zeta Alpha Plan", "attribute": "launched", "value": d("2026-01-12")})[0] == UNVERIFIED
    assert v.truth("fact", {"subject": "Zeta Alpha Plan", "attribute": "launched", "value": d("2026-01")})[0] == PASS


def test_relation_needs_overlapping_evidence():
    lex = {"subjects": [{"name": "Zeta Co", "aliases": [], "attributes": ["cto"]},
                        {"name": "Ana Zed", "aliases": [], "attributes": ["joined"]}], "events": []}
    name, d = (lambda v: {"kind": "name", "value": v}), (lambda v: {"kind": "date", "value": v})
    recs = [{"subject": "Zeta Co", "attribute": "cto", "date": "2026-06-01", "value": name("Ana Zed")},
            {"subject": "Zeta Co", "attribute": "cto", "date": "2026-07-01", "value": name("Ana Zed")},
            {"subject": "Ana Zed", "attribute": "joined", "date": "2026-09-01", "value": d("2026-05-01")}]
    form = {"subject": "Zeta Co", "link": "cto", "target": "Ana Zed", "attribute": "joined", "value": d("2026-05-01")}
    assert ClaimValidator(FactInventory.from_records(recs, lex)).truth("relation", form)[0] == UNVERIFIED
    recs.append({"subject": "Ana Zed", "attribute": "joined", "date": "2026-05-15", "value": d("2026-05-01")})
    assert ClaimValidator(FactInventory.from_records(recs, lex)).truth("relation", form)[0] == PASS


def test_malformed_form_is_a_counted_failure(V):
    before = V.errors
    st, why, _ = V.truth("fact", {"subject": "Hooly"})  # no attribute, no value
    assert st == FAIL and V.errors == before + 1


def test_text_must_say_what_the_form_says(V):
    # the form says lt, the text says "more": the contract check catches it
    form = cmp_form("Hooly Search", "monthly_active_users", "lt", "Hooly Chrome", "monthly_active_users")
    v = V.check(claim("c1", "compare", "Hooly Search has more monthly active users than Hooly Chrome.", form, AMOUNT), AMOUNT)
    assert v.checks["contract"] == FAIL


def test_selected_thousand_are_all_pass_and_unique():
    from app.generate import generate

    rows = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    texts = {norm_text(r["text"]) for r in rows}
    report = {}
    claims = generate(KB(lex).build([], rows), lex, texts, 1000, report)
    assert len(claims) == 1000
    vrep = report["_validation_report"]
    assert vrep["selected"]["verdicts"] == {"PASS": 1000}
    keys = [tuple(c["semantic_key"]) for c in vrep["claims"]]
    assert len(set(keys)) == 1000
    assert report["selected_semantic_duplicates"] == 0
    assert report["semantic_duplicates_inverse_compare"] > 0  # inverse compares were detected and dropped
    assert all(c.meta["validation"]["evidence"] for c in claims)


def test_fact_tying_a_value_to_its_unnamed_subject_is_new(V):
    # p-1798 "A team of 4 recruiters reads your file." never names Hooly Hiring Programme
    form = {"subject": "Hooly Hiring Programme", "attribute": "recruiters",
            "value": {"kind": "quantity", "value": 4, "unit": "recruiter"}}
    v = V.check(claim("f2", "fact", "Hooly Hiring Programme has 4 recruiters.", form))
    assert v.checks["novelty"] == PASS, v.reasons


def test_fact_whose_sentence_names_subject_and_value_is_a_reformulation(V):
    # "Hooly Search holds 301,000,000,000 pages." names the subject and states the value
    form = {"subject": "Hooly Search", "attribute": "index_size",
            "value": {"kind": "quantity", "value": 301000000000, "unit": "page"}}
    v = V.check(claim("f3", "fact", "The index size of Hooly Search is 301,000,000,000 pages.", form))
    assert v.checks["novelty"] == FAIL

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


def test_event_only_names_count_as_named():
    lex = {"subjects": [{"name": "Zeta Plan", "aliases": [], "attributes": ["price"]}], "events": ["Zeta Summit"]}
    recs = [{"subject": "Zeta Plan", "attribute": "price", "date": "2026-05-01", "row_id": "r1",
             "text": "At Zeta Summit, Zeta Plan cost €9 per month.",
             "value": {"kind": "money", "value": 9, "currency": "EUR", "per": "month"}}]
    v = ClaimValidator(FactInventory.from_records(recs, lex))
    assert v._named_in("r1") >= {"Zeta Plan", "Zeta Summit"}


# ---- every input sentence, labelled or not, read as text (no extractor needed) -------------------
ZLEX = {"subjects": [{"name": "Zeta Drone", "aliases": ["the Drone"], "attributes": ["propellers", "max_speed", "trial", "launched", "cpu_clock", "night_mode"]},
                     {"name": "Zeta Plan", "aliases": [], "attributes": ["price"]},
                     {"name": "Zeta Hub", "aliases": [], "attributes": ["price"]}], "events": []}
Q = lambda v, u: {"kind": "quantity", "value": v, "unit": u}  # noqa: E731


def _gold(rid, text, date, *facts):
    return {"id": rid, "text": text, "date": date, "label": "true",
            "form": [{"subject": s, "attribute": a, "value": v} for s, a, v in facts]}


def zeta(*universe_texts):
    """Practice gold that states facts without naming the subject, plus unlabelled universe rows."""
    gold = [_gold("g1", "It flies on 12 propellers.", "2026-09-01", ("Zeta Drone", "propellers", Q(12, "propeller"))),
            _gold("g2", "Its chip runs at 3.2 GHz.", "2026-09-01", ("Zeta Drone", "cpu_clock", Q(3.2, "GHz"))),
            _gold("g3", "A free trial of 14 days comes with it.", "2026-09-01",
                  ("Zeta Drone", "trial", {"kind": "duration", "value": 14, "unit": "day"})),
            _gold("g4", "It went on sale on 12 March 2026.", "2026-09-01", ("Zeta Drone", "launched", {"kind": "date", "value": "2026-03-12"})),
            _gold("g5", "Price: CHF 12 a month.", "2026-09-01", ("Zeta Plan", "price", {"kind": "money", "value": 12, "currency": "CHF", "per": "month"}))]
    universe = [{"id": f"u{i}", "text": t, "date": "2026-10-01", "label": ""} for i, t in enumerate(universe_texts)]
    return ClaimValidator(FactInventory.from_kb(KB(ZLEX).build([], gold, universe)))


@pytest.mark.parametrize("form,unlabelled", [
    ({"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}, "Zeta Drone flies on twelve propellers."),
    ({"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}, "The Drone has a dozen props."),
    ({"subject": "Zeta Drone", "attribute": "cpu_clock", "value": Q(3.2, "GHz")}, "Zeta Drone: 3,200 MHz chip."),
    ({"subject": "Zeta Drone", "attribute": "trial", "value": {"kind": "duration", "value": 14, "unit": "day"}},
     "Zeta Drone comes with a fortnight free."),
    ({"subject": "Zeta Drone", "attribute": "launched", "value": {"kind": "date", "value": "2026-03-12"}},
     "Zeta Drone launched March 12, 2026."),
    ({"subject": "Zeta Plan", "attribute": "price", "value": {"kind": "money", "value": 12, "currency": "CHF", "per": "month"}},
     "Zeta Plan is CHF 12 monthly."),
])
def test_unlabelled_sentence_in_any_form_restates_a_fact(form, unlabelled):
    """Nouns, units and currencies that never appear in practice still match (no overfitting)."""
    assert zeta().text_restates("fact", form)[0] is None  # gold sentences never name the subject
    assert zeta(unlabelled).text_restates("fact", form)[0] == "strong", unlabelled


def test_same_value_for_another_subject_is_not_a_restatement():
    form = {"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}
    assert zeta("Zeta Hub ships with 12 cables.").text_restates("fact", form)[0] is None


def test_subject_far_from_the_value_is_only_a_weak_match():
    form = {"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}
    far = "Zeta Drone was shown at the fair, and after a long talk about safety, batteries and the new app, we counted 12 of them."
    v = zeta(far)
    assert v.text_restates("fact", form)[0] == "weak"
    c = claim("f9", "fact", "Zeta Drone has 12 propellers.", form)
    verdict = v.check(c)
    assert verdict.verdict == PASS and "novelty_weak" in verdict.flags


def test_compare_said_by_an_unlabelled_sentence_fails():
    lex_v = zeta("Zeta Plan costs less than Zeta Hub.")
    form = cmp_form("Zeta Plan", "price", "lt", "Zeta Hub", "price")
    assert lex_v.text_restates("compare", form)[0] == "strong"


def test_fallback_ladder_never_lets_fail_or_duplicates_in():
    from app.generate.select import select

    from app.generate.candidates import Candidate

    def cand(i, conf, weak=False):
        c = Candidate("change", f"t{i}", {"subject": f"s{i}", "attribute": "a"}, conf, ("k", i))
        c.meta = {"vkey": ("k", i), **({"novelty_weak": True} if weak else {})}
        return c

    pool = [cand(4, 0.6, weak=True), cand(3, 0.6), cand(2, 0.97, weak=True), cand(1, 0.97)]  # worst first
    report = {}
    chosen = select(pool + [cand(1, 0.5)], 4, report)  # the second ("k", 1) is a duplicate
    assert [c.text for c in chosen] == ["t1", "t2", "t3", "t4"]  # ladder order, whatever the input order
    assert report["fallback"]["selected_novelty_weak"] == 2 and report["semantic_duplicates"] == 1
    report = {}
    assert len(select(pool, 10, report)) == 4 and report["shortfall"] == 6


@pytest.mark.parametrize("text,key", [
    ("three hundred engineers", ("number", 300.0)), ("two thousand users", ("number", 2000.0)),
    ("for a week", ("time", 604800.0)), ("in a decade", ("calendar", 120.0)), ("usd 12", ("money", "USD", 12.0)),
    ("Jul 12, 2026", ("date", "2026-07-12")), ("12 users", ("number", 12.0)),
])
def test_value_forms_are_indexed(text, key):
    from app.validate.textindex import value_keys

    assert key in {k for k, _ in value_keys(text)}


@pytest.mark.parametrize("text", ["at least 12 users", "about 12 users", "12+ users", "over 12 users"])
def test_qualified_values_are_other_values(text):
    from app.validate.textindex import value_keys

    assert ("number", 12.0) not in {k for k, _ in value_keys(text)}


def test_strong_match_fails_novelty_through_check():
    v = zeta("Zeta Drone flies on twelve propellers.")
    form = {"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}
    verdict = v.check(claim("f10", "fact", "Zeta Drone has 12 propellers.", form))
    assert verdict.verdict == FAIL and verdict.checks["novelty"] == FAIL


def test_subject_and_value_in_different_sentences_is_not_strong():
    form = {"subject": "Zeta Drone", "attribute": "propellers", "value": Q(12, "propeller")}
    assert zeta("Meet Zeta Drone. It has 12 propellers.").text_restates("fact", form)[0] != "strong"


def test_yes_no_sentence_with_the_opposite_polarity_is_not_a_restatement():
    yes = {"subject": "Zeta Drone", "attribute": "night_mode", "value": {"kind": "boolean", "value": True}}
    assert zeta("Zeta Drone has no night mode.").text_restates("fact", yes)[0] is None
    assert zeta("Zeta Drone has night mode.").text_restates("fact", yes)[0] == "strong"


def _fake(i, t, conf, **form):
    from app.generate.candidates import Candidate

    c = Candidate(t, f"{t}{i}", {"subject": form.get("subject", f"s{i}"), "attribute": form.get("attribute", "a")},
                  conf, (t, i))
    c.meta = {"vkey": (t, i)}
    return c


def test_every_type_with_a_pass_candidate_is_represented():
    from app.generate.select import select

    pool = [_fake(i, "change", 0.98) for i in range(10)] + [_fake(99, "relation", 0.7)]
    report = {}
    chosen = select(pool, 10, report)
    assert len(chosen) == 10 and any(c.type == "relation" for c in chosen)  # low confidence, but the only one
    assert report["missing_types"] == ["fact", "compare", "at_event"]  # none exist: nothing forced


def test_at_event_kept_to_two_per_fact():
    from app.generate.select import select

    pool = [_fake(i, "at_event", 0.9, subject="Zeta", attribute="eu_hosted") for i in range(5)] + \
           [_fake(10 + i, "change", 0.88) for i in range(5)]
    chosen = select(pool, 7, {})
    assert sum(c.type == "at_event" for c in chosen) == 2


def test_yes_no_at_event_reads_naturally_and_validates():
    from app.generate.templates import at_event_text

    rows = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    v = ClaimValidator(FactInventory.from_kb(KB(lex).build([], rows)))
    form = {"subject": "Hooly Gemini", "attribute": "eu_hosted", "event": "Hooly Cloud Next 2026",
            "value": {"kind": "boolean", "value": True}}
    text = at_event_text("Hooly Gemini", "eu_hosted", "Hooly Cloud Next 2026", form["value"])
    assert text == "Hooly Gemini was EU hosted at Hooly Cloud Next 2026."
    assert v.check(claim("a1", "at_event", text, form)).checks["contract"] == PASS

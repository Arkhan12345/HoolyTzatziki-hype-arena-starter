import glob
import json
import os

import pytest

from app.io.reader import read_lexicon
from app.io.types import Claim
from app.validate.rules import validate_claim

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEX = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
EX = os.path.join(ROOT, "contracts", "examples", "entry")


def _claim(entry: dict, **meta) -> Claim:
    c = Claim(id=entry.get("id", "x"), type=entry.get("type"), text=entry.get("text", ""),
              form=entry.get("form") if isinstance(entry.get("form"), dict) else {"_": entry.get("form")},
              confidence=entry.get("confidence", 0.9) if isinstance(entry.get("confidence"), (int, float)) else 0.9)
    c.meta = meta
    return c


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(EX, "good", "*.json"))), ids=os.path.basename)
def test_good_examples_pass(path):
    entry = _load(path)
    between = "date" if "dates" in path else "amount"
    assert validate_claim(_claim(entry, between=between), LEX, set()) == []


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(EX, "bad", "*.json"))), ids=os.path.basename)
def test_bad_examples_fail(path):
    entry = _load(path)
    extra = set(entry) - {"id", "type", "text", "form", "tag", "confidence"}
    if extra or entry.get("tag") not in (None, "true") or "confidence" not in entry or "text" not in entry:
        pytest.skip("rejected by the writer/schema outside validate_claim (tag, confidence or text missing)")
    assert validate_claim(_claim(entry), LEX, set()) != []


PRICE = {"subject": "Hooly AI Pro", "attribute": "price",
         "value": {"kind": "money", "value": 59, "currency": "EUR", "per": "month"}}
SOC2 = {"subject": "Hooly", "attribute": "soc2_certified", "value": {"kind": "boolean", "value": False}}


@pytest.mark.parametrize("text,ok", [
    ("Hooly AI Pro costs €59 per month.", True),
    ("At €59 per month, Hooly AI Pro is our best seller.", True),
    ("Hooly AI Pro costs over €59 per month.", False),
    ("Hooly AI Pro costs about €59 per month.", False),
    ("Hooly AI Pro costs €59.", False),
    ("It costs €59 per month.", False),
    ("Hooly AI Pro is our plan. It costs €59 per month.", False),
    ("Hooly costs €59 per month.", False),
    ("Hooly AI Pro costs €59 per month, and never more.", False),
    ("Hooly AI Pro costs €59 per month, up from €49 per month.", False),
    ("In March 2026, Hooly AI Pro cost €59 per month.", False),
    ("In April, Hooly AI Pro cost €59 per month.", False),
    ("Hooly AI Pro used to cost €59 per month.", False),
])
def test_fact_table_price(text, ok):
    assert (validate_claim(Claim("e1", "fact", text, PRICE, 0.9), LEX, set()) == []) == ok


@pytest.mark.parametrize("text,ok", [
    ("Hooly is not SOC 2 certified.", True),
    ("Hooly has never been SOC2-certified.", True),
    ("Hooly is fully SOC 2 certified.", False),
    ("Hooly has never passed a SOC 2 audit.", False),
    ("Hooly is not profitable.", False),
])
def test_fact_table_yes_no(text, ok):
    assert (validate_claim(Claim("e1", "fact", text, SOC2, 0.9), LEX, set()) == []) == ok


CMP = {"left": {"subject": "Hooly Workspace", "attribute": "price"},
       "right": {"subject": "Hooly AI Pro", "attribute": "price"}, "op": "gt"}


@pytest.mark.parametrize("text,op,ok", [
    ("Hooly Workspace costs more than Hooly AI Pro.", "gt", True),
    ("Hooly Workspace costs the same as Hooly AI Pro.", "eq", True),
    ("Hooly Workspace costs more than Hooly AI Pro.", "lt", False),
    ("Hooly AI Pro costs less than Hooly Workspace.", "lt", False),
    ("Hooly Workspace costs more than its rivals.", "gt", False),
    ("Hooly Workspace outprices Hooly AI Pro.", "gt", False),
    ("Hooly Workspace does not cost more than Hooly AI Pro.", "gt", False),
    ("Hooly Workspace costs more than Hooly AI Pro, and less than Hooly AI Free.", "gt", False),
    ("Hooly Workspace costs more than Hooly.", "gt", False),
    ("Hooly Workspace comes after Hooly AI Pro.", "gt", False),
    ("Hooly Workspace costs more per month than Hooly AI Pro.", "gt", True),
    ("Hooly Workspace costs €100 more than Hooly AI Pro.", "gt", False),
    ("Hooly Workspace costs 100 more than Hooly AI Pro.", "gt", False),
    ("Hooly Workspace costs a hundred more than Hooly AI Pro.", "gt", False),
    ("In June, Hooly Workspace cost more than Hooly AI Pro.", "gt", False),
    ("Hooly Workspace used to cost more than Hooly AI Pro.", "gt", False),
    ("Hooly Workspace cost more than Hooly AI Pro before the outage.", "gt", False),
])
def test_compare_table(text, op, ok):
    c = Claim("e1", "compare", text, dict(CMP, op=op), 0.9)
    c.meta = {"between": "amount"}
    assert (validate_claim(c, LEX, set()) == []) == ok


CHANGE = {"subject": "Hooly AI Pro", "attribute": "price",
          "from": {"kind": "money", "value": 49, "currency": "EUR", "per": "month"},
          "to": {"kind": "money", "value": 59, "currency": "EUR", "per": "month"}}


@pytest.mark.parametrize("text,ok", [
    ("Hooly AI Pro went from €49 per month to €59 per month.", True),
    ("Hooly AI Pro went from €59 per month to €49 per month.", False),
    ("Hooly AI Pro went from €49 to €59 per month.", False),
    ("Hooly AI Pro cost €49 per month and now costs €59 per month.", False),
    ("Hooly AI Pro never went from €49 per month to €59 per month.", False),
    ("Hooly AI Pro went down from €49 per month to €59 per month.", False),
])
def test_change_table(text, ok):
    assert (validate_claim(Claim("e1", "change", text, CHANGE, 0.9), LEX, set()) == []) == ok


def test_copy_of_input_is_rejected():
    c = Claim("e1", "fact", "Hooly AI Pro costs €59 per month.", PRICE, 0.9)
    assert "copy of an input row" in validate_claim(c, LEX, {"hooly ai pro costs €59 per month"})


AT_EVENT = {"subject": "Hooly Gemini", "attribute": "languages", "event": "Hooly I/O 2026",
            "value": {"kind": "quantity", "value": 12, "unit": "language"}}


@pytest.mark.parametrize("text,ok", [
    ("Hooly Gemini spoke 12 languages at Hooly I/O 2026.", True),
    # the referee also reads "a dozen" and a bare alias here; our validator only knows our own rendering
    pytest.param("At Hooly I/O, Hooly Gemini spoke a dozen languages.", True, marks=pytest.mark.xfail(strict=True)),
    ("Hooly Gemini supported 12 languages on the day of Hooly I/O 2026.", True),
    ("Hooly Gemini spoke 12 languages at the time of Hooly I/O 2026.", True),
    ("Hooly Gemini spoke 12 languages before Hooly I/O 2026.", False),
    ("Hooly Gemini spoke 12 languages in March.", False),
    ("At Hooly I/O 2026, Hooly spoke 12 languages.", False),
    ("Hooly Gemini spoke 12 languages at Hooly I/O 2026, up from 6 languages.", False),
    ("In March 2026, at Hooly I/O 2026, Hooly Gemini spoke 12 languages.", False),
])
def test_at_event_table(text, ok):
    assert (validate_claim(Claim("e4", "at_event", text, AT_EVENT, 0.9), LEX, set()) == []) == ok


RELATION = {"subject": "Hooly", "link": "cto", "target": "Bertram Gilfoil", "attribute": "joined",
            "value": {"kind": "date", "value": "2026-07-01"}}


@pytest.mark.parametrize("text,ok", [
    ("The CTO of Hooly joined on 1 July 2026.", True),
    # ISO dates read for the referee; we never write them, so the validator rejects them (safe side)
    pytest.param("Hooly's CTO started on 2026-07-01.", True, marks=pytest.mark.xfail(strict=True)),
    ("Hooly's chief technology officer joined on 1 July 2026.", False),
    ("Bertram Gilfoil joined Hooly on 1 July 2026.", False),
    ("The CTO joined on 1 July 2026.", False),
    ("The CTO of Hooly, who left on 3 May 2027, joined on 1 July 2026.", False),
    ("The CTO of Hooly joined on 1 July 2026, after Richard Hendrix.", False),
])
def test_relation_table(text, ok):
    assert (validate_claim(Claim("e5", "relation", text, RELATION, 0.9), LEX, set()) == []) == ok


DATES = {"left": {"subject": "Hooly I/O 2026", "attribute": "date"},
         "right": {"subject": "The May 2026 outage", "attribute": "date"}}


@pytest.mark.parametrize("text,op,ok", [
    ("Hooly I/O 2026 took place before the May 2026 outage.", "lt", True),
    ("Hooly I/O 2026 came earlier than the outage.", "lt", True),
    ("Hooly I/O took place on the same day as the outage.", "eq", True),
    ("Hooly I/O 2026 was more than the May 2026 outage.", "gt", False),
    ("Hooly I/O 2026 took place two weeks before the May 2026 outage.", "lt", False),
    ("Hooly I/O 2026 took place before the outage, in March 2026.", "lt", False),
])
def test_compare_dates_table(text, op, ok):
    c = Claim("e6", "compare", text, dict(DATES, op=op), 0.9)
    c.meta = {"between": "date"}
    assert (validate_claim(c, LEX, set()) == []) == ok


@pytest.mark.parametrize("name,ok", [("Bertram Gilfoil", True), ("bertram gilfoil", True), ("Lisbon", False)])
def test_name_values_must_resolve_in_the_lexicon(name, ok):
    form = {"subject": "Hooly", "attribute": "cto", "value": {"kind": "name", "value": name}}
    c = Claim("e7", "fact", f"The cto of Hooly is {name}.", form, 0.9)
    assert (validate_claim(c, LEX, set()) == []) == ok


USERS = {"subject": "Hooly Drive", "attribute": "active_users", "value": {"kind": "quantity", "value": 10, "unit": "user"}}
MAU = {"subject": "Hooly Maps", "attribute": "monthly_active_users",
       "value": {"kind": "quantity", "value": 118000000, "unit": "user"}}


@pytest.mark.parametrize("form,text,ok", [
    (MAU, "Hooly Maps has 118,000,000 monthly active users.", True),  # two words of the attribute between
    (USERS, "Hooly Drive has 10 active users.", True),
    (USERS, "Hooly Drive has 10 million users.", False),          # a scale word is not an attribute word
    (USERS, "Hooly Drive has 10 more than users.", False),        # nor a comparison
    (USERS, "Hooly Drive has 110 users.", False),                 # 10 inside a larger number
    (USERS, "Hooly Drive has 1.10 users.", False),
])
def test_counted_noun_words_are_the_attributes_own(form, text, ok):
    assert (validate_claim(Claim("e8", "fact", text, form, 0.9), LEX, set()) == []) == ok


def test_generated_yes_no_wordings_validate():
    from app.generate.templates import fact_text

    for attribute, subject in (("soc2_certified", "Hooly"), ("ai_sidebar", "Hooly Chrome"), ("raised_series_a", "Hooly")):
        for yes in (True, False):
            form = {"subject": subject, "attribute": attribute, "value": {"kind": "boolean", "value": yes}}
            text = fact_text(subject, attribute, form["value"])
            assert validate_claim(Claim("e9", "fact", text, form, 0.9), LEX, set()) == [], text

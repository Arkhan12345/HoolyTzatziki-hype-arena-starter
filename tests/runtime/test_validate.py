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

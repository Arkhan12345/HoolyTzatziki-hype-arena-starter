"""Steps for features/extraction.feature (eda/extract_simple.py against contracts/value-forms.md)."""
import importlib

from pytest_bdd import given, parsers, scenarios, then, when

scenarios("extraction.feature")


@given("the examples of contracts/value-forms.md", target_fixture="ctx")
def examples():
    tv = importlib.import_module("test_values_simple")
    return {"tv": tv, "examples": tv.examples()}


@when("the simple parser reads each example")
def read_all(ctx):
    tv, fails = ctx["tv"], []
    for section, written, expected in ctx["examples"]:
        got = tv.reads(written, expected)
        ok = (not got) if section == "Not read" else got
        if not ok:
            fails.append((section, written, expected))
    ctx["fails"] = fails


@then("every example passes")
def all_pass(ctx):
    assert not ctx["fails"], ctx["fails"][:5]


@then(parsers.parse("there are at least {n:d} examples"))
def enough_examples(ctx, n):
    assert len(ctx["examples"]) >= n


@when(parsers.parse('the simple parser reads "{text}"'), target_fixture="ctx")
def read_one(text):
    from extract_simple import parse_values
    return {"values": [v for _, _, v in parse_values(text)]}


@then(parsers.parse("it finds a money value of {amount:d} {cur} per {per}"))
def finds_money(ctx, amount, cur, per):
    money = [v for v in ctx["values"] if v.get("kind") == "money"]
    assert any(float(v["value"]) == amount and v.get("currency") == cur and v.get("per") == per for v in money), \
        ctx["values"]

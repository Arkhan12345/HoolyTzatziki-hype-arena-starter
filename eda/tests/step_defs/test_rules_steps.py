"""Steps for features/rules.feature."""
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("rules.feature")


@given("the practice rows with their research features", target_fixture="ctx")
def practice_rows(features):
    return {"R": features}


@when("I select the rows whose worst fact repeats an older value")
def select_stale(ctx):
    ctx["sel"] = ctx["R"][ctx["R"]["rel_stale_older"] == 1]


@when("I select the rows whose pairs have one value in the corpus")
def select_single(ctx):
    ctx["sel"] = ctx["R"][ctx["R"]["pair_n_values_max"] == 1]


@when("I select the rows with a joke tail")
def select_joke(ctx):
    ctx["sel"] = ctx["R"][ctx["R"]["joke_tail"] == 1]


@then(parsers.parse("at least {n:d} rows are selected"))
def at_least_rows(ctx, n):
    assert len(ctx["sel"]) >= n, f"only {len(ctx['sel'])} rows"


@then(parsers.parse("at least {pct:d} percent of them are labelled false"))
def mostly_false(ctx, pct):
    rate = (ctx["sel"]["label"] == "false").mean()
    assert rate >= pct / 100, f"false rate {rate:.3f}"


@then("none of them is labelled false")
def none_false(ctx):
    assert (ctx["sel"]["label"] == "false").sum() == 0


@then(parsers.parse("their unsure rate is at least {k:d} times the overall unsure rate"))
def unsure_lift(ctx, k):
    base = (ctx["R"]["label"] == "unsure").mean()
    rate = (ctx["sel"]["label"] == "unsure").mean()
    assert rate >= k * base, f"unsure rate {rate:.3f} vs base {base:.3f}"


@then(parsers.parse("there are {n:d} rows"))
def n_rows(ctx, n):
    assert len(ctx["R"]) == n


@then(parsers.parse("the false rate is between {lo:f} and {hi:f}"))
def false_rate(ctx, lo, hi):
    rate = (ctx["R"]["label"] == "false").mean()
    assert lo <= rate <= hi, rate

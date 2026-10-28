"""Steps for features/overfit.feature. Skips cleanly until eda/11_overfit.py has written its summary."""
import pandas as pd
import pytest
from pytest_bdd import given, parsers, scenarios, then

from conftest import OUT

scenarios("overfit.feature")


@given("the overfitting summary", target_fixture="summary")
def summary():
    p = OUT / "overfit_summary.csv"
    if not p.exists():
        pytest.skip("eda/out/overfit_summary.csv not created yet (run eda/11_overfit.py)")
    return pd.read_csv(p)


def _rows(summary, prefix):
    r = summary[summary.test.str.startswith(prefix)]
    if r.empty:
        pytest.skip(f"test '{prefix}' not in the summary yet")
    return r


@then(parsers.parse('the train-vs-test macro-F1 gap of "{model}" is at most {limit:f}'))
def train_gap(summary, model, limit):
    r = _rows(summary, "1 train-vs-test")
    v = r[r.model == model]["value"]
    assert not v.empty, f"no train-vs-test row for {model}"
    assert float(v.iloc[0]) <= limit, r


@then("the label-permutation test passes")
def permutation(summary):
    r = _rows(summary, "3 label permutation")
    assert (r.verdict == "PASS").all(), r


@then("the seed-stability test passes")
def seeds(summary):
    r = _rows(summary, "5 seed stability")
    assert (r.verdict == "PASS").all(), r


@then("every fold-scheme gain is positive")
def schemes(summary):
    r = _rows(summary, "7 gain vs L0")
    assert (r["value"].astype(float) > 0).all(), r

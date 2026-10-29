"""Steps for features/rules_round.feature. Skips cleanly until the rules round has produced its files."""
import hashlib
import json

import pandas as pd
import pytest
from pytest_bdd import given, scenarios, then

from conftest import EDA

scenarios("rules_round.feature")
RR = EDA / "selfimprove" / "rules_round"
GOAL = EDA / "selfimprove" / "goal.json"
TRUE = {"true", "1", "yes"}


def _flag(series):
    return series.astype(str).str.lower().isin(TRUE)


@given("the rules-round state", target_fixture="state")
def state():
    p = RR / "state.json"
    if not p.exists():
        pytest.skip("rules round has not run yet")
    return json.loads(p.read_text())


@given("the rules-round trial log", target_fixture="trials")
def trials():
    p = RR / "trials.csv"
    if not p.exists():
        pytest.skip("rules round has not run yet")
    T = pd.read_csv(p)
    return T[T["knob"] != "base"]


@given("the rules-round pre-filter table", target_fixture="pre")
def prefilter_table():
    p = RR / "prefilter.csv"
    if not p.exists():
        pytest.skip("rules round pre-filter not written yet")
    return pd.read_csv(p)


@then("the goal hash recorded at start equals the current goal hash")
def goal_unchanged(state):
    assert state["goal_sha256"] == hashlib.sha256(GOAL.read_bytes()).hexdigest()


@then("every kept trial passed gates a, b, c and the fold_row5 check")
def kept_pass_gates(trials):
    kept = trials[_flag(trials["kept"])]
    for g in ("gate_a_nested", "gate_b_gap", "gate_c_seeds", "gate_d_row5"):
        assert _flag(kept[g]).all(), kept[["knob", g]]


@then("every kept trial met the accept rule with a gain of at least 0.005 on at least 4 of 5 folds")
def kept_accept(trials):
    kept = trials[_flag(trials["kept"])]
    assert _flag(kept["accept_rule"]).all()
    assert (pd.to_numeric(kept["mean_gain"]) >= 0.005 - 1e-9).all()
    assert (pd.to_numeric(kept["folds_better"]) >= 4).all()


@then("no trial that failed a gate is marked kept")
def failed_not_kept(trials):
    failed = ~(_flag(trials["gate_a_nested"]) & _flag(trials["gate_b_gap"]) & _flag(trials["gate_c_seeds"])
               & _flag(trials["gate_d_row5"]))
    assert not _flag(trials.loc[failed, "kept"]).any()


@then("every rule turned into a knob has support of at least 20 and lift at least 1.5 or at most 0.5 or a strong "
      "unsure rate")
def prefilter_ok(pre):
    k = pre[pre["verdict"] == "candidate -> knob"]
    assert (k["support"] >= 20).all() and (k["pages"] >= 10).all()
    strong = (k["lift_false"] >= 1.5) | (k["lift_false"] <= 0.5) | (k["p_unsure"] >= 3 * 0.021)
    assert strong.all(), k[["rule", "lift_false", "p_unsure"]]


@then("the permutation test scored at most 0.50 macro-F1")
def permutation_ok(state):
    """The LEARNED part (hard rules off) must collapse on shuffled labels. The two deterministic hard rules need no
    labels, so the full pipeline keeps a floor above all-true; it may not exceed that floor by more than 0.05."""
    perm = state.get("permutation")
    if not perm:
        pytest.skip("permutation test not run yet")
    if "max_learned_tf" not in perm:
        pytest.skip("permutation result predates the learned-part split; rerun loop_rules.py --refresh")
    assert perm["max_learned_tf"] <= 0.50, perm
    assert perm["max_excess_over_rules"] <= 0.05, perm

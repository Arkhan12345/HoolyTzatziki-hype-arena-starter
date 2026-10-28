"""Steps for features/selfimprove.feature. Skips cleanly until the loop has produced its files."""
import hashlib
import json

import pandas as pd
import pytest
from pytest_bdd import given, scenarios, then

from conftest import EDA

scenarios("selfimprove.feature")
SI = EDA / "selfimprove"


def _find(d, words):
    """First value in a (nested) dict whose key contains one of the words."""
    for k, v in d.items():
        if isinstance(v, dict):
            got = _find(v, words)
            if got is not None:
                return got
        elif any(w in k.lower() for w in words):
            return v
    return None


@given("the self-improvement goal file", target_fixture="ctx")
def goal_file():
    p = SI / "goal.json"
    if not p.exists():
        pytest.skip("eda/selfimprove/goal.json not created yet")
    raw = p.read_bytes()
    return {"goal": json.loads(raw), "sha": hashlib.sha256(raw).hexdigest()}


@given("the self-improvement trial log")
def trial_log(ctx):
    p = SI / "trials.csv"
    if not p.exists():
        pytest.skip("eda/selfimprove/trials.csv not created yet")
    ctx["T"] = pd.read_csv(p)


@then("its hash matches the hash the loop recorded at start")
def goal_unchanged(ctx):
    p = SI / "state.json"
    if not p.exists():
        pytest.skip("eda/selfimprove/state.json not created yet")
    recorded = _find(json.loads(p.read_text(encoding="utf-8")), ("sha", "hash"))
    if recorded is None:
        pytest.skip("state.json records no goal hash")
    assert recorded in (ctx["sha"], ctx["sha"][:len(str(recorded))]), (recorded, ctx["sha"])


@then("every kept trial meets the minimum gain and the folds-better rule")
def kept_trials_ok(ctx):
    T, goal = ctx["T"], ctx["goal"]
    kept_col = next((c for c in T.columns if c.lower() in ("kept", "accepted", "accept", "keep")), None)
    gain_col = next((c for c in T.columns if "gain" in c.lower()), None)
    better_col = next((c for c in T.columns if "better" in c.lower() and "fold" in c.lower()), None)
    if not (kept_col and gain_col and better_col):
        pytest.skip(f"trials.csv columns not recognised: {list(T.columns)}")
    min_gain = _find(goal, ("min_gain",))
    k = _find(goal, ("folds_better",))
    if min_gain is None or k is None:
        pytest.skip("goal.json has no min_gain / folds_better")
    kept = T[T[kept_col].astype(str).str.lower().isin(["true", "1", "yes", "kept", "accepted"])]
    # the base row writes "-" in these columns, so pandas reads them as text
    kept = kept.assign(**{gain_col: pd.to_numeric(kept[gain_col], errors="coerce"),
                          better_col: pd.to_numeric(kept[better_col], errors="coerce")})
    assert (kept[gain_col] >= float(min_gain) - 1e-9).all(), kept[[gain_col]]
    assert (kept[better_col] >= int(k)).all(), kept[[better_col]]

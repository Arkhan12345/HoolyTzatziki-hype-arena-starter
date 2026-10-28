"""Steps for features/rules_model.feature (G1-G5 against eda/improve/rules_model.py)."""
import numpy as np
import pandas as pd
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from app.kb.normalize import key as value_key
from improve import rules_model as rm

scenarios("rules_model.feature")

PRIOR = 0.146


def _vkey(n, unit="user"):
    return value_key({"kind": "quantity", "value": n, "unit": unit})


def _one_fact(stated, held):
    facts = pd.DataFrame([{"id": "r1", "vkey": _vkey(stated), "page": 1}])
    tl = pd.DataFrame([{"held_vkey": _vkey(held), "tl_agree": float(stated == held),
                        "tl_was_held_before": 0.0, "tl_held_later": 0.0}], index=facts.index)
    rows = pd.DataFrame([{"id": "r1", "label": "", "speaker": None, "hedge": 0}])
    return facts, rows, tl


@pytest.fixture(scope="module")
def oof():
    return rm.practice_oof_rule_features()


# ------------------------------------------------------------------ G1 synthetic
@given(parsers.parse("a fact stating {s:d} users while the timeline holds {h:d} users"), target_fixture="ctx")
def fact_vs_held(s, h):
    facts, rows, tl = _one_fact(s, h)
    return {"facts": facts, "rows": rows, "tl": tl}


@when("the rule features are computed")
def compute(ctx):
    ctx["out"] = rm.rule_features(ctx["facts"], ctx["rows"], ctx["tl"], ctx.get("train_ids", set()))


@then(parsers.parse("ratio_to_held is {r:f}"))
def ratio_is(ctx, r):
    assert ctx["out"].loc["r1", "ratio_to_held"] == pytest.approx(r)


@then(parsers.parse("perturb_ratio_hit is {h:d}"))
def hit_is(ctx, h):
    assert int(ctx["out"].loc["r1", "perturb_ratio_hit"]) == h


# ------------------------------------------------------------------ data checks
@given("the out-of-fold rule features for the practice CV rows", target_fixture="ctx")
def practice(oof):
    return {"oof": oof}


@then(parsers.parse("rows with {col} are at least {pct:d} percent false"))
def mostly_false(ctx, col, pct):
    d = ctx["oof"]
    hit = d[d[col] == 1]
    assert len(hit) >= 5, f"only {len(hit)} rows with {col}"
    rate = (hit["label"] == "false").mean()
    assert rate >= pct / 100, f"{col}: {rate:.3f} false over {len(hit)} rows"


# ------------------------------------------------------------------ G4
@when("the labels of one fold's test rows are flipped and the features rebuilt")
def flipped(ctx):
    ctx["fold"] = 0
    ctx["flip"] = rm.practice_oof_rule_features(flip_fold=0)


@then("the speaker trust of those test rows is unchanged")
def trust_unchanged(ctx):
    a, b = ctx["oof"], ctx["flip"]
    ids = a.index[a["fold"] == ctx["fold"]]
    pd.testing.assert_series_equal(a.loc[ids, "speaker_trust"], b.loc[ids, "speaker_trust"])
    for c in ["ratio_to_held", "perturb_ratio_hit", "hedge_unbacked"]:
        pd.testing.assert_series_equal(a.loc[ids, c], b.loc[ids, c])


@given("a speaker who appears once in the training rows", target_fixture="ctx")
def speaker_once():
    facts, rows, tl = _one_fact(100, 100)
    train = pd.DataFrame([{"id": "t1", "label": "false", "speaker": "Gavin", "hedge": 0}])
    rows = pd.concat([rows.assign(speaker="Gavin"), train], ignore_index=True)
    facts = pd.concat([facts, pd.DataFrame([{"id": "t1", "vkey": _vkey(5), "page": 2}])], ignore_index=True)
    tl = pd.concat([tl, pd.DataFrame([{"held_vkey": _vkey(5), "tl_agree": 1.0, "tl_was_held_before": 0.0,
                                       "tl_held_later": 0.0}])], ignore_index=True)
    return {"facts": facts, "rows": rows, "tl": tl, "train_ids": {"t1"}}


@then(parsers.parse("speaker_trust is the prior {p:f}"))
def trust_prior(ctx, p):
    assert ctx["out"].loc["r1", "speaker_trust"] == pytest.approx(p)


# ------------------------------------------------------------------ G2 / G5 synthetic
def _row_feats(**kw):
    base = {"joke_tail": 0, "n_facts": 1, "kv_line": 0, "hearsay": 0, "today_date_mismatch": 0,
            "pair_n_values_max": 3, "same_value_rows_min": 2, "tl_n_pages_for_max": 2}
    base.update(kw)
    return pd.DataFrame([base])


def _proba(label, p):
    rest = (1 - p) / 2
    v = {"false": [p, 1 - p - 0.01, 0.01], "true": [1 - p - 0.01, p, 0.01]}[label]
    return np.array([v])


@given(parsers.parse("a single-fact row with a joke tail that the model calls {lab} at {p:f}"), target_fixture="ctx")
def joke_row(lab, p):
    return {"feats": _row_feats(joke_tail=1), "proba": _proba(lab, p)}


@given(parsers.parse("a row about a one-value pair, never repeated, that the model calls {lab} at {p:f}"),
       target_fixture="ctx")
def single_value_row(lab, p):
    return {"feats": _row_feats(pair_n_values_max=1, same_value_rows_min=0, tl_n_pages_for_max=0),
            "proba": _proba(lab, p)}


@given(parsers.parse("a single-fact row with hearsay that the model calls {lab} at {p:f}"), target_fixture="ctx")
def hearsay_row(lab, p):
    return {"feats": _row_feats(hearsay=1), "proba": _proba(lab, p)}


@given(parsers.parse("a row with a today date mismatch that the model calls {lab} at {p:f}"), target_fixture="ctx")
def today_row(lab, p):
    return {"feats": _row_feats(today_date_mismatch=1), "proba": _proba(lab, p)}


KNOB = {"joke": "joke", "single-value": "single_value", "hearsay": "hearsay", "today": "today"}


@when(parsers.parse("the unsure layer runs with the {name} knob on"))
def layer_one(ctx, name):
    ctx["labels"], ctx["conf"] = rm.unsure_layer(ctx["proba"], ctx["feats"], {KNOB[name]: True})


@when(parsers.parse("the unsure layer runs with the {name} knob on and t {t:f}"))
def layer_one_t(ctx, name, t):
    ctx["labels"], ctx["conf"] = rm.unsure_layer(ctx["proba"], ctx["feats"], {KNOB[name]: True, "t": t})


@when("the unsure layer runs with every knob off")
def layer_off(ctx):
    ctx["labels"], ctx["conf"] = rm.unsure_layer(ctx["proba"], ctx["feats"], {})


@when("the unsure layer runs with every knob on over model probabilities that predict false for every gold-false row")
def layer_all_practice(ctx):
    d = ctx["oof"]
    p = np.tile([0.1, 0.85, 0.05], (len(d), 1))
    p[(d["label"] == "false").to_numpy()] = [0.6, 0.35, 0.05]
    ctx["labels"], _ = rm.unsure_layer(p, d, {"joke": True, "single_value": True, "hearsay": True,
                                              "today": True, "t": 0.7})


@then(parsers.parse("the row is labelled {lab}"))
def labelled(ctx, lab):
    assert rm.CLASSES[ctx["labels"][0]] == lab


@then(parsers.parse("the row is not labelled {lab}"))
def not_labelled(ctx, lab):
    assert rm.CLASSES[ctx["labels"][0]] != lab


@then("no gold-false row is labelled true")
def no_false_to_true(ctx):
    d = ctx["oof"]
    gold_false = (d["label"] == "false").to_numpy()
    assert not (rm.CLASSES[ctx["labels"][gold_false]] == "true").any()

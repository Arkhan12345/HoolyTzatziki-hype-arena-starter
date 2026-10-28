"""Steps for features/ladder.feature."""
import pickle

import pandas as pd
from pytest_bdd import given, parsers, scenarios, then
from sklearn.metrics import f1_score

from conftest import EDA, OUT

scenarios("ladder.feature")


def _row(df, variant, cv):
    r = df[(df["variant"] == variant) & (df["cv"] == cv)]
    assert len(r) == 1, f"{variant}/{cv} missing"
    return r.iloc[0]


@given("the ladder results", target_fixture="ctx")
def ladder_results(improve_labels):
    return {"L": improve_labels}


@given("the ladder out-of-fold predictions", target_fixture="ctx")
def ladder_oof(improve_labels):
    with open(OUT / "improve_oof.pkl", "rb") as f:
        oof = pickle.load(f)
    return {"L": improve_labels, "oof": oof, "S": pd.read_csv(EDA / "splits.csv").set_index("id")}


@given("the neural results")
def neural(ctx, improve_neuro):
    ctx["N"] = improve_neuro


@then(parsers.parse("the bake-off reproduction row has accuracy {acc:f} and macro-F1 {f1:f} within {tol:f}"))
def reproduces(ctx, acc, f1, tol):
    r = _row(ctx["L"], "L0_base(bakeoff GroupKFold all rows)", "groupkfold_all")
    assert abs(r["acc"] - acc) <= tol and abs(r["macro_f1_tf"] - f1) <= tol, (r["acc"], r["macro_f1_tf"])


def _per_fold(ctx, variant):
    d = ctx["oof"]["fold_group"][variant]
    df = pd.DataFrame({"y": d["y"], "p": d["pred"]}, index=d["ids"])
    df["fold"] = ctx["S"].loc[df.index, "fold_group"].values
    # labels are encoded as in eda/improve/base.CLASSES: 0 false, 1 true, 2 unsure
    return {k: f1_score(g["y"], g["p"], labels=[1, 0], average="macro", zero_division=0)
            for k, g in df.groupby("fold")}


@then(parsers.parse("L6_judge has a higher macro-F1 than L0_base on at least {k:d} of the 5 fold_group folds"))
def beats_on_folds(ctx, k):
    a, b = _per_fold(ctx, "L0_base"), _per_fold(ctx, "L6_judge")
    wins = sum(b[f] > a[f] for f in a)
    assert wins >= k, (a, b)


@then("L6_judge has a lower calibration error than L0_base on fold_group")
def better_ece(ctx):
    assert _row(ctx["L"], "L6_judge", "fold_group")["ece"] < _row(ctx["L"], "L0_base", "fold_group")["ece"]


@then("no neural row has a macro-F1 above the L6_judge fold_group macro-F1")
def neural_below(ctx):
    best = _row(ctx["L"], "L6_judge", "fold_group")["macro_f1_tf"]
    top = ctx["N"]["macro_f1_tf"].dropna().max()
    assert top <= best, (top, best)

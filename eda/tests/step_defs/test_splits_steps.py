"""Steps for features/splits.feature."""
import csv
import importlib.util

import pandas as pd
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("splits.feature")


@given("the shared split file", target_fixture="ctx")
def split_file(splits, root):
    practice = pd.read_csv(root / "data" / "practice.csv", encoding="utf-8-sig")
    return {"S": splits, "practice": practice, "root": root}


@then("it has one row per practice row and no duplicate ids")
def one_row_each(ctx):
    S, P = ctx["S"], ctx["practice"]
    assert len(S) == len(P)
    assert S["id"].is_unique
    assert set(S["id"]) == set(P["id"])


@then("every page of the cross-validation rows belongs to exactly one fold_group")
def page_one_fold(ctx):
    cv = ctx["S"][ctx["S"]["holdout"] == 0]
    per_page = cv.groupby("page")["fold_group"].nunique()
    assert (per_page == 1).all(), per_page[per_page > 1].index.tolist()
    assert set(cv["fold_group"]) == {0, 1, 2, 3, 4}


@then("no holdout page has a row with a fold_group other than -1")
def holdout_clean(ctx):
    S = ctx["S"]
    hold_pages = set(S.loc[S["holdout"] == 1, "page"])
    on_hold_pages = S[S["page"].isin(hold_pages)]
    assert (on_hold_pages["fold_group"] == -1).all()
    assert (on_hold_pages["holdout"] == 1).all()


@then(parsers.parse("the holdout holds between {lo:d} and {hi:d} percent of the pages"))
def holdout_share(ctx, lo, hi):
    S = ctx["S"]
    share = S.loc[S["holdout"] == 1, "page"].nunique() / S["page"].nunique()
    assert lo / 100 <= share <= hi / 100, share


@when("I run the arena's input builder on the practice data")
def run_input_builder(ctx, tmp_path):
    spec = importlib.util.spec_from_file_location("arena_input", ctx["root"] / "scripts" / "input.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "universe.csv"
    mod.main(str(ctx["root"] / "data" / "practice.csv"), str(out))
    with open(out, encoding="utf-8", newline="") as f:
        ctx["hidden"] = {r["id"] for r in csv.DictReader(f) if r["label"] == ""}


@then("the rows it hides are exactly the rows with fold_row5 equal to 0")
def hidden_match(ctx):
    S = ctx["S"]
    assert ctx["hidden"] == set(S.loc[S["fold_row5"] == 0, "id"])

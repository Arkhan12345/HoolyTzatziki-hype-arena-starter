"""Steps for features/timeline.feature (synthetic pairs against eda/improve/timeline.py)."""
import numpy as np
import pandas as pd
from pytest_bdd import given, parsers, scenarios, then, when

from improve.timeline import Timeline, anchors_for

scenarios("timeline.feature")

BASE = pd.Timestamp("2025-06-01")
STRONG = 5.0  # well above SWITCH, so the dynamic program alone would switch


def _row(rid, page, day, value, anchor, weight):
    return {"id": rid, "page": page, "date": BASE + pd.Timedelta(days=day), "vkey": value,
            "anchor": anchor, "weight": weight}


@given(parsers.parse('a pair anchored to value "{v}" on page {p:d} at day {d:d}'), target_fixture="ctx")
def anchored_pair(v, p, d):
    return {"rows": [_row("r0", p, d, v, 1, 1.0)]}


@given(parsers.parse('page {p:d} strongly votes for value "{v}" at day {d:d}'))
def strong_vote(ctx, p, v, d):
    ctx["rows"].append(_row(f"r{len(ctx['rows'])}", p, d, v, 0, STRONG))


@given(parsers.parse('a pair with an unlabelled row on page {p:d} at day {d:d} stating "{v}"'),
       target_fixture="ctx")
def unlabelled_pair(p, d, v):
    return {"rows": [_row("scored", p, d, v, 0, 1.0)], "scored": "scored"}


@given(parsers.parse('another row on page {p:d} at day {d:d} also stating "{v}"'))
def same_page_row(ctx, p, d, v):
    ctx["rows"].append(_row("same_page", p, d, v, 0, 1.0))


@given(parsers.parse('a row on page {p:d} at day {d:d} stating "{v}"'))
def other_page_row(ctx, p, d, v):
    ctx["rows"].append(_row(f"r{len(ctx['rows'])}", p, d, v, 0, 1.0))


@given("a pair whose rows are all labelled true", target_fixture="ctx")
def all_true_pair():
    rows = [_row(f"t{i}", i, 10 * i, "A", 0, 1.0) for i in range(3)]
    return {"rows": rows, "labels": {r["id"]: "true" for r in rows}}


def _build(ctx):
    df = pd.DataFrame(ctx["rows"])
    df["subject"], df["attribute"] = "Hooly Test", "price"
    tl = Timeline(df)
    return df, tl


@when("the timeline is built")
def build(ctx):
    df, tl = _build(ctx)
    anchor = df["anchor"].to_numpy()
    weight = df["weight"].to_numpy()
    feats, _ = tl.run(anchor, weight, np.ones(tl.n, dtype=bool))
    idx = tl.groups[0]
    days_u, vals_u, _, _, path = tl._path(idx, anchor, weight)
    names = pd.factorize(df["vkey"])[1]
    day0 = (BASE - pd.Timestamp("2025-01-01")).days
    ctx["held"] = {int(d) - day0: names[vals_u[path[t]]] for t, d in enumerate(days_u)}
    ctx["feats"], ctx["df"] = feats, df


@when("anchors are built with an empty training set")
def build_anchors(ctx):
    df, tl = _build(ctx)
    ctx["anchor"] = anchors_for(tl, set(), ctx["labels"], {i: 1 for i in ctx["labels"]})


@then(parsers.parse('the value held at day {d:d} is "{v}"'))
def held_value(ctx, d, v):
    assert ctx["held"][d] == v, ctx["held"]


@then(parsers.parse("the scored row is supported by exactly {n:d} other page"))
def own_page_excluded(ctx, n):
    k = ctx["df"].index[ctx["df"]["id"] == ctx["scored"]][0]
    assert ctx["feats"].loc[k, "n_pages_for"] == n


@then("no fact is anchored")
def none_anchored(ctx):
    assert (ctx["anchor"] == 0).all()
